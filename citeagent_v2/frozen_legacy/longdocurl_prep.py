#!/usr/bin/env python3
"""LongDocURL 数据准备 —— 把每题的 30 页 images 窗口建成 CiteAgent 的 Document。

与 MMLongBench(data_prep.py)的差异 & 为什么单独写一个 builder:
1. 文档范围 = 每题 `images` 字段的 30 张页图窗口(对齐官方 eval_open_lvlms.py:
   官方把这 30 张一次性喂给模型,证据页一定落在窗口内)。**不是整篇 PDF**。
   同一 doc_no 在不同题里窗口不同,故 Document 必须「按题」构建,不能按 doc 缓存。
2. 页图/crop 命名和目录结构与 MMLong 不同:
   - 页图:  pdf_pngs/<bucket>/<doc_no[:4]>/<doc_no>_<page>.png     (page 真实 0-based)
   - crop:  visual_crops_stage3/<bucket>/<doc_no[:4]>/<doc_no>_<page>.png_b<bid>.jpg
   - OCR :  stage2_blocks_cache.jsonl,key=image_path(basename 反推 doc_no+page)
   - caption: crop_captions.jsonl,{elem_id:"p<page>_b<bid>", doc_id, page, block_id, caption}
3. 页号对齐:evidence_pages 是 1-based 全文档页号,窗口内页号是真实 0-based,
   两者差 offset=1(config.PAGE_INDEX_OFFSET),与 MMLong 口径一致。

复用策略:直接复用 data_prep.Document(它只依赖 self.pages/self._image_paths +
全局 crop-caption 缓存,与数据来源无关)。本模块:
- 把 stage2 OCR 建成 {(doc_no, page): blocks} 索引(全量 16230 页,一次性加载)。
- 把 LongDocURL 的 crop_captions 注入 data_prep._CROP_CAPTIONS(loader 见到已填充
  就直接返回,从而覆盖掉指向 mmlongbench 的默认路径)→ page_search_text 用对 caption。
- build_document(sample):按该题 30 页窗口逐页填充 Document,元素挂 LongDocURL 的 crop_path。
下游 retriever / locator / memory / pipeline 完全不用改。
"""
import os
import re
import json
import threading
from collections import defaultdict

try:
    from . import config
    from . import data_prep
except ImportError:
    import config
    import data_prep

# ---------- 路径根 ----------
_HERE = os.path.dirname(os.path.abspath(__file__))
LONGDOC_DIR = os.path.abspath(os.path.join(_HERE, "..", "data", "LongDocURL"))
PDF_PNGS_DIR = os.path.join(LONGDOC_DIR, "pdf_pngs")
CROP_DIR = os.path.join(LONGDOC_DIR, "visual_crops_stage3")
OCR_CACHE_PATH = os.path.join(LONGDOC_DIR, "stage2_blocks_cache.jsonl")
CROP_CAPTIONS_PATH = os.path.join(LONGDOC_DIR, "crop_captions.jsonl")
SAMPLES_PATH = os.path.join(LONGDOC_DIR, "LongDocURL_public.jsonl")

# 有 crop 子图的视觉本体 label(与 data_prep.CROP_LABELS 一致)
CROP_LABELS = ("image", "chart")

_PAGE_RE = re.compile(r"_(\d+)\.png$")


def _bucket(doc_no):
    """doc_no 前 4 位所属的千号桶目录名,如 '4026369' → '4000-4999'。"""
    hi = int(doc_no[:4])
    lo = hi // 1000 * 1000
    return f"{lo}-{lo + 999}"


def _parse_doc_page(image_path):
    """从 image_path(任意前缀)反推 (doc_no, page)。

    basename 形如 '4026369_40.png' → ('4026369', 40)。
    """
    base = os.path.basename(image_path)
    m = _PAGE_RE.search(base)
    if not m:
        raise ValueError(f"无法从 {image_path} 解析页号")
    doc_no = base[: base.rfind("_")]
    return doc_no, int(m.group(1))


# ---------- OCR 索引(全量,一次性) ----------
_OCR_INDEX = None  # {(doc_no:str, page:int): [blocks]}
# 并发评测(concurrency>1)时多个线程会同时首次触发惰性加载:
# 若不加锁,线程 A 刚执行完 `_OCR_INDEX = {}`(已非 None)、还没填充完,
# 线程 B 就看到非 None 直接返回半成品空索引 → build_document 拿到空 blocks →
# 整篇 30 页全空 token → BM25 建索引时 avgdl/idf 除零 ZeroDivisionError。
# 用锁 + "先建局部再原子赋值" 保证任何线程要么看到 None、要么看到完整索引。
_INIT_LOCK = threading.Lock()


def _load_ocr_index():
    global _OCR_INDEX
    if _OCR_INDEX is not None:
        return _OCR_INDEX
    with _INIT_LOCK:
        if _OCR_INDEX is not None:   # 双检:等锁期间别的线程可能已建好
            return _OCR_INDEX
        index = {}
        with open(OCR_CACHE_PATH) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                doc_no, page = _parse_doc_page(rec["image_path"])
                # 同 (doc,page) 若重复,保留首次(与 Document.add_page 去重语义一致)
                index.setdefault((doc_no, page), rec.get("blocks", []))
        _OCR_INDEX = index   # 原子赋值:填充完毕后才对外可见
    return _OCR_INDEX


def _inject_crop_captions():
    """把 LongDocURL 的 crop_captions 注入 data_prep 的全局缓存。

    data_prep._load_crop_captions() 见 _CROP_CAPTIONS 非 None 即直接返回,
    因此在这里预填充可覆盖掉它默认读 mmlongbench 路径的行为,让
    page_search_text 用到 LongDocURL 的图表 caption。
    同样用 _INIT_LOCK + 原子赋值,避免并发下别的线程看到半填充的 caption 缓存。
    """
    if data_prep._CROP_CAPTIONS is not None:
        return data_prep._CROP_CAPTIONS
    with _INIT_LOCK:
        if data_prep._CROP_CAPTIONS is not None:
            return data_prep._CROP_CAPTIONS
        caps = defaultdict(dict)
        if os.path.exists(CROP_CAPTIONS_PATH):
            with open(CROP_CAPTIONS_PATH) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        o = json.loads(line)
                        did = str(o["doc_id"])
                        cap = (o.get("caption") or "").strip()
                        if cap:
                            caps[did][o["elem_id"]] = cap
                    except Exception:
                        continue
        data_prep._CROP_CAPTIONS = caps   # 原子赋值
    return data_prep._CROP_CAPTIONS


def _crop_path(doc_no, page, block_id):
    """LongDocURL crop 路径(命名与 MMLong 不同,故不能用 Document.add_page 的默认拼法)。"""
    return os.path.join(CROP_DIR, _bucket(doc_no), doc_no[:4],
                        f"{doc_no}_{page}.png_b{block_id}.jpg")


def _page_image_path(doc_no, page):
    return os.path.join(PDF_PNGS_DIR, _bucket(doc_no), doc_no[:4],
                        f"{doc_no}_{page}.png")


def build_document(sample):
    """按一题的 30 页 images 窗口构建 Document(元素级,挂 LongDocURL crop)。

    - 文档范围 = sample["images"] 里那 30 张页图的真实页号(0-based)。
    - 每页元素来自 stage2 OCR;image/chart 元素若 crop 存在则挂 crop_path。
    - elem_id = p<page>_b<block_id>(与 data_prep 一致,crop_captions 用同一 key 命中)。
    - Document.doc_id = doc_no(crop_captions 也按 doc_no 建键)。
    直接手工填 Document 内部结构(pages/_image_paths),不走 add_page(其 crop 拼法是
    MMLong 命名)。下游 retriever/locator/memory 只读 Document 的公开方法,无需改动。
    """
    _inject_crop_captions()
    ocr = _load_ocr_index()

    doc_no = str(sample["doc_no"])
    doc = data_prep.Document(doc_no)

    # 解析窗口页号(真实 0-based),按题给定的 images 顺序
    window_pages = [_parse_doc_page(p)[1] for p in sample["images"]]

    for page in window_pages:
        if page in doc.pages:
            continue  # 窗口内理论不重复,防御性去重
        blocks = ocr.get((doc_no, page), [])
        img = _page_image_path(doc_no, page)
        doc._image_paths[page] = img
        for b in blocks:
            block_id = b["block_id"]
            elem = {
                "doc_id": doc_no,
                "page": page,
                "elem_id": f"p{page}_b{block_id}",
                "label": b.get("label", ""),
                "bbox": b.get("bbox", []),
                "content": b.get("content", ""),
            }
            if elem["label"] in CROP_LABELS:
                cp = _crop_path(doc_no, page, block_id)
                if os.path.exists(cp):
                    elem["crop_path"] = cp
            doc.pages[page].append(elem)
        # 保证窗口内空 OCR 的页也在 sorted_pages 里(Document 需要页存在才能 read/建索引)
        if not blocks:
            doc.pages[page]  # touch defaultdict 生成空列表

    return doc


def load_samples(jsonl_path=None):
    """读全部题目,补 _sample_index。"""
    jsonl_path = jsonl_path or SAMPLES_PATH
    samples = []
    with open(jsonl_path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            o["_sample_index"] = i
            samples.append(o)
    return samples


# ---------- 自测 / 冒烟 ----------
if __name__ == "__main__":
    import sys

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    samples = load_samples()
    print(f"总题数: {len(samples)}")
    for s in samples[:n]:
        doc = build_document(s)
        pages = doc.sorted_pages()
        # 页图存在性
        img0 = doc.page_image_path(pages[0])
        # evidence 对齐检查
        ev = s["evidence_pages"]
        ev = ev if isinstance(ev, list) else json.loads(ev)
        ev_internal = [e - config.PAGE_INDEX_OFFSET for e in ev]
        ev_in_window = all(p in doc.pages for p in ev_internal) if ev else True
        n_crop = sum(1 for p in pages for e in doc.get_page_elements(p)
                     if e.get("crop_path"))
        print(f"\n=== {s['question_id']} ===")
        print(f"  doc_no={doc.doc_id} 窗口页={pages[0]}~{pages[-1]}({doc.num_pages}页) "
              f"元素={doc.num_elements} crop={n_crop}")
        print(f"  首页图 {img0} 存在={os.path.exists(img0)}")
        print(f"  evidence(1-based)={ev} → 内部={ev_internal} 落在窗口内={ev_in_window}")
        print(f"  问题: {s['question'][:80]}")
