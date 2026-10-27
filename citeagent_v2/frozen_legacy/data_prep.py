#!/usr/bin/env python3
"""MMLongBench-Doc 数据准备。

职责:
1. 读取 OCR 结果 jsonl(每行=一页,含 image_path + blocks)。
2. 修正 image_path 前缀(他人目录 -> 本地 images 目录)。
3. 把「页 + block」展开为带全局锚点的元素级记录,补 doc_id/page/elem_id。
4. 按文档聚合成 {doc_id: Document},Document 提供页分组、outline、页图路径等。

其它模块通过 load_documents() / load_document(doc_id) 取用。
"""
import os
import re
import json
from collections import defaultdict

try:
    from . import config
except ImportError:
    import config

# 标题类标签,用于拼 outline
TITLE_LABELS = ("doc_title", "paragraph_title")
# outline 里可选纳入的图表标题
CAPTION_LABELS = ("figure_title",)
# caption 标签(图表标题由 OCR 标成 figure_title)
FIGURE_CAPTION_LABEL = "figure_title"
# 真正的图表标题都以 "Figure N / Table N / Chart N" 开头;
# Type 从 caption 前缀取(本体 label 不可靠,OCR 会把 figure 标成 table)。
# 过滤掉 OCR 误标的零碎 figure_title(如 "GPT-4V:"、"(b)"、"Prompt: ...")。
_FIGURE_CAPTION_RE = re.compile(
    r"^(Figure|Fig|Table|Chart)\.?\s*(\d+|[IVXLC]+)\b", re.IGNORECASE)
# caption 前缀词 -> 清单里显示的规范 Type
_FIGURE_TYPE_MAP = {"figure": "Figure", "fig": "Figure",
                    "table": "Table", "chart": "Chart"}
# 有 crop 子图的视觉本体 label(同事只 crop 了 image/chart,table 走 OCR HTML 文本)
CROP_LABELS = ("image", "chart")

_PAGE_RE = re.compile(r"page_(\d+)")


# ---------- crop caption 加载(BM25 检索 / Global Selector 共用) ----------
# 离线生成的 image/chart 描述(由 tools/gen_crop_captions.py 生成)。
# 结构:{doc_id: {elem_id: caption}}。图表本体 OCR content 为空,导致"答案只在
# 图里"的页 BM25 召回不到(Case 5);把 caption 并进检索文本以提召回。
# 不写回元素 content；各调用方按实验开关决定是否使用。
_CROP_CAPTIONS = None


def _load_crop_captions():
    """惰性加载 crop_captions.jsonl → {doc_id: {elem_id: caption}}。

    文件缺失或某行损坏都容错跳过(检索退回纯 OCR 文本,不崩)。
    """
    global _CROP_CAPTIONS
    if _CROP_CAPTIONS is not None:
        return _CROP_CAPTIONS
    _CROP_CAPTIONS = defaultdict(dict)
    path = getattr(config, "CROP_CAPTIONS_PATH", None)
    if not path or not os.path.exists(path):
        return _CROP_CAPTIONS
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
                did = normalize_doc_id(o["doc_id"])
                cap = (o.get("caption") or "").strip()
                if cap:
                    _CROP_CAPTIONS[did][o["elem_id"]] = cap
            except Exception:
                continue
    return _CROP_CAPTIONS


def normalize_doc_id(doc_id):
    """统一 doc_id:去掉 .pdf 后缀。

    samples.json 里 doc_id 形如 'PH_2016.06.08_Economy-Final.pdf',
    而 OCR 聚合的 doc_id(来自 image_path 目录名)不带 .pdf。
    统一以「不带 .pdf」为准,供各模块对齐。
    """
    if doc_id.endswith(".pdf"):
        return doc_id[:-4]
    return doc_id


def _parse_doc_and_page(image_path):
    """从 image_path 解析 (doc_id, page)。

    doc_id = 倒数第二级目录名;page = 文件名 page_N.png 里的 N。
    """
    parts = image_path.rstrip("/").split("/")
    doc_id = parts[-2]
    m = _PAGE_RE.search(parts[-1])
    if not m:
        raise ValueError(f"无法从 {image_path} 解析页号")
    return doc_id, int(m.group(1))


def _fix_image_path(image_path):
    """Resolve cached machine-specific paths against the local data root."""
    if os.path.exists(image_path):
        return image_path

    if config.OLD_IMAGE_PREFIX and image_path.startswith(config.OLD_IMAGE_PREFIX):
        rewritten = config.NEW_IMAGE_PREFIX + image_path[len(config.OLD_IMAGE_PREFIX):]
        if os.path.exists(rewritten):
            return rewritten

    # OCR records identify a page by the final ``<doc_id>/page_N.png`` pair.
    # Reconstructing that suffix makes the archive portable without rewriting
    # the large source cache or knowing the path used on the original machine.
    parts = image_path.rstrip("/").split("/")
    if len(parts) >= 2:
        return os.path.join(config.IMAGES_DIR, parts[-2], parts[-1])
    return image_path


class Document:
    """单篇文档:页 -> 元素列表。"""

    def __init__(self, doc_id):
        self.doc_id = doc_id
        self.pages = defaultdict(list)   # {page:int -> [element dict]}
        self._image_paths = {}           # {page:int -> 修正后的页图路径}

    def add_page(self, page, image_path, blocks):
        # 去重:同一 doc 内某页已加载过则跳过后续重复行,
        # 保证一个 doc 内 elem_id = p{page}_b{block_id} 唯一。
        # (已核实 jsonl 里 125 个重复 (doc,page),重复行 blocks 内容完全一致,skip 安全)
        if page in self.pages:
            return
        self._image_paths[page] = image_path
        for b in blocks:
            block_id = b["block_id"]
            elem = {
                "doc_id": self.doc_id,
                "page": page,
                "elem_id": f"p{page}_b{block_id}",
                "label": b.get("label", ""),
                "bbox": b.get("bbox", []),
                "content": b.get("content", ""),
            }
            # image/chart 元素:若同事 crop 好的子图存在,挂上 crop_path。
            # crop 命名:visual_crops/<doc>/page_<page>.png_b<block_id>.jpg
            if elem["label"] in CROP_LABELS:
                cp = os.path.join(config.CROP_ROOT, self.doc_id,
                                  f"page_{page}.png_b{block_id}.jpg")
                if os.path.exists(cp):
                    elem["crop_path"] = cp
            self.pages[page].append(elem)

    # ---- 便捷访问 ----
    @property
    def num_pages(self):
        return len(self.pages)

    @property
    def num_elements(self):
        return sum(len(v) for v in self.pages.values())

    def sorted_pages(self):
        return sorted(self.pages.keys())

    def get_page_elements(self, page):
        return self.pages.get(page, [])

    def get_element(self, elem_id):
        for elems in self.pages.values():
            for e in elems:
                if e["elem_id"] == elem_id:
                    return e
        return None

    def page_image_path(self, page):
        """页图路径:优先用解析出的路径,否则按约定拼。"""
        if page in self._image_paths:
            return self._image_paths[page]
        return os.path.join(config.NEW_IMAGE_PREFIX, self.doc_id, f"page_{page}.png")

    def page_text(self, page):
        """把一页所有 block 的 content 拼成一个文本(供 BM25 建索引)。"""
        return "\n".join(e["content"] for e in self.get_page_elements(page) if e["content"])

    def page_search_text(self, page):
        """供 BM25 建索引的页文本 = OCR content + 该页 image/chart 的 crop caption。

        与 page_text 的区别:额外把图表元素的离线 caption 拼进来(仅检索用),
        让"答案只在图里"的图表页也能被内容词/颜色/数值召回(修 Case 5)。
        USE_CROP_CAPTION_IN_SEARCH=False 时退化为 page_text(纯 OCR)。
        caption 不写回元素 content,故 read_page/Reasoner 的展示完全不受影响。
        """
        base = self.page_text(page)
        if not getattr(config, "USE_CROP_CAPTION_IN_SEARCH", False):
            return base
        caps = _load_crop_captions().get(self.doc_id, {})
        if not caps:
            return base
        extra = [caps[e["elem_id"]] for e in self.get_page_elements(page)
                 if e["elem_id"] in caps]
        if not extra:
            return base
        return "\n".join([base] + extra) if base else "\n".join(extra)

    def figures(self):
        """图表清单:caption 锚定 + 前缀校验。

        只认 content 以 "Figure N / Table N / Chart N" 开头的 figure_title
        (真正的图表标题),过滤 OCR 误标的零碎 figure_title(如 "GPT-4V:"、
        "(b)"、"Prompt: ...")。返回 [{page, type, caption, elem_id}],按页序。
        - type 从 caption 前缀取(Figure/Table/Chart),**不看本体 label**
          (OCR 会把 figure 本体误标成 table,前缀更可靠)。
        - elem_id 取该 caption 元素的(留作将来引用/多模态用;prompt 不显示)。
        """
        result = []
        for page in self.sorted_pages():
            for e in self.get_page_elements(page):
                if e["label"] != FIGURE_CAPTION_LABEL:
                    continue
                caption = " ".join(e["content"].strip().split())
                m = _FIGURE_CAPTION_RE.match(caption)
                if not m:
                    continue  # 非规范图表标题,过滤
                result.append({
                    "page": page,
                    "type": _FIGURE_TYPE_MAP[m.group(1).lower()],
                    "caption": caption,
                    "elem_id": e["elem_id"],
                })
        return result

    def outline(self, include_captions=False, include_figures=True):
        """按页顺序拼章节树,返回多行字符串。

        include_figures=True 时在章节标题后附「Figures & Tables」小节
        (只列有 caption 的 image/table/chart,页号 0-indexed,不显示 elem_id)。
        """
        labels = set(TITLE_LABELS)
        if include_captions:
            labels |= set(CAPTION_LABELS)
        lines = []
        for page in self.sorted_pages():
            for e in self.get_page_elements(page):
                if e["label"] in labels:
                    t = " ".join(e["content"].strip().split())
                    if not t:
                        continue
                    lines.append(f"[p{page}] ({e['label']}) {t}")

        if include_figures:
            figs = self.figures()
            if figs:
                lines.append("")
                lines.append("=== Figures & Tables ===")
                for fig in figs:
                    lines.append(f"[p{fig['page']}] ({fig['type']}) {fig['caption']}")
        return "\n".join(lines)


def load_documents(jsonl_path=None):
    """读取整个 jsonl,返回 {doc_id: Document}。"""
    jsonl_path = jsonl_path or config.JSONL_PATH
    docs = {}
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            image_path = _fix_image_path(d["image_path"])
            doc_id, page = _parse_doc_and_page(image_path)
            if doc_id not in docs:
                docs[doc_id] = Document(doc_id)
            docs[doc_id].add_page(page, image_path, d["blocks"])
    return docs


def load_document(doc_id, jsonl_path=None):
    """只加载单篇文档(遍历一次 jsonl,过滤 doc_id)。

    doc_id 会先经 normalize_doc_id() 去掉 .pdf,以兼容 samples.json 的命名。
    """
    jsonl_path = jsonl_path or config.JSONL_PATH
    doc_id = normalize_doc_id(doc_id)
    doc = Document(doc_id)
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            image_path = _fix_image_path(d["image_path"])
            did, page = _parse_doc_and_page(image_path)
            if did != doc_id:
                continue
            doc.add_page(page, image_path, d["blocks"])
    if doc.num_pages == 0:
        raise ValueError(f"文档 {doc_id} 未找到任何页")
    return doc


# ---------- 自测 ----------
if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "2310.07609v1"
    print(f"加载单篇文档: {target}")
    doc = load_document(target)
    print(f"  页数: {doc.num_pages}, 元素数: {doc.num_elements}")
    print(f"  页号: {doc.sorted_pages()}")

    print("\n=== 抽查第 5 页前 3 个元素 ===")
    for e in doc.get_page_elements(5)[:3]:
        c = e["content"]
        c = (c[:70] + "...") if len(c) > 70 else c
        print(f"  {e['elem_id']} [{e['label']}] bbox={e['bbox']} content={c!r}")

    print("\n=== 页图路径存在性抽查 ===")
    for p in doc.sorted_pages()[:3]:
        ip = doc.page_image_path(p)
        print(f"  p{p}: {ip}  存在={os.path.exists(ip)}")

    print("\n=== outline(含图表清单) ===")
    print(doc.outline())

    print("\n=== figures() 结构化 ===")
    for fig in doc.figures():
        print(f"  [p{fig['page']}] {fig['type']:6s} {fig['elem_id']:8s} {fig['caption'][:70]}")

    print("\n=== 全量加载冒烟(文档总数) ===")
    all_docs = load_documents()
    print(f"  文档总数: {len(all_docs)}")
    print(f"  总元素数: {sum(d.num_elements for d in all_docs.values())}")
