#!/usr/bin/env python3
"""Search and page-reading tools used by the Locator.

- BM25 页级检索:把每页所有 block 的 content 拼成一个"文档"建索引,
  索引单位是「页」。用 rank_bm25 的 BM25Okapi。
- read_page:把某页"展开",返回该页全部元素(elem_id/label/bbox/content 摘要),
  供 Locator 从中精选。

这里只提供绑定到具体 Document 的纯函数式能力；工具的 JSON schema 与调度在 locator.py。
"""
import os
import re

from rank_bm25 import BM25Okapi

try:
    from . import config
except ImportError:
    import config


def _tokenize(text):
    """简单分词:小写 + 抽取字母数字词(mmlongbench 以英文为主,够用)。

    同时保留中文字符为单字 token,兼顾少量中文内容。
    """
    text = text.lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    tokens += re.findall(r"[一-鿿]", text)
    return tokens


class PageRetriever:
    """单篇文档的 BM25 页级检索器。"""

    def __init__(self, document):
        self.doc = document
        self.pages = document.sorted_pages()        # 有序页号列表
        # 建索引用 page_search_text(OCR + 图表 caption),让图表页能被召回(Case 5);
        # KWIC snippet 仍用纯 OCR 的 page_text(caption 不是页面真实文字,不宜展示)。
        corpus = [document.page_search_text(p) for p in self.pages]
        self._tokenized = [_tokenize(t) for t in corpus]
        self.bm25 = BM25Okapi(self._tokenized)

    def _kwic_snippet(self, page, query, snippet_chars=240):
        """关键词邻域 snippet(KWIC):截取查询词【命中最密集】的一段。

        1. 用与 BM25 一致的 _tokenize(query) 得查询词;在该页 OCR 文本(小写)里
           用词边界正则找所有命中位置。
        2. 双指针滑动:找一个宽 snippet_chars 的字符窗口,使其覆盖的命中词最多
           (并列时取更靠前的)。以该窗口为中心截取,截到词边界,首尾加省略号。
        3. 无任何命中 → 退回该页前 snippet_chars 字(保证 snippet 非空)。
        """
        text = self.doc.page_text(page).replace("\n", " ")
        low = text.lower()
        qwords = set(_tokenize(query))
        if not qwords:
            return text[:snippet_chars]

        # 所有命中位置(词边界匹配,避免子串误命中)
        hits = []
        for m in re.finditer(r"[a-z0-9]+|[一-鿿]", low):
            if m.group() in qwords:
                hits.append(m.start())
        if not hits:
            return text[:snippet_chars]

        # 双指针:窗口 [hits[i], hits[i]+snippet_chars] 内命中数最多
        best_i, best_cnt, j = 0, 0, 0
        for i in range(len(hits)):
            if j < i:
                j = i
            while j + 1 < len(hits) and hits[j + 1] - hits[i] <= snippet_chars:
                j += 1
            cnt = j - i + 1
            if cnt > best_cnt:
                best_cnt, best_i = cnt, i

        # 以最密集簇的中点为中心取窗
        center = (hits[best_i] + hits[min(best_i + best_cnt - 1, len(hits) - 1)]) // 2
        half = snippet_chars // 2
        start = max(0, center - half)
        end = min(len(text), start + snippet_chars)
        # 截到词边界(避免切半个词)
        if start > 0:
            sp = text.find(" ", start)
            if sp != -1 and sp < center:
                start = sp + 1
        if end < len(text):
            sp = text.rfind(" ", center, end)
            if sp != -1:
                end = sp
        snippet = text[start:end].strip()
        if start > 0:
            snippet = "…" + snippet
        if end < len(text):
            snippet = snippet + "…"
        return snippet

    def search(self, query, top_k=5, snippet_chars=240):
        """返回 [{page, score, snippet}],按分数降序。

        snippet 用 KWIC(关键词命中最密集的邻域),便于模型判断该页要不要 read。
        """
        scores = self.bm25.get_scores(_tokenize(query))
        ranked = sorted(zip(self.pages, scores), key=lambda x: x[1], reverse=True)
        results = []
        for page, score in ranked[:top_k]:
            results.append({
                "page": page,
                "score": round(float(score), 4),
                "snippet": self._kwic_snippet(page, query, snippet_chars),
            })
        return results

    def _read_one_page(self, page):
        """读单页,返回 {"page", "elements":[...], "page_image"} 或 {"page","error"}。"""
        elems = self.doc.get_page_elements(page)
        if not elems:
            valid = self.doc.sorted_pages()
            hint = (f"有效页号: {valid[0]}~{valid[-1]}" if valid else "文档无页")
            return {"page": page, "error": f"页 {page} 不存在或无元素;{hint}"}
        out = []
        for e in elems:
            item = {
                "elem_id": e["elem_id"],
                "label": e["label"],
                "bbox": e["bbox"],
                "content": e.get("content", ""),
            }
            if e.get("crop_path"):
                item["crop_path"] = e["crop_path"]
            out.append(item)
        # 只有页图文件真实存在才给路径,避免编码时崩
        img = self.doc.page_image_path(page)
        return {
            "page": page,
            "elements": out,
            "page_image": img if img and os.path.exists(img) else None,
        }

    def read_page(self, pages, max_pages=None):
        """展开一页或多页,返回该页全部元素(完整 content,不截断)+ 整页原图路径。

        pages:单个页号(int)或页号列表(list[int]);允许模型一次读多页,
        以聚合散在不同页的证据(如定义/身份页 + 数据页)。单次最多 max_pages 页。

        返回 {"pages": [ {page, elements, page_image} | {page, error}, ... ]}。
        - 每个 element:{elem_id, label, bbox, content, crop_path?},content 不截断。
        - page_image:整页原图路径(文件不存在则 None,防编码崩)。
        - 非法页号不整批失败,单页返回 {"page", "error"}。
        - 超过 max_pages 只取前 max_pages 页,并在结果里附 truncated 提示。
        """
        if max_pages is None:
            max_pages = config.READ_PAGE_MAX
        # 归一化为有序去重的页号列表
        if isinstance(pages, int):
            pages = [pages]
        norm, seen = [], set()
        for p in pages:
            try:
                p = int(p)
            except (ValueError, TypeError):
                continue
            if p not in seen:
                seen.add(p)
                norm.append(p)
        result = {"pages": [self._read_one_page(p) for p in norm[:max_pages]]}
        if len(norm) > max_pages:
            result["truncated"] = (
                f"一次最多读 {max_pages} 页,已只处理前 {max_pages} 页 "
                f"({norm[:max_pages]});其余请分批读取。")
        return result


# ---------- 自测 ----------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    import data_prep
    import config

    doc = data_prep.load_document("2310.07609v1")
    r = PageRetriever(doc)

    # gold 用 samples.json 的原始 1-indexed evidence_pages;
    # 内部页号 0-indexed,对照时需减 PAGE_INDEX_OFFSET。
    cases = [
        ("What is the average performance gap between ProgramFC and our system in HOVER?", [6]),
        ("How many modules are implemented with Large Language Models backbone?", [2]),
        ("How many prompts instruction examples are listed in the main body of the paper?", [3, 4]),
        ("What is the F1 evaluation results table for HOVER and FEVEROUS datasets?", [6]),
    ]
    for q, gold_1idx in cases:
        gold0 = [g - config.PAGE_INDEX_OFFSET for g in gold_1idx]
        res = r.search(q, top_k=5)
        hit_pages = [x["page"] for x in res]
        hit = any(g in hit_pages for g in gold0)
        print(f"\nQ: {q}")
        print(f"  gold(1idx)={gold_1idx} -> gold(0idx)={gold0}  召回top5页={hit_pages}  命中={hit}")
        for x in res[:3]:
            print(f"    p{x['page']} score={x['score']} | {x['snippet'][:70]}")

    print("\n=== read_page([1,5]) 多页 ===")
    rp = r.read_page([1, 5])
    for pg in rp["pages"]:
        if "error" in pg:
            print(f"  p{pg['page']}: ERROR {pg['error']}")
        else:
            print(f"  p{pg['page']}: {len(pg['elements'])} 元素, page_image={'✓' if pg['page_image'] else '✗'}")

    print("\n=== 非法/超限保护 ===")
    print("  read_page(999):", r.read_page(999)["pages"])
    over = r.read_page([0, 1, 2, 3, 4])
    print(f"  read_page([0,1,2,3,4]) 实际处理 {len(over['pages'])} 页, truncated={'truncated' in over}")
