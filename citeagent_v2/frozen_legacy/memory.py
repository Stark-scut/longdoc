#!/usr/bin/env python3
"""证据记忆 E:Locator/Global Selector 选中的证据,Reasoner 的唯一输入来源。

支持两种证据粒度(默认使用【页级】):
- 元素级(旧):add(element) 按 elem_id 存单个元素(保留兼容 Verifier/测试)。
- 页级(新):add_page(page, page_image, elements) 以"页"为单位存证据页;每页含
  整页截图 + 该页全部 OCR 元素(elem_id/label/bbox/content/crop_path)。

无论哪种粒度,所有元素都同时进 flat 索引 `_items`(elem_id -> element),使
__contains__ / get_element / elem_ids 在两种模式下语义一致。
render_parts() 生成页级【图文交叉】content 片段。
"""
try:
    from . import llm
except ImportError:
    import llm


class EvidenceMemory:
    def __init__(self):
        # flat 索引:elem_id -> element dict(去重、旧接口用)
        self._items = {}
        # 页级结构:page -> {"page_image": path|None, "elements": [element,...]}
        self._pages = {}

    def add(self, element):
        """加入一个元素(按 elem_id 去重,重复则忽略)。"""
        eid = element["elem_id"]
        if eid not in self._items:
            self._items[eid] = element

    def add_many(self, elements):
        for e in elements:
            self.add(e)

    def add_page(self, page, page_image, elements):
        """加入一个证据页(整页截图 + 该页全部 OCR 元素)。

        page 已存在则忽略(去重,保证同页只进一次)。所有元素同时并入 flat 索引,
        使 elem_id ∈ E 判定在页级模式下仍成立。

        retriever.read_page() 构造的元素只带 elem_id/label/bbox/content/crop_path,
        不含 page;此处统一回填 page(缺失才补,不覆盖已有),避免 to_prompt /
        pipeline._evidence_brief 读 el["page"] 时 KeyError。回填在副本上做,不改
        _read_page_history 里的原始 dict。
        """
        if page in self._pages:
            return
        normalized = []
        for e in elements:
            if "page" not in e:
                e = {**e, "page": page}
            normalized.append(e)
        self._pages[page] = {
            "page_image": page_image,
            "elements": normalized,
        }
        for e in normalized:
            self.add(e)


    def get(self, elem_id):
        return self._items.get(elem_id)

    # get_element 是 get 的别名(与 document.get_element 命名对齐)
    def get_element(self, elem_id):
        return self._items.get(elem_id)

    def remove(self, elem_id):
        """移除某元素(Verifier 反事实掩盖时用);返回被移除的元素或 None。

        同步从页级结构 _pages 里删除该元素(避免 render_parts 仍显示已移除元素);
        若某页元素被删空则移除该页。
        """
        removed = self._items.pop(elem_id, None)
        if removed is not None:
            for page in list(self._pages.keys()):
                rec = self._pages[page]
                rec["elements"] = [e for e in rec["elements"]
                                   if e.get("elem_id") != elem_id]
                if not rec["elements"]:
                    del self._pages[page]
        return removed

    def elem_ids(self):
        return list(self._items.keys())

    def elements(self):
        return list(self._items.values())

    def pages(self):
        """返回有序证据页号列表。"""
        return sorted(self._pages.keys())

    def __len__(self):
        return len(self._items)

    def __contains__(self, elem_id):
        return elem_id in self._items

    def copy_without(self, elem_id):
        """返回一个去掉指定 elem_id 的浅拷贝(供 Verifier,不改动原记忆)。

        同步复制并维护页级结构 _pages(去掉该 elem_id;删空的页丢弃),
        使页级 render_parts 在拷贝对象上保持一致。
        """
        new = EvidenceMemory()
        for eid, el in self._items.items():
            if eid != elem_id:
                new._items[eid] = el
        for page, rec in self._pages.items():
            kept = [e for e in rec["elements"] if e.get("elem_id") != elem_id]
            if kept:
                new._pages[page] = {"page_image": rec.get("page_image"),
                                    "elements": list(kept)}
        return new

    def to_prompt(self, max_content_chars=1500):
        """渲染成给 Reasoner 的文本块:每个元素列出 elem_id/type/page/content。

        max_content_chars=None 时不截断(Locator 上下文里 E 需要完整内容)。
        (纯文本视图;页级图文交叉视图见 render_parts。)
        """
        if not self._items:
            return "(证据集合为空)"
        lines = []
        for el in self._items.values():
            content = el.get("content", "")
            if max_content_chars is not None and len(content) > max_content_chars:
                content = content[:max_content_chars] + "...(截断)"
            crop = " | 图像见下方" if el.get("crop_path") else ""
            lines.append(
                f"- elem_id: {el['elem_id']} | type: {el['label']} | page: {el['page']}{crop}\n"
                f"  content: {content}"
            )
        return "\n".join(lines)

    def render_parts(self, header=""):
        """页级图文交叉渲染:
        每证据页 = [text '[PAGE N] 整页截图:'] [image 整页图]
                   [text 'page N OCR 元素:' + 逐元素行]
                   遇 image/chart 元素 → 在其 OCR 行后插 [image crop]。

        返回 content 片段列表 [{type:text|image_url,...}]。image_url 用
        llm._encode_image 预编码。表格及其他 label 只出文本、不插图。
        page 顺序按页号升序;元素按其在 elements 里的原顺序(= block_id 顺序)。
        """
        import os
        parts = []
        if header:
            parts.append({"type": "text", "text": header})
        if not self._pages:
            parts.append({"type": "text", "text": "(证据集合为空)"})
            return parts

        for page in sorted(self._pages.keys()):
            rec = self._pages[page]
            page_image = rec.get("page_image")
            if page_image and os.path.exists(page_image):
                parts.append({"type": "text", "text": f"[PAGE {page}] 整页截图:"})
                parts.append({"type": "image_url",
                              "image_url": {"url": llm._encode_image(page_image)}})
            else:
                parts.append({"type": "text", "text": f"[PAGE {page}] (整页截图缺失)"})
            parts.append({"type": "text", "text": f"page {page} OCR 元素:"})
            for e in rec.get("elements", []):
                label = e.get("label", "")
                content = e.get("content", "")
                crop_path = e.get("crop_path")
                is_visual = label in ("image", "chart")
                if is_visual and not str(content).strip():
                    content = f"[{label}, 见下图]"
                parts.append({
                    "type": "text",
                    "text": (f"- elem_id: {e['elem_id']} | {label} | "
                             f"bbox: {e.get('bbox', [])} | content: {content}"),
                })
                # image/chart 元素:crop 插在该 block 位置;table 及其他不插图
                if is_visual and crop_path and os.path.exists(crop_path):
                    parts.append({"type": "image_url",
                                  "image_url": {"url": llm._encode_image(crop_path)}})
        return parts

    def crop_items(self):
        """返回带 crop 子图的证据元素:[(elem_id, label, crop_path)]。

        供 Locator/Reasoner 把 image/chart 证据的 crop 图喂给模型。
        """
        return [(el["elem_id"], el["label"], el["crop_path"])
                for el in self._items.values() if el.get("crop_path")]

    def summary(self):
        """一行摘要:elem_id 及类型,供 Locator 观察当前证据。"""
        if not self._items:
            return "(空)"
        return ", ".join(f"{el['elem_id']}({el['label']})" for el in self._items.values())


# ---------- 自测 ----------
if __name__ == "__main__":
    E = EvidenceMemory()
    E.add_many([
        {"doc_id": "d", "page": 5, "elem_id": "p5_b0", "label": "table", "bbox": [], "content": "表格内容 A"},
        {"doc_id": "d", "page": 5, "elem_id": "p5_b1", "label": "text", "bbox": [], "content": "正文内容 B"},
    ])
    E.add({"doc_id": "d", "page": 5, "elem_id": "p5_b0", "label": "table", "bbox": [], "content": "重复"})
    print("len:", len(E), "(应为 2,去重生效)")
    print("elem_ids:", E.elem_ids())
    print("summary:", E.summary())
    print("contains p5_b0:", "p5_b0" in E)
    print("\nto_prompt:\n" + E.to_prompt())
    masked = E.copy_without("p5_b0")
    print("\ncopy_without('p5_b0') 后:", masked.elem_ids(), "| 原记忆不变:", E.elem_ids())
