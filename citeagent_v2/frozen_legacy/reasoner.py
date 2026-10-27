#!/usr/bin/env python3
"""Reasoner: evidence consumer and sufficiency judge.

只看「问题 Q + 当前证据记忆 E(页级:整页截图 + OCR 元素 + crop 图)」,绝不看全文。
输出严格 JSON,二选一:
- answerable  : {status, answer}
- insufficient: {status, missing:[{missing_claim, expected_type, search_hint}]}
"""
try:
    from . import llm, config
except ImportError:
    import llm
    import config


SYSTEM_PROMPT = """你是一个严谨的证据推理器。你只能依据【提供的证据页】回答问题,绝对不能使用证据之外的知识或猜测。

你会收到:
- 一个问题 Q
- 一组证据页,按页组织:每页先给整页截图,再给该页的 OCR 解析元素列表
- 若页内有 image/chart 元素,其裁剪图会紧跟在对应 OCR 行之后(图表数据在图里,OCR 的 content 可能为空,须看图)

你的任务:判断这些证据是否足以回答 Q。
- 若足够:直接给出答案。
- 若不够:不要硬答。用结构化 missing 指出还缺什么。

作答前请逐页阅读:结合整页截图与 OCR 文本,认真观察每张图像/图表的细节(数值、颜色、位置、数量、坐标轴、图例等),先推理【问题需要哪些信息】,再核对【证据是否覆盖这些信息】,据此判断能否作答。思考要清晰高效:证据足够时果断作答,不要反复自我怀疑或无谓地循环验证。

严格只输出如下 JSON 之一,不要有多余文字、不要 markdown 代码块:

// 情形A:证据足够
{"status":"answerable","answer":"..."}

// 情形B:证据不足
{"status":"insufficient","missing":[{"missing_claim":"缺什么信息","expected_type":"table|text|figure|chart","search_hint":"用于关键词检索的判别性词/短语(空格分隔)"}]}

约束:
- answer 必须完全由给定证据支撑,不得掺入外部知识。
- missing 每项都要具体:missing_claim 说清缺的是什么事实。
- search_hint 用于词法(关键词)检索:给判别性强、能在文档原文中逐字出现的词或短语(如专有名词、术语、数字、表格/图表标题、图例词),空格分隔;不要写成整句问题,不要含无信息量的停用词。
- 若证据完全无关或为空,输出 insufficient。"""


def _page_level_header(question):
    """页级 E 的交叉输入头部说明(拼在 render_parts 最前)。"""
    return (
        f"问题 Q:{question}\n\n"
        "下面按【证据页】给出证据集合 E:每页先给整页截图,再给该页的 OCR 元素列表;"
        "image/chart 元素的裁剪图紧跟在其 OCR 行之后。请结合整页截图、OCR 文本与裁剪图判断。\n"
    )


_PAGE_LEVEL_TAIL = (
    "\n请判断证据是否足以回答 Q,按系统要求只输出一个 JSON 对象。"
)


def _normalize_missing(missing, question):
    """逐项补齐 missing 的三个字段,保证 Locator 取用时不会 KeyError。

    - missing 非 list 或为空 → 造一个默认项。
    - 每项缺 missing_claim/expected_type/search_hint → 用默认值补齐
      (search_hint 缺失时退回原问题,保证补检索仍可进行)。
    """
    if not isinstance(missing, list) or not missing:
        return [{
            "missing_claim": "证据不足(推理器未给出结构化缺失)",
            "expected_type": "text",
            "search_hint": question,
        }]
    normed = []
    for m in missing:
        if not isinstance(m, dict):
            m = {}
        m.setdefault("missing_claim", "证据不足")
        m.setdefault("expected_type", "text")
        # search_hint 缺失或空 → 退回原问题
        if not m.get("search_hint"):
            m["search_hint"] = question
        normed.append(m)
    return normed


def reason(question, evidence_memory, document=None):
    """对 Q + E 做一次充分性判断,返回规整后的 dict。

    document:保留参数(兼容旧调用),当前不再使用。
    输入为页级证据:用 memory.render_parts() 构造【图文交叉】content(每证据页:
    整页截图 → OCR 元素,image/chart 的 crop 插在对应 block 位置),
    走 llm.chat_json_parts。
    返回始终含 status;answerable 时含 answer,insufficient 时含规整后的 missing。
    """
    try:
        content_parts = evidence_memory.render_parts(
            header=_page_level_header(question))
        content_parts.append({"type": "text", "text": _PAGE_LEVEL_TAIL})
        result = llm.chat_json_parts(SYSTEM_PROMPT, content_parts,
                                     max_tokens=config.REASONER_MAX_TOKENS,
                                     timeout=config.REASONER_TIMEOUT,
                                     max_retries=config.REASONER_MAX_RETRY,
                                     enable_thinking=config.REASONER_ENABLE_THINKING,
                                     base_url=config.REASONER_BASE_URL,
                                     api_key=config.REASONER_API_KEY,
                                     model=config.REASONER_MODEL)
    except Exception as e:
        # JSON 解析失败 / 请求超时(不收敛难题):降级 insufficient,避免上层崩溃/卡死
        return {
            "status": "insufficient",
            "missing": [{
                "missing_claim": "推理器输出解析失败或超时",
                "expected_type": "text",
                "search_hint": question,
            }],
            "_error": f"{type(e).__name__}: {e}",
        }

    status = result.get("status")

    if status == "answerable":
        return result

    if status == "insufficient":
        # 逐项补齐 missing 的三个字段(missing_claim/expected_type/search_hint),
        # 避免模型漏字段导致 Locator 补检索时 KeyError。
        result["missing"] = _normalize_missing(result.get("missing"), question)
        return result

    # 未知 status:兜底
    return {
        "status": "insufficient",
        "missing": [{
            "missing_claim": f"推理器返回未知 status: {status}",
            "expected_type": "text",
            "search_hint": question,
        }],
        "_error": f"未知 status: {status}",
    }


# ---------- 自测 ----------
if __name__ == "__main__":
    from memory import EvidenceMemory

    # 构造一个「表格证据 + 正文」的场景
    table_elem = {
        "doc_id": "demo", "page": 5, "elem_id": "p5_b0", "label": "table", "bbox": [],
        "content": "<table><tr><td>Model</td><td>F1</td></tr>"
                   "<tr><td>QACHECK</td><td>82.3</td></tr>"
                   "<tr><td>Baseline</td><td>75.1</td></tr></table>",
    }
    text_elem = {
        "doc_id": "demo", "page": 5, "elem_id": "p5_b2", "label": "text", "bbox": [],
        "content": "Table 1 reports F1 scores. QACHECK outperforms the baseline.",
    }

    print("=== 场景1:证据充分(含答案所在表格)→ 期望 answerable ===")
    E1 = EvidenceMemory()
    E1.add_many([table_elem, text_elem])
    r1 = reason("What is the F1 score of QACHECK?", E1)
    import json
    print(json.dumps(r1, ensure_ascii=False, indent=2))

    print("\n=== 场景2:删掉关键表格,只剩无关正文 → 期望 insufficient + missing ===")
    E2 = EvidenceMemory()
    E2.add(text_elem)
    r2 = reason("What is the exact F1 score of QACHECK?", E2)
    print(json.dumps(r2, ensure_ascii=False, indent=2))

    print("\n=== 场景3:空证据 → 期望 insufficient ===")
    E3 = EvidenceMemory()
    r3 = reason("What is the F1 score of QACHECK?", E3)
    print(json.dumps(r3, ensure_ascii=False, indent=2))
