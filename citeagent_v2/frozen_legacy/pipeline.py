#!/usr/bin/env python3
"""Pipeline: outer state machine and final JSON assembly.

职责分层:
- Locator.run() 已在内部跑完整个 function-calling 工具循环,并暴露
  evidence(E) / last_reason / trace / stop_reason。
- pipeline 只做【结果裁决 + 兜底补调 + 拒答准则 + 最终 JSON】,不再驱动第二层循环。

流程:
1. locator.run(Q) 跑工具循环。
2. 兜底补调:若 E 非空但模型全程没 call_reasoner(last_reason is None),
   pipeline 主动补调一次 Reasoner,避免"白跑"。
3. 裁决:
   - answerable → status=answerable,取 Reasoner 的 answer。
   - 否则(insufficient / E 空 / 无结果)→ status=abstain,附部分证据 + missing。
4. 产出最终 JSON:{doc_id, question, answer, status, evidence_pages, missing?,
   trace, meta}。page 保持 0-indexed(与 page_N.png、bbox 对齐,便于核验)。
"""
import json

try:
    from . import config
    from .locator import Locator
except ImportError:
    import config
    from locator import Locator


def _evidence_brief(evidence):
    """把证据集合渲染成简要列表(用于 abstain 时展示"确实找过了什么")。"""
    return [{
        "elem_id": el["elem_id"],
        "page": el["page"],
        "bbox": el["bbox"],
        "label": el["label"],
    } for el in evidence.elements()]


def answer_question(document, question, max_rounds=None, discipline=None):
    """对一篇文档 + 一个问题跑完整闭环,返回最终 JSON(dict)。

    包含一次保守重试:
    - B(保守重试):若疑似"空跑"(abstain && E空 && stop_reason=max_rounds &&
      模型全程没 call_reasoner),自动重试 1 次;第二次仍空跑则接受 abstain,
      并在 meta 记 retry_count / final_failure。

    discipline:Locator 纪律块变体(strict/gentle/none);
               None 时用 Locator 默认(strict)。
    """
    result = _run_once(document, question, max_rounds=max_rounds, discipline=discipline)

    # ---- B:仅对"疑似空跑"重试 1 次 ----
    if _is_empty_spin(result):
        first_meta = dict(result["meta"])  # 留存首次(空跑)的 meta 供诊断
        retry = _run_once(document, question, max_rounds=max_rounds, discipline=discipline)
        retry["meta"]["retry_count"] = 1
        retry["meta"]["first_attempt_meta"] = first_meta
        if _is_empty_spin(retry):
            retry["meta"]["final_failure"] = "no_evidence_selected"
        result = retry

    return result


def _is_empty_spin(result):
    """疑似空跑:模型反复调工具却没选证据、也没触发 Reasoner。"""
    meta = result["meta"]
    return (
        result["status"] == "abstain"
        and meta["num_evidence"] == 0
        and meta["stop_reason"] == "max_rounds"
        and meta.get("abstain_reason") == "no_evidence_no_reasoner"
    )


def _run_once(document, question, max_rounds=None, discipline=None):
    """跑一遍闭环(Locator 循环 + 裁决 + 兜底补调),返回最终 JSON。"""
    loc = Locator(document, max_rounds=max_rounds, discipline=discipline)
    run_out = loc.run(question)

    evidence = run_out["evidence"]
    last_reason = run_out["last_reason"]

    # ---- 兜底补调:Global Select 也允许从已读候选补出空的局部 E ----
    fallback_reasoner = False
    has_global_candidates = bool(getattr(loc, "_read_page_history", {}))
    should_fallback = len(evidence) > 0 or (
        (config.GLOBAL_EVIDENCE_SELECTION or getattr(config, "PAGE_LEVEL_EVIDENCE", False))
        and has_global_candidates
    )
    if last_reason is None and should_fallback:
        last_reason = loc._tool_call_reasoner()
        evidence = loc.evidence
        fallback_reasoner = True

    meta = {
        "rounds": run_out["rounds"],
        "stop_reason": run_out["stop_reason"],
        "num_evidence": len(evidence),
        "fallback_reasoner": fallback_reasoner,
        "read_context": run_out.get("read_context", {}),
        "global_selection": {
            "enabled": config.GLOBAL_EVIDENCE_SELECTION,
            "calls": len(loc._global_selection_history),
            "failures": sum(
                1 for item in loc._global_selection_history if item.get("error")
            ),
            "total_added": sum(
                len(item.get("added_ids", []))
                for item in loc._global_selection_history
            ),
            "history": loc._global_selection_history,
        },
    }

    result = {
        "doc_id": document.doc_id,
        "question": question,
        "trace": run_out["trace"],
        "meta": meta,
    }
    # Locator 轨迹采集:透传每轮 (messages, teacher assistant),供 SFT 原料落盘
    if run_out.get("traj_turns"):
        result["traj_turns"] = run_out["traj_turns"]

    # ---- 裁决 ----
    status = last_reason.get("status") if last_reason else None
    # 证据集合 E 的页集合(0-indexed),供评测侧算证据页命中率
    result["evidence_pages"] = evidence.pages()

    if status == "answerable":
        result.update({
            "status": "answerable",
            "answer": last_reason.get("answer", ""),
        })
    else:
        # insufficient / E 空 / 无结果 → 拒答
        result.update({
            "status": "abstain",
            "answer": None,
            "partial_evidence": _evidence_brief(evidence),
            "missing": last_reason.get("missing", []) if last_reason else [],
        })
        if last_reason is None:
            meta["abstain_reason"] = "no_evidence_no_reasoner"
        elif len(evidence) == 0:
            meta["abstain_reason"] = "empty_evidence"
        else:
            meta["abstain_reason"] = "reasoner_insufficient"

    return result


# ---------- 自测 ----------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    import data_prep

    doc = data_prep.load_document("2310.07609v1")
    q = "What is the average performance gap between ProgramFC and our system in HOVER?"
    print(f"文档: {doc.doc_id} | 问题: {q}\n")
    res = answer_question(doc, q)

    # 精简打印(trace 太长,单独摘要)
    printable = {k: v for k, v in res.items() if k != "trace"}
    print(json.dumps(printable, ensure_ascii=False, indent=2))
    print(f"\n=== trace 轮数: {len(res['trace'])} | meta: {res['meta']} ===")
