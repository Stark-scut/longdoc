#!/usr/bin/env python3
"""Run CiteAgent on MMLongBench-Doc using the official evaluation protocol.

与官方端到端脚本(一次喂全文页图)不同,这里跑的是 CiteAgent Agent 系统:
Locator 自主 search / read_page(可多页,含整页图) / call_global_selector 选
【证据页】(整页截图+全 OCR 元素+crop 进 E)/ call_reasoner,pipeline 裁决出 answer。
评测口径完全复用官方:
  pipeline 自由文本答案 → extract_answer(gpt-oss-120b)抽短答案 →
  eval_score(gt, pred, answer_format) 判分;abstain → "Not answerable"。

产出:
- 主结果 JSON:官方样本字段(score/pred/response 写回),可直接跑官方 show_results
  得到分层报告(Single/Cross/Unanswerable/evidence_source/doc_type)。
- CiteAgent 额外指标:证据页命中率(证据集合 E 的页 +offset 对照 gold evidence_pages)。
- 轨迹 JSONL:每行一题,记每回合工具调用。search/select/reasoner 记结果,
  read_page 只记参数(页内容太大,按需回查)。

工程能力对齐官方:--limit 冒烟 / --num_shards+--shard_id 分片 / --save_every 断点续跑 / 串行。
"""
import os
import re
import sys
import json
import time
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

# The official evaluator creates its client at import time.  Configure it only
# from the environment; no service address or credential is embedded here.
_evaluator_url = os.getenv("CITEAGENT_EVALUATOR_BASE_URL")
if _evaluator_url:
    os.environ.setdefault("OPENAI_BASE_URL", _evaluator_url)
os.environ.setdefault(
    "OPENAI_API_KEY", os.getenv("CITEAGENT_EVALUATOR_API_KEY", "EMPTY")
)

_HERE = os.path.dirname(os.path.abspath(__file__))
_MMLB = os.path.join(_HERE, "..", "MMLongBench-Doc")
sys.path.insert(0, _MMLB)
sys.path.insert(0, _HERE)

from eval.eval_score import eval_score, eval_acc_and_f1, show_results   # noqa: E402
from eval.extract_answer import extract_answer                          # noqa: E402

import config          # noqa: E402
import data_prep       # noqa: E402
import pipeline        # noqa: E402

EXTRACTOR_MODEL = os.getenv("CITEAGENT_EVALUATOR_MODEL", "gpt-oss-120b")
EXTRACTOR_PROMPT_PATH = os.path.join(_MMLB, "eval", "prompt_for_answer_extraction.md")


def pipeline_response_text(res):
    """做法 A:pipeline 结果 → 喂抽取器的自由文本。answerable 用 answer;abstain 用 Not answerable。"""
    if res["status"] == "answerable":
        ans = res.get("answer", "")
        if not isinstance(ans, str):
            ans = json.dumps(ans, ensure_ascii=False) if ans is not None else ""
        return ans or "Not answerable"
    return "Not answerable"


def compact_trace(trace):
    """把 Locator trace 压成紧凑轨迹:分工具决定记多细。

    - search:记 args + 完整返回(page/score/snippet)。
    - read_page:只记 args(pages),不记页内容。
    - call_global_selector(页级选页):记 args + 选中页。
    - call_reasoner:记完整 result(status/answer/missing)。
    """
    steps = []
    for step in trace:
        rnd = step["round"]
        if step.get("nudged"):
            steps.append({"round": rnd, "event": "nudge",
                          "assistant_text": (step.get("assistant_text") or "")[:200]})
        for c in step["tool_calls"]:
            name, args, r = c["name"], c["args"], c["result"]
            item = {"round": rnd, "tool": name, "args": args}
            if name == "read_page":
                if isinstance(r, dict):
                    item["result"] = {k: r.get(k) for k in (
                        "requested_pages", "repeat_requested_pages",
                        "skipped_already_read", "persistent_read_pages",
                        "persistent_read_evicted_pages", "persistent_read_chars",
                        "unique_read_pages", "repeat_read_requests",
                        "repeat_read_blocked",
                    ) if k in r}
            elif name == "search":
                item["result"] = r if isinstance(r, list) else r
            elif name == "select_evidence":
                if isinstance(r, dict):
                    item["result"] = {"added": r.get("added"), "invalid": r.get("invalid")}
                else:
                    item["result"] = r
            elif name == "call_reasoner":
                if isinstance(r, dict):
                    item["result"] = {k: r.get(k) for k in
                                      ("status", "answer", "missing")
                                      if k in r}
                else:
                    item["result"] = r
                if c.get("global_selection") is not None:
                    item["global_selection"] = c["global_selection"]
            else:
                item["result"] = r
            steps.append(item)
    return steps


def eval_one(sample, extractor_prompt):
    """跑 pipeline + 官方判分 + 页命中 + 轨迹。就地写回 sample,返回轨迹 dict。"""
    question = sample["question"]
    gt = sample["answer"]
    answer_format = sample["answer_format"]
    gold_pages = eval(sample["evidence_pages"]) if isinstance(sample["evidence_pages"], str) \
        else sample["evidence_pages"]

    doc = data_prep.load_document(sample["doc_id"])
    res = pipeline.answer_question(doc, question)
    response = pipeline_response_text(res)

    # ---- 官方抽取 + 判分 ----
    extracted_res = extract_answer(question, response, extractor_prompt,
                                   model_name=EXTRACTOR_MODEL)
    try:
        pred_ans = extracted_res.split("Answer format:")[0].split("Extracted answer:")[1].strip()
        score = eval_score(gt, pred_ans, answer_format)
    except Exception:
        pred_ans = "Not answerable" if response == "Not answerable" else "Failed to extract"
        score = eval_score(gt, pred_ans, answer_format) if pred_ans == "Not answerable" else 0.0

    # ---- 证据页命中(E 的页,内部 0-idx → +offset 转 1-idx 对照 gold) ----
    pred_pages_1idx = sorted({p + config.PAGE_INDEX_OFFSET
                              for p in res.get("evidence_pages", [])})
    page_hit = None
    if gt != "Not answerable" and gold_pages:
        page_hit = bool(set(pred_pages_1idx) & set(gold_pages))

    # 写回官方字段(供 eval_acc_and_f1 / show_results)
    sample["response"] = response
    sample["extracted_res"] = extracted_res
    sample["pred"] = pred_ans
    sample["score"] = float(score)
    # CiteAgent 额外字段
    sample["citeagent"] = {
        "status": res["status"],
        "pred_pages_1idx": pred_pages_1idx,
        "page_hit": page_hit,
        "meta": res["meta"],
    }

    # 轨迹(单独落 jsonl)
    trace_rec = {
        "sample_index": sample["_sample_index"],
        "doc_id": sample["doc_id"],
        "question": question,
        "answer_format": answer_format,
        "gold": gt,
        "gold_pages": gold_pages,
        "status": res["status"],
        "score": float(score),
        "pred": pred_ans,
        "page_hit": page_hit,
        "rounds": res["meta"]["rounds"],
        "stop_reason": res["meta"]["stop_reason"],
        "meta": res["meta"],
        "steps": compact_trace(res["trace"]),
    }
    # Locator SFT 采集:带上每轮 (messages, teacher assistant) 原料(仅当开启采集时存在)
    if res.get("traj_turns"):
        trace_rec["traj_turns"] = res["traj_turns"]
    return trace_rec


def _load_done_traces(trace_path):
    """已有轨迹文件里已完成的 (doc_id, question) 集合,用于续跑跳过。"""
    done = set()
    if os.path.exists(trace_path):
        with open(trace_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    t = json.loads(line)
                    done.add((t["doc_id"], t["question"]))
                except json.JSONDecodeError:
                    continue
    return done


def main():
    ap = argparse.ArgumentParser(description="MMLongBench 全量评测 CiteAgent")
    ap.add_argument("--input_path", default=config.SAMPLES_PATH)
    ap.add_argument("--output_path",
                    default=os.path.join(_MMLB, "results", "citeagent_mmlb.json"))
    ap.add_argument("--extractor_prompt_path", default=EXTRACTOR_PROMPT_PATH)
    ap.add_argument("--limit", type=int, default=0, help=">0 只跑前 N 题(冒烟)")
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--save_every", type=int, default=10)
    ap.add_argument("--concurrency", type=int, default=1,
                    help="卡内并发线程数(>1 时用线程池并发跑题;实测单卡最佳约 24)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    trace_path = re.sub(r"\.json$", "_traces.jsonl", args.output_path)

    with open(args.input_path) as f:
        samples = json.load(f)
    for sample_index, sample in enumerate(samples):
        sample["_sample_index"] = sample_index
    if args.limit > 0:
        samples = samples[:args.limit]

    # 多卡分片(单卡 num_shards=1 时取全部)
    if args.num_shards > 1:
        assert 0 <= args.shard_id < args.num_shards
        samples = [s for i, s in enumerate(samples) if i % args.num_shards == args.shard_id]
        # 分片时结果/轨迹带 shard 后缀,避免多进程互相覆盖
        args.output_path = re.sub(r"\.json$", f".shard{args.shard_id}.json", args.output_path)
        trace_path = re.sub(r"\.json$", "_traces.jsonl", args.output_path)
        print(f"[shard] {args.shard_id}/{args.num_shards},分到 {len(samples)} 条")

    # 断点续跑:合并已有主结果里带 score 的样本
    if os.path.exists(args.output_path):
        with open(args.output_path) as f:
            done = {d.get("_sample_index"): d for d in json.load(f)
                    if "score" in d and isinstance(d.get("_sample_index"), int)}
        if done:
            merged = 0
            for s in samples:
                prev = done.get(s["_sample_index"])
                if prev is not None:
                    s.update(prev)
                    merged += 1
            print(f"[resume] 合并已完成 {merged} 条")

    with open(args.extractor_prompt_path) as f:
        extractor_prompt = f.read()

    todo = [s for s in samples if "score" not in s]
    print(f"[run] 待处理 {len(todo)} / 全量 {len(samples)} 条;并发={args.concurrency} "
          f"(MAX_ROUNDS={config.MAX_ROUNDS}, READ_PAGE_IMAGE={config.READ_PAGE_IMAGE})")

    # 线程安全:落盘 + 轨迹 append 各一把锁(samples 里各元素被就地写,互不干扰)
    io_lock = threading.Lock()

    def flush():
        tmp_path = f"{args.output_path}.tmp.{os.getpid()}"
        with open(tmp_path, "w") as f:
            json.dump(samples, f, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, args.output_path)

    def _process(sample):
        """跑一题(异常兜底)+ 追加轨迹。返回 None,结果就地写回 sample。"""
        working = dict(sample)
        try:
            trace = eval_one(working, extractor_prompt)
        except Exception as e:
            working["response"] = "Failed"
            working["pred"] = "Failed to extract"
            working["score"] = 0.0
            working["citeagent"] = {"status": "error", "error": f"{type(e).__name__}: {e}"}
            trace = {"sample_index": sample["_sample_index"],
                     "doc_id": sample["doc_id"], "question": sample["question"],
                     "error": f"{type(e).__name__}: {e}", "steps": []}
        with io_lock:
            sample.update(working)
            with open(trace_path, "a") as tf:
                tf.write(json.dumps(trace, ensure_ascii=False) + "\n")
        return None

    t_start = time.time()
    done_cnt = 0
    if args.concurrency <= 1:
        # 串行执行
        for sample in tqdm(todo):
            _process(sample)
            done_cnt += 1
            if done_cnt % args.save_every == 0:
                flush()
    else:
        # 卡内并发:线程池并发跑题(pipeline 每题独立、无共享可变状态,线程安全)
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = {ex.submit(_process, s): s for s in todo}
            for _ in tqdm(as_completed(futs), total=len(futs)):
                done_cnt += 1
                if done_cnt % args.save_every == 0:
                    with io_lock:
                        flush()
    flush()
    elapsed = time.time() - t_start

    # ---- 官方口径汇总 ----
    acc, f1 = eval_acc_and_f1(samples)
    # CiteAgent 页命中率(可答题)
    hit_rows = [s for s in samples
                if s.get("citeagent", {}).get("page_hit") is not None]
    page_hit_rate = (sum(1 for s in hit_rows if s["citeagent"]["page_hit"]) / len(hit_rows)
                     if hit_rows else None)
    n = len(todo)
    print("=" * 64)
    print(f"[TIMING] {n} 条 | {elapsed:.1f}s | 均 {elapsed / max(n,1):.2f}s/条")
    print(f"[RESULT] Overall Acc={acc:.4f} | F1={f1:.4f} | 全量 {len(samples)} 条")
    if page_hit_rate is not None:
        print(f"[CiteAgent] 证据页命中率(可答题)= {page_hit_rate:.4f} "
              f"({sum(1 for s in hit_rows if s['citeagent']['page_hit'])}/{len(hit_rows)})")
    print("=" * 64)

    # 官方分层报告(需 evidence_pages/sources 为原始字符串形式,show_results 内部 eval)
    try:
        show_results(samples, show_path=re.sub(r"\.json$", ".txt", args.output_path))
        print(f"[报告] {re.sub(r'.json$', '.txt', args.output_path)}")
    except Exception as e:
        print(f"[warn] show_results 失败(不影响主结果): {type(e).__name__}: {e}")
    print(f"[落盘] 主结果 {args.output_path}")
    print(f"[落盘] 轨迹   {trace_path}")


if __name__ == "__main__":
    main()
