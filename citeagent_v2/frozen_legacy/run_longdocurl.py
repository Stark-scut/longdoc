#!/usr/bin/env python3
"""LongDocURL 全量评测 CiteAgent —— 判分口径完全对齐 LongDocURL 官方。

与 run_mmlongbench.py 的关键差异(都源于「对齐官方 leaderboard」):
1. 文档 = 每题 images 的 30 页窗口(longdocurl_prep.build_document,按题建,不缓存)。
2. 判分用 **LongDocURL 官方** utils_score_v3.eval_score(不是 MMLong 的 eval_score)。
3. 答案抽取用 **官方 concise_answer/<answer_format> tag 式** prompt(LongDocURL/eval/
   prompt_for_answer_extraction.md),解析同官方 eval_open_lvlms.py:
     concise_answer = re.findall(r"<concise_answer>(.*?)</concise_answer>", ...)
     pred_ans = eval(concise_answer)(set→list),失败则取原串
   score = eval_score(gt, pred_ans, sample["answer_format"])(用数据集原 answer_format)。
4. Not answerable / Fail to extract → score 0(官方同)。

工程能力对齐 MMLong runner:--limit / --num_shards+--shard_id / --save_every 续跑 /
--concurrency 卡内并发 / 轨迹 jsonl。多卡起 vLLM 的 shell 由 run_longdocurl_8gpu.sh 负责。
"""
import os
import re
import sys
import json
import ast
import time
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
_LONGDOC = os.path.join(_ROOT, "LongDocURL")
sys.path.insert(0, _LONGDOC)   # 官方 utils/ 在这
sys.path.insert(0, _HERE)

# 官方判分(utils/utils_score_v3.py)
from utils.utils_score_v3 import eval_score   # noqa: E402

import config              # noqa: E402
import longdocurl_prep     # noqa: E402
import pipeline            # noqa: E402
import llm                 # noqa: E402

# ---- 抽取器:复用 MMLong runner 的 gpt-oss-120b 端点,但用官方 tag 式 prompt ----
EXTRACTOR_MODEL = os.getenv(
    "LONGDOC_EXTRACTOR_MODEL",
    os.getenv("CITEAGENT_EVALUATOR_MODEL", "gpt-oss-120b"),
)
EXTRACTOR_BASE_URL = os.getenv(
    "LONGDOC_EXTRACTOR_BASE_URL",
    os.getenv("CITEAGENT_EVALUATOR_BASE_URL", "http://localhost:8010/v1"),
)
EXTRACTOR_API_KEY = os.getenv(
    "LONGDOC_EXTRACTOR_API_KEY",
    os.getenv("CITEAGENT_EVALUATOR_API_KEY", "EMPTY"),
)
EXTRACTOR_PROMPT_PATH = os.path.join(_LONGDOC, "eval",
                                     "prompt_for_answer_extraction.md")
SYSTEM_PROMPT = ("You are an expert in visual document question-answering, "
                 "please answer our questions based on the given images.\n")

_extractor_client = None


def _get_extractor_client():
    global _extractor_client
    if _extractor_client is None:
        _extractor_client = llm._get_client(EXTRACTOR_BASE_URL, EXTRACTOR_API_KEY)
    return _extractor_client


def extract_concise(question, analysis, extractor_prompt):
    """官方口径抽取:返回 (pred_ans, raw_extractor_output)。

    prompt = system + extractor_prompt + Question + Analysis(与官方 eval_open_lvlms 同);
    解析 <concise_answer>/<answer_format> tag;pred_ans = literal_eval(concise)(set→list),
    失败取原串;抽取失败返回 ("Fail to extract", raw)。

    max_tokens 对齐官方 call_llm 的 4096(旧值 256 会把长 List 答案的
    <concise_answer> 截断在闭合标签之前 → 正则匹配不到 → 误判 Fail to extract,
    实测 185 个 answerable 的 List 题里 55 个明确是这种截断,还有更多长度不够)。
    另加容错:若只有开标签、没闭合标签(仍被截断或模型漏写),就从开标签之后
    取到末尾当 concise,不直接判 Fail。
    """
    prompt = (SYSTEM_PROMPT + extractor_prompt
              + "\nQuestion: " + question + "\nAnalysis: " + analysis)
    try:
        resp = _get_extractor_client().chat.completions.create(
            model=EXTRACTOR_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=4096, top_p=1,
        )
        raw = resp.choices[0].message.content
    except Exception as e:
        return "Fail to extract", f"__extractor_error__: {type(e).__name__}: {e}"

    m = re.findall(r"<concise_answer>(.*?)</concise_answer>", raw, re.DOTALL)
    if m:
        concise = m[0]
    elif "<concise_answer>" in raw:
        # 只有开标签没闭合(截断/漏写):取开标签之后,遇到任何 "<"(残缺标签起点)即止。
        concise = raw.split("<concise_answer>", 1)[1].split("<")[0]
    else:
        return "Fail to extract", raw

    try:
        val = ast.literal_eval(concise)
        pred_ans = list(val) if isinstance(val, set) else val
    except Exception:
        pred_ans = concise.strip()
    return pred_ans, raw


def compact_trace(trace):
    """轨迹压缩,与 run_mmlongbench.compact_trace 同口径。"""
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
                        "unique_read_pages", "repeat_read_requests",
                        "repeat_read_blocked") if k in r}
            elif name == "search":
                item["result"] = r
            elif name == "select_evidence":
                item["result"] = ({"added": r.get("added"), "invalid": r.get("invalid")}
                                  if isinstance(r, dict) else r)
            elif name == "call_reasoner":
                item["result"] = ({k: r.get(k) for k in ("status", "answer", "missing")
                                   if k in r} if isinstance(r, dict) else r)
                if c.get("global_selection") is not None:
                    item["global_selection"] = c["global_selection"]
            else:
                item["result"] = r
            steps.append(item)
    return steps


def pipeline_response_text(res):
    """pipeline 结果 → 喂抽取器的自由文本。answerable 用 answer,abstain → Not answerable。"""
    if res["status"] == "answerable":
        ans = res.get("answer", "")
        if not isinstance(ans, str):
            ans = json.dumps(ans, ensure_ascii=False) if ans is not None else ""
        return ans or "Not answerable"
    return "Not answerable"


def eval_one(sample, extractor_prompt):
    """跑 pipeline + 官方抽取 + 官方判分 + 页命中 + 轨迹。就地写回 sample,返回轨迹 dict。"""
    question = sample["question"]
    gt = sample["answer"]
    answer_format = sample["answer_format"]
    gold_pages = sample["evidence_pages"]
    if isinstance(gold_pages, str):
        gold_pages = json.loads(gold_pages)

    doc = longdocurl_prep.build_document(sample)
    res = pipeline.answer_question(doc, question)
    response = pipeline_response_text(res)

    # ---- 官方口径:tag 式抽取 + utils_score_v3.eval_score ----
    if response == "Not answerable":
        pred_ans = "Not answerable"
        raw_extract = "__abstain__"
    else:
        pred_ans, raw_extract = extract_concise(question, response, extractor_prompt)
    if pred_ans == "Fail to extract":
        score = 0.0
    else:
        try:
            score = float(eval_score(gt, pred_ans, answer_format))
        except Exception:
            score = 0.0

    # ---- 证据页命中(E 内部 0-idx → +offset 对照 gold 1-idx) ----
    pred_pages_1idx = sorted({p + config.PAGE_INDEX_OFFSET
                              for p in res.get("evidence_pages", [])})
    page_hit = None
    if gt != "Not answerable" and gold_pages:
        page_hit = bool(set(pred_pages_1idx) & set(gold_pages))

    # 写回(官方 calculate_acc_and_f1 需 score_v3 + answer + pred)
    sample["response"] = response
    sample["extracted_res"] = raw_extract
    sample["pred"] = pred_ans
    sample["score_v3"] = score
    sample["citeagent"] = {
        "status": res["status"],
        "pred_pages_1idx": pred_pages_1idx,
        "page_hit": page_hit,
        "meta": res["meta"],
    }

    return {
        "sample_index": sample["_sample_index"],
        "question_id": sample.get("question_id"),
        "doc_no": sample["doc_no"],
        "question": question,
        "answer_format": answer_format,
        "task_tag": sample.get("task_tag"),
        "gold": gt,
        "gold_pages": gold_pages,
        "status": res["status"],
        "score_v3": score,
        "pred": pred_ans,
        "page_hit": page_hit,
        "rounds": res["meta"]["rounds"],
        "stop_reason": res["meta"]["stop_reason"],
        "meta": res["meta"],
        "steps": compact_trace(res["trace"]),
    }


def main():
    ap = argparse.ArgumentParser(description="LongDocURL 全量评测 CiteAgent(官方判分口径)")
    ap.add_argument("--input_path", default=longdocurl_prep.SAMPLES_PATH)
    ap.add_argument("--output_path",
                    default=os.path.join(_LONGDOC, "results", "citeagent_longdoc.json"))
    ap.add_argument("--extractor_prompt_path", default=EXTRACTOR_PROMPT_PATH)
    ap.add_argument("--limit", type=int, default=0, help=">0 只跑前 N 题(冒烟)")
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--save_every", type=int, default=10)
    ap.add_argument("--concurrency", type=int, default=1)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    trace_path = re.sub(r"\.json$", "_traces.jsonl", args.output_path)

    samples = longdocurl_prep.load_samples(args.input_path)
    if args.limit > 0:
        samples = samples[:args.limit]

    if args.num_shards > 1:
        assert 0 <= args.shard_id < args.num_shards
        samples = [s for i, s in enumerate(samples) if i % args.num_shards == args.shard_id]
        args.output_path = re.sub(r"\.json$", f".shard{args.shard_id}.json", args.output_path)
        trace_path = re.sub(r"\.json$", "_traces.jsonl", args.output_path)
        print(f"[shard] {args.shard_id}/{args.num_shards},分到 {len(samples)} 条")

    # 断点续跑:合并已有主结果里带 score_v3 的样本
    if os.path.exists(args.output_path):
        with open(args.output_path) as f:
            done = {d.get("_sample_index"): d for d in json.load(f)
                    if "score_v3" in d and isinstance(d.get("_sample_index"), int)}
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

    todo = [s for s in samples if "score_v3" not in s]
    print(f"[run] 待处理 {len(todo)} / 全量 {len(samples)} 条;并发={args.concurrency} "
          f"(MAX_ROUNDS={config.MAX_ROUNDS}, READ_PAGE_IMAGE={config.READ_PAGE_IMAGE}, "
          f"REASONER_TIMEOUT={config.REASONER_TIMEOUT})")

    io_lock = threading.Lock()

    def flush():
        tmp_path = f"{args.output_path}.tmp.{os.getpid()}"
        with open(tmp_path, "w") as f:
            json.dump(samples, f, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, args.output_path)

    def _process(sample):
        working = dict(sample)
        try:
            trace = eval_one(working, extractor_prompt)
        except Exception as e:
            working["response"] = "Failed"
            working["pred"] = "Fail to extract"
            working["score_v3"] = 0.0
            working["citeagent"] = {"status": "error", "error": f"{type(e).__name__}: {e}"}
            trace = {"sample_index": sample["_sample_index"],
                     "question_id": sample.get("question_id"),
                     "doc_no": sample["doc_no"], "question": sample["question"],
                     "error": f"{type(e).__name__}: {e}", "steps": []}
        with io_lock:
            sample.update(working)
            with open(trace_path, "a") as tf:
                tf.write(json.dumps(trace, ensure_ascii=False) + "\n")
        return None

    t_start = time.time()
    done_cnt = 0
    if args.concurrency <= 1:
        for sample in tqdm(todo):
            _process(sample)
            done_cnt += 1
            if done_cnt % args.save_every == 0:
                flush()
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = {ex.submit(_process, s): s for s in todo}
            for _ in tqdm(as_completed(futs), total=len(futs)):
                done_cnt += 1
                if done_cnt % args.save_every == 0:
                    with io_lock:
                        flush()
    flush()
    elapsed = time.time() - t_start

    # ---- 官方口径汇总(acc = mean score_v3;F1 见官方 calculate_acc_and_f1) ----
    scored = [s for s in samples if "score_v3" in s]
    acc = sum(s["score_v3"] for s in scored) / len(scored) if scored else 0.0
    ans_rows = [s for s in scored if s["answer"] != "Not answerable"]
    recall = (sum(s["score_v3"] for s in ans_rows) / len(ans_rows)) if ans_rows else 0.0
    pred_pos = [s for s in scored if s["pred"] != "Not answerable"]
    precision = (sum(s["score_v3"] for s in pred_pos) / len(pred_pos)) if pred_pos else 0.0
    f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0.0

    hit_rows = [s for s in samples if s.get("citeagent", {}).get("page_hit") is not None]
    page_hit_rate = (sum(1 for s in hit_rows if s["citeagent"]["page_hit"]) / len(hit_rows)
                     if hit_rows else None)
    n = len(todo)
    print("=" * 64)
    print(f"[TIMING] {n} 条 | {elapsed:.1f}s | 均 {elapsed / max(n,1):.2f}s/条")
    print(f"[RESULT] Overall Acc(score_v3)={acc:.4f} | F1={f1:.4f} | 全量 {len(scored)} 条")
    if page_hit_rate is not None:
        print(f"[CiteAgent] 证据页命中率(可答题)= {page_hit_rate:.4f} "
              f"({sum(1 for s in hit_rows if s['citeagent']['page_hit'])}/{len(hit_rows)})")
    print("=" * 64)
    print(f"[落盘] 主结果 {args.output_path}")
    print(f"[落盘] 轨迹   {trace_path}")


if __name__ == "__main__":
    main()
