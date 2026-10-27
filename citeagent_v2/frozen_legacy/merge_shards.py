#!/usr/bin/env python3
"""合并评测分片的结果与轨迹，并重算官方指标和证据页命中率。

scripts/evaluate.sh 启动的各分片写:
  citeagent_mmlb.shard<i>.json         (主结果,官方字段)
  citeagent_mmlb.shard<i>_traces.jsonl (轨迹,每行一题)
本脚本合并为:
  citeagent_mmlb.json / citeagent_mmlb.txt / citeagent_mmlb_traces.jsonl
"""
import os
import re
import sys
import json
import argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
_MMLB = os.path.join(_HERE, "..", "MMLongBench-Doc")
sys.path.insert(0, _MMLB)
from eval.eval_score import eval_acc_and_f1, show_results   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_out", required=True, help="总结果路径,如 .../citeagent_mmlb.json")
    ap.add_argument("--num_shards", type=int, default=8)
    args = ap.parse_args()

    merged = []
    for sid in range(args.num_shards):
        shard = re.sub(r"\.json$", f".shard{sid}.json", args.base_out)
        if not os.path.exists(shard):
            print(f"[merge][WARN] 缺分片文件 {shard},跳过")
            continue
        with open(shard) as f:
            merged.extend(json.load(f))
    if not merged:
        print("[merge][FATAL] 没有任何分片结果可合并")
        sys.exit(1)

    with open(args.base_out, "w") as f:
        json.dump(merged, f, ensure_ascii=False)

    # 合并轨迹
    trace_out = re.sub(r"\.json$", "_traces.jsonl", args.base_out)
    with open(trace_out, "w") as tf:
        for sid in range(args.num_shards):
            st = re.sub(r"\.json$", f".shard{sid}_traces.jsonl", args.base_out)
            if os.path.exists(st):
                with open(st) as f:
                    for line in f:
                        if line.strip():
                            tf.write(line if line.endswith("\n") else line + "\n")

    # 汇总
    acc, f1 = eval_acc_and_f1(merged)
    hit_rows = [s for s in merged if s.get("citeagent", {}).get("page_hit") is not None]
    phr = (sum(1 for s in hit_rows if s["citeagent"]["page_hit"]) / len(hit_rows)
           if hit_rows else None)
    print("=" * 64)
    print(f"[MERGED] 全量 {len(merged)} 条 | Overall Acc={acc:.4f} | F1={f1:.4f}")
    if phr is not None:
        print(f"[CiteAgent] 证据页命中率(可答题)= {phr:.4f} "
              f"({sum(1 for s in hit_rows if s['citeagent']['page_hit'])}/{len(hit_rows)})")
    print("=" * 64)
    try:
        show_results(merged, show_path=re.sub(r"\.json$", ".txt", args.base_out))
        print(f"[报告] {re.sub(r'.json$', '.txt', args.base_out)}")
    except Exception as e:
        print(f"[warn] show_results 失败: {type(e).__name__}: {e}")
    print(f"[落盘] 总结果 {args.base_out}")
    print(f"[落盘] 总轨迹 {trace_out}")


if __name__ == "__main__":
    main()
