#!/usr/bin/env python3
"""Freeze, run, compare, validate and replay CiteAgent context experiments."""
from __future__ import annotations
import argparse
import ast
from collections import Counter
import contextlib
import datetime
import fcntl
import importlib.util
import io
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
sys.dont_write_bytecode = True

from common import ARCHIVE, HERE, DEFAULT_MODEL, Audit, Assets, digest, read_json, save_json, json_hash
from freeze import freeze, DEFAULT_DATA

def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    f = sub.add_parser("freeze", help="Freeze code/config/cases/input assets/archived results")
    f.add_argument("--baseline", type=Path, default=HERE / "baseline")
    f.add_argument("--samples", type=Path, default=DEFAULT_DATA / "samples.json")
    f.add_argument("--blocks", type=Path, default=DEFAULT_DATA / "stage2_blocks_cache.jsonl")
    f.add_argument("--images-dir", type=Path, default=DEFAULT_DATA / "images_1024")
    f.add_argument("--crop-root", type=Path, default=DEFAULT_DATA / "visual_crops")
    f.add_argument("--captions", type=Path, default=DEFAULT_DATA / "crop_captions.jsonl")
    f.add_argument("--indices", default="281,283,289,291")
    f.add_argument("--max-pages", type=int, default=15)
    f.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    f.add_argument("--old-results", type=Path, nargs="+")
    f.add_argument("--url", default="http://127.0.0.1:8067/v1")
    f.add_argument("--model", default="citeagent-qwen35")
    for role in ["locator", "reasoner", "selector", "extractor"]:
        f.add_argument("--" + role + "-url")
        f.add_argument("--" + role + "-model")
    f.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=True)
    f.add_argument("--reasoner-thinking", action=argparse.BooleanOptionalAction, default=False)
    f.add_argument("--locator-max-tokens", type=int, default=8192)
    f.add_argument("--reasoner-max-tokens", type=int, default=8192)
    f.add_argument("--selector-max-tokens", type=int, default=4096)
    f.add_argument("--extract-max-tokens", type=int, default=2048)
    f.add_argument("--max-rounds", type=int, default=8)
    f.add_argument("--read-page-max", type=int, default=3)
    f.add_argument("--timeout", type=float, default=120)
    f.add_argument("--wall-seconds", type=float, default=900)
    f.add_argument("--max-model-len", type=int, default=65536)
    f.add_argument("--max-pixels", type=int, default=1048576)
    f.add_argument("--request-attempts", type=int, default=2)
    f.add_argument("--json-attempts", type=int, default=2)
    f.add_argument("--seed", type=int, default=42)
    f.add_argument("--discipline", choices=["strict", "gentle", "none"], default="strict")
    f.add_argument("--controller-fallback", action="store_true", help="Native only: explicitly permit one controller Reasoner fallback")
    for name in ["run", "compare", "verify", "doctor"]:
        q = sub.add_parser(name)
        q.add_argument("--baseline", type=Path, default=HERE / "baseline")
        if name in {"run", "compare"}:
            q.add_argument("--output", type=Path, default=HERE / "runs/small_comparison")
            q.add_argument("--stage", choices=["all", "agent", "score"], default="all")
            q.add_argument("--retry-failed", action="store_true")
        if name == "run":
            q.add_argument("--mode", choices=["legacy", "native"], required=True)
    r = sub.add_parser("replay", help="Rebuild a logged request without submitting it")
    r.add_argument("--trace", type=Path, required=True)
    r.add_argument("--sequence", type=int, required=True)
    r.add_argument("--output", type=Path, required=True)
    sub.add_parser("validate", help="Run offline invariant and error-path tests")
    return p

def verify(baseline):
    from runtime import load
    runtime = load(baseline)
    source = read_json(HERE / "source_manifest.json")
    original = Path(source["original_directory"])
    changes = [name for name, value in source["copied_source_sha256"].items()
               if not (original / name).is_file() or digest(original / name) != value]
    return {"baseline": str(baseline.resolve()), "baseline_valid": True,
            "frozen_legacy_source_valid": True, "original_repository_changed": changes,
            "sample_indices": read_json(baseline / "manifest.json")["sample_indices"]}

def legacy_run(runtime, document, question, audit):
    config, _, _, _, _, pipeline, _, settings = runtime
    raw = pipeline.answer_question(document, question, max_rounds=settings["max_rounds"],
                                   discipline=settings["discipline"])
    status = "completed" if raw["status"] == "answerable" and raw.get("answer") else "incomplete"
    last_errors = [call["result"].get("_error") for row in raw.get("trace", [])
                   for call in row.get("tool_calls", [])
                   if isinstance(call.get("result"), dict) and call["result"].get("_error")]
    result = {"execution_status": status, "semantic_status": raw["status"],
              "stop_reason": raw["meta"]["stop_reason"], "raw_answer": raw.get("answer"),
              "legacy_benchmark_response": pipeline.pipeline_response_text(raw) if hasattr(pipeline, "pipeline_response_text") else raw.get("answer") or "Not answerable",
              "rounds": raw["meta"]["rounds"], "evidence_pages": raw["evidence_pages"],
              "controller_fallback_calls": int(raw["meta"].get("fallback_reasoner", False)),
              "tool_calls": sum(len(row.get("tool_calls", [])) for row in raw.get("trace", [])),
              "protocol_valid": False, "protocol_note": "Intentional legacy user-observation/state-rebuild baseline.",
              "legacy_raw_result": raw, "legacy_technical_errors": last_errors,
              "legacy_retry_count": raw["meta"].get("retry_count", 0)}
    audit.emit("legacy_session_end", result=result)
    return result

def score(result, case, baseline, transport, audit):
    result["score"] = 0.0
    if result["execution_status"] != "completed" or not result.get("raw_answer"):
        result["scoring_status"] = "not_answered"
        result["evaluation_note"] = "Incomplete/technical failures count as zero, including on unanswerable GT."
        return
    prompt = (baseline / "extract_prompt.md").read_text()
    response = transport.complete("extractor", [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": "\n\nQuestion:{}\nAnalysis:{}\n".format(case["question"], result["raw_answer"])}],
        attempts=transport.settings["request_attempts"])
    from transport import require_completed, Failure
    import re
    choice = require_completed(response)
    text = choice["message"].get("content") or ""
    result["extracted_res"] = text
    match = re.search(r"Extracted answer:\s*(.*?)\s*Answer format:", text, re.S)
    if not match or not match.group(1).strip():
        raise Failure("extraction_parse_error", "Missing extracted answer/format markers")
    prediction = match.group(1).strip()
    spec = importlib.util.spec_from_file_location("_frozen_official_scorer", baseline / "eval_score.py")
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    gt, pred = case["answer"], prediction
    if case["answer_format"] not in {"Int", "Float", "Str", "None"}:
        if isinstance(gt, str) and gt.startswith("["):
            gt = ast.literal_eval(gt)
        if pred.startswith("["):
            pred = ast.literal_eval(pred)
    with contextlib.redirect_stdout(io.StringIO()):
        value = float(scorer.eval_score(gt, pred, case["answer_format"]))
    result.update(score=value, pred=prediction, scoring_status="scored", answer_correct=value == 1.0)
    gold = case.get("evidence_pages", [])
    if isinstance(gold, str):
        gold = ast.literal_eval(gold)
    returned = [page + 1 for page in result.get("evidence_pages", [])]
    result["evidence_page_hit"] = bool(set(gold) & set(returned)) if gold else None
    audit.emit("offline_score", score=value, pred=prediction, evidence_page_hit=result["evidence_page_hit"])

def summarize(cases, output):
    rows = []
    for case in cases:
        path = output / f"sample_{case['_sample_index']:04d}/result.json"
        if path.exists():
            rows.append(read_json(path))
    scored = sum(r.get("scoring_status") == "scored" for r in rows)
    summary = {"total": len(cases), "attempted": len(rows),
        "completed": sum(r["execution_status"] == "completed" for r in rows),
        "incomplete": sum(r["execution_status"] == "incomplete" for r in rows),
        "errors": sum(r["execution_status"] == "error" for r in rows),
        "scored": scored, "scoring_errors": sum(r.get("scoring_status") == "error" for r in rows),
        "accuracy_with_incomplete_as_zero": sum(r.get("score", 0.0) for r in rows) / len(cases),
        "evaluation_complete": len(rows) == len(cases) and all(r.get("scoring_status") in {"scored", "not_answered"} for r in rows),
        "protocol_valid_sessions": sum(r.get("protocol_valid") is True for r in rows),
        "stop_reasons": dict(Counter(r["stop_reason"] for r in rows)),
        "median_rounds": statistics.median([r["rounds"] for r in rows]) if rows else None,
        "total_llm_requests": sum(r.get("cost", {}).get("http_requests", 0) for r in rows),
        "total_reported_tokens": sum(r.get("cost", {}).get("usage", {}).get("total_tokens", 0) for r in rows),
        "total_sample_seconds": sum(r.get("cost", {}).get("elapsed_seconds", 0) for r in rows),
        "note": "Tool/session completion is independent of QA score. Legacy/native changes include termination/error semantics; this is not a role-only causal ablation."}
    save_json(output / "summary.json", summary)
    save_json(output / "results.json", rows)
    return summary

def run(args):
    from runtime import load, configure_transport, implementation_hash
    from transport import Transport, Failure
    from agent import build_agent_class
    baseline, output = args.baseline.resolve(), args.output.resolve()
    runtime = load(baseline)
    settings = runtime[-1]
    cases = read_json(baseline / "cases.json")
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / ".run.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    fingerprint = {"mode": args.mode, "baseline_manifest_sha256": digest(baseline / "manifest.json"),
                   "implementation_sha256": implementation_hash()}
    config_path = output / "run_config.json"
    if config_path.exists() and read_json(config_path) != fingerprint:
        raise ValueError("Output directory belongs to a different baseline/mode/implementation")
    save_json(config_path, fingerprint)
    for number, case in enumerate(cases, 1):
        index = case["_sample_index"]
        directory = output / f"sample_{index:04d}"
        directory.mkdir(exist_ok=True)
        result_path = directory / "result.json"
        old = read_json(result_path) if result_path.exists() else None
        skip_agent = old is not None and not (args.retry_failed and old["execution_status"] != "completed")
        if args.stage == "score" and old is None:
            print(f"[{args.mode}] #{index}: no saved agent output; skipped", flush=True)
            continue
        if old and skip_agent and (args.stage == "agent" or old.get("scoring_status") in {"scored", "not_answered"}):
            print(f"[{args.mode}] #{index}: reused", flush=True)
            continue
        attempt = len(list(directory.glob("attempt_*.jsonl"))) + 1
        audit = Audit(directory / f"attempt_{attempt:03d}.jsonl")
        audit.emit("sample_start", sample_index=index, mode=args.mode, fingerprint=fingerprint,
                   question=case["question"], doc_id=case["doc_id"], settings=settings)
        transport = Transport(settings, audit)
        configure_transport(runtime, transport, args.mode == "native")
        try:
            if skip_agent:
                result = old
                audit.emit("reuse_agent_output", previous_cost=old.get("cost"), raw_answer=old.get("raw_answer"))
            else:
                document = runtime[1].load_document(case["doc_id"])
                transport.deadline = time.monotonic() + settings["max_wall_seconds"]
                try:
                    if args.mode == "legacy":
                        result = legacy_run(runtime, document, case["question"], audit)
                    else:
                        result = build_agent_class(runtime)(document, transport, audit).run_native(case["question"])
                except Exception as exc:
                    failure = exc if isinstance(exc, Failure) else Failure("pipeline_exception", f"{type(exc).__name__}: {exc}")
                    result = {"execution_status": "error", "semantic_status": None,
                              "raw_answer": None, "stop_reason": failure.code, "error": failure.as_dict(),
                              "rounds": 0, "evidence_pages": [], "protocol_valid": False}
                    audit.emit("pipeline_error", error=failure.as_dict())
                result.update(sample_index=index, doc_id=case["doc_id"], question=case["question"],
                              mode=args.mode, agent_cost=audit.costs())
                save_json(result_path, result)  # Always save QA result before answer extraction.
            transport.deadline = None
            if args.stage != "agent":
                audit.context = {"source": "offline_evaluation"}
                try:
                    score(result, case, baseline, transport, audit)
                except Exception as exc:
                    result.update(score=0.0, scoring_status="error",
                                  scoring_error=f"{type(exc).__name__}: {exc}")
                    audit.emit("scoring_error", error=result["scoring_error"])
            current_cost = audit.costs()
            if skip_agent:
                previous = result.get("cost", {})
                result["cost"] = {
                    **current_cost,
                    "http_requests": current_cost["http_requests"] + previous.get("http_requests", 0),
                    "http_failures": current_cost["http_failures"] + previous.get("http_failures", 0),
                    "elapsed_seconds": current_cost["elapsed_seconds"] + previous.get("elapsed_seconds", 0),
                    "usage": {k: current_cost["usage"].get(k, 0) + previous.get("usage", {}).get(k, 0)
                              for k in set(current_cost["usage"]) | set(previous.get("usage", {}))},
                    "previous_attempt_cost": previous}
            else:
                result["cost"] = current_cost
            result["latest_trace"] = str(audit.path)
            save_json(result_path, result)
            summary = summarize(cases, output)
            print(f"[{args.mode} {number}/{len(cases)}] #{index} {result['execution_status']} "
                  f"stop={result['stop_reason']} answer={result.get('raw_answer')!r} "
                  f"score={result.get('score')} seconds={result['cost']['elapsed_seconds']:.1f}", flush=True)
        finally:
            transport.close()
    print(json.dumps(summarize(cases, output), ensure_ascii=False, indent=2))
    lock.close()

def compare(args):
    args.output.mkdir(parents=True, exist_ok=True)
    for mode in ["legacy", "native"]:
        command = [sys.executable, str(HERE / "run.py"), "run", "--mode", mode,
                   "--baseline", str(args.baseline), "--output", str(args.output / mode), "--stage", args.stage]
        if args.retry_failed:
            command.append("--retry-failed")
        log = args.output / (mode + ".log")
        with log.open("a") as f:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    f.write(line)
                    f.flush()
                if process.wait():
                    raise RuntimeError(mode + " worker failed; see " + str(log))
            except BaseException:
                process.terminate()
                process.wait()
                raise
    summaries = {mode: read_json(args.output / mode / "summary.json") for mode in ["legacy", "native"]}
    save_json(args.output / "comparison.json", summaries)
    lines = ["# CiteAgent 小样本对照", "", "工具完成率与答案分数分别统计；未完成题按零分计入固定分母。",
             "", "| 模式 | 正常完成 | 判分样本 | QA 分数 | 请求数 | 总 token | 总耗时秒 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for mode, s in summaries.items():
        lines.append(f"| {mode} | {s['completed']}/{s['total']} | {s['scored']} | "
                     f"{s['accuracy_with_incomplete_as_zero']:.4f} | {s['total_llm_requests']} | "
                     f"{s['total_reported_tokens']} | {s['total_sample_seconds']:.1f} |")
    lines.extend(["", "两组均使用冻结题集、数据和推理参数。新模式还修正终止/错误语义，不能将差异完全归因于 role 字段。", ""])
    (args.output / "comparison.md").write_text("\n".join(lines))

def doctor(args):
    import httpx
    settings = read_json(args.baseline / "settings.json")
    rows = []
    with httpx.Client(trust_env=False, timeout=5) as client:
        for name, spec in settings["roles"].items():
            try:
                response = client.get(spec["url"].rstrip("/") + "/models")
                response.raise_for_status()
                models = [m["id"] for m in response.json()["data"]]
                rows.append({"role": name, "url": spec["url"], "configured_model": spec["model"],
                             "served_models": models, "model_available": spec["model"] in models})
            except Exception as exc:
                rows.append({"role": name, "url": spec["url"], "error": f"{type(exc).__name__}: {exc}"})
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    return 0 if all(row.get("model_available") for row in rows) else 1

def main():
    args = parser().parse_args()
    if args.command == "freeze":
        freeze(args)
    elif args.command == "run":
        run(args)
    elif args.command == "compare":
        compare(args)
    elif args.command == "verify":
        print(json.dumps(verify(args.baseline), ensure_ascii=False, indent=2))
    elif args.command == "doctor":
        return doctor(args)
    elif args.command == "replay":
        events = [json.loads(line) for line in args.trace.read_text().splitlines()]
        event = next(e for e in events if e["sequence"] == args.sequence and e["event"] == "llm_request")
        assets = Assets(args.trace.parent / "assets")
        body = assets.unpack(event["body"])
        save_json(args.output, body)
        print("Restored request (not submitted):", args.output)
    elif args.command == "validate":
        return subprocess.call([sys.executable, str(HERE / "tests.py")], cwd=HERE)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
