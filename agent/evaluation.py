"""Immutable run snapshots, per-slot evaluation workers and aggregate results."""
from __future__ import annotations
import copy
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import time
import uuid
from .client import Client, Failure
from .documents import Documents, select_cases, validate_images, doc_key
from .logging import Audit, atomic, read, sha, fingerprint, stamp, add_cost
from .loop import Agent
from .scoring import score, metrics

def result_path(root, index):
    return Path(root) / "samples" / f"sample_{index:04d}" / "result.json"

def needs_work(result, stage, retry_failed=False):
    if result is None:
        return stage != "score"
    if stage != "score" and retry_failed and result.get("execution_status") != "completed":
        return True
    if stage == "agent":
        return False
    return result.get("scoring_status") not in {"scored", "not_answered", "error"} or (
        retry_failed and result.get("scoring_status") == "error")

def asset_inventory(documents, hashing=True):
    result = {}
    for doc_id, document in documents.documents.items():
        entries = []
        for page in document.pages.values():
            for kind, path in [("page", page.image), *[("crop", p) for p in page.crops.values()]]:
                stat = path.stat()
                entries.append({"path": str(path.resolve()), "page": page.index, "kind": kind,
                                "size": stat.st_size, **({"sha256": sha(path)} if hashing else {})})
        result[doc_id] = entries
    return result

def prepare(root, config, cases):
    root = Path(root)
    snapshot = root / "snapshot"
    if snapshot.exists():
        raise ValueError("Incomplete snapshot exists; use a new output directory")
    docs = Documents(config, [c["doc_id"] for c in cases])
    for doc in docs.documents.values():
        validate_images(doc, config["max_pixels"])
    inventory = asset_inventory(docs)
    snapshot.mkdir(parents=True)
    shutil.copytree(Path(__file__).parent, snapshot / "agent",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shared = snapshot / "inputs"
    shared.mkdir()
    wanted = set(docs.documents)
    # Copy only selected documents' OCR/captions; large images stay at their original paths.
    with Path(config["blocks"]).open() as src, (shared / "blocks.jsonl").open("w") as dst:
        for line in src:
            if Path(json.loads(line)["image_path"]).parent.name in wanted:
                dst.write(line)
    captions = Path(config["captions"])
    with (shared / "captions.jsonl").open("w") as dst:
        if captions.is_file():
            with captions.open() as src:
                for line in src:
                    if doc_key(json.loads(line)["doc_id"]) in wanted:
                        dst.write(line)
    atomic(shared / "cases.json", cases)
    atomic(shared / "assets.json", inventory)
    model = Path(config["model_path"])
    identity = {"metadata": {}, "weights": {}}
    if model.is_dir():
        metadata = snapshot / "model_metadata"; metadata.mkdir()
        for name in ("config.json", "generation_config.json", "tokenizer_config.json",
                     "chat_template.jinja", "model.safetensors.index.json"):
            p = model / name
            if p.is_file():
                shutil.copy2(p, metadata / name)
                identity["metadata"][name] = sha(p)
        identity["weights"] = {p.name: [p.stat().st_size, p.stat().st_mtime_ns] for p in model.glob("*.safetensors")}
    snapshot_files = {str(p.relative_to(snapshot)): sha(p) for p in sorted(snapshot.rglob("*")) if p.is_file()}
    run = {"schema": 2, "implementation": "standalone-agent", "config": config,
           "model_identity": identity, "snapshot_files": snapshot_files,
           "total": len(cases), "created_at": stamp()}
    run["fingerprint"] = fingerprint({"config": config, "files": snapshot_files, "model": identity})
    atomic(root / "run_config.json", run)
    return run

def verify(root, run, config=None):
    root = Path(root)
    if run.get("implementation") != "standalone-agent" or run.get("schema") != 2:
        raise ValueError("Output uses a different Agent protocol; use a new directory or the original code branch")
    if config is not None and config != run["config"]:
        raise ValueError("Configuration differs from saved run; use its config or a new output directory")
    for name, expected in run["snapshot_files"].items():
        if sha(root / "snapshot" / name) != expected:
            raise ValueError("Frozen run source/input changed: " + name)
    identity = run["model_identity"]
    model = Path(run["config"]["model_path"])
    for name, expected in identity["metadata"].items():
        if sha(model / name) != expected:
            raise ValueError("Model metadata changed: " + name)
    if identity["weights"]:
        current = {p.name: [p.stat().st_size, p.stat().st_mtime_ns] for p in model.glob("*.safetensors")}
        if current != identity["weights"]:
            raise ValueError("Model weights inventory changed")

def aggregate(root, stage="all", persist=False):
    root = Path(root)
    run = read(root / "run_config.json")
    cases = read(root / "snapshot/inputs/cases.json")
    rows = []
    counts = {"processed": 0, "completed": 0, "incomplete": 0, "errors": 0,
              "scored": 0, "scoring_errors": 0}
    stop_reasons = {}
    for case in cases:
        p = result_path(root, case["_sample_index"])
        result = read(p) if p.exists() else {}
        if result and result.get("input_fingerprint") != run["fingerprint"]:
            raise ValueError("Result fingerprint mismatch: " + str(p))
        terminal = result.get("execution_status") in {"completed", "incomplete", "error"}
        settled = terminal and (stage == "agent" or result.get("scoring_status") in {"scored", "not_answered", "error"})
        counts["processed"] += bool(settled)
        counts["completed"] += result.get("execution_status") == "completed"
        counts["incomplete"] += result.get("execution_status") == "incomplete"
        counts["errors"] += result.get("execution_status") == "error"
        counts["scored"] += result.get("scoring_status") == "scored"
        counts["scoring_errors"] += result.get("scoring_status") == "error"
        if terminal:
            reason = result.get("stop_reason")
            stop_reasons[reason] = stop_reasons.get(reason, 0) + 1
        # Failed extraction remains zero even if an interrupted attempt had a previous prediction.
        value = result.get("score", 0.0) if result.get("scoring_status") == "scored" else 0.0
        row = {**case, **result, "score": value}
        rows.append(row)
    acc, f1 = metrics(rows)
    summary = {"implementation": "standalone-agent", "total": len(cases), **counts,
               "pending": len(cases) - counts["processed"], "accuracy": acc, "f1": f1,
               "stop_reasons": stop_reasons, "run_finished": counts["processed"] == len(cases),
               "evaluation_complete": counts["scored"] == len(cases),
               "note": "Fixed selected denominator; missing, incomplete and technical failures count as zero.",
               "updated_at": stamp()}
    if persist:
        atomic(root / "summary.json", summary)
        atomic(root / "merged.json", rows)
        atomic(root / "progress.json", summary)
    return summary

class Interrupted(BaseException):
    pass

def worker(root, replica, slot):
    root = Path(root).resolve()
    run = read(root / "run_config.json"); runtime = read(root / "runtime.json")
    config = copy.deepcopy(run["config"])
    original_url = config["roles"]["locator"]["url"]
    if runtime.get("map_replicas", runtime["local_servers"]):
        for spec in config["roles"].values():
            if spec["url"] == original_url:
                spec["url"] = f"http://127.0.0.1:{runtime['ports'][replica]}/v1"
    config["blocks"] = str(root / "snapshot/inputs/blocks.jsonl")
    config["captions"] = str(root / "snapshot/inputs/captions.jsonl")
    cases = {c["_sample_index"]: c for c in read(root / "snapshot/inputs/cases.json")}
    assigned = runtime["assignments"].get(f"{replica}:{slot}", [])
    documents = Documents(config, [cases[i]["doc_id"] for i in assigned]) if assigned else None
    inventory = read(root / "snapshot/inputs/assets.json")
    verified = set()
    heartbeat_path = root / "workers" / f"replica{replica}_slot{slot}.json"
    state = {"pid": os.getpid(), "replica": replica, "slot": slot, "run_id": runtime["run_id"], "status": "starting"}
    def heartbeat(event):
        state.update(event=event["event"], round=event.get("round"), role=event.get("role", state.get("role")),
                     updated_at=stamp(), elapsed_seconds=event["elapsed_seconds"])
        if event["event"] == "llm_request":
            state["request_started_at"] = time.time()
        atomic(heartbeat_path, state)
    def interrupt(signum, frame):
        raise Interrupted()
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupt)
    atomic(heartbeat_path, state)
    for index in assigned:
        case = cases[index]; path = result_path(root, index); path.parent.mkdir(parents=True, exist_ok=True)
        with (path.parent / ".sample.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            old = read(path) if path.exists() else None
            if old and old.get("input_fingerprint") != run["fingerprint"]:
                raise ValueError("Sample fingerprint mismatch")
            if not needs_work(old, runtime["stage"], runtime["retry_failed"]):
                continue
            attempts = [int(p.stem.split("_")[-1]) for p in path.parent.glob("attempt_*.jsonl")]
            trace = path.parent / f"attempt_{max(attempts, default=0) + 1:04d}.jsonl"
            audit = Audit(trace, root / "assets", heartbeat)
            client = Client(config, audit)
            result = None
            state.update(status="running", sample_index=index, phase="agent")
            audit.emit("sample_start", sample_index=index, question=case["question"],
                       doc_id=case["doc_id"], run_id=runtime["run_id"])
            try:
                reuse = old and (old.get("execution_status") == "completed" or runtime["stage"] == "score")
                if reuse:
                    result = copy.deepcopy(old)
                    audit.emit("reuse_agent_output", agent_trace=result.get("agent_trace"))
                else:
                    try:
                        doc_id = doc_key(case["doc_id"])
                        if doc_id not in verified:
                            for item in inventory[doc_id]:
                                if sha(item["path"]) != item["sha256"]:
                                    raise Failure("input_asset_changed", item["path"])
                            verified.add(doc_id)
                        result = Agent(documents[case["doc_id"]], client, audit, config).run(case["question"])
                    except Exception as exc:
                        failure = exc if isinstance(exc, Failure) else Failure("input_or_agent_error", str(exc))
                        result = {"execution_status": "error", "stop_reason": failure.code,
                                  "raw_answer": None, "rounds": 0, "error": failure.as_dict(), "evidence_pages": []}
                    result.update(sample_index=index, doc_id=case["doc_id"], question=case["question"],
                                  input_fingerprint=run["fingerprint"], agent_trace=str(trace),
                                  scoring_status="pending", phase="agent_saved", agent_cost=audit.costs())
                    atomic(path, {**result, "cost": add_cost((old or {}).get("cost"), audit.costs())})
                if runtime["stage"] != "agent":
                    state["phase"] = "score"; client.deadline = None
                    for key in ("pred", "extracted_res", "score", "answer_correct", "scoring_error"):
                        result.pop(key, None)
                    try:
                        audit.context = {"source": "offline_evaluation"}
                        score(result, case, client)
                    except Exception as exc:
                        result.update(score=0.0, scoring_status="error", scoring_error=str(exc))
                        audit.emit("scoring_error", error=str(exc))
                result.update(latest_trace=str(trace), run_id=runtime["run_id"], phase="done",
                              cost=add_cost((old or {}).get("cost"), audit.costs()), updated_at=stamp())
                atomic(path, result)
                print(f"#{index:04d} {result['execution_status']} stop={result['stop_reason']} score={result.get('score')}", flush=True)
            except Interrupted:
                if result is not None:
                    atomic(path, {**result, "cost": add_cost((old or {}).get("cost"), audit.costs()),
                                  "latest_trace": str(trace), "updated_at": stamp()})
                state.update(status="interrupted", updated_at=stamp()); atomic(heartbeat_path, state)
                return 130
            finally:
                client.close()
        state.update(status="between_samples", updated_at=stamp()); atomic(heartbeat_path, state)
    state.update(status="finished", updated_at=stamp()); atomic(heartbeat_path, state)
    return 0
