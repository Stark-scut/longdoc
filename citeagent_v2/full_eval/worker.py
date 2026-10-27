#!/usr/bin/env python3
"""An isolated process per replica/slot. No shared mutable Locator globals."""
from __future__ import annotations
import argparse
import base64
import copy
import fcntl
import hashlib
import json
import mimetypes
import os
from pathlib import Path
import signal
import sys
import time
import uuid
sys.dont_write_bytecode = True
from support import read, atomic, sha, stamp, result_path, needs_work

class Interrupted(BaseException):
    pass

def interrupt(signum, frame):
    raise Interrupted()

def add_cost(old, new):
    result = copy.deepcopy(new)
    for field in ["http_requests", "http_failures", "elapsed_seconds"]:
        result[field] = old.get(field, 0) + new.get(field, 0)
    result["usage"] = {k: old.get("usage", {}).get(k, 0) + new.get("usage", {}).get(k, 0)
                       for k in set(old.get("usage", {})) | set(new.get("usage", {}))}
    result["by_role"] = copy.deepcopy(old.get("by_role", {}))
    for role, values in new.get("by_role", {}).items():
        target = result["by_role"].setdefault(role, {})
        for k, v in values.items():
            target[k] = target.get(k, 0) + v
    return result

def run_worker(root, shard, slot):
    root = Path(root).resolve()
    config, running = read(root / "run_config.json"), read(root / "runtime.json")
    sys.path.insert(0, str(root / "snapshot/core"))
    from common import Audit, Assets
    from runtime import load, configure_transport
    from transport import Transport, Failure
    from agent import build_agent_class
    from run import score

    baseline_shard = 0 if config.get("flexible_replicas") else shard
    baseline = root / "snapshot" / f"shard{baseline_shard}"
    runtime = list(load(baseline))
    settings = runtime[-1] = copy.deepcopy(runtime[-1])
    for role, spec in settings["roles"].items():
        spec["url"] = f"http://127.0.0.1:{running['ports'][shard]}/v1"
        spec["timeout"] = running["timeout"]
        if role == "extractor" and config["parameters"]["extractor_base_url"]:
            spec["url"] = config["parameters"]["extractor_base_url"]
            spec["model"] = config["parameters"]["extractor_model"]
    settings["max_wall_seconds"] = running["wall_seconds"]
    runtime[0].REASONER_TIMEOUT = running["timeout"]
    runtime[0].GLOBAL_SELECTOR_TIMEOUT = running["timeout"]
    if os.getenv("EXTRACTOR_API_KEY") and not os.getenv("CITEAGENT_EXTRACTOR_API_KEY"):
        os.environ["CITEAGENT_EXTRACTOR_API_KEY"] = os.environ["EXTRACTOR_API_KEY"]
    indices = running["assignments"].get(f"{shard}:{slot}", [])
    cases = {c["_sample_index"]: c for c in read(root / "snapshot/cases.json")}
    inventory = read(root / "snapshot/assets_manifest.json")
    wanted = {cases[i]["doc_id"].removesuffix(".pdf") for i in indices}
    documents = {doc: runtime[1].Document(doc) for doc in wanted}
    with (root / "snapshot/data/blocks.jsonl").open() as f:
        for line in f:
            row = json.loads(line)
            path = Path(row["image_path"])
            doc = path.parent.name
            if doc in documents:
                page = int(path.stem.split("_")[-1])
                image = baseline / "images" / doc / path.name
                documents[doc].add_page(page, str(image), row["blocks"])
    for doc, document in documents.items():
        for resource in inventory[doc]:
            if resource["kind"] == "image":
                page = resource["page"]
                document.pages.setdefault(page, [])
                document._image_paths[page] = resource["path"]
    verified_docs = set()
    heartbeat = root / "workers" / f"shard{shard}_slot{slot}.json"
    state = {"pid": os.getpid(), "shard": shard, "slot": slot,
             "run_id": running["run_id"], "status": "starting"}

    class SafeAssets(Assets):
        def pack(self, value):
            if isinstance(value, list):
                return [self.pack(v) for v in value]
            if isinstance(value, dict):
                return {k: self.pack(v) for k, v in value.items()}
            if isinstance(value, str) and value.startswith("data:image/") and ";base64," in value:
                head, body = value.split(",", 1)
                raw = base64.b64decode(body, validate=True)
                ext = mimetypes.guess_extension(head[5:].split(";")[0]) or ".bin"
                name = hashlib.sha256(raw).hexdigest() + ext
                target = self.directory / name
                if not target.exists() or target.stat().st_size != len(raw):
                    temporary = target.with_name(name + "." + uuid.uuid4().hex + ".tmp")
                    try:
                        temporary.write_bytes(raw)
                        temporary.replace(target)
                    finally:
                        temporary.unlink(missing_ok=True)
                return "asset://" + name
            return value

    class LiveAudit(Audit):
        def emit(self, event, **fields):
            result = super().emit(event, **fields)
            state.update(event=event, role=fields.get("role", state.get("role")),
                         round=self.context.get("round"), updated_at=stamp(),
                         http_requests=self.requests, elapsed_seconds=time.monotonic() - self.started)
            if event == "llm_request":
                state["request_started_at"] = time.time()
            atomic(heartbeat, state)
            return result

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    for index in indices:
        case = cases[index]
        path = result_path(root, index)
        path.parent.mkdir(parents=True, exist_ok=True)
        with (path.parent / ".sample.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError(f"Another process owns sample {index}")
            old = read(path) if path.exists() else None
            if old and old.get("input_fingerprint") != config["fingerprint"]:
                raise ValueError("Saved result fingerprint mismatch")
            if not needs_work(old, running["stage"], running["retry_failed"]):
                continue
            if running["stage"] == "score" and not old:
                continue
            reuse = old is not None and (running["stage"] == "score" or old.get("execution_status") == "completed")
            attempt = len(list(path.parent.glob("attempt_*.jsonl"))) + 1
            trace = path.parent / f"attempt_{attempt:04d}.jsonl"
            assets = root / "assets"
            assets.mkdir(exist_ok=True)
            if not (path.parent / "assets").exists():
                (path.parent / "assets").symlink_to(assets, target_is_directory=True)
            audit = LiveAudit(trace)
            audit.assets = SafeAssets(assets)
            state.update(status="running", sample_index=index, phase="score" if reuse else "agent",
                         sample_started_at=time.time())
            atomic(heartbeat, state)
            transport = Transport(settings, audit)
            configure_transport(runtime, transport, True)
            result = copy.deepcopy(old) if reuse else None
            try:
                audit.emit("sample_start", sample_index=index, mode="native",
                           question=case["question"], doc_id=case["doc_id"],
                           run_id=running["run_id"], input_fingerprint=config["fingerprint"], settings=settings)
                if not reuse:
                    try:
                        doc = case["doc_id"].removesuffix(".pdf")
                        if doc not in verified_docs:
                            for entry in inventory[doc]:
                                if sha(entry["path"]) != entry["sha256"]:
                                    raise Failure("input_asset_changed", "Image/crop changed: " + entry["path"])
                            verified_docs.add(doc)
                        transport.deadline = time.monotonic() + running["wall_seconds"]
                        result = build_agent_class(runtime)(documents[doc], transport, audit).run_native(case["question"])
                    except Exception as exc:
                        failure = exc if isinstance(exc, Failure) else Failure("input_or_agent_error", f"{type(exc).__name__}: {exc}")
                        result = {"execution_status": "error", "stop_reason": failure.code,
                            "semantic_status": None, "raw_answer": None, "rounds": 0,
                            "evidence_pages": [], "protocol_valid": False, "error": failure.as_dict()}
                    result.update(sample_index=index, doc_id=case["doc_id"], question=case["question"],
                        mode="native", input_fingerprint=config["fingerprint"], run_id=running["run_id"],
                        scoring_status="pending", phase="agent_saved", agent_cost=audit.costs(),
                        cost=add_cost((old or {}).get("cost", {}), audit.costs()), latest_trace=str(trace))
                    atomic(path, result)
                else:
                    audit.emit("reuse_agent_output", raw_answer=result.get("raw_answer"),
                               original_trace=result.get("latest_trace"))
                result.update(run_id=running["run_id"], latest_trace=str(trace))
                transport.deadline = None
                if running["stage"] != "agent":
                    state["phase"] = "score"
                    audit.context = {"source": "offline_evaluation"}
                    for key in ["pred", "extracted_res", "score", "scoring_error", "answer_correct"]:
                        result.pop(key, None)
                    try:
                        score(result, case, baseline, transport, audit)
                    except Exception as exc:
                        result.update(score=0.0, scoring_status="error",
                                      scoring_error=f"{type(exc).__name__}: {exc}")
                        audit.emit("scoring_error", error=result["scoring_error"])
                result.update(phase="done", cost=add_cost((old or {}).get("cost", {}), audit.costs()),
                              updated_at=stamp())
                atomic(path, result)
                print(f"#{index:04d} {result['execution_status']} stop={result['stop_reason']} "
                      f"score={result.get('score')} answer={result.get('raw_answer')!r}", flush=True)
            except Interrupted:
                if result is not None:
                    result.update(cost=add_cost((old or {}).get("cost", {}), audit.costs()),
                                  latest_trace=str(trace), updated_at=stamp())
                    atomic(path, result)
                state.update(status="interrupted", updated_at=stamp())
                atomic(heartbeat, state)
                print(f"#{index:04d} interrupted; saved QA output is preserved", flush=True)
                return 130
            finally:
                transport.close()
        state.update(status="between_samples", updated_at=stamp())
        atomic(heartbeat, state)
    state.update(status="finished", updated_at=stamp())
    atomic(heartbeat, state)
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--slot", type=int, required=True)
    args = parser.parse_args()
    raise SystemExit(run_worker(args.output_dir, args.shard, args.slot))
