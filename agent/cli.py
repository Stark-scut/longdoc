"""Single entry point for the independent Agent."""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time
import httpx
from .config import ROOT, load, validate, with_data_dir
from .logging import Audit, atomic, read, replay

def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for command in ("doctor", "run", "eval"):
        q = sub.add_parser(command)
        q.add_argument("--config", type=Path)
        q.add_argument("--data-dir", type=Path)
        for name in ("samples", "blocks", "images-dir", "crop-root", "captions", "model-path"):
            q.add_argument("--" + name, type=Path)
        q.add_argument("--base-url")
        q.add_argument("--model")
        q.add_argument("--extractor-base-url")
        q.add_argument("--extractor-model")
        for name in ("max-rounds", "read-page-max", "max-model-len", "max-pixels", "seed",
                     "locator-max-tokens", "reasoner-max-tokens"):
            q.add_argument("--" + name, type=int)
        q.add_argument("--timeout", type=float)
        q.add_argument("--wall-seconds", type=float)
        q.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=None)
        q.add_argument("--reasoner-thinking", action=argparse.BooleanOptionalAction, default=None)
    q = sub.choices["run"]
    q.add_argument("--doc-id", required=True)
    q.add_argument("--question", required=True)
    q.add_argument("--output-dir", type=Path)
    q = sub.choices["eval"]
    q.add_argument("--gpu-ids")
    q.add_argument("--ports")
    q.add_argument("--no-serve", action="store_true")
    q.add_argument("--concurrency-per-gpu", "--concurrency", type=int, default=1)
    q.add_argument("--max-num-seqs", type=int)
    q.add_argument("--max-num-batched-tokens", type=int)
    q.add_argument("--mm-processor-cache-gb", type=float)
    q.add_argument("--enable-prefix-caching", action=argparse.BooleanOptionalAction, default=None)
    q.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=None)
    q.add_argument("--gpu-memory-utilization", type=float)
    q.add_argument("--output-dir", type=Path, default=ROOT / "output/agent/native")
    q.add_argument("--indices")
    q.add_argument("--limit", type=int, default=0)
    q.add_argument("--stage", choices=("all", "agent", "score"), default="all")
    q.add_argument("--retry-failed", action="store_true")
    q.add_argument("--ready-timeout", type=float, default=1200)
    mode = q.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--prepare-only", action="store_true")
    q = sub.add_parser("watch")
    q.add_argument("--output-dir", type=Path, required=True)
    q.add_argument("--interval", type=float, default=5)
    q = sub.add_parser("merge")
    q.add_argument("--output-dir", type=Path, required=True)
    q = sub.add_parser("replay")
    q.add_argument("--trace", type=Path, required=True)
    q.add_argument("--sequence", type=int, required=True)
    q.add_argument("--output", type=Path, required=True)
    q = sub.add_parser("_worker", help=argparse.SUPPRESS)
    q.add_argument("--output-dir", type=Path, required=True)
    q.add_argument("--replica", type=int, required=True)
    q.add_argument("--slot", type=int, required=True)
    return p

def configuration(args):
    config = load(args.config)
    if args.data_dir:
        config = with_data_dir(config, args.data_dir)
    for name in ("samples", "blocks", "images_dir", "crop_root", "captions", "model_path",
                 "max_rounds", "read_page_max", "max_model_len", "max_pixels", "seed",
                 "wall_seconds"):
        value = getattr(args, name, None)
        if value is not None:
            config[name] = str(value) if isinstance(value, Path) else value
    for role, spec in config["roles"].items():
        if args.base_url:
            spec["url"] = args.base_url
        if args.model:
            spec["model"] = args.model
        if args.timeout is not None:
            spec["timeout"] = args.timeout
    for role, attr in (("locator", "thinking"), ("reasoner", "reasoner_thinking")):
        value = getattr(args, attr)
        if value is not None:
            config["roles"][role]["thinking"] = value
    for role in ("locator", "reasoner"):
        value = getattr(args, role + "_max_tokens")
        if value is not None:
            config["roles"][role]["max_tokens"] = value
    if args.extractor_base_url:
        config["roles"]["extractor"]["url"] = args.extractor_base_url
    if args.extractor_model:
        config["roles"]["extractor"]["model"] = args.extractor_model
    for name in config["server"]:
        value = getattr(args, name, None)
        if value is not None:
            config["server"][name] = value
    return validate(config)

def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "_worker":
            from .evaluation import worker
            return worker(args.output_dir, args.replica, args.slot)
        if args.command in {"watch", "merge"}:
            from .evaluation import aggregate
            if args.command == "merge":
                print(json.dumps(aggregate(args.output_dir, persist=True), ensure_ascii=False, indent=2))
                return 0
            if args.interval <= 0:
                raise ValueError("interval must be positive")
            while True:
                print(json.dumps(aggregate(args.output_dir), ensure_ascii=False), flush=True)
                status = read(args.output_dir / "run_status.json")
                if status["status"] in {"finished", "failed", "interrupted"}:
                    return 0
                time.sleep(args.interval)
        if args.command == "replay":
            replay(args.trace, args.sequence, args.output)
            print("Restored request: " + str(args.output))
            return 0
        config = configuration(args)
        if args.command == "doctor":
            results = []
            with httpx.Client(trust_env=False, timeout=10) as client:
                for role, spec in config["roles"].items():
                    response = client.get(spec["url"].rstrip("/") + "/models",
                        headers={"Authorization": "Bearer " + os.getenv(spec["key_env"], "EMPTY")})
                    response.raise_for_status()
                    names = [v["id"] for v in response.json()["data"]]
                    results.append({"role": role, "url": spec["url"], "model": spec["model"],
                                    "served_models": names, "available": spec["model"] in names})
            print(json.dumps(results, ensure_ascii=False, indent=2))
            return 0 if all(r["available"] for r in results) else 1
        if args.command == "eval":
            from .launcher import launch
            return launch(args, config)
        from .documents import Documents, validate_images
        from .client import Client
        from .loop import Agent
        root = args.output_dir or ROOT / "output/agent" / ("question_" + time.strftime("%Y%m%d_%H%M%S") + "_" + str(os.getpid()))
        root = Path(root).resolve()
        if root.exists() and any(root.iterdir()):
            raise ValueError("Use a new output directory for a standalone question")
        if Path(__file__).parent.resolve() in root.parents:
            raise ValueError("Outputs must be outside agent/")
        doc = Documents(config, [args.doc_id])[args.doc_id]
        validate_images(doc, config["max_pixels"])
        audit = Audit(root / "trace.jsonl")
        client = Client(config, audit)
        try:
            result = Agent(doc, client, audit, config).run(args.question)
            result["cost"] = audit.costs()
            atomic(root / "result.json", result)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result["execution_status"] == "completed" else 1
        finally:
            client.close()
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
