#!/usr/bin/env python3
"""Independent single-GPU vLLM replicas, native Agent evaluation and resume."""
from __future__ import annotations
import argparse
import contextlib
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import uuid
sys.dont_write_bytecode = True
import httpx
from tqdm import tqdm
from support import (SOURCE_CORE, ARCHIVE, DATA, MODEL, read, sha, atomic, stamp,
                     prepare, verify_snapshot, select_cases, inspect_inputs,
                     result_path, needs_work, aggregate, duration, verify_model)

DEFAULTS = {
    "input": str(DATA / "samples.json"), "blocks": str(DATA / "stage2_blocks_cache.jsonl"),
    "images_dir": str(DATA / "images_1024"), "crop_root": str(DATA / "visual_crops"),
    "captions": str(DATA / "crop_captions.jsonl"), "model_path": str(MODEL),
    "max_rounds": 15, "read_page_max": 3, "thinking": True, "reasoner_thinking": False,
    "locator_max_tokens": 8192, "reasoner_max_tokens": 8192, "selector_max_tokens": 4096,
    "extract_max_tokens": 2048, "max_pixels": 1048576, "max_model_len": 65536,
    "request_attempts": 2, "json_attempts": 2, "seed": 42, "controller_fallback": False,
    "timeout": 600.0, "wall_seconds": 3600.0, "limit": 0, "indices": None,
    "extractor_base_url": None, "extractor_model": None}

def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu-ids", help="Physical GPU IDs: one full model per GPU, e.g. 1,4,5 or 1,2")
    p.add_argument("--ports", help="One distinct port per GPU; defaults to 8071 onward")
    p.add_argument("--output-dir", type=Path, default=SOURCE_CORE / "runs/native_4gpu")
    for name in ["input", "blocks", "images_dir", "crop_root", "captions", "model_path"]:
        p.add_argument("--" + name.replace("_", "-"), type=Path)
    for name in ["max_rounds", "read_page_max", "locator_max_tokens", "reasoner_max_tokens",
                 "selector_max_tokens", "extract_max_tokens", "max_pixels", "max_model_len",
                 "request_attempts", "json_attempts", "seed", "limit"]:
        p.add_argument("--" + name.replace("_", "-"), type=int)
    for name in ["thinking", "reasoner_thinking", "controller_fallback"]:
        p.add_argument("--" + name.replace("_", "-"), action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--indices", help="Diagnostic subset of original indices; omitted means all samples")
    p.add_argument("--timeout", type=float, help="Per HTTP request; fresh-run default 600 seconds")
    p.add_argument("--wall-seconds", type=float, help="Soft per-question Agent budget; default 3600")
    p.add_argument("--extractor-base-url")
    p.add_argument("--extractor-model")
    p.add_argument("--concurrency", "--concurrency-per-gpu", dest="concurrency", type=int, default=1,
                   help="Concurrent questions per GPU (isolated worker processes); default 1")
    p.add_argument("--max-num-seqs", type=int, help="Per replica scheduler limit; defaults to concurrency")
    p.add_argument("--max-num-batched-tokens", type=int, default=2048)
    p.add_argument("--max-images", type=int, default=128, help="Images allowed in one accumulated Agent request")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    p.add_argument("--mm-processor-cache-gb", type=float, default=1,
                   help="Per vLLM process MM preprocessor cache GiB; default 1 (enabled), 0 disables")
    p.add_argument("--enable-prefix-caching", action=argparse.BooleanOptionalAction, default=True,
                   help="Enable prefix KV/state reuse by default; --no-enable-prefix-caching opts out; restart required")
    p.add_argument("--ready-timeout", type=float, default=1200)
    p.add_argument("--progress-interval", type=float, default=5)
    p.add_argument("--stage", choices=["all", "agent", "score"], default="all")
    p.add_argument("--retry-failed", action="store_true", help="Retry failed/incomplete questions; re-score saved successful QA without rerunning it")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Check inputs and show deployment commands; no writes/GPU/API requests")
    mode.add_argument("--prepare-only", action="store_true", help="Prepare frozen full input/code, without deploying")
    mode.add_argument("--watch", action="store_true", help="Read progress of an existing run")
    mode.add_argument("--merge-only", action="store_true")
    p.add_argument("--no-serve", action="store_true", help="Use existing compatible services, one per port; never stop them")
    return p

def parameters(args, previous=None):
    saved = (previous or {}).get("parameters", {})
    values = {}
    for name, default in DEFAULTS.items():
        given = getattr(args, name)
        value = saved.get(name, default) if given is None else given
        if name in {"input", "blocks", "images_dir", "crop_root", "captions", "model_path"}:
            value = str(Path(value).resolve())
        values[name] = value
    if bool(values["extractor_base_url"]) != bool(values["extractor_model"]):
        raise ValueError("Set both --extractor-base-url and --extractor-model")
    if any(values[k] <= 0 for k in ["max_rounds", "read_page_max", "locator_max_tokens",
        "reasoner_max_tokens", "selector_max_tokens", "extract_max_tokens", "max_model_len",
        "max_pixels", "timeout", "wall_seconds", "request_attempts", "json_attempts"]):
        raise ValueError("Budgets must be positive")
    if values["limit"] < 0:
        raise ValueError("--limit must be nonnegative")
    if previous:
        changes = {k: (saved.get(k), v) for k, v in values.items()
                   if k not in {"timeout", "wall_seconds"} and saved.get(k) != v}
        if changes:
            raise ValueError("Semantic/data configuration changed; use a new --output-dir: " + repr(changes))
    return values

def gpu_ports(args, require_gpus=True, default_replicas=4):
    gpus = [int(x.strip()) for x in args.gpu_ids.split(",")] if args.gpu_ids else []
    if gpus and (len(set(gpus)) != len(gpus) or min(gpus) < 0):
        raise ValueError("--gpu-ids must contain distinct nonnegative GPU IDs")
    if not gpus and require_gpus:
        raise ValueError("Specify idle cards using --gpu-ids 1,4,5 (any positive number of cards)")
    count = len(gpus) or default_replicas
    ports = [int(x.strip()) for x in args.ports.split(",")] if args.ports else list(range(8071, 8071 + count))
    if not ports or len(set(ports)) != len(ports) or any(p < 1024 or p > 65535 for p in ports):
        raise ValueError("--ports must contain distinct ports in 1024..65535")
    if gpus and len(ports) != len(gpus):
        raise ValueError("--ports must contain exactly one port per GPU")
    return gpus, ports

def server_command(args, p, port):
    return [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", p["model_path"], "--served-model-name", "citeagent-qwen35",
        "--host", "127.0.0.1", "--port", str(port),
        "--tensor-parallel-size", "1", "--dtype", "bfloat16", "--cpu-offload-gb", "0",
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--max-model-len", str(p["max_model_len"]), "--max-num-seqs", str(args.max_num_seqs or args.concurrency),
        "--max-num-batched-tokens", str(args.max_num_batched_tokens),
        "--enable-chunked-prefill",
        "--enable-prefix-caching" if args.enable_prefix_caching else "--no-enable-prefix-caching",
        "--enable-prompt-tokens-details", "--enforce-eager",
        "--limit-mm-per-prompt", json.dumps({"image":{"count":args.max_images,"width":1024,"height":1024},"video":0}),
        "--mm-processor-kwargs", json.dumps({"min_pixels":1024,"max_pixels":p["max_pixels"]}),
        "--mm-processor-cache-gb", str(args.mm_processor_cache_gb), "--chat-template-content-format", "openai",
        "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder", "--reasoning-parser", "qwen3",
        "--default-chat-template-kwargs", json.dumps({"enable_thinking":p["thinking"]})]

def check_hardware(args, gpus, ports):
    for port in ports:
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                raise ValueError(f"Port {port} is occupied; choose --ports or use --no-serve")
    output = subprocess.check_output(["nvidia-smi",
        "--query-gpu=index,memory.free,memory.total,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
    rows = {int(v[0]): list(map(float, v[1:])) for v in
            ([x.strip() for x in line.split(",")] for line in output.splitlines())}
    for gpu in gpus:
        if gpu not in rows:
            raise ValueError(f"GPU {gpu} not found")
        free, total, util = rows[gpu]
        if util != 0 or free < total * args.gpu_memory_utilization:
            raise ValueError(f"GPU {gpu} not idle/enough memory: free={free}MiB, util={util}%; existing jobs will not be stopped")
    return rows

def stop_groups(processes, sig=signal.SIGTERM, grace=15):
    for proc in reversed(processes):
        if proc.poll() is not None and Path(f"/proc/{proc.pid}").exists():
            continue  # PID has been reused; do not signal an unrelated process.
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, sig)
    deadline = time.monotonic() + grace
    for proc in processes:
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=max(0.01, deadline - time.monotonic()))
    for proc in reversed(processes):
        if proc.poll() is not None and Path(f"/proc/{proc.pid}").exists():
            continue
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()

def install_shutdown_handlers():
    """First stop signal enters finally; subsequent signals cannot abort cleanup."""
    requested = False
    def request_stop(signum, frame):
        nonlocal requested
        if requested:
            return
        requested = True
        raise KeyboardInterrupt
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, request_stop)

@contextlib.contextmanager
def cleanup_guard():
    """Protect normal-exit cleanup as well as cleanup after an interruption."""
    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    try:
        for sig in saved:
            signal.signal(sig, signal.SIG_IGN)
        yield
    finally:
        for sig, handler in saved.items():
            signal.signal(sig, handler)

def cleanup_children(workers, servers):
    errors = []
    for phase, processes, sig, grace in [
        ("workers", workers, signal.SIGINT, 8),
        ("servers", servers, signal.SIGTERM, 15),
    ]:
        try:
            stop_groups(processes, sig, grace=grace)
        except Exception as exc:
            errors.append({"phase": phase, "error": f"{type(exc).__name__}: {exc}"})
            print(f"清理 {phase} 出错：{exc}", file=sys.stderr, flush=True)
    return errors

class Dashboard:
    def __init__(self, root, config, running, write=True):
        self.root, self.config, self.running, self.write = root, config, running, write
        self.started = time.monotonic()
        initial = aggregate(root, config, running, persist=False)["processed"]
        self.initial = initial
        self.bar = tqdm(total=config["total"], initial=initial, desc="Native Agent", unit="题", dynamic_ncols=True)
        self.last = None
    def update(self):
        s = aggregate(self.root, self.config, self.running, persist=False)
        self.bar.update(s["processed"] - self.bar.n)
        elapsed = time.monotonic() - self.started
        added = s["processed"] - self.initial
        rate = added / elapsed if elapsed > 0 and added > 0 else 0
        eta = s["pending"] / rate if rate else (0 if not s["pending"] else None)
        shard_text = []
        for shard in s["shards"]:
            active = []
            for path in (self.root / "workers").glob(f"shard{shard['shard']}_slot*.json"):
                state = read(path)
                if state.get("run_id") == self.running.get("run_id") and state.get("status") == "running":
                    age = int(max(0, time.time() - state.get("request_started_at", time.time())))
                    active.append(f"#{state.get('sample_index')} r{state.get('round') or '-'}:{state.get('role') or state.get('phase')} {age}s")
            gpu = self.running.get("gpu_ids", [f"R{i}" for i in range(len(s["shards"]))])[shard["shard"]]
            label = f"G{gpu}" if isinstance(gpu, int) else gpu
            text = f"{label} {shard['processed']}/{shard['total']}"
            if active:
                text += " " + ",".join(active)
            shard_text.append(text)
        self.bar.set_postfix_str(
            f"已判分{s['scored']} 技术失败{s['agent_errors']+s['scoring_errors']} 预算/未完成{s['agent_incomplete']} | "
            + " | ".join(shard_text) + f" | {rate*3600:.1f}题/h ETA {duration(eta)}", refresh=True)
        s.update(session_elapsed_seconds=elapsed, questions_per_hour=rate*3600, eta_seconds=eta)
        if self.write:
            atomic(self.root / "progress.json", s)
        self.last = s
        return s
    def close(self):
        self.bar.close()

def readiness(ports, servers, timeout):
    pending, deadline = set(range(len(ports))), time.monotonic() + timeout
    with httpx.Client(trust_env=False, timeout=3) as client:
        while pending and time.monotonic() < deadline:
            for i in list(pending):
                if servers and servers[i].poll() is not None:
                    raise RuntimeError(f"Replica {i} exited; see server{i}/server.log")
                try:
                    response = client.get(f"http://127.0.0.1:{ports[i]}/v1/models")
                    if response.status_code == 200 and "citeagent-qwen35" in [m["id"] for m in response.json()["data"]]:
                        pending.remove(i)
                        print(f"副本 {i} 已就绪：:{ports[i]}", flush=True)
                except (httpx.RequestError, ValueError, KeyError):
                    pass
            if pending:
                time.sleep(2)
        if pending:
            raise TimeoutError("Replicas not ready: " + repr(sorted(pending)))

def main():
    args = parser().parse_args()
    root = args.output_dir.resolve()
    if root == SOURCE_CORE or SOURCE_CORE not in root.parents:
        raise ValueError("--output-dir must be a new/result subdirectory inside citeagent_v2")
    previous = read(root / "run_config.json") if (root / "run_config.json").exists() else None
    if args.watch or args.merge_only:
        if not previous:
            raise ValueError("No prepared run found")
        if args.merge_only:
            with (root / ".run.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                s = aggregate(root, previous, persist=True)
                print(json.dumps(s, ensure_ascii=False, indent=2))
            return 0
        running = read(root / "runtime.json") if (root / "runtime.json").exists() else {"stage":"all"}
        view = Dashboard(root, previous, running, write=False)
        try:
            while True:
                s = view.update()
                if s["run_finished"] or read(root / "run_status.json").get("status") in {"interrupted","failed","finished","prepared"}:
                    break
                time.sleep(args.progress_interval)
        finally:
            view.close()
        return 0
    p = parameters(args, previous)
    last_runtime = read(root / "runtime.json") if (root / "runtime.json").exists() else {}
    default_replicas = last_runtime.get("replica_count", (previous or {}).get("replica_count", 4))
    gpus, ports = gpu_ports(args, not (args.prepare_only or args.no_serve), default_replicas)
    replicas = len(ports)
    if previous and not previous.get("flexible_replicas") and replicas != previous.get("replica_count", 4):
        raise ValueError("Old snapshot has fixed shards; use a new --output-dir to change replica count")
    if args.concurrency < 1 or args.concurrency > 16 or (args.max_num_seqs is not None and args.max_num_seqs < args.concurrency):
        raise ValueError("concurrency must be 1..16 and max-num-seqs >= concurrency")
    if min(args.max_num_batched_tokens, args.max_images, args.progress_interval, args.ready_timeout) <= 0:
        raise ValueError("Scheduling/time budgets must be positive")
    if not 0 < args.gpu_memory_utilization < 1 or args.mm_processor_cache_gb < 0:
        raise ValueError("Invalid GPU/cache allocation")
    if previous:
        verify_snapshot(root, previous)
        cases = read(root / "snapshot/cases.json")
    else:
        cases = select_cases(read(p["input"]), p["limit"], p["indices"])
    if args.dry_run:
        info = inspect_inputs(p, cases)
        print(json.dumps({"selected":len(cases), "documents":len(info), "per_replica":[sum(c["_sample_index"]%replicas==i for c in cases) for i in range(replicas)],
            "gpu_ids":gpus, "ports":ports, "per_gpu_concurrency":args.concurrency,
            "parameters":p, "server_commands":[server_command(args,p,port) for port in ports],
            "note":"Dry run: no writes, API requests or model deployment."}, ensure_ascii=False, indent=2))
        return 0
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("This output directory is already running")
        if not previous:
            marker = root / ".prepare_state.json"
            if (root / "snapshot").exists():
                if not marker.is_file() or read(marker).get("owner") != "citeagent_native_4gpu":
                    raise ValueError("Unrecognized incomplete snapshot; choose another output directory")
                shutil.rmtree(root / "snapshot")  # Only our unfinished generated snapshot; directory links are not followed.
            atomic(marker, {"owner":"citeagent_native_4gpu"})
            previous = prepare(root, p, cases, replica_count=replicas)
            marker.unlink(missing_ok=True)
        if args.prepare_only:
            atomic(root / "run_status.json", {"status":"prepared", "updated_at":stamp()})
            print("全量输入和代码已冻结：", root, flush=True)
            return 0
        todo = []
        for case in cases:
            path = result_path(root, case["_sample_index"])
            result = read(path) if path.exists() else None
            if result and result.get("input_fingerprint") != previous["fingerprint"]:
                raise ValueError("Saved result fingerprint mismatch")
            if needs_work(result, args.stage, args.retry_failed):
                todo.append(case["_sample_index"])
        if not todo:
            print("没有待处理题目；已有结果保留。")
            print(json.dumps(aggregate(root, previous), ensure_ascii=False, indent=2))
            return 0
        assignments = {}
        for i in range(replicas):
            ids = [n for n in todo if n%replicas==i]
            for slot in range(args.concurrency):
                assignments[f"{i}:{slot}"] = ids[slot::args.concurrency]
        running = {"run_id":uuid.uuid4().hex, "started_at":stamp(), "stage":args.stage, "launcher_pid":os.getpid(),
            "retry_failed":args.retry_failed, "timeout":p["timeout"], "wall_seconds":p["wall_seconds"],
            "enable_prefix_caching":args.enable_prefix_caching,
            "mm_processor_cache_gb":args.mm_processor_cache_gb, "concurrency_per_gpu":args.concurrency,
            "max_num_seqs":args.max_num_seqs if args.max_num_seqs is not None else args.concurrency,
            "replica_count":replicas, "gpu_ids":gpus or [f"R{i}" for i in range(replicas)], "ports":ports, "todo":todo, "assignments":assignments}
        servers, workers, logs = [], [], []
        view, code, status = None, 1, "failed"
        install_shutdown_handlers()
        atomic(root / "runtime.json", running)
        with (root / "run_history.jsonl").open("a") as history:
            history.write(json.dumps({**running, "parameters":p, "no_serve":args.no_serve, "enable_prefix_caching":args.enable_prefix_caching})+"\n")
        atomic(root / "run_status.json", {"status":"starting", "run_id":running["run_id"], "updated_at":stamp()})
        try:
            verify_model(previous)
            if not args.no_serve:
                check_hardware(args, gpus, ports)
                for i, (gpu, port) in enumerate(zip(gpus, ports)):
                    directory = root / f"server{i}"
                    directory.mkdir(exist_ok=True)
                    env = os.environ.copy()
                    env.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(gpu),
                               OMP_NUM_THREADS="4", PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
                    for key in ["NPROC_PER_NODE","NNODES","NODE_RANK","RANK","LOCAL_RANK","WORLD_SIZE","LOCAL_WORLD_SIZE"]:
                        env.pop(key, None)
                    for key, suffix in {"TRITON_CACHE_DIR":"triton", "TORCHINDUCTOR_CACHE_DIR":"inductor",
                                        "VLLM_CACHE_ROOT":"vllm", "XDG_CACHE_HOME":"xdg"}.items():
                        cache = directory / "cache" / suffix
                        cache.mkdir(parents=True, exist_ok=True)
                        env[key] = str(cache)
                    command = server_command(args, p, port)
                    log = (directory / "server.log").open("a", buffering=1)
                    logs.append(log)
                    log.write("\nRUN "+running["run_id"]+" "+stamp()+"\n")
                    proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    servers.append(proc)
                    atomic(directory / "launch.json", {"run_id":running["run_id"], "gpu":gpu, "port":port,
                        "pid":proc.pid, "pgid":proc.pid, "session_id":proc.pid, "command":command, "dtype":"bfloat16", "tensor_parallel_size":1, "cpu_offload_gb":0})
            print(f"native 全量 {len(cases)} 题，本次待处理 {len(todo)}；{replicas} 个独立 BF16 / TP=1 副本，无 CPU offload。", flush=True)
            print(f"Locator thinking={p['thinking']}；Reasoner thinking={p['reasoner_thinking']}；每卡并发={args.concurrency}。", flush=True)
            print(f"Prefix caching={args.enable_prefix_caching}；MM processor cache={args.mm_processor_cache_gb} GiB；"
                  f"每副本 max_num_seqs={running['max_num_seqs']}；总评测进程={replicas * args.concurrency}。", flush=True)
            readiness(ports, servers, args.ready_timeout)
            (root / "workers").mkdir(exist_ok=True)
            for i in range(replicas):
                for slot in range(args.concurrency):
                    if not assignments[f"{i}:{slot}"]:
                        continue
                    log = (root / "workers" / f"shard{i}_slot{slot}.log").open("a", buffering=1)
                    logs.append(log)
                    env = os.environ.copy()
                    env.update(PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
                    workers.append(subprocess.Popen([sys.executable, str(root / "snapshot/full_eval/worker.py"),
                        "--output-dir",str(root),"--shard",str(i),"--slot",str(slot)],
                        env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
            atomic(root / "run_status.json", {"status":"running","run_id":running["run_id"],"updated_at":stamp()})
            view = Dashboard(root, previous, running)
            while any(w.poll() is None for w in workers):
                if any(w.poll() not in (None,0) for w in workers):
                    raise RuntimeError("Worker exited unexpectedly; see workers/shard*_slot*.log")
                if any(s.poll() is not None for s in servers):
                    raise RuntimeError("vLLM replica exited unexpectedly; see server*/server.log")
                view.update()
                time.sleep(args.progress_interval)
            if any(w.returncode != 0 for w in workers):
                raise RuntimeError("A worker failed; saved samples can be resumed")
            s = view.update()
            status, code = "finished", 0 if s["run_finished"] else 1
        except KeyboardInterrupt:
            status, code = "interrupted", 130
            print("\n正在停止工作进程并保存结果，然后清理模型服务；清理期间重复 Ctrl+C 会被忽略，请等待退出。", flush=True)
        except Exception as exc:
            print("\nERROR:", exc, file=sys.stderr, flush=True)
            status, code = "failed", 1
        finally:
            with cleanup_guard():
                # A status-write failure must not prevent terminating owned processes.
                with contextlib.suppress(OSError):
                    atomic(root / "run_status.json", {"status":"stopping","run_id":running["run_id"],"updated_at":stamp()})
                cleanup_errors = cleanup_children(workers, servers)
                if cleanup_errors:
                    status, code = "failed", 1
                if view:
                    view.close()
                for log in logs:
                    with contextlib.suppress(OSError):
                        log.close()
                s = aggregate(root, previous, running)
                atomic(root / "progress.json", s)
                atomic(root / "run_status.json", {"status":status,"run_id":running["run_id"],
                    "updated_at":stamp(),"cleanup_errors":cleanup_errors})
                print(f"结果：{root}\n已处理 {s['processed']}/{s['total']}；已判分 {s['scored']}；"
                      f"ACC={s['accuracy_percent']:.2f}% F1={s['f1_percent']:.2f}%（缺失/失败计零）", flush=True)
        return code

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError) as exc:
        print("ERROR:", exc, file=sys.stderr)
        raise SystemExit(2)
