"""One full vLLM replica per selected GPU; isolated worker slots and safe cleanup."""
from __future__ import annotations
import contextlib
import copy
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import uuid
import httpx
from tqdm import tqdm
from .documents import select_cases
from .evaluation import prepare, verify, aggregate, needs_work, result_path
from .logging import atomic, read, stamp

def gpu_ports(gpu_ids, ports, local):
    ids = [] if not gpu_ids else [int(v.strip()) for v in gpu_ids.split(",")]
    if len(ids) != len(set(ids)) or any(i < 0 for i in ids):
        raise ValueError("GPU IDs must be distinct nonnegative integers")
    if local and not ids:
        raise ValueError("Pass --gpu-ids or use --no-serve for an existing service")
    numbers = [int(v.strip()) for v in ports.split(",")] if ports else list(range(8071, 8071 + max(1, len(ids))))
    if len(numbers) != len(set(numbers)) or any(not 1 <= p <= 65535 for p in numbers):
        raise ValueError("Ports must be distinct values in 1..65535")
    if ids and len(numbers) != len(ids):
        raise ValueError("One port is required per selected GPU")
    return ids, numbers

def server_command(config, port, concurrency, max_num_seqs=None):
    settings = config["server"]
    base_url = config["roles"]["locator"]["url"]
    names = list(dict.fromkeys(s["model"] for s in config["roles"].values() if s["url"] == base_url))
    command = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", config["model_path"], "--served-model-name", *names,
        "--host", "127.0.0.1", "--port", str(port), "--tensor-parallel-size", "1",
        "--dtype", "bfloat16", "--cpu-offload-gb", "0",
        "--gpu-memory-utilization", str(settings["gpu_memory_utilization"]),
        "--max-model-len", str(config["max_model_len"]),
        "--max-num-seqs", str(max_num_seqs or concurrency),
        "--max-num-batched-tokens", str(settings["max_num_batched_tokens"]),
        "--enable-chunked-prefill", "--mm-encoder-tp-mode", "data",
        "--limit-mm-per-prompt", json.dumps({"image": {"count": settings["max_images"],
             "width": 1024, "height": 1024}, "video": 0}),
        "--mm-processor-kwargs", json.dumps({"min_pixels": config["min_pixels"], "max_pixels": config["max_pixels"]}),
        "--mm-processor-cache-gb", str(settings["mm_processor_cache_gb"]),
        "--chat-template-content-format", "openai",
        "--default-chat-template-kwargs", '{"enable_thinking":true}',
        "--reasoning-parser", "qwen3", "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder",
        "--seed", str(config["seed"]), "--enable-prompt-tokens-details"]
    if settings["enable_prefix_caching"]:
        command.append("--enable-prefix-caching")
    else:
        command.append("--no-enable-prefix-caching")
    if settings["enforce_eager"]:
        command.append("--enforce-eager")
    return command

def port_available(port):
    with socket.socket() as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True

def check_gpus(ids):
    proc = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free,memory.total,utilization.gpu",
                           "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True)
    found = {}
    for line in proc.stdout.splitlines():
        i, free, total, utilization = [v.strip() for v in line.split(",")]
        found[int(i)] = (int(free), int(total), int(utilization))
    for gpu in ids:
        if gpu not in found:
            raise ValueError("GPU not found: " + str(gpu))
        free, total, utilization = found[gpu]
        if free < total * .85 or utilization > 10:
            raise ValueError(f"GPU {gpu} appears occupied (free={free}/{total} MiB, utilization={utilization}%)")
        print(f"GPU {gpu}: free={free}/{total} MiB", flush=True)

@contextlib.contextmanager
def interrupt_handler():
    previous = {}
    requested = [False]
    def stop(signum, frame):
        if not requested[0]:
            requested[0] = True
            raise KeyboardInterrupt
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        previous[sig] = signal.signal(sig, stop)
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)

@contextlib.contextmanager
def cleanup_guard():
    previous = {sig: signal.signal(sig, signal.SIG_IGN)
                for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)

def stop_groups(processes, sig, grace):
    # All supplied process groups were created by this launcher. Kill descendants
    # even if the group leader already exited after an EngineCore failure.
    for process in processes:
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline and any(p.poll() is None for p in processes):
        time.sleep(.1)
    for process in processes:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            pass

def wait_ready(servers, ports, config, timeout):
    deadline = time.monotonic() + timeout
    pending = set(range(len(servers)))
    with httpx.Client(trust_env=False, timeout=3) as client:
        while pending:
            for i in list(pending):
                if servers[i].poll() is not None:
                    raise RuntimeError(f"vLLM replica {i} exited; inspect server{i}/server.log")
                try:
                    response = client.get(f"http://127.0.0.1:{ports[i]}/v1/models")
                    names = {v["id"] for v in response.json().get("data", [])} if response.is_success else set()
                    if config["roles"]["locator"]["model"] in names:
                        pending.remove(i)
                        print(f"Replica {i} ready on :{ports[i]}", flush=True)
                except (httpx.HTTPError, ValueError):
                    pass
            if pending:
                if time.monotonic() >= deadline:
                    raise TimeoutError("vLLM readiness timeout")
                time.sleep(.5)

def launch(args, config):
    root = Path(args.output_dir).resolve()
    source_dir = Path(__file__).resolve().parent
    if root == source_dir or source_dir in root.parents:
        raise ValueError("Outputs must be outside the agent source directory")
    if args.limit < 0 or args.ready_timeout <= 0:
        raise ValueError("limit must be nonnegative and ready-timeout positive")
    if args.concurrency_per_gpu < 1 or args.concurrency_per_gpu > 16:
        raise ValueError("concurrency-per-gpu must be in 1..16")
    if args.max_num_seqs is not None and args.max_num_seqs < args.concurrency_per_gpu:
        raise ValueError("max-num-seqs must be at least concurrency-per-gpu")
    local = not args.no_serve
    ids, ports = gpu_ports(args.gpu_ids, args.ports, local)
    if args.dry_run:
        cases = select_cases(config["samples"], args.indices, args.limit)
        for key in ("blocks", "images_dir"):
            if not Path(config[key]).exists():
                raise ValueError("Missing " + key + ": " + config[key])
        print(json.dumps({"total": len(cases), "gpu_ids": ids, "ports": ports,
                          "concurrency_per_gpu": args.concurrency_per_gpu,
                          "commands": [server_command(config, p, args.concurrency_per_gpu, args.max_num_seqs)
                                       for p in ports] if local else [],
                          "config": config, "writes": False}, ensure_ascii=False, indent=2))
        return 0
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another launcher owns this output directory") from exc
        if (root / "run_config.json").exists():
            run = read(root / "run_config.json"); verify(root, run, config)
            cases = read(root / "snapshot/inputs/cases.json")
            if args.indices is not None or args.limit:
                chosen = select_cases(config["samples"], args.indices, args.limit)
                if [c["_sample_index"] for c in chosen] != [c["_sample_index"] for c in cases]:
                    raise ValueError("Selected sample set differs from saved run")
        else:
            cases = select_cases(config["samples"], args.indices, args.limit)
            run = prepare(root, config, cases)
        if args.prepare_only:
            print("Prepared independent Agent snapshot: " + str(root), flush=True)
            return 0
        pending = [c["_sample_index"] for c in cases if needs_work(
            read(result_path(root, c["_sample_index"])) if result_path(root, c["_sample_index"]).exists() else None,
            args.stage, args.retry_failed)]
        if not pending:
            summary = aggregate(root, args.stage, True)
            print(json.dumps(summary, ensure_ascii=False, indent=2)); return 0
        if local:
            if not (Path(config["model_path"]) / "config.json").is_file():
                raise ValueError("Missing model config.json: " + config["model_path"])
            if any(not port_available(p) for p in ports):
                raise ValueError("A requested port is occupied; existing services will not be stopped")
            check_gpus(ids)
        replicas = len(ports) if local else (len(ids) or len(ports))
        assignments = {f"{i}:{j}": [] for i in range(replicas) for j in range(args.concurrency_per_gpu)}
        for n, index in enumerate(pending):
            replica = n % replicas
            slot = (n // replicas) % args.concurrency_per_gpu
            assignments[f"{replica}:{slot}"].append(index)
        runtime = {"run_id": uuid.uuid4().hex, "gpu_ids": ids, "ports": ports, "local_servers": local, "map_replicas": local or args.ports is not None,
                   "concurrency_per_gpu": args.concurrency_per_gpu, "assignments": assignments,
                   "stage": args.stage, "retry_failed": args.retry_failed, "created_at": stamp()}
        atomic(root / "runtime.json", runtime)
        with (root / "run_history.jsonl").open("a") as f:
            f.write(json.dumps(runtime, ensure_ascii=False) + "\n")
        servers, workers, handles, caches = [], [], [], []
        interrupted, failure = False, None
        summary = aggregate(root, args.stage, True)
        initial = summary["processed"]; started = time.monotonic()
        bar = tqdm(total=len(cases), initial=initial, desc="Native Agent", unit="question", dynamic_ncols=True)
        status = {"status": "starting", "run_id": runtime["run_id"], "updated_at": stamp()}
        atomic(root / "run_status.json", status)
        try:
            with interrupt_handler():
                if local:
                    for i, gpu in enumerate(ids):
                        folder = root / f"server{i}"; folder.mkdir(exist_ok=True)
                        handle = (folder / "server.log").open("a"); handles.append(handle)
                        cache = tempfile.mkdtemp(prefix="longdoc_vllm_"); caches.append(cache)
                        env = os.environ.copy()
                        env.update(CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER="PCI_BUS_ID",
                                   PYTHONUNBUFFERED="1", TRITON_CACHE_DIR=cache + "/triton",
                                   TORCHINDUCTOR_CACHE_DIR=cache + "/inductor", VLLM_CACHE_ROOT=cache + "/vllm")
                        env["LIBRARY_PATH"] = "/usr/lib/x86_64-linux-gnu:" + env.get("LIBRARY_PATH", "")
                        for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "NPROC_PER_NODE") :
                            env.pop(key, None)
                        command = server_command(config, ports[i], args.concurrency_per_gpu, args.max_num_seqs)
                        process = subprocess.Popen(command, env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
                        servers.append(process)
                        atomic(folder / "launch.json", {"pid": process.pid, "gpu": gpu, "command": command})
                    wait_ready(servers, ports, config, args.ready_timeout)
                folder = root / "workers"; folder.mkdir(exist_ok=True)
                for key, indices in assignments.items():
                    if not indices:
                        continue
                    replica, slot = key.split(":")
                    handle = (folder / f"replica{replica}_slot{slot}.log").open("a"); handles.append(handle)
                    env = os.environ.copy(); env["PYTHONPATH"] = str(root / "snapshot")
                    env["PYTHONDONTWRITEBYTECODE"] = "1"
                    command = [sys.executable, "-m", "agent", "_worker", "--output-dir", str(root),
                               "--replica", replica, "--slot", slot]
                    process = subprocess.Popen(command, cwd=root / "snapshot", env=env,
                        stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
                    workers.append(process)
                status["status"] = "running"; atomic(root / "run_status.json", status)
                while any(p.poll() is None for p in workers):
                    if any(p.poll() not in (None, 0) for p in workers):
                        raise RuntimeError("A worker failed; inspect workers/*.log")
                    if any(p.poll() is not None for p in servers):
                        raise RuntimeError("A vLLM service exited during evaluation")
                    summary = aggregate(root, args.stage, True)
                    bar.update(summary["processed"] - bar.n)
                    added = summary["processed"] - initial
                    rate = added / max(.001, time.monotonic() - started)
                    eta = summary["pending"] / rate if rate else None
                    active = []
                    for p in sorted(folder.glob("*.json")):
                        state = read(p)
                        if state.get("run_id") == runtime["run_id"] and state.get("status") == "running":
                            active.append(f"R{state['replica']} #{state.get('sample_index')} r{state.get('round')}:{state.get('role')}")
                    bar.set_postfix_str(f"scored={summary['scored']} errors={summary['errors']} incomplete={summary['incomplete']} "
                                        f"{rate*3600:.1f}q/h ETA={eta:.0f}s " + " ".join(active[:4])
                                        if eta is not None else "waiting " + " ".join(active[:4]))
                    time.sleep(.5)
                if any(p.returncode for p in workers):
                    raise RuntimeError("A worker failed; inspect workers/*.log")
        except KeyboardInterrupt:
            interrupted = True
        except Exception as exc:
            failure = str(exc)
        finally:
            with cleanup_guard():
                status.update(status="stopping", updated_at=stamp()); atomic(root / "run_status.json", status)
                cleanup_errors = []
                for name, processes, sig, grace in [("workers", workers, signal.SIGINT, 8),
                                                    ("servers", servers, signal.SIGTERM, 15)]:
                    try:
                        stop_groups(processes, sig, grace)
                    except Exception as exc:
                        cleanup_errors.append({"phase": name, "error": str(exc)})
                for handle in handles:
                    handle.close()
                for cache in caches:
                    shutil.rmtree(cache, ignore_errors=True)
                summary = aggregate(root, args.stage, True)
                bar.update(summary["processed"] - bar.n)
                bar.set_postfix_str(f"scored={summary['scored']} errors={summary['errors']} incomplete={summary['incomplete']} pending={summary['pending']}")
                bar.close()
                status.update(status="interrupted" if interrupted else "failed" if failure or cleanup_errors else "finished",
                              error=failure, cleanup_errors=cleanup_errors, updated_at=stamp())
                atomic(root / "run_status.json", status)
        print(f"Results: {root}\nProcessed {summary['processed']}/{summary['total']}; "
              f"ACC={summary['accuracy']:.2%} F1={summary['f1']:.2%}", flush=True)
        if failure:
            raise RuntimeError(failure)
        return 130 if interrupted else 1 if cleanup_errors else 0
