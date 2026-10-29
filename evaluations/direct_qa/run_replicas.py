#!/usr/bin/env python3
"""任意GPU数量的MMLongBench直接问答：每卡BF16/TP=1副本，无Agent。"""

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import socket
import subprocess
import sys
import time

import requests
from tqdm import tqdm

import evaluate_qwen35_mmlongbench as evaluation

WORKDIR = next(p for p in Path(__file__).resolve().parents if (p / ".gitignore").is_file())


def replica_count(args):
    return len(args.gpu_ids.split(","))

def port_available(port):
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False

def native_context(model_path):
    p = Path(model_path) / "config.json"
    if not p.is_file():
        raise ValueError("Missing model config.json; specify --model-path")
    c = json.loads(p.read_text())
    n = c.get("text_config", c).get("max_position_embeddings")
    if type(n) is not int or n <= 0:
        raise ValueError("Specify --max-model-len; model lacks max_position_embeddings")
    return n

@contextlib.contextmanager
def cleanup_guard():
    saved = {sig: signal.signal(sig, signal.SIG_IGN)
             for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    try:
        yield
    finally:
        for sig, handler in saved.items(): signal.signal(sig, handler)

def install_interrupts():
    saved, requested = {}, False
    def stop(signum, frame):
        nonlocal requested
        if not requested:
            requested = True
            raise KeyboardInterrupt
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        saved[sig] = signal.signal(sig, stop)
    return saved


def duration(seconds):
    if seconds is None:
        return "待估算"
    seconds = max(0, int(seconds))
    return f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


def logged_rate(path):
    """从子进程已有 tqdm 日志读取本轮实际吞吐，兼容正在运行的旧入口。"""
    if not path.exists():
        return None, None
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - 32768))
        text = stream.read().decode("utf-8", errors="replace")
    matches = list(re.finditer(r"(?:all|score|predict):[^\r\n]*?(\d+)/(\d+)\s*\[([\d:]+)<", text))
    if not matches:
        return None, None
    count = int(matches[-1][1])
    elapsed = 0
    for part in matches[-1][3].split(":"):
        elapsed = elapsed * 60 + int(part)
    elapsed += max(0, time.time() - path.stat().st_mtime)
    return (count / elapsed if count and elapsed > 0 else None), elapsed


class Progress:
    def __init__(self, args, total):
        self.args, self.total = args, total
        self.started = time.monotonic()
        self.cache = {}
        self.baseline = None
        self.bar = tqdm(total=total, desc="MMLongBench", unit="题", dynamic_ncols=True,
                        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{postfix}]")

    def update(self, persist=False):
        shards = []
        monitor_elapsed = time.monotonic() - self.started
        for shard in range(replica_count(self.args)):
            total = len(range(shard, self.total, replica_count(self.args)))
            path = self.args.output_dir / f"shard{shard}.json"
            rows = []
            if path.exists():
                stat = path.stat()
                fingerprint = (stat.st_mtime_ns, stat.st_size)
                if self.cache.get(shard, (None,))[0] != fingerprint:
                    self.cache[shard] = (fingerprint, json.loads(path.read_text()))
                rows = self.cache[shard][1]
            complete = sum(row.get("status") == "complete" for row in rows)
            errors = sum(row.get("status") == "error" for row in rows)
            predicted = sum(row.get("status") == "predicted" for row in rows)
            done = complete + errors
            rate, elapsed = logged_rate(self.args.output_dir / f"eval{shard}.log")
            if rate is None and self.baseline is not None and monitor_elapsed > 0:
                change = done - self.baseline[shard]
                if change > 0:
                    rate = change / monitor_elapsed
            remaining = max(0, total - done)
            shards.append({"shard": shard, "gpu": self.args.gpu_ids.split(",")[shard].strip(),
                           "total": total, "complete": complete, "errors": errors,
                           "generated_waiting_for_extraction": predicted,
                           "processed": done, "remaining": remaining,
                           "questions_per_second": rate, "elapsed_seconds": elapsed,
                           "eta_seconds": remaining / rate if remaining and rate else (0 if not remaining else None)})
        if self.baseline is None:
            self.baseline = [shard["processed"] for shard in shards]
        done = sum(shard["processed"] for shard in shards)
        complete = sum(shard["complete"] for shard in shards)
        errors = sum(shard["errors"] for shard in shards)
        eta = None if any(s["eta_seconds"] is None for s in shards) else max(s["eta_seconds"] for s in shards)
        rate = sum(s["questions_per_second"] or 0 for s in shards)
        elapsed = max((s["elapsed_seconds"] or 0 for s in shards), default=0) or monitor_elapsed
        snapshot = {"updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "total": self.total,
                    "processed": done, "complete": complete, "errors": errors,
                    "remaining": max(0, self.total - done), "questions_per_hour": rate * 3600,
                    "elapsed_seconds": elapsed, "eta_seconds": eta, "shards": shards,
                    "note": "processed 包含成功和失败；ETA 按当前各分片吞吐取较慢者估算，文档长度变化会使其波动。"}
        self.bar.n = done
        status = (f"成功{complete} 失败{errors} | " +
            " ".join(f"G{s['gpu']} {s['processed']}/{s['total']}" for s in shards) +
            f" | {rate * 3600:.1f}题/h | 耗时{duration(elapsed)} ETA {duration(eta)}")
        self.bar.bar_format = "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [" + status + "]"
        self.bar.refresh()
        if persist:
            evaluation.save_json(self.args.output_dir / "progress.json", snapshot)
        return snapshot

    def close(self):
        self.bar.close()


def watch(args, total):
    if not args.output_dir.exists():
        raise ValueError(f"结果目录不存在：{args.output_dir}")
    print(f"只读监控：{args.output_dir}；Ctrl-C 退出监控，不影响评测。", flush=True)
    progress = Progress(args, total)
    try:
        while True:
            snapshot = progress.update()
            if snapshot["processed"] >= total:
                break
            time.sleep(args.progress_interval)
    except KeyboardInterrupt:
        pass
    finally:
        progress.close()
    return 0


def stop_processes(processes):
    # 只终止本入口用 start_new_session 创建的进程组，包括 vLLM 子进程。
    for process in reversed(processes):
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 15
    for process in reversed(processes):
        try:
            process.wait(timeout=max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass
    for process in reversed(processes):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def merge_results(args, samples, selected):
    records = {i: {**sample, "_sample_index": i, "status": "pending"}
               for i, sample in enumerate(selected)}
    input_hash = hashlib.sha256(args.input.read_bytes()).hexdigest()
    shard_metadata = []
    for shard in range(replica_count(args)):
        path = args.output_dir / f"shard{shard}.json"
        if not path.exists():
            continue
        meta = json.loads(path.with_suffix(".meta.json").read_text())
        if (meta["input_sha256"] != input_hash or
                meta.get("partition") != ({"num_shards": replica_count(args), "shard_index": shard} if replica_count(args) > 1 else None)):
            raise ValueError(f"分片配置不匹配：{path}")
        shard_metadata.append(meta)
        seen = set()
        for sample in json.loads(path.read_text()):
            index = sample["_sample_index"]
            if index in seen or index % replica_count(args) != shard or index not in records:
                raise ValueError(f"分片包含重复或越界题号 {index}：{path}")
            if any(sample[key] != samples[index][key] for key in ("doc_id", "question", "answer")):
                raise ValueError(f"分片题目与输入不匹配：{path} #{index}")
            seen.add(index)
            records[index] = sample
    merged = [records[i] for i in sorted(records)]
    output = args.output_dir / "merged.json"
    evaluation.save_json(output, merged)
    evaluation.save_json(output.with_suffix(".meta.json"), {
        "input_sha256": input_hash, "shards": shard_metadata,
        "note": "每卡独立 TP=1；合并后按全部选中题目计算，错误和未完成计零。",
    })
    evaluation.report(merged, output)
    print(f"合并结果：{output}", flush=True)
    return 0 if all(s.get("status") == "complete" for s in merged) else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-ids", default="1,2", help="任意数量不同物理卡，例如0,1,2,3或1,4,5。")
    parser.add_argument("--ports", help="每卡一个端口，默认从8047连续分配")
    parser.add_argument("--model-path", type=Path, default=WORKDIR.parent / "models/Qwen3.5-9B")
    parser.add_argument("--input", type=Path, default=evaluation.ROOT / "data/mmlongbench/samples.json")
    parser.add_argument("--images-dir", type=Path, default=evaluation.ROOT / "data/mmlongbench/images_1024")
    parser.add_argument("--output-dir", type=Path,
                        default=None)
    parser.add_argument("--concurrency", type=int, default=1, help="每个模型副本的客户端并发")
    parser.add_argument("--max-num-seqs", type=int, default=8, help="每张卡的 vLLM 调度上限。")
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096, help="每卡每轮预填充/解码 token 调度预算。")
    parser.add_argument("--mm-processor-cache-gb", type=float, default=1,
                        help="每个 API/engine 进程的 CPU 图像预处理缓存 GiB；可先试 1，不等同于 GPU KV 缓存。")
    parser.add_argument("--enable-prefix-caching", action=argparse.BooleanOptionalAction, default=True,
                        help="默认开启Prefix caching；Qwen3.5使用align模式。")
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=True,
                        help="默认 eager；--no-enforce-eager 尝试编译/CUDA Graphs，会增加显存与启动时间。")
    parser.add_argument("--max-cudagraph-capture-size", type=int,
                        help="非 eager 时默认等于 max-num-seqs，限制图捕获规模，避免默认大 batch 的额外显存。")
    parser.add_argument("--cpu-offload-gb", type=float, default=0,
                        help="默认 0：每卡完整模型权重留在 GPU；仅显式指定正数时启用 CPU offload。")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--max-model-len", type=int, help="默认读取模型原生上限，Qwen3.5-9B为262144；不自动开启YaRN。")
    parser.add_argument("--truncate-long-documents", action=argparse.BooleanOptionalAction, default=True,
                        help="默认保留可容纳的整页前缀，超长时从末尾裁页")
    parser.add_argument("--input-token-limit", type=int, default=260000,
                        help="输入总token目标上限，实际还需受上下文减输出预算约束")
    parser.add_argument("--max-pixels", type=int, default=1048576,
                        help="足以保留全部 images_1024 页图；不足时拒绝运行，避免静默缩图。")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--extract-max-tokens", type=int, default=2048)
    parser.add_argument("--extractor-base-url")
    parser.add_argument("--extractor-model")
    parser.add_argument("--limit", type=int, default=0, help="全局前 N 题，再分片；0 全量。")
    parser.add_argument("--timeout", type=float, default=1800, help="单次回答/抽取请求超时。")
    parser.add_argument("--ready-timeout", type=float, default=1200)
    parser.add_argument("--dry-run", action="store_true", help="检查命令、数据、原图与上下文，不启动 GPU 服务。")
    parser.add_argument("--merge-only", action="store_true", help="只合并当前副本数对应的全部分片。")
    parser.add_argument("--watch", action="store_true", help="只读显示现有任务进度、吞吐和 ETA，不启动或停止服务。")
    parser.add_argument("--progress-interval", type=float, default=5, help="主进度条或 watch 刷新间隔，秒。")
    args = parser.parse_args(argv)
    if (args.watch or args.merge_only) and args.output_dir:
        saved_path = args.output_dir / "run_config.json"
        if saved_path.is_file():
            saved = json.loads(saved_path.read_text())
            args.gpu_ids, args.ports = saved["gpu_ids"], saved["ports"]
            args.input = Path(saved["input"])
            args.images_dir = Path(saved["images_dir"])
            args.model_path = Path(saved["model_path"])
            args.truncate_long_documents = saved.get("truncate_long_documents", False)
            args.input_token_limit = saved.get("input_token_limit", 260000)
            if not args.limit: args.limit = saved.get("selected_count",0)
            if args.max_model_len is None: args.max_model_len = saved.get("max_model_len")
    gpus = [v.strip() for v in args.gpu_ids.split(",")]
    if not gpus or len(set(gpus)) != len(gpus) or not all(v.isdigit() for v in gpus):
        parser.error("gpu-ids必须是任意正数个不同的非负卡号")
    args.gpu_ids = ",".join(gpus)
    try:
        ports = [int(v) for v in args.ports.split(",")] if args.ports else list(range(8047,8047+len(gpus)))
    except ValueError:
        parser.error("每卡需一个整数端口")
    if len(ports)!=len(gpus) or len(set(ports))!=len(ports) or any(not 1024<=p<=65535 for p in ports):
        parser.error("ports数量应与GPU相同，且各自是不同有效端口")
    args.ports = ",".join(map(str, ports))
    if args.output_dir is None:
        args.output_dir = WORKDIR / f"output/mmlongbench/qwen35_1024_{len(gpus)}replicas_bf16"
    if args.max_model_len is None:
        try: args.max_model_len = native_context(args.model_path)
        except ValueError as exc: parser.error(str(exc))
    if not (args.watch or args.merge_only) and args.max_model_len > native_context(args.model_path):
        parser.error("本入口使用原生上下文；请勿超过模型配置上限")
    if args.input_token_limit <= 0:
        parser.error("input-token-limit必须为正")
    if (min(args.concurrency, args.max_num_seqs, args.max_tokens, args.extract_max_tokens,
            args.max_num_batched_tokens) <= 0
            or args.limit < 0 or args.cpu_offload_gb < 0 or args.mm_processor_cache_gb < 0
            or (args.max_cudagraph_capture_size is not None and args.max_cudagraph_capture_size <= 0)
            or not 0 < args.gpu_memory_utilization < 1
            or args.max_model_len <= args.max_tokens or args.max_pixels < 65536
            or min(args.timeout, args.ready_timeout, args.progress_interval) <= 0):
        parser.error("检查并发、显存比例、offload、上下文、输出预算、超时和 limit 参数。")
    args.output_dir = args.output_dir.resolve()
    args.images_dir = args.images_dir.resolve()
    args.input = args.input.resolve()
    args.model_path = args.model_path.resolve()
    samples = json.loads(args.input.read_bytes())
    selected = samples[:args.limit] if args.limit else samples
    if not selected:
        parser.error("输入样本为空。")
    if args.watch and args.merge_only:
        parser.error("watch 与 merge-only 不能同时使用。")
    if args.watch:
        if not args.output_dir.exists():
            parser.error(f"结果目录不存在：{args.output_dir}")
        # 监控时使用实际保存的 GPU 编号；不改变任何运行配置或结果。
        config_path = args.output_dir / "run_config.json"
        if config_path.exists():
            args.gpu_ids = json.loads(config_path.read_text()).get("gpu_ids", args.gpu_ids)
        return watch(args, len(selected))
    if args.merge_only:
        if not any((args.output_dir / f"shard{i}.json").exists() for i in range(len(gpus))):
            parser.error("output-dir 中没有分片结果。")
        return merge_results(args, samples, selected)
    print(f"{len(gpus)}个独立BF16/TP=1副本："+"；".join(f"GPU {g} → :{p}" for g,p in zip(gpus,ports)),flush=True)
    if args.cpu_offload_gb == 0:
        print(f"共 {len(selected)} 题；按原始题号取模分片；每卡完整模型权重留在 GPU，CPU offload=0。", flush=True)
    else:
        print(f"共 {len(selected)} 题；按原始题号取模分片；显式启用每卡 CPU offload={args.cpu_offload_gb} GiB。", flush=True)
    print(f"images={args.images_dir}；thinking=false；结果目录={args.output_dir}", flush=True)
    print(f"max_model_len={args.max_model_len}；超长前缀截断={args.truncate_long_documents}；"
          f"请求输入上限={args.input_token_limit}；有效输入预算={min(args.input_token_limit, args.max_model_len-args.max_tokens)}", flush=True)
    print(f"每卡客户端并发={args.concurrency}；vLLM max_num_seqs={args.max_num_seqs}；"
          f"max_num_batched_tokens={args.max_num_batched_tokens}。", flush=True)
    print(f"图像预处理缓存={args.mm_processor_cache_gb} GiB/进程；"
          f"prefix caching={args.enable_prefix_caching}；enforce eager={args.enforce_eager}。", flush=True)

    # 整体检查一次；各工作进程还会检查自己的分片，保证直接调用它们也安全。
    documents = {sample["doc_id"] for sample in selected}
    infos = {doc: evaluation.native_page_info(evaluation.page_paths(args.images_dir, doc)) for doc in documents}
    largest = max(value[1] for value in infos.values())
    if largest > args.max_pixels:
        parser.error(f"需要 max-pixels >= {largest} 才能保留当前页图分辨率。")
    oversized = {doc for doc, (tokens, _) in infos.items()
                 if tokens + args.max_tokens >= args.max_model_len}
    print(f"分辨率检查通过：最大对齐面积={largest}；明确超限文档={len(oversized)}，"
          f"相关 {sum(s['doc_id'] in oversized for s in selected)} 题超出容量；" +
          ("将按完整页前缀截断，记录保留/丢弃页数。" if args.truncate_long_documents else "保留全部页，超限记失败。"), flush=True)

    plans = []
    for shard, (gpu, port) in enumerate(zip(gpus, ports)):
        env = os.environ.copy()
        env.update(GPU_IDS=gpu, TENSOR_PARALLEL_SIZE="1", PORT=str(port), DTYPE="bfloat16",
                   MAX_PIXELS=str(args.max_pixels), MIN_PIXELS="1024",
                   MAX_MODEL_LEN=str(args.max_model_len), MAX_NUM_SEQS=str(args.max_num_seqs),
                   MAX_NUM_BATCHED_TOKENS=str(args.max_num_batched_tokens), PROFILE_IMAGE_EDGE="1024",
                   CPU_OFFLOAD_GB=str(args.cpu_offload_gb), MODEL_PATH=str(args.model_path.resolve()),
                   VLLM_GPU_MEMORY_UTILIZATION=str(args.gpu_memory_utilization),
                   MM_PROCESSOR_CACHE_GB=str(args.mm_processor_cache_gb),
                   ENABLE_PREFIX_CACHING=str(int(args.enable_prefix_caching)),
                   ENFORCE_EAGER=str(int(args.enforce_eager)),
                   MAX_CUDAGRAPH_CAPTURE_SIZE=str(args.max_cudagraph_capture_size or args.max_num_seqs),
                   SERVER_LOGDIR=str(args.output_dir / f"server{shard}"),
                   PYTHONUNBUFFERED="1", PYTHON_BIN=sys.executable)
        env.pop("HF_OVERRIDES", None)
        env.pop("VLLM_ALLOW_LONG_MAX_MODEL_LEN", None)
        command = [sys.executable, str(Path(__file__).resolve().with_name("evaluate_qwen35_mmlongbench.py")),
                   "--input", str(args.input.resolve()), "--images-dir", str(args.images_dir),
                   "--output", str(args.output_dir / f"shard{shard}.json"),
                   "--base-url", f"http://127.0.0.1:{port}/v1", "--model", "qwen35-mmlongbench",
                   "--num-shards", str(len(gpus)), "--shard-index", str(shard),
                   "--concurrency", str(args.concurrency), "--native-resolution",
                   "--max-model-len", str(args.max_model_len), "--max-pixels", str(args.max_pixels),
                   "--max-tokens", str(args.max_tokens), "--extract-max-tokens", str(args.extract_max_tokens),
                   "--timeout", str(args.timeout), "--limit", str(args.limit), "--no-enable-thinking"]
        if args.truncate_long_documents:
            command += ["--truncate-long-documents", "--input-token-limit", str(args.input_token_limit)]
        if args.extractor_base_url:
            command += ["--extractor-base-url", args.extractor_base_url]
        if args.extractor_model:
            command += ["--extractor-model", args.extractor_model]
        if args.dry_run:
            print(f"GPU {gpu}: BF16, TP=1, max_model_len={args.max_model_len}, "
                  f"max_num_seqs={args.max_num_seqs}, cpu_offload_gb={args.cpu_offload_gb}", flush=True)
            print(shlex.join(command), flush=True)
        plans.append((env, command))
    if args.dry_run:
        return 0
    for port in ports:
        if not port_available(port):
            parser.error(f"端口{port}已占用，请设置--ports；不会终止已有服务。")

    # 同一目录禁止并行运行；允许 limit 扩大，其余配置变化要另建目录。
    import fcntl
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock = (args.output_dir / ".run.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error("该 output-dir 已有评测进程在运行。")
    config = {key: value for key, value in vars(args).items()
              if key not in {"limit", "concurrency", "timeout", "ready_timeout", "dry_run", "merge_only"}}
    config = json.loads(json.dumps(config, default=str))
    config["concurrency"] = args.concurrency
    config["input_sha256"] = hashlib.sha256(args.input.read_bytes()).hexdigest()
    config["num_replicas"] = len(gpus)
    config["selected_count"] = len(selected)
    config_path = args.output_dir / "run_config.json"
    # 调度配置可续跑；截断模式下上下文容量会改变保留页数，必须保持一致。
    # 实际部署配置与变化写入 run_history。
    runtime_keys = {"concurrency", "max_num_seqs", "max_num_batched_tokens", "progress_interval", "watch",
                    "cpu_offload_gb", "max_model_len", "mm_processor_cache_gb", "enable_prefix_caching",
                    "enforce_eager", "max_cudagraph_capture_size", "gpu_ids", "selected_count"}
    if args.truncate_long_documents:
        runtime_keys.discard("max_model_len")
    if config_path.exists():
        old_config = json.loads(config_path.read_text())
        old_config.setdefault("num_replicas", len(old_config["gpu_ids"].split(",")))
        old_config.setdefault("selected_count",len(selected))
        old_config.setdefault("truncate_long_documents", False)
        old_config.setdefault("input_token_limit", 260000)
        if ({k: v for k, v in old_config.items() if k not in runtime_keys} !=
                {k: v for k, v in config.items() if k not in runtime_keys}):
            parser.error("该目录保存的模型/输入/分辨率等配置不同，请设置新的 --output-dir。")
        changes = {key: {"old": old_config.get(key), "new": config.get(key)} for key in runtime_keys
                   if key not in {"watch", "progress_interval"} and old_config.get(key) != config.get(key)}
        if changes:
            print("续跑部署配置变化：" + json.dumps(changes, ensure_ascii=False) +
                  "；保留已保存的完整回答，超限题按当前上下文检查。", flush=True)
    for shard in range(replica_count(args)):
        result = args.output_dir / f"shard{shard}.json"
        if result.exists() and any(s["_sample_index"] >= len(selected) for s in json.loads(result.read_text())):
            parser.error("limit 小于已有结果范围，请沿用或扩大。")
    evaluation.save_json(config_path, config)
    with (args.output_dir / "run_history.jsonl").open("a") as history:
        history.write(json.dumps({"started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                  "limit": args.limit, **config}, ensure_ascii=False) + "\n")
    processes, log_files = [], []
    progress = None
    exit_code = 1
    handlers = install_interrupts()
    try:
        servers = []
        for shard, (env, _) in enumerate(plans):
            log = (args.output_dir / f"server{shard}.launcher.log").open("a", buffering=1)
            log_files.append(log)
            process = subprocess.Popen(["bash", str(Path(__file__).resolve().with_name("serve_qwen35_mmlongbench.sh"))],
                                       env=env, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            processes.append(process)
            servers.append(process)
        session = requests.Session()
        session.trust_env = False
        try:
            pending = set(range(replica_count(args)))
            deadline = time.monotonic() + args.ready_timeout
            print(f"等待{len(gpus)}个vLLM服务；日志见server*/server.log。", flush=True)
            while pending:
                if time.monotonic() > deadline:
                    raise RuntimeError("服务启动超时，请检查 server*/server.log。")
                for shard in list(pending):
                    if servers[shard].poll() is not None:
                        raise RuntimeError(f"GPU {gpus[shard]} 服务已退出，请检查 server{shard}.launcher.log。")
                    try:
                        response = session.get(f"http://127.0.0.1:{ports[shard]}/v1/models", timeout=2)
                        if response.ok and any(m["id"] == "qwen35-mmlongbench" for m in response.json().get("data", [])):
                            pending.remove(shard)
                            print(f"GPU {gpus[shard]} 服务就绪。", flush=True)
                    except (requests.RequestException, ValueError):
                        pass
                if pending:
                    time.sleep(2)
        finally:
            session.close()
        workers = []
        for shard, (env, command) in enumerate(plans):
            if not any(i % len(gpus) == shard for i in range(len(selected))):
                continue
            log = (args.output_dir / f"eval{shard}.log").open("a", buffering=1)
            log_files.append(log)
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            processes.append(process)
            workers.append(process)
        progress = Progress(args, len(selected))
        progress.update(persist=True)
        last_update = time.monotonic()
        while any(worker.poll() is None for worker in workers):
            if any(server.poll() is not None for server in servers):
                raise RuntimeError("vLLM 服务中途退出，请检查服务日志；结果已逐题保存。")
            if time.monotonic() - last_update >= args.progress_interval:
                progress.update(persist=True)
                last_update = time.monotonic()
            time.sleep(min(2, args.progress_interval))
        exit_code = int(any(worker.returncode != 0 for worker in workers))
    except KeyboardInterrupt:
        print("已中断，保留分片结果；重跑同一命令可继续。", flush=True)
        exit_code = 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        exit_code = 1
    finally:
        with cleanup_guard():
            stop_processes(processes)
            if progress is not None:
                progress.update(persist=True)
                progress.close()
            for log in log_files: log.close()
            exit_code = max(exit_code, merge_results(args, samples, selected))
            lock.close()
        for sig, handler in handlers.items(): signal.signal(sig, handler)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
