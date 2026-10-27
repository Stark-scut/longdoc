#!/usr/bin/env python3
"""Sequential cache probe against the existing GPU-0 vLLM service (no deployment)."""
from __future__ import annotations
import argparse
import base64
import hashlib
import json
import mimetypes
from pathlib import Path
import re
import statistics
import sys
import time
import uuid

import httpx

ROOT = Path(__file__).resolve().parent
COUNTERS = ("prefix_cache_queries", "prefix_cache_hits", "mm_cache_queries", "mm_cache_hits")


def parse_metrics(text):
    counters, config = {}, {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        match = re.match(r"([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(.*)\})?\s+(\S+)", line)
        if not match:
            continue
        name, labels, value = match.groups()
        if name == "vllm:cache_config_info":
            fields = dict(re.findall(r'(\w+)="([^"]*)"', labels or ""))
            config.update({k: fields[k] for k in (
                "enable_prefix_caching", "block_size", "mamba_block_size", "mamba_cache_mode"
            ) if k in fields})
        for key in COUNTERS:
            if name in ("vllm:" + key, "vllm:" + key + "_total"):
                counters[key] = counters.get(key, 0) + float(value)
    return {"counters": counters, "config": config}


def scrape(client, origin):
    try:
        response = client.get(origin + "/metrics", timeout=10)
        response.raise_for_status()
        return parse_metrics(response.text)
    except (httpx.HTTPError, ValueError) as exc:
        return {"counters": {}, "config": {}, "error": str(exc)}


def counter_delta(before, after):
    a, b = before["counters"], after["counters"]
    result = {k: b[k] - a[k] for k in COUNTERS if k in a and k in b}
    for family in ("prefix", "mm"):
        q, h = result.get(family + "_cache_queries"), result.get(family + "_cache_hits")
        result[family + "_hit_rate_percent"] = 100 * h / q if q and h is not None and q > 0 else None
    return result


def request(client, url, model, messages, max_tokens):
    body = {
        "model": model, "messages": messages, "temperature": 0,
        "max_tokens": max_tokens, "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    started, first, usage, answer, finish = time.perf_counter(), None, {}, [], None
    with client.stream("POST", url + "/chat/completions", json=body) as response:
        if response.status_code != 200:
            response.read()
            raise RuntimeError(f"HTTP {response.status_code}: {response.text[:1200]}")
        for line in response.iter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            if event.get("error"):
                raise RuntimeError(str(event["error"]))
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                chunks = [delta.get(k) for k in ("content", "reasoning", "reasoning_content")]
                if any(chunks) and first is None:
                    first = time.perf_counter() - started
                if delta.get("content"):
                    answer.append(delta["content"])
                finish = choice.get("finish_reason") or finish
    if finish is None:
        raise RuntimeError("Stream ended without finish_reason; request was not completed.")
    return {
        "ttft_seconds": first, "total_seconds": time.perf_counter() - started,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
        "finish_reason": finish, "answer": "".join(answer), "usage": usage,
        "messages_sha256": hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest(),
    }


def run(args, client):
    url = args.base_url.rstrip("/")
    origin = url[:-3] if url.endswith("/v1") else url
    if not url.endswith("/v1"):
        url += "/v1"
    response = client.get(url + "/models")
    response.raise_for_status()
    models = [m["id"] for m in response.json()["data"]]
    if args.model not in models:
        raise ValueError(f"Model {args.model!r} not served; available={models}")
    before = scrape(client, origin)
    config = before["config"]
    print("服务：", url, "模型：", args.model, flush=True)
    print("缓存配置：", json.dumps(config, ensure_ascii=False), flush=True)
    if str(config.get("enable_prefix_caching")).lower() == "false":
        print("当前 Prefix caching 已关闭：重复请求也无法命中。先测关闭基线，重启时加 "
              "--enable-prefix-caching --enable-prompt-tokens-details 后重跑。", flush=True)
    if before.get("error"):
        print("无法读取 /metrics；继续记录请求耗时与 usage：", before["error"], flush=True)
    print("TTFT 包含排队时间；若服务同时跑评测，耗时和全局计数会受到其他请求影响。", flush=True)
    nonce = uuid.uuid4().hex
    document = f"Cache experiment {nonce}\n" + "".join(
        f"Record {i:04d}: The document describes page retrieval and visual evidence for long document questions.\n"
        for i in range(args.lines)
    )
    system = {"role": "system", "content": "Follow the final instruction. Reply briefly."}
    def text_messages(question):
        return [system, {"role": "user", "content": document + "\n" + question}]
    same = text_messages("Reply with OK only.")
    jobs = [(f"text_same_{i+1}", same) for i in range(args.repeats)]
    jobs.append(("text_changed_suffix", text_messages("Reply with YES only.")))
    if not args.text_only:
        image_path = args.image.resolve()
        raw = image_path.read_bytes()
        mime = mimetypes.guess_type(image_path.name)[0] or "image/png"
        image = {"type": "image_url", "image_url": {
            "url": f"data:{mime};base64," + base64.b64encode(raw).decode()}}
        def image_messages(changed=False):
            prefix = f"Different prefix {nonce}\n" if changed else f"Image experiment {nonce}\n"
            return [system, {"role": "user", "content": [
                {"type": "text", "text": prefix + document + "\nDescribe the main image colors briefly."},
                image,
            ]}]
        same_image = image_messages()
        jobs += [(f"image_same_{i+1}", same_image) for i in range(args.repeats)]
        jobs.append(("same_image_changed_prefix", image_messages(True)))
    report = {
        "base_url": url, "model": args.model, "nonce": nonce, "config": config,
        "metrics_before": before, "requests": [],
        "note": "Server counters include concurrent clients; cached_tokens is per request. "
                "No cache reset or service restart is performed. First image may already be in MM cache.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"{'case':30} {'TTFT(s)':>10} {'total(s)':>10} {'prompt':>9} {'cached':>9}", flush=True)
    try:
        for label, messages in jobs:
            start = scrape(client, origin)
            result = request(client, url, args.model, messages, args.max_tokens)
            end = scrape(client, origin)
            result.update(case=label, server_counter_delta=counter_delta(start, end))
            report["requests"].append(result)
            ttft = f"{result['ttft_seconds']:.3f}" if result["ttft_seconds"] is not None else "N/A"
            print(f"{label:30} {ttft:>10} {result['total_seconds']:10.3f} "
                  f"{str(result['prompt_tokens']):>9} {str(result['cached_tokens']):>9}", flush=True)
            save()
    finally:
        after = scrape(client, origin)
        report.update(metrics_after=after, server_counter_delta=counter_delta(before, after))
        report["per_request_prefix_hit_observed"] = any(
            (r["cached_tokens"] or 0) > 0 for r in report["requests"])
        save()
        print("本次期间服务全局计数增量：", json.dumps(report["server_counter_delta"], ensure_ascii=False), flush=True)
        print("结果：", args.output, flush=True)
    for group in ("text_same_", "image_same_"):
        rows = [r for r in report["requests"] if r["case"].startswith(group)]
        if len(rows) > 1 and all(r["ttft_seconds"] is not None for r in rows):
            cold = rows[0]["ttft_seconds"]
            warm = statistics.median(r["ttft_seconds"] for r in rows[1:])
            if warm > 0:
                print(f"{group} 首次 TTFT={cold:.3f}s；后续中位数={warm:.3f}s；比值={cold/warm:.2f}x", flush=True)
    if report["per_request_prefix_hit_observed"]:
        print("已从本次请求的 cached_tokens 观察到 Prefix cache 命中。", flush=True)
    else:
        print("尚未从本次请求的 cached_tokens 证实 Prefix 命中。None 表示服务未返回该字段，不代表零；"
              "应开启 --enable-prompt-tokens-details，并结合配置和计数判断。", flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default="http://127.0.0.1:8071/v1")
    p.add_argument("--model", default="citeagent-qwen35")
    p.add_argument("--image", type=Path, default=ROOT / "a_tool_image.png")
    p.add_argument("--text-only", action="store_true")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--lines", type=int, default=192, help="Shared document length; usually several thousand tokens")
    p.add_argument("--max-tokens", type=int, default=8)
    p.add_argument("--timeout", type=float, default=900)
    p.add_argument("--output", type=Path, default=ROOT / "output/cache_probe" / (time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8] + ".json"))
    args = p.parse_args()
    if args.repeats < 2 or min(args.lines, args.max_tokens, args.timeout) <= 0:
        p.error("repeats must be >=2 and budgets must be positive")
    with httpx.Client(timeout=args.timeout, trust_env=False) as client:
        run(args, client)


if __name__ == "__main__":
    try:
        main()
    except (httpx.HTTPError, ValueError, OSError, RuntimeError) as exc:
        print("ERROR:", exc, file=sys.stderr)
        raise SystemExit(1)
