#!/usr/bin/env python3
"""Qwen3.5 + vLLM：全部 PDF 页图 → 回答 → 模型抽取短答案 → 官方评分。

基于归档 scripts/evaluate_mmlongbench_direct.py，使用本地部署的 Qwen3.5。
可选整页前缀截断超长文档。依赖 requests、Pillow、tqdm。API key 仅从环境变量读取；本地无认证服务无需 key。
先运行 serve_qwen35_mmlongbench.sh；--dry-run 检查数据，--limit 2 做冒烟测试。
"""

import argparse
import ast
import base64
import copy
import hashlib
import importlib.util
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

import requests
from tqdm import tqdm

WORKDIR = next(p for p in Path(__file__).resolve().parents if (p / ".gitignore").is_file())
ROOT = WORKDIR / "aaai_commit_20260803" / "aaai_commit"
EVAL_DIR = ROOT / "MMLongBench-Doc" / "eval"
SCORER_PATH = EVAL_DIR / "eval_score.py"
if not SCORER_PATH.is_file(): SCORER_PATH = WORKDIR / "agent/metrics/mmlongbench.py"
EXTRACTION_PROMPT_PATH = EVAL_DIR / "prompt_for_answer_extraction.md"
if not EXTRACTION_PROMPT_PATH.is_file(): EXTRACTION_PROMPT_PATH = WORKDIR / "agent/prompts/extract.txt"
spec = importlib.util.spec_from_file_location("mmlb_official_score", SCORER_PATH)
scorer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scorer)
# 原评分函数会打印 List 的比较过程；仅禁用该模块的打印，避免并发时重定向全局 stdout。
scorer.print = lambda *args, **kwargs: None
INSTRUCTION = (
    "Read all pages of the document and answer the question based on the document. "
    "If the question cannot be answered from the document, answer 'Not answerable'."
)


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def page_paths(images_dir, doc_id):
    folder = images_dir / Path(doc_id).stem
    pages = sorted(folder.glob("page_*.png"), key=lambda p: int(p.stem.split("_")[-1]))
    indexes = [int(p.stem.split("_")[-1]) for p in pages]
    if not pages or indexes != list(range(len(pages))):
        raise ValueError(f"页图不存在或页号不连续（应从 page_0.png 开始）：{folder}")
    return tuple(pages)


@lru_cache(maxsize=256)
def native_page_info(paths):
    """Qwen3.5-9B: patch_size=16, merge_size=2，预处理对齐到 32。

    只读取文件头；这里的视觉 token 估计不包括文本和图像分隔符。
    """
    from PIL import Image
    tokens, largest = 0, 0
    for path in paths:
        with Image.open(path) as im:
            width, height = im.size
        aligned_width = max(32, round(width / 32) * 32)
        aligned_height = max(32, round(height / 32) * 32)
        pixels = aligned_width * aligned_height
        tokens += pixels // 1024
        largest = max(largest, pixels)
    return tokens, largest


@lru_cache(maxsize=1)
def page_content(paths):
    content = [{"type": "text", "text": INSTRUCTION}]
    for number, path in enumerate(paths, 1):
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        content.extend([
            {"type": "text", "text": f"Page {number}:"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + encoded}},
        ])
    return content

class InputBudgetExceeded(ValueError):
    pass


@lru_cache(maxsize=256)
def page_visual_tokens(paths):
    return tuple(native_page_info((path,))[0] for path in paths)


def prefix_pages(paths, question, context_limit, output_tokens, input_limit):
    """Keep a whole-page prefix, reserving output plus conservative text/template room."""
    budget = min(input_limit, context_limit - output_tokens)
    # UTF-8 byte count conservatively covers question/instruction text tokens;
    # page labels/vision delimiters have their own allowance.
    text_reserve = len((INSTRUCTION + "\nQuestion: " + question).encode("utf-8")) + 512
    used = text_reserve
    count = 0
    costs = page_visual_tokens(paths)
    for cost in costs:
        if used + cost + 16 > budget:
            break
        used += cost + 16
        count += 1
    if count == 0:
        raise ValueError("Input budget cannot fit the question and one complete page image")
    selected = paths[:count]
    metadata = {
        "policy": "whole-page-prefix-v1", "applied": count < len(paths),
        "original_pages": len(paths), "kept_pages": count, "dropped_pages": len(paths) - count,
        "kept_page_indices": list(range(count)), "input_token_limit": budget,
        "requested_input_token_limit": input_limit, "context_limit": context_limit,
        "output_token_budget": output_tokens, "estimated_visual_tokens": sum(costs[:count]),
        "estimated_prompt_tokens_with_reserve": used, "text_template_token_reserve": text_reserve,
        "context_retries": 0,
        "note": "Image token counts are estimated at native 32-pixel alignment; text/template room is reserved. Whole images are never cut.",
    }
    return selected, metadata


def context_overflow(error):
    text = str(error).lower()
    return "http 400" in text and any(term in text for term in (
        "maximum context", "context length", "max_model_len", "longer than the maximum"))


def generation_messages(paths, question):
    return [{"role": "user", "content": page_content(paths) +
             [{"type": "text", "text": "Question: " + question}]}]


def chat(session, base_url, model, key, messages, max_tokens, timeout, thinking,
         temperature=0.0, max_pixels=None, min_pixels=65536):
    payload = {
        "model": model, "messages": messages, "temperature": temperature,
        "max_tokens": max_tokens, "stream": False,
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
    if max_pixels is not None:
        payload["mm_processor_kwargs"] = {"min_pixels": min_pixels, "max_pixels": max_pixels}
    headers = {"Authorization": "Bearer " + key} if key else {}
    started = time.monotonic()
    for attempt in range(3):
        try:
            response = session.post(
                base_url.rstrip("/") + "/chat/completions", json=payload,
                headers=headers, timeout=(10, timeout),
            )
            if response.status_code == 429 or response.status_code >= 500:
                response.raise_for_status()
            if not response.ok:
                raise ValueError(f"HTTP {response.status_code}: {response.text[:2000]}")
            result = response.json()
            choice = result["choices"][0]
            text = choice["message"].get("content") or ""
            if not text.strip():
                raise ValueError(
                    f"正文为空，finish_reason={choice.get('finish_reason')}；"
                    "若思考耗尽输出预算，可关闭思考或增加 max_tokens。"
                )
            message = choice["message"]
            return text, {"finish_reason": choice.get("finish_reason"),
                          "usage": result.get("usage"), "id": result.get("id"),
                          "model": result.get("model"), "elapsed_seconds": time.monotonic() - started,
                          "reasoning": message.get("reasoning") or message.get("reasoning_content")}
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def parse_prediction(extracted):
    match = re.search(r"Extracted answer:\s*(.*?)\s*Answer format:", extracted, re.S)
    if not match or not match.group(1).strip():
        raise ValueError("抽取回复缺少 Extracted answer: / Answer format:，请检查 extracted_res。")
    return match.group(1).strip()


def official_score(sample, prediction):
    gt = sample["answer"]
    # 原评分函数对 List 字符串使用 eval；先安全解析，再传入原函数。
    if sample["answer_format"] not in {"Int", "Float", "Str", "None"}:
        if isinstance(gt, str) and gt.startswith("["):
            gt = ast.literal_eval(gt)
        if prediction.startswith("["):
            prediction = ast.literal_eval(prediction)
    return scorer.eval_score(gt, prediction, sample["answer_format"])


def report(records, output):
    # 未完成和接口失败按 0 分计入总体分母，同时单独报告覆盖率，避免虚高。
    rows = [{**s, "score": s.get("score", 0.0),
             "pred": s.get("pred", "Failed to answer")} for s in records]
    acc, f1 = scorer.eval_acc_and_f1(rows)
    complete = sum(s.get("status") == "complete" for s in records)
    errors = sum(s.get("status") == "error" for s in records)
    summary = {
        "total": len(records), "complete": complete, "errors": errors,
        "pending": len(records) - complete - errors,
        "accuracy_with_incomplete_as_zero": acc, "f1_with_incomplete_as_zero": f1,
        "accuracy_percent": 100 * acc, "f1_percent": 100 * f1,
        "evaluation_complete": complete == len(records),
        "truncated_inputs": sum(bool(s.get("input_truncation", {}).get("applied")) for s in records),
        "truncated_answers": sum(s.get("generation", {}).get("finish_reason") == "length"
                                 for s in records),
    }
    save_json(output.with_suffix(".summary.json"), summary)
    report_rows = copy.deepcopy(rows)
    for row in report_rows:
        for field in ("evidence_pages", "evidence_sources"):
            if not isinstance(row[field], str):
                row[field] = repr(row[field])
    report_path = output.with_suffix(".txt")
    scorer.show_results(report_rows, show_path=str(report_path))
    report_path.write_text(
        f"Coverage: {complete}/{len(records)} complete; {errors} errors. "
        "Incomplete/error samples count as zero.\n" + report_path.read_text(), encoding="utf-8",
    )
    print(f"完成 {complete}/{len(records)}，失败 {errors}；ACC={acc:.4f}，F1={f1:.4f}")
    if complete != len(records):
        print("这是未完成的评测：失败和未完成题目按 0 分计入；重跑相同命令可以继续。")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/mmlongbench/samples.json")
    parser.add_argument("--images-dir", type=Path, default=ROOT / "data/mmlongbench/images_1024")
    parser.add_argument("--output", type=Path,
                        help="默认按思考开关区分结果文件，避免混用两种评测设置。")
    parser.add_argument("--base-url", default=os.getenv("QWEN_BASE_URL", "http://127.0.0.1:8037/v1"))
    parser.add_argument("--model", default=os.getenv("QWEN_MODEL", "qwen35-mmlongbench"))
    parser.add_argument("--extractor-base-url", default=os.getenv("EXTRACTOR_BASE_URL") or os.getenv("CITEAGENT_EVALUATOR_BASE_URL"))
    parser.add_argument("--extractor-model", default=os.getenv("EXTRACTOR_MODEL") or os.getenv("CITEAGENT_EVALUATOR_MODEL"))
    parser.add_argument("--stage", choices=("all", "predict", "score"), default="all")
    parser.add_argument("--limit", type=int, default=0, help="前 N 题；0 为全部。扩大 limit 可以续跑。")
    parser.add_argument("--concurrency", type=int, default=1, help="同时处理的题目数；服务端决定实际推理 batch。")
    parser.add_argument("--num-shards", type=int, default=1, help="按原始题号取模分片，配合独立 output 使用。")
    parser.add_argument("--shard-index", type=int, default=0, help="分片编号，从 0 开始。")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--extract-max-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-pixels", type=int, default=262144,
                        help="每页视觉预处理的最大像素数；默认约 512x512，保留所有页面。")
    parser.add_argument("--native-resolution", action="store_true",
                        help="检查 max-pixels 足以保留源页图分辨率，只允许模型要求的 32 像素对齐。")
    parser.add_argument("--max-model-len", type=int,
                        help="与服务一致；原分辨率模式预检查上下文，是否截页由truncate-long-documents控制。")
    parser.add_argument("--truncate-long-documents", action=argparse.BooleanOptionalAction, default=False,
                        help="按整页边界保留文档前部，不缩图；记录截断信息")
    parser.add_argument("--input-token-limit", type=int, default=260000)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=False,
                        help="与归档脚本一致，回答默认关闭思考；--enable-thinking 可开启。抽取固定关闭思考。")
    parser.add_argument("--dry-run", action="store_true", help="仅检查题目和所有页图，不调用 API、不写结果。")
    args = parser.parse_args(argv)
    if args.output is None:
        filename = "qwen35_direct.json" if args.enable_thinking else "qwen35_direct_no_thinking.json"
        if args.native_resolution:
            filename = filename.replace(".json", "_native.json")
        if args.num_shards > 1:
            filename = filename.replace(".json", f".shard{args.shard_index}.json")
        args.output = WORKDIR / "output/mmlongbench" / filename
    if (args.limit < 0 or args.concurrency <= 0 or args.max_tokens <= 0
            or args.extract_max_tokens <= 0 or args.timeout <= 0
            or args.temperature < 0 or args.max_pixels < 65536):
        parser.error("limit/temperature 必须非负；并发、token 预算、timeout 必须为正；max-pixels 至少 65536。")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("num-shards 必须为正，且 0 <= shard-index < num-shards。")
    if args.max_model_len is not None and args.max_model_len <= args.max_tokens:
        parser.error("max-model-len 必须大于 max-tokens。")
    if args.truncate_long_documents and (not args.native_resolution or args.max_model_len is None):
        parser.error("前缀截断需要--native-resolution和--max-model-len")
    if args.input_token_limit <= 0:
        parser.error("input-token-limit必须为正")
    args.extractor_base_url = args.extractor_base_url or args.base_url
    args.extractor_model = args.extractor_model or args.model
    input_bytes = args.input.read_bytes()
    samples = json.loads(input_bytes)
    selected_pairs = [(i, s) for i, s in enumerate(samples[:args.limit] if args.limit else samples)
                      if i % args.num_shards == args.shard_index]
    selected = [s for _, s in selected_pairs]
    if not selected:
        parser.error("样本为空。")
    for sample in selected:
        for field in ("doc_id", "question", "answer", "answer_format", "evidence_pages", "evidence_sources", "doc_type"):
            if field not in sample:
                parser.error(f"样本缺少字段：{field}")
    if args.stage != "score":
        documents = {s["doc_id"] for s in selected}
        counts = [len(page_paths(args.images_dir, doc)) for doc in documents]
        print(f"数据检查：{len(selected)} 题 / {len(documents)} 份文档 / {sum(counts)} 张页图；"
              f"最长 {max(counts)} 页。前缀截断={args.truncate_long_documents}。")
        print(f"页图目录：{args.images_dir.resolve()}")
        print(f"回答服务：{args.model} @ {args.base_url}；thinking={args.enable_thinking}；"
              f"每页 max_pixels={args.max_pixels}。服务端需允许至少 {max(counts)} 张图片。")
        print("不检索、不发送GT或证据页标签；" +
              ("超长文档仅保留可容纳的整页前缀。" if args.truncate_long_documents else "保留全部页，超限记失败。"))
        if args.num_shards > 1:
            print(f"分片 {args.shard_index}/{args.num_shards}：{len(selected)} 题，保留原始题号。")
        if args.native_resolution:
            infos = {doc: native_page_info(page_paths(args.images_dir, doc)) for doc in documents}
            largest = max(info[1] for info in infos.values())
            if largest > args.max_pixels:
                parser.error(f"max-pixels={args.max_pixels} 会缩图：需要至少 {largest}；请提高上限。")
            oversized = {doc for doc, (tokens, _) in infos.items()
                         if args.max_model_len and tokens + args.max_tokens >= args.max_model_len}
            print(f"原分辨率检查通过：最大对齐面积 {largest} 像素；仅做 32 像素对齐。")
            if oversized:
                print(f"明确超出上下文：{len(oversized)} 份文档 / "
                      f"{sum(s['doc_id'] in oversized for s in selected)} 题；" +
                      ("将保留整页前缀并记录截断。" if args.truncate_long_documents else "保留全部页，超限记失败。"))
    if args.dry_run:
        return 0

    prompt = EXTRACTION_PROMPT_PATH.read_text(encoding="utf-8")
    generation_config = {"base_url": args.base_url.rstrip("/"), "model": args.model,
                         "images_dir": str(args.images_dir.resolve()), "max_tokens": args.max_tokens,
                         "enable_thinking": args.enable_thinking, "instruction": INSTRUCTION,
                         "temperature": args.temperature, "min_pixels": 65536, "max_pixels": args.max_pixels}
    if args.native_resolution:
        generation_config.update(min_pixels=1024, native_resolution=True,
                                 spatial_factor=32, max_model_len=args.max_model_len)
    if args.truncate_long_documents:
        generation_config["input_truncation"] = {
            "policy": "whole-page-prefix-v1", "input_token_limit": args.input_token_limit,
            "context_limit": args.max_model_len, "output_token_budget": args.max_tokens}
    extraction_config = {"base_url": args.extractor_base_url.rstrip("/"), "model": args.extractor_model,
                         "max_tokens": args.extract_max_tokens, "enable_thinking": False,
                         "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
    metadata_path = args.output.with_suffix(".meta.json")
    metadata = {"input_sha256": hashlib.sha256(input_bytes).hexdigest(),
                "generation": generation_config, "extraction": None}
    if args.num_shards > 1:
        metadata["partition"] = {"num_shards": args.num_shards, "shard_index": args.shard_index}
    previous = {}
    if args.output.exists():
        if not metadata_path.exists():
            parser.error("已有结果缺少 .meta.json 配置记录，请换一个 output 路径。")
        old_meta = json.loads(metadata_path.read_text())
        if old_meta["input_sha256"] != metadata["input_sha256"]:
            parser.error("输入数据与已有结果不同，请换一个 output 路径。")
        if old_meta.get("partition") != metadata.get("partition"):
            parser.error("分片设置与已有结果不同，请换一个 output 路径。")
        old_generation = old_meta["generation"]
        if args.stage != "score":
            # 不截断时允许容量变化后复用完整答案；截断时容量会改变输入，须一致。
            old_comparable = {k: v for k, v in old_generation.items() if k != "max_model_len" or args.truncate_long_documents}
            new_comparable = {k: v for k, v in generation_config.items() if k != "max_model_len" or args.truncate_long_documents}
            if old_comparable != new_comparable:
                parser.error("生成配置与已有结果不同，请换一个 output 路径。")
        if args.stage != "predict" and old_meta["extraction"] not in (None, extraction_config):
            parser.error("抽取配置与已有结果不同，请换一个 output 路径。")
        metadata = old_meta
        previous = {s["_sample_index"]: s for s in json.loads(args.output.read_text())}
        if args.native_resolution:
            for sample in previous.values():
                if sample.get("response") and isinstance(sample.get("generation"), dict):
                    sample["generation"].setdefault("context_limit", old_generation.get("max_model_len"))
        if args.stage != "score" and old_generation != generation_config:
            history = metadata.setdefault("generation_history", [copy.deepcopy(old_generation)])
            history.append(copy.deepcopy(generation_config))
            metadata["generation"] = generation_config
    if args.stage != "predict":
        metadata["extraction"] = extraction_config
    records = [previous.get(i, {**s, "_sample_index": i, "status": "pending"})
               for i, s in selected_pairs]
    if not set(previous).issubset({i for i, _ in selected_pairs}):
        parser.error("选择范围不包含已有结果；请沿用或扩大 limit，避免丢失已有结果。")
    if args.stage == "score" and not any(s.get("response") for s in records):
        parser.error("没有可评分的 response；请先用 --stage predict 或 --stage all 生成回答。")

    key = ""
    if args.stage != "score":
        key = os.getenv("QWEN_API_KEY") or os.getenv("API_KEY") or ""
    extractor_key = ""
    if args.stage != "predict":
        extractor_key = os.getenv("EXTRACTOR_API_KEY") or os.getenv("CITEAGENT_EVALUATOR_API_KEY", "")
        if not extractor_key and args.extractor_base_url.rstrip("/") == args.base_url.rstrip("/"):
            extractor_key = key or os.getenv("QWEN_API_KEY") or os.getenv("API_KEY") or ""
        print(f"短答案抽取模型：{args.extractor_model} @ {args.extractor_base_url}")
    metadata["concurrency"] = args.concurrency
    save_json(metadata_path, metadata)
    save_json(args.output, records)
    io_lock = threading.Lock()
    stopping = threading.Event()

    def process(original):
        if stopping.is_set():
            return
        sample = dict(original)
        session = requests.Session()
        session.trust_env = False

        def commit():
            # 线程在各自副本中处理；更新共享结果和写文件必须在同一把锁内完成。
            with io_lock:
                original.update(sample)
                for field in ("error", "score", "pred"):
                    if field not in sample:
                        original.pop(field, None)
                save_json(args.output, records)

        try:
            sample.pop("error", None)
            sample.pop("score", None)
            if not sample.get("response"):
                all_paths = page_paths(args.images_dir, sample["doc_id"])
                paths = all_paths
                truncation = None
                if args.truncate_long_documents:
                    paths, truncation = prefix_pages(all_paths, sample["question"], args.max_model_len,
                                                      args.max_tokens, args.input_token_limit)
                    sample["input_truncation"] = truncation
                if args.native_resolution:
                    tokens, _ = native_page_info(paths)
                    sample.update(page_count=len(paths), original_page_count=len(all_paths),
                                  estimated_visual_tokens=tokens)
                    if not args.truncate_long_documents and args.max_model_len and tokens + args.max_tokens >= args.max_model_len:
                        raise ValueError(f"原分辨率上下文超限：视觉约{tokens}+输出{args.max_tokens} >= {args.max_model_len}")
                while True:
                    try:
                        text, info = chat(session, args.base_url, args.model, key,
                                          generation_messages(paths, sample["question"]),
                                          args.max_tokens, args.timeout, args.enable_thinking,
                                          temperature=args.temperature, max_pixels=args.max_pixels,
                                          min_pixels=1024 if args.native_resolution else 65536)
                        actual = (info.get("usage") or {}).get("prompt_tokens")
                        if truncation and isinstance(actual, int) and actual > truncation["input_token_limit"]:
                            truncation.setdefault("discarded_generations", []).append({
                                "usage": info.get("usage"), "elapsed_seconds": info.get("elapsed_seconds")})
                            raise InputBudgetExceeded(f"Actual prompt {actual} exceeds input budget {truncation['input_token_limit']}")
                        break
                    except ValueError as exc:
                        if (not truncation or not (context_overflow(exc) or isinstance(exc, InputBudgetExceeded)) or len(paths) <= 1 or
                            truncation["context_retries"] >= 8):
                            raise
                        paths = paths[:-1]
                        truncation.update(applied=True, kept_pages=len(paths),
                                          dropped_pages=len(all_paths)-len(paths),
                                          kept_page_indices=list(range(len(paths))),
                                          estimated_visual_tokens=native_page_info(paths)[0],
                                          estimated_prompt_tokens_with_reserve=(native_page_info(paths)[0] + 16*len(paths) + truncation["text_template_token_reserve"]),
                                          context_retries=truncation["context_retries"]+1)
                        truncation.setdefault("context_errors", []).append(str(exc)[:1000])
                if truncation:
                    truncation["actual_prompt_tokens"] = (info.get("usage") or {}).get("prompt_tokens")
                if args.native_resolution:
                    info["context_limit"] = args.max_model_len
                sample.update(response=text, generation=info, page_count=len(paths), status="predicted")
                if args.native_resolution:
                    sample["estimated_visual_tokens"] = native_page_info(paths)[0]
                commit()  # 抽取失败时保留已生成的回答。
            if args.stage == "predict" or stopping.is_set():
                sample["status"] = "predicted"
                return
            extracted, info = chat(
                session, args.extractor_base_url, args.extractor_model, extractor_key,
                [{"role": "user", "content": prompt},
                 {"role": "assistant", "content": "\n\nQuestion:{}\nAnalysis:{}\n".format(
                     sample["question"], sample["response"])}],
                args.extract_max_tokens, args.timeout, False,
            )
            sample.update(extracted_res=extracted, extraction=info)
            pred = parse_prediction(extracted)
            score = official_score(sample, pred)
            sample.update(pred=pred, score=score, status="complete")
        except Exception as exc:
            sample.update(status="error", error=f"{type(exc).__name__}: {exc}")
            sample.pop("score", None)
            sample.pop("pred", None)
            tqdm.write(f"[失败] #{sample['_sample_index']} {sample['doc_id']}: {sample['error']}")
        finally:
            session.close()
            commit()

    todo = [s for s in records if s.get("status") != "complete"
            and not (args.stage == "score" and not s.get("response"))
            and not (args.stage == "predict" and s.get("response"))]
    print(f"本次待处理 {len(todo)} 题，客户端并发={args.concurrency}。")
    pool = ThreadPoolExecutor(max_workers=args.concurrency)
    futures = []
    started = time.monotonic()
    try:
        futures = [pool.submit(process, sample) for sample in todo]
        for future in tqdm(as_completed(futures), total=len(todo), desc=args.stage, unit="题"):
            future.result()
    except BaseException:
        stopping.set()
        for future in futures:
            future.cancel()
        print("正在停止提交新题目，等待进行中的请求结束并保存结果……", flush=True)
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        save_json(args.output, records)
        if args.stage != "predict":
            report(records, args.output)
    elapsed = time.monotonic() - started
    print(f"本次耗时 {elapsed:.1f}s，处理 {len(todo)} 题；吞吐 {len(todo) / max(elapsed, 0.001):.3f} 题/s。")
    print(f"结果：{args.output}")
    if args.stage == "predict":
        return int(any(not s.get("response") for s in records))
    return int(any(s.get("status") != "complete" for s in records))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n已中断；进度已保存，重跑相同命令可续跑。", file=sys.stderr)
        raise SystemExit(130)
