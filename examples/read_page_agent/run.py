#!/usr/bin/env python3
"""最小 read_page Agent：原始问题 → thinking/tool_calls → role=tool 页图 → 回答。

先启动 serve_qwen35_read_page_agent.sh。依赖 requests、Pillow；无 Agent 框架、OCR、检索器或抽取器。
"""

import argparse
import base64
import copy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys

import requests
from PIL import Image

WORKDIR = next(p for p in Path(__file__).resolve().parents if (p / ".gitignore").is_file())
DATA = WORKDIR / "aaai_commit_20260803/aaai_commit/data/mmlongbench"


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def compact(value):
    """落盘时省略重复 base64；实际发给 API 的 messages 始终包含真实图片。"""
    if isinstance(value, dict):
        return {key: compact(item) for key, item in value.items()}
    if isinstance(value, list):
        return [compact(item) for item in value]
    if isinstance(value, str) and value.startswith("data:image/"):
        return f"[image data URL omitted; sha256={hashlib.sha256(value.encode()).hexdigest()}]"
    return value


def tools(k, total):
    return [{"type": "function", "function": {
        "name": "read_page",
        "description": f"Read 1 to {k} document pages. Returns the original page images. "
                       f"Page numbers are 1-based, from 1 to {total}; page 1 is the first PDF page, "
                       "not necessarily the page number printed on the page.",
        "parameters": {"type": "object", "properties": {
            "pages": {"type": "array", "items": {"type": "integer", "minimum": 1, "maximum": total},
                      "minItems": 1, "maxItems": k, "uniqueItems": True}},
            "required": ["pages"], "additionalProperties": False},
    }}]


def read_page(arguments, paths, k):
    args = json.loads(arguments) if isinstance(arguments, str) else arguments
    if not isinstance(args, dict) or set(args) != {"pages"}:
        raise ValueError('Expected {"pages": [1, 2]}.')
    pages = args["pages"]
    if (not isinstance(pages, list) or not 1 <= len(pages) <= k
            or any(type(p) is not int or not 1 <= p <= len(paths) for p in pages)):
        raise ValueError(f"pages must contain 1 to {k} integer page numbers in [1, {len(paths)}].")
    if len(set(pages)) != len(pages):
        raise ValueError("Do not repeat page numbers in the same call.")
    content = []
    for page in pages:
        raw = paths[page - 1].read_bytes()
        content += [{"type": "text", "text": f"Document page {page} of {len(paths)}:"},
                    {"type": "image_url", "image_url": {
                        "url": "data:image/png;base64," + base64.b64encode(raw).decode("ascii")}}]
    return pages, content


def assistant_history(message):
    history = {"role": "assistant", "content": message.get("content") or ""}
    if message.get("tool_calls"):
        history["tool_calls"] = copy.deepcopy(message["tool_calls"])
    reasoning = message.get("reasoning") or message.get("reasoning_content")
    if reasoning:
        # vLLM 0.25.1 将 reasoning 映射为模板使用的 reasoning_content。
        history["reasoning"] = reasoning
    return history


def run_sample(session, args, index, sample, paths, out, record):
    messages = [{"role": "system", "content": (
        "Answer the user's question using only the document. You initially see no pages. "
        f"Use read_page to inspect 1 to {args.k} pages at a time. Make at most one tool call per turn. "
        "Choose pages yourself, inspect the returned images, and continue reading if needed. "
        "Do not guess unseen page contents. You may revisit pages; all earlier images remain in context. "
        "Once sufficient evidence is available, give only a concise final answer. "
        "If the document does not contain the answer, respond 'Not answerable'. "
        f"You have at most {args.max_rounds} assistant turns, including the final answer."
    )}, {"role": "user", "content": f"Document length: {len(paths)} pages.\nQuestion: {sample['question']}"}]
    record.update(sample_index=index, doc_id=sample["doc_id"], question=sample["question"],
                  page_count=len(paths), status="running", stop_reason=None, final_answer=None,
                  tool_calls=0, tool_errors=0, read_pages=[], thinking_observed=False, steps=[])

    def checkpoint():
        # GT 只在请求构造完成后写入本地记录，不进入 messages 或 tools。
        save(out / f"sample_{index:04d}.json", {
            **record, "ground_truth": sample["answer"], "answer_format": sample["answer_format"],
            "messages": compact(messages)})

    checkpoint()
    try:
        for turn in range(1, args.max_rounds + 1):
            payload = {"model": args.model, "messages": messages, "tools": tools(args.k, len(paths)),
                       "tool_choice": "auto", "parallel_tool_calls": False, "stream": False,
                       "temperature": args.temperature, "top_p": 0.95, "seed": args.seed,
                       "max_tokens": args.max_tokens, "chat_template_kwargs": {"enable_thinking": True},
                       "mm_processor_kwargs": {"min_pixels": 1024, "max_pixels": 1048576}}
            print(f"  turn {turn}/{args.max_rounds}：请求模型……", flush=True)
            response = session.post(args.base_url.rstrip("/") + "/chat/completions", json=payload,
                                    timeout=(10, args.timeout))
            if not response.ok:
                raise RuntimeError(f"HTTP {response.status_code}: {response.text[:2000]}")
            result = response.json()
            choice = result["choices"][0]
            message = choice["message"]
            reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
            record["thinking_observed"] |= bool(reasoning)
            step = {"turn": turn, "response": result, "tool_results": []}
            record["steps"].append(step)
            messages.append(assistant_history(message))
            print(f"  finish={choice.get('finish_reason')}；thinking_chars={len(reasoning)}", flush=True)
            if args.show_thinking and reasoning:
                print(f"  Thought:\n{reasoning}", flush=True)
            calls = message.get("tool_calls") or []
            if choice.get("finish_reason") == "length":
                record.update(status="incomplete", stop_reason="max_tokens")
                break  # 不把截断输出误认作最终答案，也不执行可能被截断的参数。
            if not calls:
                if choice.get("finish_reason") != "stop" or not (message.get("content") or "").strip():
                    raise RuntimeError("未返回有效 tool_calls 或非空最终答案；请检查工具解析器/输出预算。")
                record.update(status="finished", stop_reason="final_answer", final_answer=message["content"].strip())
                print(f"  Final answer: {record['final_answer']}", flush=True)
                break
            for call_number, call in enumerate(calls):
                if not call.get("id"):
                    raise RuntimeError("tool_call 缺少 id，不能构造关联的 tool 消息。")
                record["tool_calls"] += 1
                try:
                    function = call["function"]
                    if function["name"] != "read_page":
                        raise ValueError("Only read_page is available.")
                    if call_number > 0:
                        raise ValueError("Make at most one read_page call per assistant turn.")
                    pages, content = read_page(function.get("arguments", "{}"), paths, args.k)
                    record["read_pages"].extend(pages)
                    step["tool_results"].append({"tool_call_id": call["id"], "pages": pages,
                                                 "image_paths": [str(paths[p - 1]) for p in pages]})
                    print(f"  Action: read_page(pages={pages}) → tool 消息返回 {len(pages)} 张图片", flush=True)
                except (ValueError, TypeError, KeyError) as exc:
                    record["tool_errors"] += 1
                    content = [{"type": "text", "text": f"Tool error: {exc} Please correct the call."}]
                    step["tool_results"].append({"tool_call_id": call["id"], "error": str(exc)})
                    print(f"  Tool error: {exc}", flush=True)
                # 关键：图片属于关联到真实 tool_call_id 的 role=tool 消息。
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": content})
            checkpoint()
        else:
            record.update(status="incomplete", stop_reason="max_rounds")
    except KeyboardInterrupt:
        record.update(status="interrupted", stop_reason="interrupted")
        raise
    except (requests.RequestException, RuntimeError, ValueError, KeyError, IndexError, TypeError, OSError) as exc:
        record.update(status="error", stop_reason="error", error=f"{type(exc).__name__}: {exc}")
        print(f"  ERROR: {record['error']}", flush=True)
    finally:
        record["rounds"] = len(record["steps"])
        record["unique_read_pages"] = sorted(set(record["read_pages"]))
        checkpoint()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8057/v1")
    parser.add_argument("--model", default="qwen35-read-page-agent")
    parser.add_argument("--input", type=Path, default=DATA / "samples.json")
    parser.add_argument("--images-dir", type=Path, default=DATA / "images_1024")
    parser.add_argument("--sample-indices", default="281,283,289,291", help="samples.json 的 0-based 题号，逗号分隔。")
    parser.add_argument("--k", type=int, default=2, help="每次 read_page 最多读取的页数，页号从 1 开始。")
    parser.add_argument("--max-rounds", type=int, default=8, help="最多请求模型的次数，含最终回答轮。")
    parser.add_argument("--max-tokens", type=int, default=8192, help="每轮 thinking 与正文共享的输出预算。")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--show-thinking", action="store_true", help="在终端显示服务返回的 reasoning；默认只落盘。")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if min(args.k, args.max_rounds, args.max_tokens, args.timeout) <= 0 or args.temperature < 0:
        parser.error("k、轮数、输出预算、超时必须为正，temperature 必须非负。")
    samples = json.loads(args.input.read_text())
    try:
        indices = [int(value) for value in args.sample_indices.split(",")]
    except ValueError:
        parser.error("sample-indices 应为逗号分隔的整数。")
    if len(set(indices)) != len(indices) or any(not 0 <= i < len(samples) for i in indices):
        parser.error("题号重复或超出输入范围。")
    selected = []
    for index in indices:
        sample = samples[index]
        folder = args.images_dir / Path(sample["doc_id"]).stem
        paths = sorted(folder.glob("page_*.png"), key=lambda p: int(p.stem.split("_")[-1]))
        if not paths or [int(p.stem.split("_")[-1]) for p in paths] != list(range(len(paths))):
            parser.error(f"页图缺失或不连续：{folder}")
        for path in paths:
            with Image.open(path) as image:
                width, height = image.size
            if max(32, round(width / 32) * 32) * max(32, round(height / 32) * 32) > 1048576:
                parser.error(f"图片超过测试的原分辨率上限：{path}，请使用 images_1024。")
        selected.append((index, sample, paths))
        print(f"#{index}：{len(paths)} 页；{sample['question']}", flush=True)
    print(f"thinking=true；每次最多 {args.k} 页；最多 {args.max_rounds} 轮；保留全部历史页图与 reasoning。", flush=True)
    if args.dry_run:
        return 0
    out = args.output_dir or WORKDIR / f"output/read_page_agent/test_{datetime.now():%Y%m%d_%H%M%S}_{os.getpid()}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "results.json").exists() or any(out.glob("sample_*.json")):
        parser.error("输出目录已有测试记录，请换一个 --output-dir。")
    save(out / "config.json", {key: str(value) if isinstance(value, Path) else value
                               for key, value in vars(args).items()})
    print(f"输出目录：{out}", flush=True)
    records = []
    session = requests.Session()
    session.trust_env = False
    key = os.getenv("QWEN_API_KEY") or os.getenv("API_KEY")
    if key:
        session.headers["Authorization"] = "Bearer " + key
    interrupted = False
    try:
        for index, sample, paths in selected:
            print(f"\n===== sample #{index} / {len(paths)} pages =====", flush=True)
            record = {}
            records.append(record)
            run_sample(session, args, index, sample, paths, out, record)
            save(out / "results.json", records)
    except KeyboardInterrupt:
        interrupted = True
    finally:
        session.close()
        save(out / "results.json", records)
        summary = {"selected": len(selected), "attempted": len(records),
                   "finished": sum(r.get("status") == "finished" for r in records),
                   "finished_after_reading_images": sum(r.get("status") == "finished" and bool(r.get("read_pages")) for r in records),
                   "thinking_observed": sum(bool(r.get("thinking_observed")) for r in records),
                   "note": "这里只统计工具循环是否完成；不调用抽取器，不代表答案正确或全量 benchmark 成绩。"}
        save(out / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if interrupted:
        return 130
    return 0 if summary["finished_after_reading_images"] == len(selected) else 1


if __name__ == "__main__":
    raise SystemExit(main())
