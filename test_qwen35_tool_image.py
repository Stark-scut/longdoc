"""用 a_tool_image.png 测试 Qwen3.5 能否识别 tool 消息中的左右色块。

终端一：bash serve_qwen35_tool_image.sh
终端二：python test_qwen35_tool_image.py
依赖：requests、Pillow（qwen35 环境已安装）。
退出码 0 表示工具调用与图片识别均通过；其他情况查看输出目录。
"""

import argparse
import base64
from datetime import datetime
import json
import os
from pathlib import Path
import re
import sys

import requests
from PIL import Image


TOOLS = [{"type": "function", "function": {
    "name": "get_test_image",
    "description": "返回一张左右各有一个色块的图片。工具只返回图片，不返回色块颜色的文字答案。",
    "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
}}]
INSTRUCTION = (
    "请观察图片的左半边和右半边，分别判断两个色块的颜色。"
    "使用常见的英文颜色名称，只输出 LEFT=颜色; RIGHT=颜色，不要猜测。"
)


def load_image(path):
    """读取用户提供的原图；参考答案仅在本地校验使用。"""
    source = Path(__file__).resolve().with_name("a_tool_image.png")
    raw = source.read_bytes()
    with Image.open(source) as image:
        image.verify()
        if image.format != "PNG":
            raise RuntimeError("a_tool_image.png 必须为 PNG 图片")
    path.write_bytes(raw)
    expected = {"left": "yellow", "right": "white"}
    return expected, "data:image/png;base64," + base64.b64encode(raw).decode("ascii")


def image_content(image_url):
    # 关键：图片作为结构化 image_url part，而不是 JSON 字符串或图片路径。
    return [{"type": "text", "text": INSTRUCTION},
            {"type": "image_url", "image_url": {"url": image_url}}]


def history_message(message):
    """仅回传消息字段，并保留 vLLM 的轮间 thinking 信息。"""
    result = {"role": "assistant", "content": message.get("content")}
    if message.get("tool_calls"):
        result["tool_calls"] = message["tool_calls"]
    reasoning = message.get("reasoning") or message.get("reasoning_content")
    if reasoning:
        result["reasoning"] = reasoning
    return result


def read_colors(message):
    text = message.get("content") or ""
    colors = {}
    aliases = {"黄色": "yellow", "黄": "yellow", "白色": "white", "白": "white",
               "红色": "red", "红": "red", "绿色": "green", "绿": "green",
               "蓝色": "blue", "蓝": "blue", "黑色": "black", "黑": "black"}
    for side in ("left", "right"):
        match = re.search(rf"\b{side}\s*[=:：]\s*([a-z]+|[\u4e00-\u9fff]+)", text, re.IGNORECASE)
        if not match:
            return None
        color = match.group(1).lower()
        colors[side] = aliases.get(color, color)
    return colors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8027/v1")
    parser.add_argument("--model", default="qwen35-tool-image")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    out = args.output_dir or Path(__file__).resolve().parent / "output" / (
        f"tool_image_test_{datetime.now():%Y%m%d_%H%M%S}_{os.getpid()}")
    out.mkdir(parents=True, exist_ok=True)
    print(f"输出目录：{out}", flush=True)
    expected, data_url = load_image(out / "tool_image.png")
    print("测试图片：a_tool_image.png；本地参考答案：左边黄色、右边白色（不放入请求文本）", flush=True)

    def get_test_image():
        """实际执行模型请求的工具；返回结构化的图片内容。"""
        return image_content(data_url)

    session = requests.Session()
    session.trust_env = False  # 本地直连，避免代理转发 localhost 请求。
    if os.environ.get("API_KEY"):
        session.headers["Authorization"] = "Bearer " + os.environ["API_KEY"]

    def chat(stage, messages, tool_choice=None):
        payload = {"model": args.model, "messages": messages, "stream": False,
                   "max_tokens": args.max_tokens, "temperature": 0.6, "top_p": 0.95,
                   "chat_template_kwargs": {"enable_thinking": True}}
        if tool_choice is not None:
            payload.update(tools=TOOLS, tool_choice=tool_choice, parallel_tool_calls=False)
        (out / f"{stage}_request.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        response = session.post(args.base_url.rstrip("/") + "/chat/completions", json=payload,
                                timeout=(10, args.timeout))
        (out / f"{stage}_response.txt").write_text(response.text)
        if not response.ok:
            raise RuntimeError(f"{stage}: HTTP {response.status_code}: {response.text[:3000]}")
        choice = response.json()["choices"][0]
        message = choice["message"]
        reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
        print(f"{stage}: finish_reason={choice.get('finish_reason')}, thinking_chars={len(reasoning)}", flush=True)
        if choice.get("finish_reason") == "length":
            raise RuntimeError(f"{stage}: 输出达到 token 上限；可增大 --max-tokens，必要时增加服务 MAX_MODEL_LEN")
        return message, bool(reasoning)

    result = {"image_source": str(Path(__file__).resolve().with_name("a_tool_image.png")),
              "expected_colors": expected, "tool_called": False, "tool_image_accepted": False,
              "tool_image_correct": False, "user_image_correct": False, "thinking_observed": False}
    try:
        messages = [{"role": "user", "content": (
            "请先调用 get_test_image 获取图片，然后观察工具返回图片的左半边和右半边，"
            "分别判断两个色块的颜色。收到图片后使用常见英文颜色名称，"
            "只输出 LEFT=颜色; RIGHT=颜色。") }]
        print("1. 请求模型自动调用 get_test_image……", flush=True)
        assistant, thinking = chat("01_tool_call", messages, "auto")
        result["thinking_observed"] |= thinking
        calls = assistant.get("tool_calls") or []
        if len(calls) != 1 or calls[0]["function"]["name"] != "get_test_image":
            raise RuntimeError("未收到一个有效的 get_test_image tool_call，请查看 01_tool_call_response.txt")
        if json.loads(calls[0]["function"].get("arguments") or "{}") != {}:
            raise RuntimeError("get_test_image 应使用空参数 {}")
        result["tool_called"] = True
        print("tool_calls：" + json.dumps(calls, ensure_ascii=False), flush=True)
        messages.append(history_message(assistant))
        # 执行工具，将用户提供的原图作为 role=tool 的结果回传。
        messages.append({"role": "tool", "tool_call_id": calls[0]["id"],
                         "content": get_test_image()})
        print("2. 将图片直接放入 role=tool 消息，再请求模型读取……", flush=True)
        answer, thinking = chat("02_tool_image", messages, "none")
        result["thinking_observed"] |= thinking
        result["tool_image_accepted"] = True
        result["tool_answer"] = answer.get("content")
        result["tool_colors"] = read_colors(answer)
        result["tool_image_correct"] = result["tool_colors"] == expected
        print("工具图片回复：", answer.get("content"), flush=True)
    except (requests.RequestException, RuntimeError, ValueError, KeyError, IndexError, TypeError) as exc:
        result["tool_error"] = str(exc)
        print(f"工具图片测试失败：{exc}", flush=True)

    # 对照请求在工具测试之后、独立上下文中进行，不会提前把答案泄露给工具测试。
    print("3. 对照：把同一张图片放入独立的 user 消息……", flush=True)
    try:
        answer, _ = chat("03_user_image_control", [{"role": "user", "content": image_content(data_url)}])
        result["user_colors"] = read_colors(answer)
        result["user_image_correct"] = result["user_colors"] == expected
        result["user_answer"] = answer.get("content")
        print("用户图片回复：", answer.get("content"), flush=True)
    except (requests.RequestException, RuntimeError, ValueError, KeyError, IndexError, TypeError) as exc:
        result["control_error"] = str(exc)

    result["passed"] = result["tool_called"] and result["tool_image_correct"]
    (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if result["passed"]:
        print("PASS：收到真实 tool_calls，且模型从 tool 消息中的原图正确识别左黄、右白。")
        if not result["thinking_observed"]:
            print("注意：请求已开启 thinking，但响应未观察到非空 reasoning，请检查响应与服务日志。")
        return 0
    if result["user_image_correct"] and result["tool_called"]:
        print("FAIL：普通图片识别通过，工具图片路径未通过；查看 02_tool_image_response.txt。")
    else:
        print("FAIL：请结合三个阶段的响应区分工具解析、视觉识别及请求长度问题。")
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
