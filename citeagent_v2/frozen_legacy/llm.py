#!/usr/bin/env python3
"""OpenAI-compatible LLM client helpers.

统一封装对本地 vLLM(Qwen3.5-9B, OpenAI 兼容)的三类调用:
1. chat():普通对话,返回文本(可选一并返回 reasoning)。
2. chat_with_tools():function calling —— 传 tools,返回原始 message(含 tool_calls)。
3. chat_json():强制模型输出严格 JSON,内置解析 + 失败重试。

均支持可选的图像输入(为后续多模态开关预留,text-first 阶段不传)。
访问 localhost 需绕过代理,这里在客户端层面显式关闭代理。
"""
import os
import json
import base64
import re

import httpx
from openai import OpenAI

try:
    from . import config
except ImportError:
    import config


# 显式关闭代理:localhost 不能走公司代理,否则连不上本地服务。
_http_client = httpx.Client(trust_env=False)
client = OpenAI(base_url=config.BASE_URL, api_key=config.API_KEY, http_client=_http_client)

# 额外端点的 client 缓存(如 Reasoner 独立服务)。key=(base_url, api_key)。
_client_cache = {}


def _get_client(base_url=None, api_key=None):
    """返回指向指定端点的 OpenAI client;不指定则用全局 client(原行为)。"""
    if not base_url and not api_key:
        return client
    key = (base_url or config.BASE_URL, api_key or config.API_KEY)
    if key not in _client_cache:
        _client_cache[key] = OpenAI(base_url=key[0], api_key=key[1],
                                    http_client=_http_client)
    return _client_cache[key]


def _encode_image(image_path):
    """把本地图片编码为 data URL(base64)。"""
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    ext = os.path.splitext(image_path)[1].lstrip(".").lower() or "png"
    return f"data:image/{ext};base64,{b64}"


def _build_user_content(text, images=None):
    """构造 user 消息的 content。

    无图 -> 直接返回字符串;有图 -> 返回 [text 块, image_url 块...] 的多模态数组。
    """
    if not images:
        return text
    content = [{"type": "text", "text": text}]
    for img in images:
        content.append({"type": "image_url", "image_url": {"url": _encode_image(img)}})
    return content


def chat(system, user, images=None, temperature=0.0, max_tokens=4096,
         return_reasoning=False):
    """普通对话,返回 assistant 文本(可选连同 reasoning 一起返回)。

    思考型模型坑:若输出 token 额度被思考(reasoning)占满、还没产出 content 就被
    截断(finish_reason == 'length' 且 content 为空),则自动加倍额度重试,最多 2 次。
    始终保证返回 str(不会是 None);彻底失败返回 "" 并打警告到 stderr。

    注意:max_tokens 是「输出」上限,不是上下文窗口。输入+输出总和才受 256k 限制,
    所以这里给合理值(默认 4096)即可,不要设成 256k(会浪费 KV cache 且可能超窗口)。
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": _build_user_content(user, images)})

    cur_max = max_tokens
    msg = None
    choice = None
    for _attempt in range(3):
        req = dict(
            model=config.MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=cur_max,
        )
        if getattr(config, "REQUEST_SEED", -1) >= 0:
            req["seed"] = config.REQUEST_SEED
        resp = client.chat.completions.create(**req)
        choice = resp.choices[0]
        msg = choice.message
        content = msg.content
        if content:  # 拿到非空文本,成功
            if return_reasoning:
                return content, getattr(msg, "reasoning", None)
            return content
        # content 为空:若是被长度截断(思考没写完),加倍额度重试
        if choice.finish_reason == "length":
            cur_max *= 2
            continue
        break  # 非截断原因的空内容,不再重试

    import sys
    fr = choice.finish_reason if choice else "?"
    print(f"[llm.chat 警告] 未获得非空 content(finish_reason={fr}, "
          f"max_tokens={cur_max})", file=sys.stderr)
    content = (msg.content or "") if msg else ""
    if return_reasoning:
        return content, (getattr(msg, "reasoning", None) if msg else None)
    return content


def chat_with_tools(messages, tools, temperature=0.0, max_tokens=4096,
                    base_url=None, api_key=None, model=None):
    """function calling:传入完整 messages + tools,返回原始 message 对象。

    base_url/api_key/model:可选，指定后走独立端点/模型；
      不传则用全局 client + config.MODEL。

    调用方负责判断 message.tool_calls、执行工具、把结果以 role=tool 追加后再次调用。
    """
    cli = _get_client(base_url, api_key)
    resp = cli.chat.completions.create(
        model=model or config.MODEL,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        temperature=temperature,
        max_tokens=max_tokens,
        **({"seed": config.REQUEST_SEED} if getattr(config, "REQUEST_SEED", -1) >= 0 else {}),
    )
    return resp.choices[0].message


def _extract_json(text):
    """从模型输出里提取第一个完整 JSON 对象。

    先尝试整体解析;失败则用括号配平找第一个 {...} 块。
    """
    text = text.strip()
    # 去掉可能的 ```json ... ``` 包裹
    fence = re.search(r"```(?:json)?\s*(.+?)\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 括号配平扫描
    start = text.find("{")
    if start == -1:
        raise ValueError("输出中未找到 JSON 对象")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError("JSON 括号不配平")


def chat_json(system, user, images=None, temperature=0.0, max_tokens=4096,
              max_retries=3, timeout=None, enable_thinking=None,
              base_url=None, api_key=None, model=None):
    """强制严格 JSON 输出;解析失败自动重试,把错误反馈给模型纠正。

    base_url/api_key/model:可选，指定后本次调用走独立端点/模型；
      不传则用全局 client + config.MODEL。

    timeout:单次请求超时秒数(防不收敛难题无限生成拖死评测)。超时时:
      - 关掉 SDK 自身的自动重试(max_retries=0),否则单次超时会被 SDK 放大成
        timeout×3;由本函数的重试循环统一管理次数。
      - 超时异常计入 max_retries;重试耗尽后把最后一次异常抛出,由调用方兜底。
    这样单题最坏耗时 ≈ timeout × max_retries,可控。
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": _build_user_content(user, images)})
    return _chat_json_loop(messages, temperature, max_tokens, max_retries,
                           timeout, enable_thinking, base_url, api_key, model)


def chat_json_parts(system, content_parts, temperature=0.0, max_tokens=4096,
                    max_retries=3, timeout=None, enable_thinking=None,
                    base_url=None, api_key=None, model=None):
    """与 chat_json 相同的严格 JSON 语义,但 user content 由调用方给定的【交叉片段
    列表】直接构成(图文任意顺序:[{type:text,...}|{type:image_url,...}, ...])。

    用于页级证据 Reasoner:每证据页"整页截图 → OCR 元素 → image/chart 的 crop 插在
    对应 block 位置"的交叉排列。content_parts 里的
    image_url 由调用方用 _encode_image() 预先编码好。
    """
    if not isinstance(content_parts, list) or not content_parts:
        raise ValueError("content_parts 必须是非空的 content 片段列表")
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": content_parts})
    return _chat_json_loop(messages, temperature, max_tokens, max_retries,
                           timeout, enable_thinking, base_url, api_key, model)


def _chat_json_loop(messages, temperature, max_tokens, max_retries,
                    timeout, enable_thinking, base_url, api_key, model):
    """chat_json / chat_json_parts 共用的请求+严格JSON解析+重试内核。"""
    cli = _get_client(base_url, api_key)
    cli = cli.with_options(timeout=timeout, max_retries=0) if timeout else cli
    req_model = model or config.MODEL
    last_err = None
    for attempt in range(max_retries):
        try:
            request = dict(
                model=req_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if getattr(config, "REQUEST_SEED", -1) >= 0:
                request["seed"] = config.REQUEST_SEED
            if enable_thinking is not None:
                request["extra_body"] = {
                    "chat_template_kwargs": {"enable_thinking": enable_thinking}
                }
            resp = cli.chat.completions.create(**request)
        except Exception as e:
            # 请求层异常(最常见是 timeout):计入重试,不再向模型回喂纠错
            last_err = e
            continue
        content = resp.choices[0].message.content or ""
        # 诊断钩子:CITEAGENT_LLM_DUMP=path 时,把本次调用的 reasoning_content(思考)
        # +content+finish_reason 追加到该文件(用于超时/循环诊断)。默认关,不影响正常评测。
        _dump = os.getenv("CITEAGENT_LLM_DUMP", "")
        if _dump:
            try:
                msg = resp.choices[0].message
                rec = {"model": req_model,
                       "reasoning": getattr(msg, "reasoning_content", None)
                       or getattr(msg, "reasoning", None) or "",
                       "content": content,
                       "finish_reason": resp.choices[0].finish_reason}
                with open(_dump, "a", encoding="utf-8") as _f:
                    _f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except Exception:  # noqa: BLE001
                pass
        try:
            return _extract_json(content)
        except (ValueError, json.JSONDecodeError) as e:
            last_err = e
            # 把上一次的错误输出 + 纠错提示塞回去,要求重出
            messages.append({"role": "assistant", "content": content})
            messages.append({
                "role": "user",
                "content": f"你的输出无法被解析为 JSON(错误: {e})。请只输出一个合法 JSON 对象,不要任何多余文字或解释。",
            })
    raise ValueError(f"经 {max_retries} 次重试仍未得到合法 JSON/成功响应,最后错误: {last_err}")


# ---------- 自测 ----------
if __name__ == "__main__":
    print("=== 1. 普通 chat 冒烟(含截断自动重试,保证非 None) ===")
    ans = chat("你是一个简洁的助手。", "用一句话说明什么是长文档问答。")
    print("回答:", repr(ans), "| 类型:", type(ans).__name__)

    print("\n=== 2. chat_json 冒烟(强制 JSON) ===")
    obj = chat_json(
        "你是一个信息抽取器,只输出 JSON。",
        '把这句话抽成 JSON:「张三今年 30 岁」。字段:name, age。',
    )
    print("解析出的对象:", obj, "类型:", type(obj).__name__)

    print("\n=== 3. function calling 回环冒烟 ===")
    TOOLS = [{
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "计算一个数学表达式",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    }]
    msgs = [{"role": "user", "content": "帮我算 88 乘以 12"}]
    m = chat_with_tools(msgs, TOOLS)
    if m.tool_calls:
        tc = m.tool_calls[0]
        print(f"模型请求调用: {tc.function.name}({tc.function.arguments})")
        args = json.loads(tc.function.arguments)
        result = str(eval(args["expression"], {"__builtins__": {}}, {}))
        print("本地执行结果:", result)
        # 回喂
        msgs.append({"role": "assistant", "content": m.content or "",
                     "tool_calls": [t.model_dump() for t in m.tool_calls]})
        msgs.append({"role": "tool", "tool_call_id": tc.id, "content": result})
        final = chat_with_tools(msgs, TOOLS)
        print("最终回答:", final.content)
    else:
        print("模型未请求工具,直接回答:", m.content)
