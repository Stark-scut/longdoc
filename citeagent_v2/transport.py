"""Explicit-budget HTTP transport; instrumentation shared between modes."""
from __future__ import annotations
import copy
import json
import time
from types import SimpleNamespace
import uuid
import httpx

class Failure(Exception):
    def __init__(self, code, message, retryable=False):
        self.code, self.retryable = code, retryable
        super().__init__(message)
    def __reduce__(self):
        return (type(self), (self.code, str(self), self.retryable))
    def as_dict(self):
        return {"code": self.code, "message": str(self), "retryable": self.retryable}

def namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{k: namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [namespace(v) for v in value]
    return value

def require_completed(response):
    try:
        choice = response["choices"][0]
        message = choice["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise Failure("invalid_response", "No valid choice/message") from exc
    finish = choice.get("finish_reason")
    if finish == "length":
        raise Failure("output_truncated", "Output reached max_tokens", True)
    if finish not in {"stop", "tool_calls"}:
        raise Failure("generation_" + str(finish), "Unexpected finish_reason=" + str(finish), True)
    if finish == "tool_calls" and not message.get("tool_calls"):
        raise Failure("invalid_tool_calls", "tool_calls finish without calls")
    return choice

class Transport:
    def __init__(self, settings, audit, client=None):
        self.settings, self.audit = settings, audit
        self.client = client or httpx.Client(trust_env=False)
        self.deadline = None
    def close(self):
        self.client.close()

    def complete(self, role, messages, *, tools=None, temperature=0.0,
                 max_tokens=None, thinking=None, timeout=None, attempts=1):
        spec = self.settings["roles"][role]
        body = {"model": spec["model"], "messages": copy.deepcopy(messages),
                "temperature": temperature, "max_tokens": max_tokens or spec["max_tokens"],
                "stream": False, "seed": self.settings["seed"],
                "chat_template_kwargs": {"enable_thinking":
                    spec["thinking"] if thinking is None else thinking}}
        if tools:
            body.update(tools=tools, tool_choice="auto", parallel_tool_calls=True)
        if role != "extractor":
            body["mm_processor_kwargs"] = self.settings["mm_processor_kwargs"]
        import os
        key = os.getenv(spec["key_env"], "EMPTY")
        last = None
        for attempt in range(attempts):
            remaining = None if self.deadline is None else self.deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                raise Failure("wall_budget", "Per-sample wall budget exhausted")
            request_timeout = min(timeout or spec["timeout"], remaining) if remaining is not None else (timeout or spec["timeout"])
            request_id = uuid.uuid4().hex
            start = time.monotonic()
            self.audit.emit("llm_request", request_id=request_id, role=role, attempt=attempt + 1,
                            base_url=spec["url"], timeout=request_timeout, body=body)
            try:
                response = self.client.post(spec["url"].rstrip("/") + "/chat/completions", json=body,
                    headers={"Authorization": "Bearer " + key},
                    timeout=httpx.Timeout(request_timeout, connect=min(10, request_timeout)))
                if response.status_code >= 400:
                    retry = response.status_code == 429 or response.status_code >= 500
                    code = "context_limit" if any(x in response.text.lower() for x in [
                        "maximum context", "context length", "max_model_len", "maximum model length"]) else "http_" + str(response.status_code)
                    raise Failure(code, response.text[:1500], retry and code != "context_limit")
                try:
                    result = response.json()
                except ValueError as exc:
                    raise Failure("invalid_response_json", str(exc), True) from exc
                if not isinstance(result, dict):
                    raise Failure("invalid_response", "Expected response object")
                self.audit.response(role, result, time.monotonic() - start)
                self.audit.emit("llm_response", request_id=request_id, role=role,
                                response=result, seconds=time.monotonic() - start)
                return result
            except httpx.TimeoutException as exc:
                last = Failure("timeout", str(exc), True)
            except httpx.RequestError as exc:
                last = Failure("connection_error", str(exc), True)
            except Failure as exc:
                last = exc
            self.audit.requests += 1
            self.audit.failures += 1
            role_cost = self.audit.roles.setdefault(role, {"requests": 0, "seconds": 0.0})
            role_cost["requests"] += 1
            role_cost["seconds"] += time.monotonic() - start
            role_cost["failures"] = role_cost.get("failures", 0) + 1
            self.audit.emit("llm_error", request_id=request_id, role=role,
                            error=last.as_dict(), seconds=time.monotonic() - start)
            if not last.retryable:
                break
        raise last or Failure("request_failed", "No response")

    def json_result(self, role, system, parts, *, max_tokens, timeout, attempts, thinking):
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": copy.deepcopy(parts)}]
        last = None
        for attempt in range(max(1, attempts)):
            try:
                response = self.complete(role, messages, max_tokens=max_tokens,
                                         timeout=timeout, thinking=thinking)
                choice = require_completed(response)
                content = choice["message"].get("content") or ""
                text = content.strip()
                fence = chr(96) * 3
                if text.startswith(fence) and text.endswith(fence):
                    text = text.split("\n", 1)[1].rsplit(fence, 1)[0].strip()
                result = json.loads(text)
                if not isinstance(result, dict):
                    raise ValueError("Expected a JSON object")
                return result
            except (Failure, ValueError, TypeError, KeyError) as exc:
                last = exc if isinstance(exc, Failure) else Failure("invalid_json", str(exc), True)
                self.audit.emit("json_attempt_error", role=role, attempt=attempt + 1, error=last.as_dict())
                if not last.retryable:
                    break
        raise last

class LegacyClient:
    """SDK-shaped shim; frozen legacy prompt construction stays unchanged."""
    def __init__(self, transport, role="locator", timeout=None):
        self.transport, self.role, self.timeout = transport, role, timeout
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
    def with_options(self, timeout=None, max_retries=0):
        return LegacyClient(self.transport, self.role, timeout)
    def create(self, **kwargs):
        role = "locator" if kwargs.get("tools") else self.role
        if role == "locator" and not kwargs.get("tools"):
            role = self.transport.settings.get("_active_subrole", "reasoner")
        spec = self.transport.settings["roles"][role]
        result = self.transport.complete(role, kwargs["messages"], tools=kwargs.get("tools"),
            temperature=kwargs.get("temperature", 0),
            max_tokens=spec["max_tokens"] if role == "locator" else kwargs.get("max_tokens"),
            thinking=spec["thinking"], timeout=self.timeout,
            attempts=self.transport.settings["request_attempts"] if role == "locator" else 1)
        for choice in result.get("choices", []):
            message = choice.get("message", {})
            message.setdefault("tool_calls", None)
            message.setdefault("content", None)
        return namespace(result)

def install(module, legacy_reasoner, locator_module, transport, native):
    """Instrument only private copied modules, never source modules."""
    module.client = LegacyClient(transport)
    module._get_client = lambda base_url=None, api_key=None: LegacyClient(transport)
    if not hasattr(module, "_original_json_helpers"):
        module._original_json_helpers = (module.chat_json, module.chat_json_parts)
    original_json, original_parts = module._original_json_helpers
    def scope(system):
        return "reasoner" if system == legacy_reasoner.SYSTEM_PROMPT else "selector"
    def call_json(system, user=None, images=None, temperature=0, max_tokens=4096,
                  max_retries=3, timeout=None, enable_thinking=None,
                  base_url=None, api_key=None, model=None, _parts=None):
        role = scope(system)
        if native:
            parts = _parts if _parts is not None else module._build_user_content(user, images)
            return transport.json_result(role, system, parts, max_tokens=max_tokens, timeout=timeout,
                attempts=max_retries, thinking=transport.settings["roles"][role]["thinking"])
        transport.settings["_active_subrole"] = role
        try:
            if _parts is not None:
                return original_parts(system, _parts, temperature=temperature, max_tokens=max_tokens,
                    max_retries=max_retries, timeout=timeout, enable_thinking=enable_thinking,
                    base_url=base_url, api_key=api_key, model=model)
            return original_json(system, user, images=images, temperature=temperature, max_tokens=max_tokens,
                max_retries=max_retries, timeout=timeout, enable_thinking=enable_thinking,
                base_url=base_url, api_key=api_key, model=model)
        finally:
            transport.settings.pop("_active_subrole", None)
    module.chat_json = call_json
    module.chat_json_parts = lambda system, content_parts, **kw: call_json(system, _parts=content_parts, **kw)
