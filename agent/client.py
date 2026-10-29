"""Model requests, explicit budgets and recorded JSON correction attempts."""
from __future__ import annotations
import copy
import json
import os
import time
import uuid
import httpx

class Failure(Exception):
    def __init__(self, code, message, retryable=False):
        super().__init__(message)
        self.code, self.message, self.retryable = code, message, retryable
    def as_dict(self):
        return {"code": self.code, "message": self.message, "retryable": self.retryable}

def completed(response):
    choices = response.get("choices") or []
    if not choices or not isinstance(choices[0].get("message"), dict):
        raise Failure("invalid_response", "Missing assistant choice/message")
    choice = choices[0]
    if choice.get("finish_reason") == "length":
        raise Failure("output_truncated", "Model output reached max_tokens")
    if choice.get("finish_reason") not in {"stop", "tool_calls"}:
        raise Failure("unexpected_finish", "finish_reason=" + str(choice.get("finish_reason")))
    return choice

class Client:
    def __init__(self, config, audit, transport=None):
        self.config, self.audit, self.deadline = config, audit, None
        self.http = httpx.Client(transport=transport, trust_env=False)
    def close(self):
        self.http.close()
    def complete(self, role, messages, tools=None, *, tool_choice=None):
        spec = self.config["roles"][role]
        body = {"model": spec["model"], "messages": copy.deepcopy(messages),
                "max_tokens": spec["max_tokens"], "temperature": 0, "seed": self.config["seed"],
                "stream": False, "chat_template_kwargs": {"enable_thinking": spec["thinking"]}}
        if role != "extractor":
            body["mm_processor_kwargs"] = {"min_pixels": self.config["min_pixels"],
                                           "max_pixels": self.config["max_pixels"]}
        if tools is not None:
            choice = tool_choice if tool_choice is not None else (
                self.config["locator_tool_choice"] if role == "locator" else "auto")
            body.update(tools=tools, tool_choice=copy.deepcopy(choice), parallel_tool_calls=True)
        for attempt in range(self.config["request_attempts"]):
            remaining = None if self.deadline is None else self.deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                raise Failure("wall_budget", "Per-question time budget exhausted")
            timeout = spec["timeout"] if remaining is None else min(spec["timeout"], remaining)
            request_id = uuid.uuid4().hex
            self.audit.requests[role] += 1
            self.audit.emit("llm_request", role=role, request_id=request_id, attempt=attempt + 1,
                            base_url=spec["url"], timeout=timeout, body=body)
            start = time.monotonic()
            try:
                response = self.http.post(spec["url"].rstrip("/") + "/chat/completions", json=body,
                    headers={"Authorization": "Bearer " + os.getenv(spec["key_env"], "EMPTY")},
                    timeout=httpx.Timeout(timeout, connect=min(10, timeout)))
                if response.status_code >= 400:
                    text = response.text[:1500]
                    context = any(s in text.lower() for s in ("maximum context", "context length",
                                                               "max_model_len", "maximum model length"))
                    raise Failure("context_limit" if context else "http_" + str(response.status_code),
                                  text, response.status_code == 429 or response.status_code >= 500)
                try:
                    result = response.json()
                except ValueError as exc:
                    raise Failure("invalid_response_json", str(exc), True) from exc
                if not isinstance(result, dict):
                    raise Failure("invalid_response", "Expected a response object")
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                failure = Failure("timeout" if isinstance(exc, httpx.TimeoutException) else "network_error",
                                  str(exc), True)
            except Failure as exc:
                failure = exc
            else:
                duration = time.monotonic() - start
                self.audit.seconds[role] += duration
                for key, value in (result.get("usage") or {}).items():
                    if type(value) in (int, float):
                        self.audit.usage[key] += value
                self.audit.emit("llm_response", role=role, request_id=request_id,
                                seconds=duration, response=result)
                completed(result)
                return result
            duration = time.monotonic() - start
            self.audit.seconds[role] += duration
            self.audit.failures[role] += 1
            self.audit.emit("llm_error", role=role, request_id=request_id, seconds=duration,
                            error=failure.as_dict())
            if not failure.retryable or attempt + 1 == self.config["request_attempts"]:
                raise failure

    def json(self, role, messages):
        history = copy.deepcopy(messages)
        for attempt in range(self.config["json_attempts"]):
            response = self.complete(role, history)
            text = completed(response)["message"].get("content") or ""
            candidate = text.strip()
            fence = chr(96) * 3
            if candidate.startswith(fence):
                candidate = candidate.split("\n", 1)[-1].rsplit(fence, 1)[0].strip()
            try:
                value = json.loads(candidate)
                if not isinstance(value, dict):
                    raise ValueError("Expected a JSON object")
                return value
            except (ValueError, TypeError) as exc:
                self.audit.emit("json_attempt_error", role=role, attempt=attempt + 1, error=str(exc))
                if attempt + 1 == self.config["json_attempts"]:
                    raise Failure("invalid_json", role + ": " + str(exc)) from exc
                history.extend([{"role": "assistant", "content": text},
                    {"role": "user", "content": "Return a complete valid JSON object only. Correct formatting without inventing evidence."}])
