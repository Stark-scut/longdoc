"""Filesystem, content-addressed replay assets, and append-only audit events."""
from __future__ import annotations
import base64
import hashlib
import json
import mimetypes
import os
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
ARCHIVE = HERE.parent / "aaai_commit_20260803/aaai_commit"
DEFAULT_MODEL = HERE.parent.parent / "models/Qwen3.5-9B"

def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

class Assets:
    """Store image bytes once; asset:// URLs in traces are never sent as-is."""
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def pack(self, obj):
        if isinstance(obj, list):
            return [self.pack(item) for item in obj]
        if isinstance(obj, dict):
            return {k: self.pack(v) for k, v in obj.items()}
        if isinstance(obj, str) and obj.startswith("data:image/") and ";base64," in obj:
            head, encoded = obj.split(",", 1)
            raw = base64.b64decode(encoded, validate=True)
            extension = mimetypes.guess_extension(head[5:].split(";")[0]) or ".bin"
            name = hashlib.sha256(raw).hexdigest() + extension
            destination = self.directory / name
            if not destination.exists():
                destination.write_bytes(raw)
            return "asset://" + name
        return obj

    def unpack(self, obj):
        if isinstance(obj, list):
            return [self.unpack(item) for item in obj]
        if isinstance(obj, dict):
            return {k: self.unpack(v) for k, v in obj.items()}
        if isinstance(obj, str) and obj.startswith("asset://"):
            name = obj[len("asset://"):]
            if Path(name).name != name:
                raise ValueError("Invalid asset reference")
            path = self.directory / name
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != name.split(".")[0]:
                raise ValueError("Replay asset checksum mismatch: " + name)
            mime = mimetypes.guess_type(name)[0] or "image/png"
            return "data:" + mime + ";base64," + base64.b64encode(raw).decode()
        return obj

class Audit:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.assets = Assets(self.path.parent / "assets")
        self.started = time.monotonic()
        self.sequence = 0
        self.requests = 0
        self.failures = 0
        self.usage = {}
        self.roles = {}
        self.context = {}
        self.max_observed_prompt_tokens = 0

    def emit(self, event, **fields):
        self.sequence += 1
        item = {"sequence": self.sequence, "event": event,
                "elapsed_seconds": time.monotonic() - self.started,
                **self.context, **fields}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(self.assets.pack(item), ensure_ascii=False, allow_nan=False) + "\n")
            f.flush()
        return item

    def response(self, role, response, elapsed):
        self.requests += 1
        info = self.roles.setdefault(role, {"requests": 0, "seconds": 0.0})
        info["requests"] += 1
        info["seconds"] += elapsed
        for key, value in (response.get("usage") or {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                self.usage[key] = self.usage.get(key, 0) + value
        self.max_observed_prompt_tokens = max(self.max_observed_prompt_tokens,
            (response.get("usage") or {}).get("prompt_tokens", 0))

    def costs(self):
        return {"http_requests": self.requests, "http_failures": self.failures,
                "usage": dict(self.usage), "by_role": self.roles,
                "elapsed_seconds": time.monotonic() - self.started,
                "usage_note": "Failed requests may consume tokens not reported by the server.",
                "max_observed_prompt_tokens": self.max_observed_prompt_tokens}

def image_part(path):
    path = Path(path)
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return {"type": "image_url", "image_url": {
        "url": "data:" + mime + ";base64," + base64.b64encode(path.read_bytes()).decode()}}

def validate_history(messages, allow_pending=False):
    """Validate native tool-call pairing; actual user follow-ups are allowed."""
    pending, seen = set(), set()
    for message in messages:
        role = message["role"]
        if role == "tool":
            cid = message.get("tool_call_id")
            if cid not in pending:
                raise ValueError("Unpaired or duplicate tool response: " + str(cid))
            pending.remove(cid)
        else:
            if pending:
                raise ValueError("Assistant/user before all tool responses")
            if role == "assistant":
                for call in message.get("tool_calls") or []:
                    cid = call.get("id")
                    if not isinstance(cid, str) or not cid or cid in seen:
                        raise ValueError("Missing or reused tool call ID")
                    seen.add(cid)
                    pending.add(cid)
    if pending and not allow_pending:
        raise ValueError("Missing tool responses: " + repr(sorted(pending)))
    return True

def assistant_history(message):
    result = {"role": "assistant", "content": message.get("content")}
    reasoning = message.get("reasoning") or message.get("reasoning_content")
    if reasoning:
        result["reasoning"] = reasoning
    if message.get("tool_calls"):
        result["tool_calls"] = [
            {"id": call.get("id"), "type": "function",
             "function": {"name": call["function"]["name"],
                          "arguments": call["function"].get("arguments", "{}")}}
            for call in message["tool_calls"]]
    return result
