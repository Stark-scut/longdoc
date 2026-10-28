"""Atomic results, content-addressed image assets and replayable events."""
from __future__ import annotations
import base64
from collections import Counter
import hashlib
import json
import mimetypes
import os
from pathlib import Path
import time
import uuid

def read(path):
    return json.loads(Path(path).read_text())

def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("w") as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n"); f.flush(); os.fsync(f.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

def stamp():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")

def image_part(path):
    path = Path(path)
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return {"type": "image_url", "image_url": {"url": "data:" + mime + ";base64," +
                                             base64.b64encode(path.read_bytes()).decode()}}

class Assets:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def pack(self, value):
        if isinstance(value, list):
            return [self.pack(v) for v in value]
        if isinstance(value, dict):
            return {k: self.pack(v) for k, v in value.items()}
        if isinstance(value, str) and value.startswith("data:image/") and ";base64," in value:
            head, body = value.split(",", 1)
            raw = base64.b64decode(body, validate=True)
            mime = head[5:].split(";")[0]
            name = hashlib.sha256(raw).hexdigest() + (mimetypes.guess_extension(mime) or ".bin")
            target = self.directory / name
            if not target.exists():
                temporary = target.with_name(name + "." + uuid.uuid4().hex + ".tmp")
                try:
                    temporary.write_bytes(raw); temporary.replace(target)
                finally:
                    temporary.unlink(missing_ok=True)
            return "asset://" + name
        return value

    def unpack(self, value):
        if isinstance(value, list):
            return [self.unpack(v) for v in value]
        if isinstance(value, dict):
            return {k: self.unpack(v) for k, v in value.items()}
        if isinstance(value, str) and value.startswith("asset://"):
            name = value[8:]
            if Path(name).name != name or not name:
                raise ValueError("Invalid image asset reference")
            path = self.directory / name
            if sha(path) != name.split(".")[0]:
                raise ValueError("Image asset changed: " + name)
            return image_part(path)["image_url"]["url"]
        return value

class Audit:
    def __init__(self, path, assets=None, heartbeat=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.assets = Assets(assets or self.path.parent / "assets")
        self.started = time.monotonic()
        self.sequence = 0
        self.context = {}
        self.heartbeat = heartbeat
        self.requests = Counter()
        self.failures = Counter()
        self.usage = Counter()
        self.seconds = Counter()

    def emit(self, event, **fields):
        self.sequence += 1
        value = {"sequence": self.sequence, "event": event,
                 "elapsed_seconds": time.monotonic() - self.started, **self.context, **fields}
        with self.path.open("a") as f:
            f.write(json.dumps(self.assets.pack(value), ensure_ascii=False, allow_nan=False) + "\n")
            f.flush()
        if self.heartbeat:
            self.heartbeat(value)
        return value

    def costs(self):
        return {"http_requests": sum(self.requests.values()), "http_failures": sum(self.failures.values()),
                "elapsed_seconds": time.monotonic() - self.started, "usage": dict(self.usage),
                "by_role": {role: {"requests": count, "seconds": self.seconds[role],
                                  "failures": self.failures[role]} for role, count in self.requests.items()},
                "note": "Failed requests may consume tokens not reported by the service."}

def add_cost(old, new):
    if not old:
        return new
    result = {}
    for field in ("http_requests", "http_failures", "elapsed_seconds"):
        result[field] = old.get(field, 0) + new.get(field, 0)
    result["usage"] = dict(Counter(old.get("usage", {})) + Counter(new.get("usage", {})))
    result["by_role"] = {}
    for role in set(old.get("by_role", {})) | set(new.get("by_role", {})):
        result["by_role"][role] = {k: old.get("by_role", {}).get(role, {}).get(k, 0) +
                                  new.get("by_role", {}).get(role, {}).get(k, 0)
                                  for k in ("requests", "seconds", "failures")}
    result["note"] = new.get("note")
    return result

def replay(trace, sequence, output):
    trace = Path(trace)
    event = next((e for e in (json.loads(s) for s in trace.read_text().splitlines())
                  if e["sequence"] == sequence), None)
    if not event or event["event"] != "llm_request":
        raise ValueError("sequence must identify an llm_request event")
    local = trace.parent / "assets"
    assets = Assets(local if local.is_dir() else trace.parents[2] / "assets")
    atomic(output, assets.unpack(event["body"]))
