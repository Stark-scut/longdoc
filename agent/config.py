"""Portable settings; credentials are environment-variable names only."""
from __future__ import annotations
import copy
import json
from pathlib import Path

ROOT = next((p for p in Path(__file__).resolve().parents if (p / ".gitignore").is_file()), Path.cwd())
ROLES = ("locator", "selector", "reasoner", "extractor")

def defaults():
    data = ROOT / "aaai_commit_20260803/aaai_commit/data/mmlongbench"
    return {
        "schema": 1, "data_dir": str(data), "samples": str(data / "samples.json"),
        "blocks": str(data / "stage2_blocks_cache.jsonl"), "images_dir": str(data / "images_1024"),
        "crop_root": str(data / "visual_crops"), "captions": str(data / "crop_captions.jsonl"),
        "model_path": str(ROOT.parent / "models/Qwen3.5-9B"),
        "max_rounds": 15, "read_page_max": 3, "wall_seconds": 3600,
        "request_attempts": 2, "json_attempts": 2, "seed": 42,
        "max_model_len": 65536, "max_pixels": 1048576, "min_pixels": 1024,
        "roles": {role: {"url": "http://127.0.0.1:8071/v1", "model": "longdoc-qwen35",
                   "thinking": role == "locator", "max_tokens": {"locator": 8192, "selector": 4096,
                   "reasoner": 8192, "extractor": 2048}[role], "timeout": 600,
                   "key_env": "CITEAGENT_" + role.upper() + "_API_KEY"} for role in ROLES},
        "server": {"gpu_memory_utilization": 0.95, "max_num_batched_tokens": 2048,
                   "max_images": 128, "mm_processor_cache_gb": 1.0,
                   "enable_prefix_caching": True, "enforce_eager": True},
    }

def merge(base, extra):
    result = copy.deepcopy(base)
    for key, value in extra.items():
        if key not in base:
            raise ValueError("Unknown configuration key: " + key)
        if isinstance(value, dict) and isinstance(base[key], dict):
            result[key] = merge(base[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result

def load(path=None):
    config = defaults()
    if path:
        extra = json.loads(Path(path).read_text())
        if "data_dir" in extra:
            config = with_data_dir(config, extra["data_dir"])
        config = merge(config, extra)
    return validate(config)

def validate(config):
    for name in ("max_rounds", "read_page_max", "request_attempts", "json_attempts",
                 "max_model_len", "max_pixels", "min_pixels"):
        if type(config[name]) is not int or config[name] < 1:
            raise ValueError(name + " must be a positive integer")
    if config["wall_seconds"] <= 0 or config["min_pixels"] > config["max_pixels"]:
        raise ValueError("Invalid wall/pixel budget")
    for role, spec in config["roles"].items():
        if not spec["url"].startswith(("http://", "https://")) or not spec["model"]:
            raise ValueError("Invalid endpoint/model for " + role)
        if (type(spec["max_tokens"]) is not int or spec["max_tokens"] < 1 or
            spec["timeout"] <= 0 or type(spec["thinking"]) is not bool):
            raise ValueError("Invalid role limits for " + role)
    server = config["server"]
    if not 0 < server["gpu_memory_utilization"] <= 1:
        raise ValueError("gpu_memory_utilization must be in (0,1]")
    if server["max_num_batched_tokens"] < 1 or server["max_images"] < 1 or server["mm_processor_cache_gb"] < 0:
        raise ValueError("Invalid server limits")
    for name in ("data_dir", "samples", "blocks", "images_dir", "crop_root", "captions", "model_path"):
        p = Path(config[name]).expanduser()
        config[name] = str((p if p.is_absolute() else ROOT / p).resolve())
    return config

def with_data_dir(config, directory):
    config = copy.deepcopy(config)
    directory = Path(directory)
    config.update(data_dir=str(directory), samples=str(directory / "samples.json"),
                  blocks=str(directory / "stage2_blocks_cache.jsonl"), images_dir=str(directory / "images_1024"),
                  crop_root=str(directory / "visual_crops"), captions=str(directory / "crop_captions.jsonl"))
    return config
