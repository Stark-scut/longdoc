"""Validate a ColEmbed index and call the shared page-ranking service."""

import math
from pathlib import Path

import httpx

from .client import Failure
from .logging import fingerprint, read


def load_index(directory):
    path = Path(directory) / "manifest.json"
    if not path.is_file():
        raise ValueError(f"Missing ColEmbed index {path}; build the vLLM page index first")
    manifest = read(path)
    if (manifest.get("format") != "colembed-page-index-v1" or not manifest.get("complete") or
            manifest.get("backend") != "vllm" or manifest.get("pooling_task") != "token_embed"):
        raise ValueError(f"A complete vLLM token_embed index is required: {path}")
    content = {key: val for key, val in manifest.items() if key not in {"complete", "index_id"}}
    if fingerprint(content) != manifest.get("index_id"):
        raise ValueError(f"ColEmbed index manifest fingerprint differs: {path}")
    return manifest


def verify_index_images(config, inventory):
    manifest = load_index(config["index_dir"])
    if manifest["index_id"] != config["index_id"]:
        raise ValueError("ColEmbed index ID changed")
    for doc, items in inventory.items():
        indexed = manifest["documents"].get(doc, {}).get("pages", [])
        pages = {item["page"]: item["sha256"] for item in items if item["kind"] == "page"}
        if pages != {item["page_index"]: item["sha256"] for item in indexed}:
            raise ValueError(f"ColEmbed index images do not match evaluation pages: {doc}")


def check_service(config):
    try:
        with httpx.Client(trust_env=False, timeout=5) as client:
            response = client.get(config["url"] + "/health")
            response.raise_for_status()
            if response.json().get("index_id") != config["index_id"]:
                raise ValueError("ColEmbed service is using a different page index")
    except (httpx.HTTPError, ValueError, AttributeError) as exc:
        raise ValueError(f"ColEmbed service is not ready at {config['url']}: {exc}") from exc


def rank_pages(config, doc_id, query, top_k, valid_pages):
    try:
        with httpx.Client(trust_env=False, timeout=120) as client:
            response = client.post(config["url"] + "/search", json={
                "doc_id": doc_id, "query": query, "top_k": top_k, "index_id": config["index_id"],
            })
            response.raise_for_status()
            body = response.json()
            if body["index_id"] != config["index_id"]:
                raise ValueError("ColEmbed response used a different page index")
            ranked = [(hit["page"], hit["score"]) for hit in body["hits"]]
        if (len(ranked) != top_k or len({page for page, _ in ranked}) != top_k or
                any(type(page) is not int or page not in valid_pages or
                    type(score) not in (int, float) or not math.isfinite(score) for page, score in ranked)):
            raise ValueError("ColEmbed returned invalid or incomplete page rankings")
        return ranked
    except (httpx.HTTPError, ValueError, KeyError, TypeError, OverflowError) as exc:
        raise Failure("retrieval_error", str(exc)) from exc
