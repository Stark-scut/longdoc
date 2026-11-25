"""Serve per-PDF page rankings from a vLLM ColEmbed index on localhost."""

import argparse
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
from threading import Lock

from agent.documents import doc_key
from agent.logging import read
from agent.retrieval import load_index
from .encode_colembed_pages import ROOT, model_identity


DEFAULT_INDEX = ROOT / "output/agent/indexes/nemotron_colembed_vl_4b_v2_vllm"


class ColEmbedSearch:
    def __init__(self, engine, index_dir, manifest, cache_bytes):
        import torch

        self.engine = engine
        self.index_dir = index_dir
        self.manifest = manifest
        self.index_id = manifest["index_id"]
        self.cache_bytes = cache_bytes
        self.cache_size = 0
        self.cache = OrderedDict()
        self.lock = Lock()
        self.device = torch.device("cuda:0")
        self.tokenizer = engine.get_tokenizer()
        model_dir = Path(manifest["model"]["path"])
        processor = read(model_dir / "processor_config.json")
        self.query_prefix = processor["query_prefix"]
        self.query_max_length = processor["q_max_length"]
        self.chat_template = (model_dir / "chat_template.jinja").read_text(encoding="utf-8")

    def encode_query(self, query):
        prefixed = f"{self.query_prefix} {query}" if self.query_prefix else query
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": [{"type": "text", "text": f"Query: {prefixed}"}]}],
            chat_template=self.chat_template, tokenize=False, add_generation_prompt=True,
        )
        tokens = self.tokenizer(prompt, add_special_tokens=False, truncation=True,
                                max_length=self.query_max_length)["input_ids"]
        result = self.engine.encode([{"prompt_token_ids": tokens}], pooling_task="token_embed", use_tqdm=False)
        return result[0].outputs.data

    def page_vectors(self, doc, entry):
        from safetensors import safe_open

        if doc in self.cache:
            self.cache.move_to_end(doc)
            return self.cache[doc][0]
        path = self.index_dir / entry["embedding_file"]
        with safe_open(path, framework="pt", device="cpu") as source:
            metadata = source.metadata() or {}
            if metadata.get("index_id") != self.index_id or metadata.get("doc_id") != doc:
                raise RuntimeError(f"Incorrect ColEmbed index shard: {path}")
            indices = source.get_tensor("page_indices").tolist()
            vectors = source.get_tensor("embeddings")
        expected = [page["page_index"] for page in entry["pages"]]
        if (indices != expected or vectors.ndim != 3 or vectors.shape[0] != len(expected) or
                vectors.shape[1] == 0 or vectors.shape[2] != self.manifest["model"]["embedding_dim"]):
            raise RuntimeError(f"Invalid page vectors in {path}")
        size = vectors.numel() * vectors.element_size()
        if size <= self.cache_bytes:
            while self.cache_size + size > self.cache_bytes:
                _, (_, removed) = self.cache.popitem(last=False)
                self.cache_size -= removed
            vectors = vectors.to(self.device)
            self.cache[doc] = (vectors, size)
            self.cache_size += size
        return vectors

    def search(self, request):
        import torch

        if request["index_id"] != self.index_id:
            raise ValueError("ColEmbed index ID differs")
        doc = doc_key(request["doc_id"])
        query, top_k = request["query"], request["top_k"]
        entry = self.manifest["documents"][doc]
        indices = [page["page_index"] for page in entry["pages"]]
        if not isinstance(query, str) or not query.strip() or type(top_k) is not int or not 1 <= top_k <= len(indices):
            raise ValueError("Search requires a query and top_k within the document page count")
        # LLM.encode is synchronous. Share the single engine and index cache across HTTP clients.
        with self.lock, torch.inference_mode():
            vectors = self.page_vectors(doc, entry)
            query_vectors = self.encode_query(query).to(device=self.device, dtype=vectors.dtype)
            scores = []
            for batch in vectors.split(32):
                similarities = torch.einsum("qd,ptd->pqt", query_vectors, batch.to(self.device))
                scores.extend(similarities.amax(dim=-1).sum(dim=-1, dtype=torch.float32).tolist())
        if not all(math.isfinite(score) for score in scores):
            raise RuntimeError(f"Non-finite ColEmbed score for {doc}")
        ranking = sorted(zip(indices, scores), key=lambda pair: (-pair[1], pair[0]))[:top_k]
        return {"index_id": self.index_id, "hits": [{"page": page, "score": score} for page, score in ranking]}


class Handler(BaseHTTPRequestHandler):
    def respond(self, status, value):
        raw = json.dumps(value, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path == "/health":
            self.respond(200, {"index_id": self.server.ranker.index_id})
        else:
            self.respond(404, {"error": "Unknown endpoint"})

    def do_POST(self):
        if self.path != "/search":
            self.respond(404, {"error": "Unknown endpoint"})
            return
        try:
            size = int(self.headers.get("Content-Length", 0))
            if not 0 < size <= 65536:
                raise ValueError("Invalid search body size")
            request = json.loads(self.rfile.read(size))
            if not isinstance(request, dict) or not isinstance(request.get("doc_id"), str):
                raise ValueError("Search requires a document ID")
            response = self.server.ranker.search(request)
        except (KeyError, ValueError, TypeError) as exc:
            self.respond(400, {"error": str(exc)})
        except Exception as exc:
            self.log_error("ColEmbed search failed: %r", exc)
            self.respond(500, {"error": str(exc)})
        else:
            self.respond(200, response)

    def log_request(self, code="-", size="-"):
        if code != "-" and int(code) >= 400:
            super().log_request(code, size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.70)
    parser.add_argument("--gpu-cache-gib", type=float, default=2)
    args = parser.parse_args()
    index_dir = args.index_dir.resolve()
    try:
        manifest = load_index(index_dir)
        model_dir = Path(manifest["model"]["path"])
        if model_identity(model_dir) != manifest["model"] or version("vllm") != manifest["packages"]["vllm"]:
            raise ValueError("Use the same ColEmbed model and vLLM version that built the index")
        if not 0 < args.gpu_memory_utilization < 1 or not 0 <= args.gpu_cache_gib < float("inf"):
            raise ValueError("Invalid GPU memory settings")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        from vllm import LLM

        with ThreadingHTTPServer(("127.0.0.1", args.port), Handler) as server:
            engine = LLM(model=str(model_dir), runner="pooling", tensor_parallel_size=1,
                         trust_remote_code=True, dtype="bfloat16",
                         max_model_len=manifest["model"]["max_model_len"], max_num_seqs=1,
                         limit_mm_per_prompt={"image": 0, "video": 0}, mm_processor_cache_gb=0,
                         gpu_memory_utilization=args.gpu_memory_utilization, enforce_eager=True)
            server.ranker = ColEmbedSearch(engine, index_dir, manifest, int(args.gpu_cache_gib * 2**30))
            print(f"Ready on http://127.0.0.1:{args.port}; index {manifest['index_id']}", flush=True)
            server.serve_forever()
    except KeyboardInterrupt:
        return
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
