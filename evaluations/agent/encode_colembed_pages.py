"""Build a per-PDF ColEmbed index from MMLongBench-Doc images using vLLM."""

import argparse
import fcntl
import hashlib
from importlib.metadata import version
from io import BytesIO
import multiprocessing
import os
from pathlib import Path
import re
import signal
from tempfile import TemporaryDirectory
import uuid

from agent.documents import doc_key
from agent.logging import atomic, fingerprint, read, sha


ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent
PAGE_NAME = re.compile(r"page_(\d+)\.png")
POOLING_TASK = "token_embed"


def discover_documents(images_dir):
    documents = {}
    for folder in sorted(path for path in images_dir.iterdir() if path.is_dir()):
        pages = []
        for path in folder.glob("page_*.png"):
            match = PAGE_NAME.fullmatch(path.name)
            if not match:
                raise ValueError(f"Invalid page filename: {path}")
            pages.append({"page_index": int(match[1]), "image": str(path.relative_to(images_dir))})
        pages.sort(key=lambda page: page["page_index"])
        if not pages or [p["page_index"] for p in pages] != list(range(len(pages))):
            raise ValueError(f"Missing or noncontiguous physical pages in {folder}")
        documents[folder.name] = {"embedding_file": f"documents/{folder.name}.safetensors", "pages": pages}

    samples = read(images_dir.parent / "samples.json")
    missing = {doc_key(case["doc_id"]) for case in samples} - documents.keys()
    if not documents or missing:
        raise ValueError(f"Missing MMLongBench-Doc image folders: {', '.join(sorted(missing))}")
    return documents


def model_identity(model_dir):
    files = ("config.json", "processor_config.json", "tokenizer.json", "tokenizer_config.json",
             "chat_template.jinja", "modeling_qwen3_vl_nemotron_embed.py",
             "processing_qwen3_vl_nemotron_embed.py", "model.safetensors.index.json")
    metadata = {name: sha(model_dir / name) for name in files}
    weight_names = sorted(set(read(model_dir / "model.safetensors.index.json")["weight_map"].values()))
    weights = {}
    for name in weight_names:
        stat = (model_dir / name).stat()
        weights[name] = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    hidden_size = read(model_dir / "config.json")["text_config"]["hidden_size"]
    processor = read(model_dir / "processor_config.json")
    return {"path": str(model_dir), "metadata_sha256": metadata, "weights": weights,
            "embedding_dim": hidden_size, "passage_prefix": processor["passage_prefix"],
            "max_model_len": processor["p_max_length"]}


def assign_documents(documents, gpu_ids):
    shards = {gpu: [] for gpu in gpu_ids}
    loads = dict.fromkeys(gpu_ids, 0)
    for doc in sorted(documents, key=lambda doc: (-len(documents[doc]["pages"]), doc)):
        gpu = min(gpu_ids, key=loads.__getitem__)
        shards[gpu].append(doc)
        loads[gpu] += len(documents[doc]["pages"])
    return shards, loads


def is_encoded(path, pages, index_id, dimension):
    from safetensors import SafetensorError, safe_open

    if not path.is_file():
        return False
    try:
        with safe_open(path, framework="pt", device="cpu") as encoded:
            if (encoded.metadata() or {}).get("index_id") != index_id:
                return False
            shape = encoded.get_slice("embeddings").get_shape()
            return (len(shape) == 3 and shape[0] == len(pages) and shape[1] > 0 and shape[2] == dimension
                    and encoded.get_tensor("page_indices").tolist() == [page["page_index"] for page in pages])
    except (OSError, SafetensorError):
        return False


def encode_document(model, prompt, images_dir, output, doc, entry, index_id, batch_size, dimension):
    from PIL import Image
    from safetensors.torch import save_file
    import torch
    from torch.nn.utils.rnn import pad_sequence

    pages = entry["pages"]
    vectors = []
    for start in range(0, len(pages), batch_size):
        prompts = []
        for page in pages[start:start + batch_size]:
            path = images_dir / page["image"]
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != page["sha256"]:
                raise ValueError(f"Source image changed during encoding: {path}")
            with Image.open(BytesIO(raw)) as image:
                prompts.append({"prompt": prompt, "multi_modal_data": {"image": image.convert("RGB")}})
        results = model.encode(prompts, pooling_task=POOLING_TASK, use_tqdm=False,
                               tokenization_kwargs={"add_special_tokens": False})
        if len(results) != len(prompts):
            raise ValueError(f"Unexpected number of vLLM results for {doc}")
        for result in results:
            data = result.outputs.data
            if data.ndim != 2 or data.shape[0] == 0 or data.shape[1] != dimension:
                raise ValueError(f"Unexpected token embedding shape for {doc}: {tuple(data.shape)}")
            if not torch.isfinite(data).all():
                raise ValueError(f"Non-finite token embedding for {doc}")
            vectors.append(data.to(device="cpu", dtype=torch.bfloat16))
    embeddings = pad_sequence(vectors, batch_first=True).contiguous()
    path = output / entry["embedding_file"]
    temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        save_file({"embeddings": embeddings,
                   "page_indices": torch.tensor([page["page_index"] for page in entry["pages"]], dtype=torch.int64)},
                  temporary, metadata={"doc_id": doc, "index_id": index_id})
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def worker(gpu, docs, manifest, index_id, output):
    os.environ.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(gpu),
                      TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1",
                      VLLM_ENABLE_V1_MULTIPROCESSING="0")
    with TemporaryDirectory(prefix="longdoc_colembed_modules_") as modules:
        os.environ["HF_MODULES_CACHE"] = modules
        import torch
        from vllm import LLM

        if not torch.cuda.is_available():
            raise RuntimeError(f"GPU {gpu} is not accessible to the encoder")
        model_config = manifest["model"]
        model_dir = Path(model_config["path"])
        model = LLM(
            model=str(model_dir), runner="pooling", tensor_parallel_size=1,
            trust_remote_code=True, dtype="bfloat16", max_model_len=model_config["max_model_len"],
            max_num_seqs=manifest["batch_size"], limit_mm_per_prompt={"image": 1, "video": 0},
            mm_processor_cache_gb=0,
        )
        prefix = model_config["passage_prefix"]
        prompt = model.get_tokenizer().apply_chat_template(
            [{"role": "user", "content": [{"type": "image"},
                                         {"type": "text", "text": f"{prefix} " if prefix else ""}]}],
            chat_template=(model_dir / "chat_template.jinja").read_text(encoding="utf-8"),
            tokenize=False, add_generation_prompt=True,
        )
        images_dir = Path(manifest["images_dir"])
        for position, doc in enumerate(docs, 1):
            entry = manifest["documents"][doc]
            print(f"[GPU {gpu}] {position}/{len(docs)} {doc} ({len(entry['pages'])} pages)", flush=True)
            encode_document(model, prompt, images_dir, output, doc, entry, index_id,
                            manifest["batch_size"], model_config["embedding_dim"])


def run_workers(shards, manifest, index_id, output):
    context = multiprocessing.get_context("spawn")
    workers = []

    def stop(signum, frame):
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGTERM, stop)
    try:
        for gpu, docs in shards.items():
            if docs:
                process = context.Process(target=worker, args=(gpu, docs, manifest, index_id, output), name=f"colembed-gpu{gpu}")
                process.start()
                workers.append(process)
        for process in workers:
            process.join()
        failed = [f"{process.name} ({process.exitcode})" for process in workers if process.exitcode != 0]
        if failed:
            raise RuntimeError("Encoder workers failed: " + ", ".join(failed))
    finally:
        for process in workers:
            if process.is_alive():
                process.terminate()
            process.join()
        signal.signal(signal.SIGTERM, previous)


def build_index(documents, gpu_ids, images_dir, model, output, batch_size):
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".encode.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for document in documents.values():
            for page in document["pages"]:
                page["sha256"] = sha(images_dir / page["image"])
        manifest = {"format": "colembed-page-index-v1", "images_dir": str(images_dir),
                    "model": model, "batch_size": batch_size, "backend": "vllm", "pooling_task": POOLING_TASK,
                    "packages": {pkg: version(pkg) for pkg in (
                        "torch", "torchvision", "transformers", "vllm", "pillow")},
                    "documents": documents}
        index_id = fingerprint(manifest)
        path = output / "manifest.json"
        if path.exists() and read(path)["index_id"] != index_id:
            raise ValueError("Model, inputs or encoding settings changed; use a new --output-dir")

        pending = {doc: entry for doc, entry in documents.items()
                   if not is_encoded(output / entry["embedding_file"], entry["pages"], index_id, model["embedding_dim"])}
        atomic(path, {**manifest, "index_id": index_id, "complete": not pending})
        print(f"Index {index_id}: {len(documents) - len(pending)} PDFs already complete; "
              f"{len(pending)} pending.", flush=True)
        if pending:
            (output / "documents").mkdir(exist_ok=True)
            shards, _ = assign_documents(pending, gpu_ids)
            run_workers(shards, manifest, index_id, output)
            if any(not is_encoded(output / entry["embedding_file"], entry["pages"], index_id, model["embedding_dim"])
                   for entry in pending.values()):
                raise RuntimeError("One or more PDFs did not produce a valid embedding file")
            atomic(path, {**manifest, "index_id": index_id, "complete": True})
        print(f"Index complete: {path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, default=WORKSPACE / "data/mmlongbench/images_1024")
    parser.add_argument("--model-path", type=Path, default=WORKSPACE.parent / "models/nemotron-colembed-vl-4b-v2")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output/agent/indexes/nemotron_colembed_vl_4b_v2_vllm")
    parser.add_argument("--gpu-ids", default="0,1,2")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        gpu_ids = [int(value) for value in args.gpu_ids.split(",")]
        if len(gpu_ids) != len(set(gpu_ids)) or min(gpu_ids) < 0 or args.batch_size < 1:
            raise ValueError("Use distinct nonnegative GPU IDs and a positive batch size")
        images_dir = args.images_dir.resolve()
        documents = discover_documents(images_dir)
        model = model_identity(args.model_path.resolve())
        shards, loads = assign_documents(documents, gpu_ids)
        print(f"{len(documents)} PDFs, {sum(loads.values())} pages, batch size {args.batch_size}", flush=True)
        for gpu in gpu_ids:
            print(f"GPU {gpu}: {len(shards[gpu])} PDFs, {loads[gpu]} pages", flush=True)
        if not args.dry_run:
            build_index(documents, gpu_ids, images_dir, model, args.output_dir.resolve(), args.batch_size)
    except KeyboardInterrupt:
        parser.exit(130, "Interrupted. Re-run the same command to resume completed PDFs.\n")
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
