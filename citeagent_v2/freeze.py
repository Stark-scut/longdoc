"""Create an immutable, small, reproducible baseline without loading any model."""
from __future__ import annotations
import datetime
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

from common import ARCHIVE, HERE, DEFAULT_MODEL, digest, json_hash, read_json, save_json

OLD_RUN = ARCHIVE / "results/mmlongbench/e2e_locator_reasoner_r15_run3_20260726_053313"
DEFAULT_DATA = ARCHIVE / "data/mmlongbench"

def original_config():
    spec = importlib.util.spec_from_file_location("_baseline_original_config", HERE / "frozen_legacy/config.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {key: value for key, value in vars(module).items()
            if key.isupper() and isinstance(value, (str, bool, int, float, type(None)))
            and not any(secret in key for secret in ("API_KEY", "PASSWORD", "SECRET", "ACCESS_TOKEN"))}

def freeze(args):
    destination = args.baseline.resolve()
    if destination.exists():
        raise ValueError("Baseline already exists; use a new --baseline or run verify")
    samples = read_json(args.samples)
    indices = [int(i) for i in args.indices.split(",")]
    if not indices or len(set(indices)) != len(indices) or any(i < 0 or i >= len(samples) for i in indices):
        raise ValueError("indices must be distinct valid sample indices")
    cases = [dict(samples[i], _sample_index=i) for i in indices]
    docs = sorted({c["doc_id"].removesuffix(".pdf") for c in cases})
    page_lists = {doc: sorted((args.images_dir / doc).glob("page_*.png"),
                    key=lambda p: int(p.stem.split("_")[-1])) for doc in docs}
    if any(not paths or len(paths) > args.max_pages for paths in page_lists.values()):
        raise ValueError("Missing page images or selected document exceeds --max-pages")
    if min(args.max_rounds, args.read_page_max, args.request_attempts, args.json_attempts,
           args.locator_max_tokens, args.reasoner_max_tokens, args.selector_max_tokens) < 1:
        raise ValueError("Budgets must be positive")
    if min(args.timeout, args.wall_seconds, args.max_model_len) <= 0:
        raise ValueError("Timeout and context budgets must be positive")
    destination.mkdir(parents=True)
    try:
        from PIL import Image
        image_manifest = []
        for doc in docs:
            for source in page_lists[doc]:
                relative = Path("images") / doc / source.name
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                with Image.open(source) as im:
                    size = list(im.size)
                aligned_pixels = round(size[0] / 32) * 32 * round(size[1] / 32) * 32
                if aligned_pixels > args.max_pixels:
                    raise ValueError("max_pixels would resize " + str(source))
                image_manifest.append({"relative": str(relative), "source": str(source.resolve()),
                    "sha256": digest(target), "size": size, "kind": "full_page"})
            crop_dir = args.crop_root / doc
            if crop_dir.is_dir():
                for source in sorted(crop_dir.glob("*")):
                    if source.is_file():
                        relative = Path("crops") / doc / source.name
                        target = destination / relative
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source, target)
                        with Image.open(source) as im:
                            size = list(im.size)
                        image_manifest.append({"relative": str(relative), "source": str(source.resolve()),
                            "sha256": digest(target), "size": size, "kind": "crop"})
        data = destination / "data"
        data.mkdir()
        counts = {doc: set() for doc in docs}
        with args.blocks.open() as src, (data / "blocks.jsonl").open("w") as out:
            for line in src:
                row = json.loads(line)
                path = Path(row["image_path"])
                if path.parent.name in counts:
                    counts[path.parent.name].add(int(path.stem.split("_")[-1]))
                    # Keep OCR contents unchanged, normalize only the image root.
                    row["image_path"] = str(destination / "images" / path.parent.name / path.name)
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
        for doc, paths in page_lists.items():
            expected_pages = {int(p.stem.split("_")[-1]) for p in paths}
            if counts[doc] != expected_pages:
                raise ValueError("OCR/image page mismatch for " + doc)
        captions = args.captions
        with (data / "captions.jsonl").open("w") as out:
            if captions.is_file():
                for line in captions.read_text().splitlines():
                    if json.loads(line).get("doc_id", "").removesuffix(".pdf") in docs:
                        out.write(line + "\n")
        save_json(destination / "cases.json", cases)
        save_json(destination / "images_manifest.json", image_manifest)
        original = original_config()
        save_json(destination / "original_effective_config.json", original)
        roles = {}
        for role, thinking, tokens in [
            ("locator", args.thinking, args.locator_max_tokens),
            ("reasoner", args.reasoner_thinking, args.reasoner_max_tokens),
            ("selector", False, args.selector_max_tokens),
            ("extractor", False, args.extract_max_tokens)]:
            roles[role] = {"url": getattr(args, role + "_url") or args.url,
                "model": getattr(args, role + "_model") or args.model,
                "max_tokens": tokens, "timeout": args.timeout,
                "thinking": thinking, "key_env": "CITEAGENT_" + role.upper() + "_API_KEY"}
        effective = dict(original)
        effective.update(MAX_ROUNDS=args.max_rounds, READ_PAGE_MAX=args.read_page_max,
            REQUEST_SEED=args.seed, REASONER_MAX_TOKENS=args.reasoner_max_tokens,
            REASONER_TIMEOUT=args.timeout, REASONER_MAX_RETRY=args.json_attempts,
            REASONER_ENABLE_THINKING=args.reasoner_thinking,
            GLOBAL_SELECTOR_MAX_TOKENS=args.selector_max_tokens,
            GLOBAL_SELECTOR_TIMEOUT=args.timeout, GLOBAL_SELECTOR_MAX_RETRY=args.json_attempts)
        effective.update(BASE_URL=args.url, MODEL=args.model,
            LOCATOR_BASE_URL=roles["locator"]["url"], LOCATOR_MODEL=roles["locator"]["model"],
            REASONER_BASE_URL=roles["reasoner"]["url"], REASONER_MODEL=roles["reasoner"]["model"],
            GLOBAL_SELECTOR_BASE_URL=roles["selector"]["url"], GLOBAL_SELECTOR_MODEL=roles["selector"]["model"])
        settings = {"roles": roles, "seed": args.seed, "max_rounds": args.max_rounds,
            "request_attempts": args.request_attempts, "json_attempts": args.json_attempts,
            "max_wall_seconds": args.wall_seconds, "max_model_len": args.max_model_len,
            "discipline": args.discipline, "controller_fallback": args.controller_fallback,
            "mm_processor_kwargs": {"min_pixels": 1024, "max_pixels": args.max_pixels},
            "legacy_effective": effective,
            "controlled_changes": [
                "Both modes use only the copied frozen images_1024 pages, OCR and crops.",
                "Explicit shared thinking/token/timeout settings and instrumented HTTP transport.",
                "Native mode retains all tool history; no compaction is implemented.",
                "Native finalizes on a successful current-evidence Reasoner answer; fallback is opt-in.",
                "Legacy mode runs the unmodified copied Locator and pipeline, including its original fallback.",
                "This is not guaranteed to reproduce the historical trained checkpoints/hardware."]}
        save_json(destination / "settings.json", settings)
        model_info = {"declared_path": str(args.model_path.resolve()), "metadata": {},
                      "weight_inventory": [], "weights_hash_status": "not_hashed"}
        model_dir = destination / "model_metadata"
        model_dir.mkdir()
        for name in ["config.json", "generation_config.json", "tokenizer_config.json",
                     "chat_template.jinja", "model.safetensors.index.json"]:
            p = args.model_path / name
            if p.is_file():
                shutil.copy2(p, model_dir / name)
                model_info["metadata"][name] = digest(p)
        for p in sorted(args.model_path.glob("*.safetensors")):
            model_info["weight_inventory"].append({"name": p.name, "size_bytes": p.stat().st_size})
        if not model_info["metadata"]:
            raise ValueError("No local model metadata found at --model-path")
        save_json(destination / "model_identity.json", model_info)
        historical = destination / "historical"
        historical.mkdir()
        selected_historical = []
        old_paths = args.old_results or [OLD_RUN / "citeagent_mmlb.json"]
        pairs = {(s["doc_id"], s["question"]) for s in cases}
        for number, old_path in enumerate(old_paths):
            if not old_path.is_file():
                raise ValueError("Historical result missing: " + str(old_path))
            old = read_json(old_path)
            rows = list(old.values()) if isinstance(old, dict) else old
            selected_historical.append({"source": str(old_path.resolve()), "sha256": digest(old_path),
                "selected_records": [r for r in rows if (r.get("doc_id"), r.get("question")) in pairs],
                "provenance": "Archived results; historical weights/template/runtime are not fully identified."})
            for candidate in [old_path, old_path.with_suffix(".txt"),
                              old_path.with_name(old_path.stem + "_traces.jsonl"),
                              old_path.parent / "COMPLETE"]:
                if candidate.is_file():
                    shutil.copy2(candidate, historical / f"{number}_{candidate.name}")
        save_json(historical / "selected_records.json", selected_historical)
        evaluator = ARCHIVE / "MMLongBench-Doc/eval"
        shutil.copy2(evaluator / "eval_score.py", destination / "eval_score.py")
        shutil.copy2(evaluator / "prompt_for_answer_extraction.md", destination / "extract_prompt.md")
        versions = {}
        for package in ["vllm", "torch", "transformers", "openai", "httpx", "rank-bm25"]:
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = "not installed in this interpreter"
        env = {k: v for k, v in os.environ.items() if k.startswith("CITEAGENT_") and not any(
            token in k for token in ("KEY", "PASSWORD", "SECRET", "TOKEN"))}
        save_json(destination / "environment.json", {"python": sys.version, "executable": sys.executable,
            "packages": versions, "nonsecret_citeagent_environment": env})
        source_manifest = read_json(HERE / "source_manifest.json")
        save_json(destination / "source_manifest.json", source_manifest)
        manifest = {"created_at": datetime.datetime.now().astimezone().isoformat(),
            "sample_indices": indices, "document_pages": {d: len(v) for d, v in page_lists.items()},
            "source_samples": {"path": str(args.samples.resolve()), "sha256": digest(args.samples)},
            "source_blocks": {"path": str(args.blocks.resolve()), "sha256": digest(args.blocks)},
            "source_captions": {"path": str(captions.resolve()), "sha256": digest(captions) if captions.is_file() else None},
            "files": {str(p.relative_to(destination)): digest(p) for p in sorted(destination.rglob("*")) if p.is_file()}}
        save_json(destination / "manifest.json", manifest)
        print(f"Frozen {len(cases)} cases / {len(docs)} documents at {destination}")
        print("Frozen legacy source is byte-identical to the source repository; historical score is reference only.")
    except BaseException:
        # Failed snapshots are never presented as a valid baseline.
        (destination / "INCOMPLETE").write_text("Freeze did not finish. Use a new directory.\n")
        raise
