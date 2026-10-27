"""Snapshot preparation, resume rules and aggregate metrics for configurable replicas."""
from __future__ import annotations
import copy
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import uuid

SOURCE_CORE = Path(__file__).resolve().parent.parent
ARCHIVE = SOURCE_CORE.parent / "aaai_commit_20260803/aaai_commit"
DATA = ARCHIVE / "data/mmlongbench"
MODEL = SOURCE_CORE.parent.parent / "models/Qwen3.5-9B"
SCHEMA = 1

def read(path):
    return json.loads(Path(path).read_text())

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("w") as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)

def stamp():
    return datetime.datetime.now().astimezone().isoformat()

def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

def result_path(root, index):
    return Path(root) / "samples" / f"sample_{index:04d}" / "result.json"

def settled(result, stage):
    if not result or result.get("execution_status") not in {"completed", "incomplete", "error"}:
        return False
    if stage == "agent":
        return True
    return result.get("scoring_status") in {"scored", "not_answered", "error"}

def needs_work(result, stage, retry_failed=False):
    if not result:
        return stage != "score"
    if retry_failed and stage != "score" and result.get("execution_status") != "completed":
        return True
    if stage == "agent":
        return False
    if retry_failed and result.get("scoring_status") == "error":
        return True
    return not settled(result, stage)

def settings_for(config):
    p = config["parameters"]
    settings = copy.deepcopy(config["base_settings"])
    settings.update(max_rounds=p["max_rounds"], max_wall_seconds=p["wall_seconds"],
                    seed=p["seed"], request_attempts=p["request_attempts"],
                    json_attempts=p["json_attempts"], max_model_len=p["max_model_len"],
                    controller_fallback=p["controller_fallback"])
    for name, tokens, thinking in [
        ("locator", p["locator_max_tokens"], p["thinking"]),
        ("selector", p["selector_max_tokens"], False),
        ("reasoner", p["reasoner_max_tokens"], p["reasoner_thinking"]),
        ("extractor", p["extract_max_tokens"], False)]:
        settings["roles"][name].update(max_tokens=tokens, timeout=p["timeout"], thinking=thinking,
                                      model="citeagent-qwen35")
    settings["mm_processor_kwargs"] = {"min_pixels": 1024, "max_pixels": p["max_pixels"]}
    settings["legacy_effective"].update(
        MAX_ROUNDS=p["max_rounds"], READ_PAGE_MAX=p["read_page_max"], REQUEST_SEED=p["seed"],
        REASONER_MAX_TOKENS=p["reasoner_max_tokens"], REASONER_TIMEOUT=p["timeout"],
        REASONER_MAX_RETRY=p["json_attempts"], REASONER_ENABLE_THINKING=p["reasoner_thinking"],
        GLOBAL_SELECTOR_MAX_TOKENS=p["selector_max_tokens"],
        GLOBAL_SELECTOR_TIMEOUT=p["timeout"], GLOBAL_SELECTOR_MAX_RETRY=p["json_attempts"])
    return settings

def select_cases(samples, limit=0, indices=None):
    chosen = list(range(len(samples))) if not indices else [int(x) for x in indices.split(",")]
    if limit:
        chosen = chosen[:limit]
    if not chosen or len(set(chosen)) != len(chosen) or any(i < 0 or i >= len(samples) for i in chosen):
        raise ValueError("Empty, duplicate or invalid sample indices")
    return [dict(samples[i], _sample_index=i) for i in chosen]

def inspect_inputs(parameters, cases, hash_assets=False):
    from PIL import Image
    images, crops = Path(parameters["images_dir"]), Path(parameters["crop_root"])
    inventory = {}
    for doc in sorted({c["doc_id"].removesuffix(".pdf") for c in cases}):
        pages = sorted((images / doc).glob("page_*.png"), key=lambda p: int(p.stem.split("_")[-1]))
        if not pages:
            raise ValueError("Missing page images: " + doc)
        entries = []
        for kind, paths in [("image", pages), ("crop", sorted((crops / doc).glob("*")))]:
            for path in paths:
                if not path.is_file():
                    continue
                with Image.open(path) as im:
                    width, height = im.size
                if kind == "image":
                    area = max(32, round(width / 32) * 32) * max(32, round(height / 32) * 32)
                    if area > parameters["max_pixels"]:
                        raise ValueError(f"max_pixels would resize page {path}; need at least {area}")
                entries.append({"path": str(path.resolve()), "kind": kind,
                    "page": int(path.stem.split("_")[-1]) if kind == "image" else None,
                    "width": width, "height": height, "size": path.stat().st_size,
                    **({"sha256": sha(path)} if hash_assets else {})})
        inventory[doc] = entries
    return inventory

def prepare(root, parameters, cases, replica_count=4):
    """Copy small inputs/code; reference and hash large page/crop resources."""
    root = Path(root)
    snapshot = root / "snapshot"
    if snapshot.exists():
        raise ValueError("Incomplete snapshot exists; use a new output directory")
    snapshot.mkdir()
    core = snapshot / "core"
    core.mkdir()
    for p in SOURCE_CORE.glob("*.py"):
        shutil.copy2(p, core / p.name)
    shutil.copy2(SOURCE_CORE / "source_manifest.json", core / "source_manifest.json")
    shutil.copytree(SOURCE_CORE / "frozen_legacy", core / "frozen_legacy",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (core / "vendor").mkdir()
    shutil.copy2(SOURCE_CORE / "vendor/rank_bm25.py", core / "vendor/rank_bm25.py")
    if (SOURCE_CORE / "vendor/rank_bm25-0.2.2.dist-info").exists():
        shutil.copytree(SOURCE_CORE / "vendor/rank_bm25-0.2.2.dist-info",
                        core / "vendor/rank_bm25-0.2.2.dist-info")
    runner = snapshot / "full_eval"
    runner.mkdir()
    for name in ["support.py", "worker.py"]:
        shutil.copy2(Path(__file__).resolve().parent / name, runner / name)
    shared = snapshot / "data"
    shared.mkdir()
    shutil.copy2(parameters["input"], shared / "samples.json")
    shutil.copy2(parameters["blocks"], shared / "blocks.jsonl")
    captions = Path(parameters["captions"])
    if captions.is_file():
        shutil.copy2(captions, shared / "captions.jsonl")
    else:
        (shared / "captions.jsonl").write_text("")
    evaldir = ARCHIVE / "MMLongBench-Doc/eval"
    shutil.copy2(evaldir / "eval_score.py", shared / "eval_score.py")
    shutil.copy2(evaldir / "prompt_for_answer_extraction.md", shared / "extract_prompt.md")
    model_dir = snapshot / "model_metadata"
    model_dir.mkdir()
    for name in ["config.json", "generation_config.json", "tokenizer_config.json",
                 "chat_template.jinja", "model.safetensors.index.json"]:
        source = Path(parameters["model_path"]) / name
        if source.is_file():
            shutil.copy2(source, model_dir / name)
    if not (model_dir / "config.json").exists():
        raise ValueError("Model metadata not found")
    print("冻结页面与 crop 的资源哈希；后续逐文档校验，图片不重复复制。", flush=True)
    inventory = inspect_inputs(parameters, cases, hash_assets=True)
    atomic(snapshot / "assets_manifest.json", inventory)
    atomic(snapshot / "cases.json", cases)
    base = read(SOURCE_CORE / "baseline/settings.json")
    config = {"schema": SCHEMA, "created_at": stamp(), "parameters": parameters,
              "base_settings": base, "total": len(cases),
              "replica_count": replica_count, "flexible_replicas": True,
              "code_note": "Workers execute this output directory's copied native implementation.",
              "assets_note": "Image/crop files are referenced by path and SHA256, verified before each document is used."}
    settings = settings_for(config)
    for shard in range(1):  # Shared immutable baseline; runtime assignments define GPU shards.
        baseline = snapshot / f"shard{shard}"
        baseline.mkdir()
        (baseline / "data").symlink_to(shared, target_is_directory=True)
        (baseline / "images").symlink_to(parameters["images_dir"], target_is_directory=True)
        if Path(parameters["crop_root"]).is_dir():
            (baseline / "crops").symlink_to(parameters["crop_root"], target_is_directory=True)
        else:
            (baseline / "crops").mkdir()
        for name in ["eval_score.py", "extract_prompt.md"]:
            (baseline / name).symlink_to(shared / name)
        atomic(baseline / "settings.json", settings)
        atomic(baseline / "cases.json", cases)
        names = ["settings.json", "cases.json", "data/blocks.jsonl", "data/captions.jsonl",
                 "eval_score.py", "extract_prompt.md"]
        atomic(baseline / "manifest.json", {"files": {name: sha(baseline / name) for name in names}})
    config["model_identity"] = {
        "metadata": {p.name: sha(p) for p in model_dir.iterdir() if p.is_file()},
        "weights": {p.name: [p.stat().st_size, p.stat().st_mtime_ns]
                    for p in Path(parameters["model_path"]).glob("*.safetensors")},
        "note": "Metadata is SHA256 checked; large weight shards use name/size/mtime inventory."}
    config["snapshot_files"] = {str(p.relative_to(snapshot)): sha(p)
        for p in sorted(snapshot.rglob("*")) if p.is_file() and not p.is_symlink()
        and "__pycache__" not in p.parts}
    config["fingerprint"] = signature({"parameters": parameters, "snapshot_files": config["snapshot_files"],
                                       "model_identity": config["model_identity"]})
    atomic(root / "run_config.json", config)
    return config

def verify_snapshot(root, config):
    for name, expected in config["snapshot_files"].items():
        if sha(Path(root) / "snapshot" / name) != expected:
            raise ValueError("Snapshot checksum changed: " + name)

def verify_model(config):
    model = Path(config["parameters"]["model_path"])
    identity = config["model_identity"]
    for name, expected in identity["metadata"].items():
        if sha(model / name) != expected:
            raise ValueError("Local model metadata changed: " + name)
    weights = {p.name: [p.stat().st_size, p.stat().st_mtime_ns] for p in model.glob("*.safetensors")}
    if weights != identity["weights"]:
        raise ValueError("Local model weight inventory changed; choose another output directory")

def official_scorer(root):
    path = Path(root) / "snapshot/data/eval_score.py"
    spec = importlib.util.spec_from_file_location("_native_full_official_score", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

class Results:
    def __init__(self, root, cases):
        self.root, self.cases, self.cache = Path(root), cases, {}
    def get(self, index):
        path = result_path(self.root, index)
        try:
            stat = path.stat()
        except FileNotFoundError:
            return None
        key = (stat.st_mtime_ns, stat.st_size)
        if self.cache.get(index, (None,))[0] != key:
            self.cache[index] = (key, read(path))
        return self.cache[index][1]
    def rows(self):
        return [(c, self.get(c["_sample_index"])) for c in self.cases]

def aggregate(root, config, runtime=None, persist=True):
    root = Path(root)
    cases = read(root / "snapshot/cases.json")
    results = Results(root, cases)
    rows, counts = [], {"processed": 0, "agent_completed": 0, "agent_errors": 0,
                        "agent_incomplete": 0, "scored": 0, "scoring_errors": 0,
                        "waiting_extraction": 0}
    if runtime is None and (root / "runtime.json").exists():
        runtime = read(root / "runtime.json")
    stage = (runtime or {}).get("stage", "all")
    todo = set((runtime or {}).get("todo", []))
    run_id = (runtime or {}).get("run_id")
    replica_count = (runtime or {}).get("replica_count", config.get("replica_count", 4))
    shards = [{"shard": i, "processed": 0, "total": sum(c["_sample_index"] % replica_count == i for c in cases)}
              for i in range(replica_count)]
    for case, result in results.rows():
        row = dict(case)
        r = result or {}
        current = not (case["_sample_index"] in todo and run_id and r.get("run_id") != run_id)
        done = current and settled(r, stage)
        counts["processed"] += int(done)
        shards[case["_sample_index"] % replica_count]["processed"] += int(done)
        r_count = r if current else {}
        counts["agent_completed"] += int(r_count.get("execution_status") == "completed")
        counts["agent_errors"] += int(r_count.get("execution_status") == "error")
        counts["agent_incomplete"] += int(r_count.get("execution_status") == "incomplete")
        counts["scored"] += int(r_count.get("scoring_status") == "scored")
        counts["scoring_errors"] += int(r_count.get("scoring_status") == "error")
        counts["waiting_extraction"] += int(bool(r_count.get("raw_answer")) and not settled(r_count, "all"))
        row.update(r)
        row["answer"] = case["answer"]  # offline GT stays in the evaluator only
        row["score"] = r.get("score", 0.0) if r.get("scoring_status") == "scored" else 0.0
        row["pred"] = r.get("pred", "Fail to answer") if r.get("scoring_status") == "scored" else "Fail to answer"
        rows.append(row)
    acc, f1 = official_scorer(root).eval_acc_and_f1(rows)
    total = len(cases)
    summary = {"updated_at": stamp(), "total": total, **counts, "pending": total - counts["processed"],
        "run_finished": counts["processed"] == total,
        "evaluation_complete": counts["scored"] == total,
        "accuracy": acc, "f1": f1, "accuracy_percent": acc * 100, "f1_percent": f1 * 100,
        "replica_count": replica_count, "shards": shards,
        "note": "All selected samples are in the denominator; missing/error/incomplete samples score zero. run_finished includes failures; evaluation_complete requires all samples scored."}
    if persist:
        atomic(root / "merged.json", rows)
        atomic(root / "summary.json", summary)
    return summary

def duration(seconds):
    if seconds is None:
        return "--:--:--"
    seconds = max(0, int(seconds))
    return f"{seconds//3600:02d}:{seconds//60%60:02d}:{seconds%60:02d}"
