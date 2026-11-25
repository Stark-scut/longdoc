#!/usr/bin/env python3
"""Run Direct QA on an Agent run's answerable questions and compare category ACC."""

import argparse
import ast
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import run_replicas
from run_replicas import evaluation


ROOT = run_replicas.WORKDIR
IDENTITY = ("doc_id", "question", "answer", "answer_format", "evidence_sources")
SOURCES = {
    "TXT": "Pure-text (Plain-text)",
    "LAY": "Generalized-text (Layout)",
    "CHA": "Chart",
    "TAB": "Table",
    "FIG": "Figure",
}


def answerable_cases(agent_dir):
    path = agent_dir / "snapshot/inputs/cases.json"
    cases = []
    for case in json.loads(path.read_text()):
        if case["answer"] != "Not answerable":
            case["benchmark_sample_index"] = case.pop("_sample_index")
            cases.append(case)
    if not cases or len({c["benchmark_sample_index"] for c in cases}) != len(cases):
        raise ValueError(f"Empty selection or duplicate question IDs in {path}")
    return cases


def gold_evidence_cases(cases, agent_dir):
    inventory = json.loads((agent_dir / "snapshot/inputs/assets.json").read_text())
    physical_pages = {doc: {asset["page"] for asset in items if asset["kind"] == "page"}
                      for doc, items in inventory.items()}
    selected = []
    excluded = {"empty": [], "out_of_bounds": []}
    for case in cases:
        value = case["evidence_pages"]
        gold = {page - 1 for page in (ast.literal_eval(value) if isinstance(value, str) else value)}
        if gold and gold <= physical_pages[Path(case["doc_id"]).name.removesuffix(".pdf")]:
            selected.append(dict(case, input_page_indices=sorted(gold)))
        else:
            excluded["out_of_bounds" if gold else "empty"].append(case["benchmark_sample_index"])
    if not selected:
        raise ValueError("No answerable questions have valid gold evidence pages")
    print(f"GT-only input: {len(selected)}/{len(cases)} answerable questions; excluded original IDs "
          f"{json.dumps(excluded)}", flush=True)
    return selected


def compare(cases, agent_dir, output_dir):
    agent_rows = json.loads((agent_dir / "merged.json").read_text())
    agents = {row["_sample_index"]: row for row in agent_rows}
    direct = json.loads((output_dir / "merged.json").read_text())
    if len(agents) != len(agent_rows) or not 0 < len(direct) <= len(cases):
        raise ValueError("Invalid Agent or Direct QA result coverage")

    groups = {name: [] for name in ("ALL", *SOURCES, "SIN", "MUL")}
    for index, (case, row) in enumerate(zip(cases, direct)):
        original_index = case["benchmark_sample_index"]
        agent = agents.get(original_index, {})
        if (row["_sample_index"] != index or
                any(row.get(k) != v for k, v in case.items()) or
                any(agent.get(k) != case[k] for k in IDENTITY)):
            raise ValueError(f"Question or annotation mismatch for benchmark sample {original_index}")

        scores = (row["score"] if row["status"] == "complete" else 0.0, agent["score"])
        sources = case["evidence_sources"]
        pages = case["evidence_pages"]
        sources = ast.literal_eval(sources) if isinstance(sources, str) else sources
        pages = ast.literal_eval(pages) if isinstance(pages, str) else pages
        groups["ALL"].append(scores)
        groups["SIN" if len(pages) == 1 else "MUL"].append(scores)
        for name, source in SOURCES.items():
            if source in sources:
                groups[name].append(scores)

    completed = sum(r["status"] == "complete" for r in direct)
    errors = sum(r["status"] == "error" for r in direct)
    truncated = sum(bool(r.get("input_truncation", {}).get("applied")) for r in direct)
    lines = [
        f"Agent baseline: {agent_dir}",
        f"Answerable coverage: {len(direct)}/{len(cases)} selected; "
        f"Direct QA completed {completed}, errors {errors}, "
        f"pending {len(direct) - completed - errors}, truncated inputs {truncated}.",
        "Failed and pending questions count as zero. Evidence categories use benchmark labels.",
        "",
        "| Category | Questions | e2e ACC | Agent ACC | e2e - Agent (pp) |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, pairs in groups.items():
        size = len(pairs)
        if size:
            e2e = 100 * sum(pair[0] for pair in pairs) / size
            agent = 100 * sum(pair[1] for pair in pairs) / size
            lines.append(f"| {name} | {size} | {e2e:.2f}% | {agent:.2f}% | {e2e-agent:+.2f} |")
        else:
            lines.append(f"| {name} | 0 | — | — | — |")
    report = "\n".join(lines)
    (output_dir / "comparison.txt").write_text(report + "\n", encoding="utf-8")
    print(report, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-dir", type=Path, default=ROOT / "output/agent/a100_4gpu_tp1_status_c4_mm3")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--gold-evidence-only", action="store_true", help="Send only valid annotated evidence page images")
    parser.add_argument("--gpu-ids", default=os.getenv("GPU_IDS", "0,1,2"))
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="Smoke test on the first N answerable questions")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--images-dir", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--compare-only", action="store_true")
    args = parser.parse_args()
    agent_dir = args.agent_dir.resolve()
    name = "qwen35_a100_answerable_gold" if args.gold_evidence_only else "qwen35_a100_answerable"
    output_dir = (args.output_dir or ROOT / "output/mmlongbench" / name).resolve()
    if output_dir == agent_dir or agent_dir in output_dir.parents:
        parser.error("Use an output directory outside the Agent run")

    try:
        cases = answerable_cases(agent_dir)
        if args.gold_evidence_only:
            cases = gold_evidence_cases(cases, agent_dir)
        if args.compare_only:
            compare(cases, agent_dir, output_dir)
            return 0

        config = json.loads((agent_dir / "run_config.json").read_text())["config"]
        # The older A100 launcher used this fallback to avoid FlashInfer sampling JIT.
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        os.environ.setdefault("NCCL_IB_DISABLE", "1")
        os.environ.setdefault("NCCL_NET", "Socket")
        command = [
            "--model-path", str(args.model_path or config["model_path"]),
            "--images-dir", str(args.images_dir or config["images_dir"]),
            "--output-dir", str(output_dir), "--gpu-ids", args.gpu_ids,
            "--concurrency", str(args.concurrency), "--max-num-seqs", str(args.concurrency),
            "--shard-strategy", "document", "--document-window", "4", "--document-warmup",
            "--enable-prefix-caching", "--mm-processor-cache-gb", "4",
            "--no-truncate-long-documents" if args.gold_evidence_only else "--truncate-long-documents",
            "--input-token-limit", "260000",
            "--seed", "0", "--timeout", "900", "--limit", str(args.limit),
        ]
        if args.dry_run:
            with TemporaryDirectory(prefix="longdoc_answerable_") as temp:
                path = Path(temp) / "answerable_samples.json"
                evaluation.save_json(path, cases)
                return run_replicas.main([*command, "--input", str(path), "--dry-run"])

        path = output_dir / "answerable_samples.json"
        if path.exists():
            if json.loads(path.read_text()) != cases:
                raise ValueError(f"Frozen answerable input differs from {path}; use a new --output-dir")
        else:
            evaluation.save_json(path, cases)

        exit_code = run_replicas.main([*command, "--input", str(path)])
        if (output_dir / "merged.json").is_file():
            compare(cases, agent_dir, output_dir)
        return exit_code
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
