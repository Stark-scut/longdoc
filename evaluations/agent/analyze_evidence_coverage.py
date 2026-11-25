"""Audit answerable questions in a schema 3 Agent run using its frozen inputs and traces."""

import argparse
import ast
from collections import Counter
import contextlib
from dataclasses import dataclass, field
import io
import json
from pathlib import Path
import re

from agent.documents import doc_key
from agent.metrics.mmlongbench import eval_score


PAGE_IMAGE = re.compile(r"page_index=(\d+) full page image:")


@dataclass
class Coverage:
    index: int
    gold: set[int]
    score: float
    stop: str
    read: set[int] = field(default_factory=set)
    stored: set[int] = field(default_factory=set)
    surfaced: set[int] = field(default_factory=set)
    full_requests: int = 0
    full_request_outcomes: list[str] = field(default_factory=list)
    raw_full_score: bool = False


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def request_pages(event):
    pages = set()
    for message in event["body"]["messages"]:
        parts = message["content"]
        if isinstance(parts, list):
            for part, following in zip(parts, parts[1:]):
                match = PAGE_IMAGE.fullmatch(part.get("text", ""))
                if match and following["type"] == "image_url":
                    pages.add(int(match[1]))
    return pages


def raw_answer_full_score(case, raw_answer):
    gold = case["answer"]
    answer_type = case["answer_format"]
    try:
        if answer_type == "List":
            gold = ast.literal_eval(gold) if gold.startswith("[") else gold
            raw_answer = ast.literal_eval(raw_answer) if raw_answer.startswith("[") else raw_answer
        with contextlib.redirect_stdout(io.StringIO()):
            return eval_score(gold, raw_answer, answer_type) == 1.0
    except (ValueError, TypeError, IndexError, SyntaxError):
        return False


def audit_case(run_dir, case, row, gold):
    index = case["_sample_index"]
    result = Coverage(index, gold, row["score"], row.get("stop_reason", "pending"))
    if not row.get("agent_trace"):
        if row.get("execution_status") or row.get("read_pages") or row.get("evidence_memory"):
            raise ValueError(f"Sample {index} is missing its Agent trace reference")
        return result

    # agent_trace refers to the attempt that generated the QA result, even after a scoring retry.
    trace = run_dir / "samples" / f"sample_{index:04d}" / Path(row["agent_trace"]).name
    full_request = False
    with trace.open(encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line)
            if event["event"] == "llm_request" and event["role"] == "reasoner":
                pages = request_pages(event)
                if not pages or not pages <= result.stored:
                    raise ValueError(f"Cannot match Reasoner pages to stored evidence in {trace}")
                full_request = gold <= pages
                result.full_requests += full_request
            if event["event"] != "tool_result":
                continue
            tool, outcome = event["name"], event["outcome"]
            if tool == "call_reasoner" and full_request:
                status = outcome["result"]["status"] if outcome["ok"] else outcome["error"]["code"]
                result.full_request_outcomes.append(status)
                full_request = False
            if outcome["ok"] and tool == "read_page":
                result.read.update(outcome["result"]["pages"])
            if outcome["ok"] and tool == "store":
                result.stored.update(outcome["result"]["added_pages"])
            if outcome["ok"] and tool == "search":
                result.surfaced.update(hit["page"] for hit in outcome["result"])

    if (result.read != set(row.get("read_pages", ())) or
            result.stored != set(row.get("evidence_memory", {}).get("pages", ())) or
            not result.stored <= result.read):
        raise ValueError(f"Read or stored pages disagree with merged result for sample {index}")
    if result.full_requests and result.score < 1 and row.get("raw_answer"):
        result.raw_full_score = raw_answer_full_score(case, row["raw_answer"])
    return result


def summarize(cases):
    requested = [case for case in cases if case.full_requests]
    return {
        "valid": len(cases),
        "read_all": sum(case.gold <= case.read for case in cases),
        "stored_all": sum(case.gold <= case.stored for case in cases),
        "requested_all": len(requested),
        "full_score": sum(case.score == 1 for case in requested),
        "partial_score": sum(0 < case.score < 1 for case in requested),
        "zero_score": sum(case.score == 0 for case in requested),
        "mean_score": sum(case.score for case in requested) / len(requested) if requested else None,
    }


def analyze(run_dir):
    config = load(run_dir / "run_config.json")
    if config["schema"] != 3:
        raise ValueError("This audit uses schema 3 evidence-memory traces")
    cases = load(run_dir / "snapshot/inputs/cases.json")
    rows = load(run_dir / "merged.json")
    results = {row["_sample_index"]: row for row in rows}
    if len(results) != len(rows) or len(results) != len(cases):
        raise ValueError("Merged results do not match the frozen question set")
    inventory = load(run_dir / "snapshot/inputs/assets.json")
    physical = {doc: {item["page"] for item in items if item["kind"] == "page"}
                for doc, items in inventory.items()}

    valid = []
    excluded = {"empty": [], "out_of_bounds": []}
    for case in cases:
        if case["answer"] == "Not answerable":
            continue
        index = case["_sample_index"]
        value = case["evidence_pages"]
        gold = {page - 1 for page in (ast.literal_eval(value) if isinstance(value, str) else value)}
        if not gold:
            excluded["empty"].append(index)
            continue
        if not gold <= physical[doc_key(case["doc_id"])]:
            excluded["out_of_bounds"].append(index)
            continue
        row = results[index]
        if (any(row[key] != case[key] for key in ("doc_id", "question", "answer", "answer_format")) or
                row.get("input_fingerprint", config["fingerprint"]) != config["fingerprint"]):
            raise ValueError(f"Merged result does not match the frozen input for sample {index}")
        valid.append(audit_case(run_dir, case, row, gold))

    not_read_all = [case for case in valid if not case.gold <= case.read]
    requested = [case for case in valid if case.full_requests]
    return {
        "run": run_dir.name,
        "answerable": len(valid) + sum(map(len, excluded.values())),
        "excluded_sample_indices": excluded,
        "overall": summarize(valid),
        "single_page": summarize([case for case in valid if len(case.gold) == 1]),
        "cross_page": summarize([case for case in valid if len(case.gold) > 1]),
        "diagnostics": {
            "read_no_gold_pages": sum(not case.gold & case.read for case in not_read_all),
            "all_missed_pages_surfaced_by_search": sum(case.gold <= case.read | case.surfaced
                                                       for case in not_read_all),
            "stored_all_without_full_request": sum(case.gold <= case.stored and not case.full_requests
                                                    for case in valid),
            "full_request_final_stop_reasons": Counter(case.stop for case in requested),
            "last_full_request_outcomes": Counter(case.full_request_outcomes[-1]
                                                  for case in requested if case.full_request_outcomes),
            "full_score_without_full_request": sum(case.score == 1 and not case.full_requests for case in valid),
            "raw_full_score_but_final_zero_sample_indices": [case.index for case in requested
                                                            if case.raw_full_score and case.score == 0],
        },
    }


def markdown_row(report):
    funnel = report["overall"]

    def ratio(count, denominator):
        percent = f"{count / denominator:.2%}" if denominator else "—"
        return f"{count} / {denominator}（{percent}）"

    cells = [f"`{report['run']}`", f"{funnel['valid']} / {report['answerable']}"]
    previous = funnel["valid"]
    for stage in ("read_all", "stored_all", "requested_all", "full_score"):
        count = funnel[stage]
        overall = ratio(count, funnel["valid"])
        cells.append(overall if stage == "read_all" else f"{overall}；{ratio(count, previous)}")
        previous = count
    return "| " + " | ".join(cells) + " |"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--markdown-row", action="store_true", help="Print a single row for the experiment log")
    args = parser.parse_args()
    try:
        report = analyze(args.run_dir.resolve())
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")
    print(markdown_row(report) if args.markdown_row else json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
