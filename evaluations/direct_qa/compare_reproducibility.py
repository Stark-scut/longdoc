#!/usr/bin/env python3
"""Compare two fresh Direct-QA runs by original sample index; never calls a model."""

import argparse
import json
from pathlib import Path


def compare(reference, candidate):
    def load(path):
        rows = json.loads(path.read_text())
        indexed = {row["_sample_index"]: row for row in rows}
        if not rows or len(indexed) != len(rows):
            raise ValueError(f"Empty results or duplicate sample indices: {path}")
        if any(row.get("status") != "complete" for row in rows):
            raise ValueError(f"Evaluation incomplete: {path}")
        required = ("doc_id", "question", "answer", "answer_format", "response",
                    "extracted_res", "pred", "score")
        if any(any(field not in row for field in required) for row in rows):
            raise ValueError(f"Missing input/output fields: {path}")
        return indexed

    left, right = load(reference), load(candidate)
    if left.keys() != right.keys():
        raise ValueError("Sample index sets differ; compare identical question sets")
    fields = ("response", "extracted_res", "pred", "score")
    differences = {field: [] for field in fields}
    for index in sorted(left):
        a, b = left[index], right[index]
        if any(a[field] != b[field] for field in ("doc_id", "question", "answer", "answer_format")):
            raise ValueError(f"Different input at sample {index}")
        for field in fields:
            if a[field] != b[field]:
                differences[field].append(index)
    return {"samples": len(left), "identical": not any(differences.values()),
            "differences": differences,
            "agreement": {field: 1 - len(ids) / len(left) for field, ids in differences.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    try:
        result = compare(args.reference, args.candidate)
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["identical"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
