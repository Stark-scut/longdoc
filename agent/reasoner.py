"""Reasoner accepts only explicit, versioned evidence and no ground truth."""
from pathlib import Path
from .client import Failure

PROMPT = (Path(__file__).parent / "prompts/reasoner.txt").read_text()

def reason(question, evidence, client):
    if not evidence.pages:
        raise Failure("empty_evidence", "Read and select evidence before calling Reasoner", True)
    parts = [{"type": "text", "text": "Question: " + question}]
    parts.extend(evidence.render())
    result = client.json("reasoner", [{"role": "system", "content": PROMPT},
                                      {"role": "user", "content": parts}])
    status = result.get("status")
    if status == "answerable":
        if not isinstance(result.get("answer"), str) or not result["answer"].strip():
            raise Failure("invalid_reasoner_answer", "answerable requires a nonempty string answer")
    elif status == "insufficient":
        missing = result.get("missing")
        if not isinstance(missing, list) or not missing or not all(isinstance(v, dict) for v in missing):
            raise Failure("invalid_reasoner_missing", "insufficient requires a nonempty missing list")
    else:
        raise Failure("invalid_reasoner_status", str(status))
    return {**result, "evidence_version": evidence.version, "_input_pages": sorted(evidence.pages)}
