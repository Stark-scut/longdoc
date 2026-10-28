"""Offline short-answer extraction and attributed official benchmark metrics."""
import ast
import contextlib
import io
import json
from pathlib import Path
import re
from .client import completed
from .metrics.mmlongbench import eval_score, eval_acc_and_f1

EXTRACT_PROMPT = (Path(__file__).parent / "prompts/extract.txt").read_text()

def score(result, case, client):
    result["score"] = 0.0
    if result.get("execution_status") != "completed" or not result.get("raw_answer"):
        result["scoring_status"] = "not_answered"
        return
    # Ground truth, gold page IDs and answer format are never placed in this request.
    question = case["question"]
    payload = json.dumps({"question": question, "model_response": result["raw_answer"]}, ensure_ascii=False)
    response = client.complete("extractor", [
        {"role": "system", "content": EXTRACT_PROMPT},
        {"role": "user", "content": "Extract from the following data:\n" + payload}])
    text = completed(response)["message"].get("content") or ""
    result["extracted_res"] = text
    match = re.search(r"Extracted answer:\s*(.*?)(?:\n\s*Answer format:|\Z)", text, re.S | re.I)
    if not match or not match.group(1).strip():
        raise ValueError("Extractor response lacks Extracted answer")
    pred = match.group(1).strip()
    gt = case["answer"]
    answer_format = case["answer_format"]
    if answer_format == "List":
        # The official code uses eval on string lists. Parse them safely first.
        if isinstance(gt, str) and gt.strip().startswith("["):
            gt = ast.literal_eval(gt)
        if isinstance(pred, str) and pred.strip().startswith("["):
            pred = ast.literal_eval(pred)
    with contextlib.redirect_stdout(io.StringIO()):
        value = eval_score(gt, pred, answer_format)
    result.update(pred=pred, score=value, answer_correct=value == 1.0, scoring_status="scored")
    client.audit.emit("offline_score", score=value, pred=pred)

def metrics(rows):
    if not rows:
        return 0.0, 0.0
    values = [dict(answer=r.get("answer", ""), pred=r.get("pred", "Fail to answer"),
                   score=r.get("score", 0.0)) for r in rows]
    with contextlib.redirect_stdout(io.StringIO()):
        return eval_acc_and_f1(values)
