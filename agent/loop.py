"""Append-only native assistant/tool session, with explicit terminal outcomes."""
import copy
from pathlib import Path
import time
from .client import Failure, completed
from .evidence import Evidence
from .tools import Tools, schemas

PROMPT = (Path(__file__).parent / "prompts/locator.txt").read_text()

def validate_history(messages, allow_pending=False):
    pending, seen = [], set()
    for message in messages:
        if message["role"] == "tool":
            if not pending or message.get("tool_call_id") not in pending:
                raise Failure("invalid_history", "Tool message without a matching pending call")
            pending.remove(message["tool_call_id"])
        else:
            if pending:
                raise Failure("invalid_history", "Missing tool observations")
            calls = message.get("tool_calls") or []
            for call in calls:
                if (not isinstance(call, dict) or not isinstance(call.get("id"), str) or
                    not call["id"] or call["id"] in seen or call.get("type") != "function" or
                    not isinstance(call.get("function"), dict) or
                    not isinstance(call["function"].get("name"), str)):
                    raise Failure("invalid_tool_calls", "Malformed or duplicate tool call")
                seen.add(call["id"]); pending.append(call["id"])
    if pending and not allow_pending:
        raise Failure("invalid_history", "Pending tool calls at request boundary")

def assistant_message(raw):
    result = {"role": "assistant", "content": raw.get("content")}
    for field in ("tool_calls", "reasoning", "reasoning_content"):
        if raw.get(field):
            result[field] = copy.deepcopy(raw[field])
    return result

class Agent:
    def __init__(self, document, client, audit, config):
        self.document, self.client, self.audit, self.config = document, client, audit, config

    def run(self, question):
        evidence = Evidence(self.document)
        tools = Tools(question, evidence, self.client, self.audit, self.config)
        messages = [{"role": "system", "content": PROMPT + "\nDocument outline:\n" + self.document.outline()},
                    {"role": "user", "content": question}]
        status, stop, error = "incomplete", "max_rounds", None
        rounds = 0
        self.client.deadline = time.monotonic() + self.config["wall_seconds"]
        for rounds in range(1, self.config["max_rounds"] + 1):
            self.audit.context = {"round": rounds, "source": "model"}
            try:
                validate_history(messages)
                response = self.client.complete("locator", messages, schemas(self.config["read_page_max"]))
                assistant = assistant_message(completed(response)["message"])
                calls = assistant.get("tool_calls") or []
                if not isinstance(calls, list) or len(calls) > 32:
                    raise Failure("invalid_tool_calls", "Expected at most 32 tool calls")
                validate_history(messages + [assistant], allow_pending=True)
                messages.append(assistant)
                if not calls:
                    stop = "premature_final" if (assistant.get("content") or "").strip() else "empty_assistant"
                    break
                fatal = None
                answered = False
                for call in calls:
                    if answered or fatal:
                        # Resolve all IDs even when later calls in a parallel batch cannot execute.
                        outcome = {"ok": False, "tool": call["function"]["name"],
                                   "error": {"code": "session_terminated", "message": "An earlier call ended this batch", "retryable": False}}
                        observation = {"role": "tool", "tool_call_id": call["id"], "content": str(outcome)}
                        self.audit.emit("tool_result", name=call["function"]["name"],
                                        tool_call_id=call["id"], outcome=outcome, message=observation)
                    else:
                        observation, outcome = tools.execute(call)
                        if not outcome["ok"] and not outcome["error"]["retryable"]:
                            fatal = outcome["error"]
                        answered = (tools.last_reason or {}).get("status") == "answerable"
                    messages.append(observation)
                validate_history(messages)
                self.audit.emit("history_checkpoint", messages=messages)
                if fatal:
                    error, stop = fatal, fatal["code"]
                    status = "incomplete" if stop in {"context_limit", "wall_budget", "output_truncated"} else "error"
                    break
                if answered:
                    status, stop = "completed", "reasoner_answerable"
                    break
            except Failure as exc:
                error, stop = exc.as_dict(), exc.code
                status = "incomplete" if stop in {"context_limit", "wall_budget", "output_truncated"} else "error"
                break
        self.client.deadline = None
        result = {"execution_status": status, "stop_reason": stop, "error": error,
                  "semantic_status": (tools.last_reason or {}).get("status"),
                  "raw_answer": tools.last_reason["answer"] if status == "completed" else None,
                  "last_reasoner_result": tools.last_reason, "rounds": rounds,
                  "evidence_pages": sorted(evidence.pages), "evidence_version": evidence.version,
                  "read_pages": sorted(evidence.read_pages), "tool_calls": tools.count,
                  "controller_fallback_calls": 0, "finalized_by": "reasoner" if status == "completed" else None,
                  "protocol_valid": stop not in {"invalid_history", "invalid_tool_calls"},
                  "total_image_parts_in_history": sum(p.get("type") == "image_url" for m in messages
                      for p in (m.get("content") if isinstance(m.get("content"), list) else []))}
        self.audit.emit("session_end", result=result, messages=messages)
        return result
