"""Append-only native assistant/tool session, with explicit terminal outcomes."""
import copy
from pathlib import Path
import time
from .client import Failure, completed
from .evidence import EvidenceMemory
from .status import append_status, format_status
from .tools import Tools, schemas

PROMPT = (Path(__file__).parent / "prompts/locator.txt").read_text()
ROUTING_REMINDER = (
    "\n\nEND OF OUTLINE. Protocol: you are the Locator, not the final answerer. "
    "Your next response must call a tool. When the evidence is ready, "
    "store the relevant successfully read pages in evidence memory, then call call_reasoner with {}. "
    "Do not output the answer as plain text; only Reasoner can finish this session.")

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
        evidence = EvidenceMemory(self.document)
        tools = Tools(question, evidence, self.client, self.audit, self.config)
        initial_status = format_status(evidence, tools, self.config, 0)
        messages = [{"role": "system", "content": PROMPT + "\nDocument outline:\n" + self.document.outline() + ROUTING_REMINDER
                     + "\n\n" + initial_status},
                    {"role": "user", "content": question}]
        status, stop, error = "incomplete", "max_rounds", None
        rounds = 0
        corrections, correction_rounds = 0, 0
        locator_drafts = []
        next_choice, correcting_next = None, False
        self.client.deadline = time.monotonic() + self.config["wall_seconds"]
        for rounds in range(1, self.config["max_rounds"] + 1):
            correcting = correcting_next
            correcting_next = False
            active_choice, next_choice = next_choice, None
            correction_rounds += int(correcting)
            self.audit.context = {"round": rounds, "source": "controller_correction" if correcting else "model"}
            try:
                validate_history(messages)
                response = self.client.complete("locator", messages, schemas(self.config["read_page_max"]),
                                                tool_choice=active_choice)
                assistant = assistant_message(completed(response)["message"])
                calls = assistant.get("tool_calls") or []
                if not isinstance(calls, list) or len(calls) > 32:
                    raise Failure("invalid_tool_calls", "Expected at most 32 tool calls")
                validate_history(messages + [assistant], allow_pending=True)
                messages.append(assistant)
                if not calls:
                    text = (assistant.get("content") or "").strip()
                    locator_drafts.append({"round": rounds, "content": assistant.get("content"),
                                           "finish_reason": completed(response).get("finish_reason")})
                    if (corrections < self.config["max_protocol_corrections"] and
                        rounds < self.config["max_rounds"]):
                        corrections += 1
                        read_pages = sorted(evidence.read_pages)
                        if evidence.pages:
                            next_choice = {"type": "function", "function": {"name": "call_reasoner"}}
                            instruction = (
                                "Controller protocol correction: your previous plain-text response is only a "
                                "Locator draft and is not the final answer. Evidence memory contains pages "
                                + str(sorted(evidence.pages)) + ". Now call call_reasoner with {}. "
                                "Reasoner will independently answer from the original page evidence.")
                        elif read_pages:
                            next_choice = {"type": "function", "function": {"name": "store"}}
                            instruction = (
                                "Controller protocol correction: evidence memory is empty. Select relevant pages "
                                "from these successfully read pages: " + str(read_pages) + ". "
                                "Call store(pages=[...]) to add them to memory, then call call_reasoner with {}. "
                                "A plain-text answer cannot finish this session.")
                        else:
                            next_choice = "required"
                            instruction = (
                                "Controller protocol correction: no page has been successfully read. "
                                "Use search/read_page to obtain evidence, store relevant pages, then call call_reasoner with {}. "
                                "A plain-text answer cannot finish this session.")
                        correction = {"role": "user", "content": instruction}
                        messages.append(correction)
                        correcting_next = True
                        self.audit.emit("protocol_correction", source="controller",
                                        correction=corrections, read_pages=read_pages,
                                        memory_pages=sorted(evidence.pages), memory_version=evidence.version,
                                        next_tool_choice=next_choice, message=correction)
                        self.audit.emit("history_checkpoint", source="controller", messages=messages)
                        continue
                    stop = "premature_final" if text else "empty_assistant"
                    break
                fatal = None
                answered = False
                batch_failed = False
                batch_errors = []
                observed_read_pages = frozenset(evidence.read_pages)
                for call in calls:
                    if answered or fatal or batch_failed:
                        # Resolve all IDs even when later calls in a parallel batch cannot execute.
                        outcome = {"ok": False, "tool": call["function"]["name"],
                                   "error": {"code": "dependency_failed" if batch_failed and not fatal else "session_terminated",
                                             "message": "An earlier call ended or failed this batch",
                                             "retryable": bool(batch_failed and not fatal)},
                                   "evidence_pages": sorted(evidence.pages), "evidence_version": evidence.version}
                        observation = {"role": "tool", "tool_call_id": call["id"], "content": str(outcome)}
                        self.audit.emit("tool_result", name=call["function"]["name"],
                                        tool_call_id=call["id"], outcome=outcome, message=observation)
                    else:
                        observation, outcome = tools.execute(call, observed_read_pages=observed_read_pages)
                        batch_failed = not outcome["ok"]
                        if not outcome["ok"] and not outcome["error"]["retryable"]:
                            fatal = outcome["error"]
                        answered = (tools.last_reason or {}).get("status") == "answerable"
                    if not outcome["ok"]:
                        batch_errors.append({"tool": call["function"]["name"], **outcome["error"]})
                    messages.append(observation)
                terminal = bool(fatal or answered or rounds == self.config["max_rounds"])
                agent_status = format_status(evidence, tools, self.config, rounds, batch_errors, terminal=terminal)
                append_status(messages[-1], agent_status)
                self.audit.emit("agent_status", tool_call_id=messages[-1]["tool_call_id"],
                                completed_rounds=rounds, terminal=terminal,
                                text=agent_status, message=messages[-1])
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
                  "evidence_memory": {"pages": sorted(evidence.pages), "version": evidence.version,
                                      "mode": "append-only"},
                  "read_pages": sorted(evidence.read_pages), "tool_calls": tools.count,
                  "tool_counts": dict(tools.counts),
                  "controller_fallback_calls": 0,
                  "protocol_corrections": corrections, "protocol_correction_rounds": correction_rounds,
                  "locator_non_tool_responses": locator_drafts,
                  "finalized_by": "reasoner" if status == "completed" else None,
                  "protocol_valid": stop not in {"invalid_history", "invalid_tool_calls"},
                  "total_image_parts_in_history": sum(p.get("type") == "image_url" for m in messages
                      for p in (m.get("content") if isinstance(m.get("content"), list) else []))}
        self.audit.emit("session_end", result=result, messages=messages)
        return result
