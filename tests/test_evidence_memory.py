"""Append-only memory contracts and offline tool-loop regressions."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
from dataclasses import FrozenInstanceError
import httpx
from test_agent import Base, reply, role_of, tool
from agent.client import Client, Failure
from agent.config import validate
from agent.evaluation import prepare, select_cases, verify
from agent.evidence import EvidenceMemory
from agent.logging import Audit
from agent.logging import atomic, read
from agent.loop import Agent, validate_history
from agent.reasoner import reason
from agent.tools import Tools


class MemoryContracts(Base):
    def memory(self):
        return EvidenceMemory(self.documents["doc"])

    def test_store_accumulates_deduplicates_and_versions_only_changes(self):
        memory = self.memory(); memory.read_pages.update([0, 1])
        first = memory.store([0, 0])
        self.assertEqual(first, {"added_pages": [0], "already_stored_pages": [],
                                 "memory_pages": [0], "memory_version": 1})
        again = memory.store([0])
        self.assertEqual(again["added_pages"], [])
        self.assertEqual(again["already_stored_pages"], [0])
        self.assertEqual(again["memory_version"], 1)
        second = memory.store([1, 0, 1])
        self.assertEqual(second["added_pages"], [1])
        self.assertEqual(second["memory_pages"], [0, 1])
        self.assertEqual(second["memory_version"], 2)
        # The public view cannot be mutated to implement an implicit remove.
        with self.assertRaises(AttributeError):
            memory.pages.remove(0)

    def test_invalid_mixed_store_is_atomic_and_never_forgets(self):
        memory = self.memory(); memory.read_pages.update([0, 1])
        memory.store([0])
        for pages in ([1, 999], [1, True], [1, "0"], [1, 0.0], [], None, "1", [None]):
            with self.assertRaises(Failure, msg=repr(pages)) as error:
                memory.store(pages)
            self.assertTrue(error.exception.retryable)
            self.assertEqual(memory.pages, {0})
            self.assertEqual(memory.version, 1)

    def test_store_uses_only_previously_read_pages_and_calls_no_model(self):
        memory = self.memory()
        audit = Audit(self.root / "store/trace.jsonl")
        client = Client(self.config, audit, httpx.MockTransport(lambda req: self.fail("store called a model")))
        tools = Tools("question", memory, client, audit, self.config)
        try:
            _, rejected = tools.execute(tool("store", {"pages": [0]}, "unread")["tool_calls"][0])
            self.assertEqual(rejected["error"]["code"], "invalid_evidence_pages")
            memory.read_pages.add(0)
            _, stored = tools.execute(tool("store", {"pages": [0]}, "stored")["tool_calls"][0])
            self.assertTrue(stored["ok"])
            self.assertEqual(stored["result"]["memory_pages"], [0])
            for args in ({"pages": [0], "mode": "remove"}, {"pages": [1], "replace": True},
                         {"pages": []}, {}):
                _, result = tools.execute(tool("store", args, "invalid")["tool_calls"][0])
                self.assertFalse(result["ok"])
                self.assertEqual(memory.pages, {0})
            self.assertEqual(sum(audit.requests.values()), 0)
            self.assertFalse(list(audit.assets.directory.iterdir()))
        finally:
            client.close()

    def test_reasoner_rejects_empty_memory_and_legacy_page_arguments_without_http(self):
        memory = self.memory(); memory.read_pages.add(0)
        audit = Audit(self.root / "empty/trace.jsonl")
        client = Client(self.config, audit, httpx.MockTransport(lambda req: self.fail("Invalid call reached model")))
        tools = Tools("question", memory, client, audit, self.config)
        try:
            # Read is insufficient: only store can populate memory.
            _, result = tools.execute(tool("call_reasoner", {}, "empty")["tool_calls"][0])
            self.assertEqual(result["error"]["code"], "empty_evidence")
            self.assertTrue(result["error"]["retryable"])
            memory.store([0])
            for args in ({"pages": [0]}, {"pages": []}, {"mode": "visual"}):
                _, result = tools.execute(tool("call_reasoner", args, "old_args")["tool_calls"][0])
                self.assertEqual(result["error"]["code"], "invalid_arguments")
            self.assertEqual(audit.requests["reasoner"], 0)
        finally:
            client.close()

    def test_reasoner_response_is_attributed_to_immutable_input_snapshot(self):
        memory = self.memory(); memory.read_pages.update([0, 1]); memory.store([0])
        original = memory.snapshot()
        with self.assertRaises(FrozenInstanceError):
            original.version = 99
        bodies = []
        def handler(req):
            body = json.loads(req.content); bodies.append(body)
            memory.store([1])
            return httpx.Response(200, json=reply({"role": "assistant", "content":
                                                 '{"status":"answerable","answer":"red"}'}))
        audit = Audit(self.root / "snapshot/trace.jsonl")
        client = Client(self.config, audit, httpx.MockTransport(handler))
        try:
            result = reason("question", memory, client)
        finally:
            client.close()
        self.assertEqual(memory.pages, {0, 1})
        self.assertEqual(memory.version, 2)
        self.assertEqual(original.pages, (0,))
        self.assertEqual(result["_input_pages"], [0])
        self.assertEqual(result["evidence_version"], 1)
        self.assertNotIn("page_index=1", json.dumps(bodies[0]))

    def test_store_reasoner_batch_uses_pages_observed_on_previous_turn(self):
        def handler(req):
            body = json.loads(req.content)
            if role_of(body) == "locator":
                if not any(m["role"] == "tool" for m in body["messages"]):
                    message = tool("read_page", {"pages": [0]}, "read")
                else:
                    message = tool("store", {"pages": [0]}, "store")
                    message["tool_calls"] += tool("call_reasoner", {}, "reason")["tool_calls"]
                return httpx.Response(200, json=reply(message, "tool_calls"))
            return httpx.Response(200, json=reply({"role": "assistant", "content":
                                                 '{"status":"answerable","answer":"red"}'}))
        result, audit = self.run_agent(handler)
        self.assertEqual(result["rounds"], 2)
        self.assertEqual(result["evidence_memory"]["pages"], [0])
        self.assertEqual(result["tool_counts"], {"read_page": 1, "store": 1, "call_reasoner": 1})
        events = [json.loads(line) for line in audit.path.read_text().splitlines()]
        history = next(e["messages"] for e in reversed(events) if e["event"] == "history_checkpoint")
        validate_history(history)
        self.assertEqual([m["tool_call_id"] for m in history if m["role"] == "tool"], ["read", "store", "reason"])

    def test_failed_store_skips_same_batch_reasoner_instead_of_using_stale_memory(self):
        seen = []
        def handler(req):
            body = json.loads(req.content); role = role_of(body); seen.append((role, body))
            if role == "locator":
                turn = sum(r == "locator" for r, _ in seen)
                if turn == 1:
                    message = tool("read_page", {"pages": [0]}, "read0")
                elif turn == 2:
                    message = tool("store", {"pages": [0]}, "store0")
                elif turn == 3:
                    message = tool("store", {"pages": [1]}, "unread")
                    message["tool_calls"] += tool("call_reasoner", {}, "must_skip")["tool_calls"]
                elif turn == 4:
                    self.assertIn("dependency_failed", json.dumps(body))
                    message = tool("read_page", {"pages": [1]}, "read1")
                else:
                    message = tool("store", {"pages": [1]}, "store1")
                    message["tool_calls"] += tool("call_reasoner", {}, "reason")["tool_calls"]
                return httpx.Response(200, json=reply(message, "tool_calls"))
            self.assertIn("page_index=0", json.dumps(body))
            self.assertIn("page_index=1", json.dumps(body))
            return httpx.Response(200, json=reply({"role": "assistant", "content":
                                                 '{"status":"answerable","answer":"white"}'}))
        result, audit = self.run_agent(handler)
        self.assertEqual(result["execution_status"], "completed")
        self.assertEqual(audit.requests["reasoner"], 1)
        self.assertEqual(result["evidence_memory"]["pages"], [0, 1])
        self.assertEqual(result["evidence_version"], 2)
        self.assertEqual(result["tool_counts"]["store"], 3)

    def test_newly_read_pages_require_locator_observation_before_store(self):
        turn = 0
        def handler(req):
            nonlocal turn
            body = json.loads(req.content)
            if role_of(body) == "locator":
                turn += 1
                if turn == 1:
                    message = tool("read_page", {"pages": [0]}, "read")
                    message["tool_calls"] += tool("store", {"pages": [0]}, "premature")["tool_calls"]
                    message["tool_calls"] += tool("call_reasoner", {}, "skipped")["tool_calls"]
                else:
                    outcomes = [json.loads(m["content"][0]["text"]) for m in body["messages"]
                                if m["role"] == "tool" and m["tool_call_id"] == "premature"]
                    self.assertEqual(outcomes[0]["error"]["code"], "invalid_evidence_pages")
                    self.assertEqual(outcomes[0]["evidence_pages"], [])
                    self.assertIn("dependency_failed", json.dumps(body))
                    message = tool("store", {"pages": [0]}, "store")
                    message["tool_calls"] += tool("call_reasoner", {}, "reason")["tool_calls"]
                return httpx.Response(200, json=reply(message, "tool_calls"))
            return httpx.Response(200, json=reply({"role": "assistant", "content":
                                                 '{"status":"answerable","answer":"red"}'}))
        result, audit = self.run_agent(handler)
        self.assertEqual(result["execution_status"], "completed")
        self.assertEqual(result["rounds"], 2)
        self.assertEqual(result["evidence_memory"]["pages"], [0])
        self.assertEqual(audit.requests["reasoner"], 1)

    def test_plain_locator_response_forces_store_when_only_read_pages_exist(self):
        seen = []
        def handler(req):
            body = json.loads(req.content); role = role_of(body); seen.append((role, body))
            if role == "locator":
                turn = sum(r == "locator" for r, _ in seen)
                if turn == 1:
                    response = tool("read_page", {"pages": [0]}, "read")
                elif turn == 2:
                    response = {"role": "assistant", "content": "LOCATOR_DRAFT"}
                elif turn == 3:
                    self.assertEqual(body["tool_choice"], {"type": "function", "function": {"name": "store"}})
                    response = tool("store", {"pages": [0]}, "store")
                else:
                    response = tool("call_reasoner", {}, "reason")
                return httpx.Response(200, json=reply(response))
            self.assertNotIn("LOCATOR_DRAFT", json.dumps(body))
            return httpx.Response(200, json=reply({"role": "assistant", "content":
                                                 '{"status":"answerable","answer":"red"}'}))
        result, audit = self.run_agent(handler)
        self.assertEqual(result["raw_answer"], "red")
        self.assertEqual(result["protocol_corrections"], 1)
        self.assertEqual(audit.requests["reasoner"], 1)

    def test_question_memory_is_reset_even_when_document_and_client_are_reused(self):
        turn = [0]; question = [0]; bodies = []
        def handler(req):
            body = json.loads(req.content)
            if role_of(body) == "locator":
                n = turn[0]; turn[0] += 1
                page = question[0]
                sequence = [("read_page", {"pages": [page]}), ("store", {"pages": [page]}),
                            ("call_reasoner", {})]
                if page == 1:
                    sequence.insert(0, ("call_reasoner", {}))
                name, args = sequence[n]
                return httpx.Response(200, json=reply(tool(name, args, "id" + str(n)), "tool_calls"))
            bodies.append(body)
            return httpx.Response(200, json=reply({"role": "assistant", "content":
                                                 '{"status":"answerable","answer":"done"}'}))
        audit = Audit(self.root / "isolation/trace.jsonl")
        client = Client(self.config, audit, httpx.MockTransport(handler))
        agent = Agent(self.documents["doc"], client, audit, self.config)
        try:
            first = agent.run("first question")
            question[0] = 1; turn[0] = 0
            second = agent.run("second question")
        finally:
            client.close()
        self.assertEqual(first["evidence_memory"]["pages"], [0])
        self.assertEqual(second["evidence_memory"]["pages"], [1])
        self.assertEqual(second["evidence_version"], 1)
        self.assertEqual(len(bodies), 2)
        self.assertNotIn("page_index=0", json.dumps(bodies[1]))

    def test_memory_survives_round_exhaustion_and_context_failure_without_trimming(self):
        def handler(req):
            body = json.loads(req.content)
            if role_of(body) == "locator":
                if not any(m["role"] == "tool" for m in body["messages"]):
                    message = tool("read_page", {"pages": [0, 1]}, "read")
                else:
                    message = tool("store", {"pages": [0, 1]}, "store")
                    if self.config["max_rounds"] > 2:
                        message["tool_calls"] += tool("call_reasoner", {}, "reason")["tool_calls"]
                return httpx.Response(200, json=reply(message, "tool_calls"))
            body = json.loads(req.content)
            self.assertIn("page_index=0", json.dumps(body))
            self.assertIn("page_index=1", json.dumps(body))
            return httpx.Response(400, text="maximum context length exceeded")
        self.config["max_rounds"] = 2
        exhausted, _ = self.run_agent(handler)
        self.assertEqual(exhausted["stop_reason"], "max_rounds")
        self.assertEqual(exhausted["evidence_memory"]["pages"], [0, 1])
        self.config["max_rounds"] = 6
        limited, _ = self.run_agent(handler)
        self.assertEqual(limited["stop_reason"], "context_limit")
        self.assertEqual(limited["evidence_memory"]["pages"], [0, 1])

    def test_text_search_remains_only_search_mode(self):
        memory = self.memory(); audit = Audit(self.root / "search/trace.jsonl")
        client = Client(self.config, audit, httpx.MockTransport(lambda req: self.fail("search called a model")))
        tools = Tools("question", memory, client, audit, self.config)
        try:
            _, result = tools.execute(tool("search", {"query": "Colors", "top_k": 1}, "text")["tool_calls"][0])
            self.assertTrue(result["ok"])
            self.assertEqual(result["result"][0]["page"], 0)
            _, invalid = tools.execute(tool("search", {"query": "Colors", "mode": "visual"}, "unsupported")["tool_calls"][0])
            self.assertFalse(invalid["ok"])
            self.assertEqual(memory.pages, set())
        finally:
            client.close()

    def test_protocol_version_rejects_schema2_and_remove_configuration(self):
        for values in ({"schema": 2, "pipeline": "locator-direct-evidence"},
                       {"evidence_memory_mode": "remove"}):
            config = copy.deepcopy(self.config); config.update(values)
            with self.assertRaises(ValueError):
                validate(config)
        output = self.root / "old"
        run = prepare(output, self.config, select_cases(self.config["samples"], "0"))
        run["schema"] = 2
        with self.assertRaisesRegex(ValueError, "different Agent protocol"):
            verify(output, run, self.config)

    def test_frozen_worker_entry_runs_in_subprocess_with_no_repository_imports(self):
        output = self.root / "frozen_worker"
        prepare(output, self.config, select_cases(self.config["samples"], "0"))
        atomic(output / "runtime.json", {"run_id": "frozen", "local_servers": False,
               "map_replicas": False, "ports": [8071], "stage": "all", "retry_failed": False,
               "asset_mode": "reference", "assignments": {"0:0": [0]}})
        script = textwrap.dedent('''
            import json, sys
            from pathlib import Path
            import httpx
            import agent
            import agent.evaluation as evaluation
            from agent.client import Client
            from agent.cli import main
            assert Path(agent.__file__).resolve().is_relative_to(Path.cwd())
            def handler(request):
                body = json.loads(request.content)
                if body.get("tools"):
                    turn = sum(m["role"] == "tool" for m in body["messages"])
                    name, args = [("read_page", {"pages": [0]}), ("store", {"pages": [0]}),
                                  ("call_reasoner", {})][turn]
                    message = {"role": "assistant", "content": None, "tool_calls": [
                        {"id": "call" + str(turn), "type": "function",
                         "function": {"name": name, "arguments": json.dumps(args)}}]}
                else:
                    text = ('{"status":"answerable","answer":"red"}'
                            if body["messages"][0]["content"].startswith("Answer the question")
                            else "Extracted answer: red\\nAnswer format: String")
                    message = {"role": "assistant", "content": text}
                return httpx.Response(200, json={"choices": [{"message": message, "finish_reason": "stop"}]})
            evaluation.Client = lambda config, audit: Client(config, audit, httpx.MockTransport(handler))
            raise SystemExit(main(["_worker", "--output-dir", sys.argv[1], "--replica", "0", "--slot", "0"]))
        ''')
        env = os.environ.copy(); env["PYTHONPATH"] = str(output / "snapshot")
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        result = subprocess.run([sys.executable, "-B", "-c", script, str(output)],
                                cwd=output / "snapshot", env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        row = read(output / "samples/sample_0000/result.json")
        self.assertEqual(row["evidence_memory"], {"pages": [0], "version": 1, "mode": "append-only"})
        self.assertEqual(row["score"], 1)
        self.assertEqual(row["tool_counts"], {"read_page": 1, "store": 1, "call_reasoner": 1})


if __name__ == "__main__":
    import unittest
    unittest.main(verbosity=2)
