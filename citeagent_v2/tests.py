#!/usr/bin/env python3
"""Offline functional tests. Synthetic responses are NOT model benchmark results."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import sys
sys.dont_write_bytecode = True
from common import HERE, Audit, Assets, assistant_history, image_part, validate_history, read_json
from transport import Transport, Failure
from runtime import load, configure_transport
from agent import build_agent_class
from run import score, verify

def call(cid, name="read_page", args=None):
    return {"id": cid, "type": "function", "function": {
        "name": name, "arguments": json.dumps({"pages": [0]} if args is None else args)}}

def reply(calls=None, text="", finish=None, reasoning="TEST_REASONING"):
    return {"id": "synthetic", "choices": [{
        "finish_reason": finish or ("tool_calls" if calls else "stop"),
        "message": {"role": "assistant", "content": text, "reasoning": reasoning,
                    **({"tool_calls": calls} if calls else {})}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}}

def json_reply(value):
    return reply(text=json.dumps(value))

class FakeTransport(Transport):
    def __init__(self, settings, audit, plan):
        self.settings, self.audit, self.plan = settings, audit, copy.deepcopy(plan)
        self.deadline = None
        self.sent = []
    def close(self):
        pass
    def complete(self, role, messages, **kwargs):
        body = {"messages": copy.deepcopy(messages), **copy.deepcopy(kwargs)}
        self.sent.append((role, body))
        self.audit.emit("llm_request", role=role, body=body, synthetic=True)
        queue = self.plan.get(role, [])
        if not queue:
            raise AssertionError("Unexpected synthetic request: " + role)
        item = queue.pop(0)
        if isinstance(item, Exception):
            self.audit.failures += 1
            raise item
        self.audit.response(role, item, 0.001)
        self.audit.emit("llm_response", role=role, response=item, synthetic=True)
        return copy.deepcopy(item)

class AgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loaded = load(HERE / "baseline")

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="case_", dir=HERE / "_test_work")
        self.directory = Path(self.tmp.name)
        self.runtime = list(self.loaded)
        self.runtime[-1] = copy.deepcopy(self.loaded[-1])
        self.runtime[-1].update(max_rounds=4, controller_fallback=False)
        self.runtime[0].READ_PAGE_MAX = 3
        self.runtime[0].REASONER_MAX_RETRY = 1
        self.runtime[0].GLOBAL_SELECTOR_MAX_RETRY = 1
        from PIL import Image
        self.document = self.runtime[1].Document("synthetic")
        for p in range(3):
            path = self.directory / f"page_{p}.png"
            Image.new("RGB", (32, 32), ["yellow", "white", "blue"][p]).save(path)
            self.document.add_page(p, str(path), [{"block_id": 0, "label": "text",
                "content": "page " + str(p), "bbox": [0, 0, 32, 32]}])
        self.audit = Audit(self.directory / "events.jsonl")

    def tearDown(self):
        self.tmp.cleanup()

    def make(self, plan, native=True):
        t = FakeTransport(self.runtime[-1], self.audit, plan)
        configure_transport(self.runtime, t, native)
        return build_agent_class(self.runtime)(self.document, t, self.audit), t

    def good_plan(self):
        return {"locator": [reply([call("r0")]), reply([call("q", "call_reasoner", {})])],
                "selector": [json_reply({"selected_pages": [0]})],
                "reasoner": [json_reply({"status": "answerable", "answer": "42"})]}

    def test_native_full_chain_images_and_reasoning(self):
        agent, t = self.make(self.good_plan())
        result = agent.run_native("Original question")
        self.assertEqual(result["execution_status"], "completed")
        self.assertEqual(result["raw_answer"], "42")
        self.assertTrue(validate_history(agent.messages))
        self.assertEqual(sum(m["role"] == "user" for m in agent.messages), 1)
        later = [b for role, b in t.sent if role == "locator"][1]["messages"]
        self.assertEqual(later[2]["reasoning"], "TEST_REASONING")
        self.assertEqual(later[3]["tool_call_id"], "r0")
        self.assertTrue(any(p["type"] == "image_url" for p in later[3]["content"]))
        self.assertEqual(result["controller_fallback_calls"], 0)

    def test_multiple_calls_each_has_its_own_response(self):
        plan = self.good_plan()
        plan["locator"][0] = reply([call("r0"), call("r1", args={"pages": [1]})])
        plan["selector"][0] = json_reply({"selected_pages": [0, 1]})
        agent, _ = self.make(plan)
        result = agent.run_native("Question")
        self.assertEqual([m["tool_call_id"] for m in agent.messages if m["role"] == "tool"], ["r0", "r1", "q"])
        self.assertEqual(result["tool_calls"], 3)

    def test_malformed_arguments_return_error_then_recover(self):
        plan = self.good_plan()
        bad = call("bad")
        bad["function"]["arguments"] = "{"
        plan["locator"].insert(0, reply([bad]))
        agent, _ = self.make(plan)
        result = agent.run_native("Question")
        self.assertEqual(result["execution_status"], "completed")
        self.assertEqual(result["tool_failures"][0]["error"]["code"], "invalid_arguments_json")

    def test_strict_page_validation(self):
        for pages in [[True], [1.3], [-1], [99], [], [0, 1, 2, 0]]:
            with self.subTest(pages=pages):
                self.runtime[-1]["max_rounds"] = 1
                agent, _ = self.make({"locator": [reply([call("x", args={"pages": pages})])]})
                result = agent.run_native("Question")
                self.assertEqual(result["tool_failures"][0]["error"]["code"], "invalid_arguments")

    def test_truncated_output_never_executes_tools(self):
        agent, _ = self.make({"locator": [reply([call("x")], finish="length")]})
        result = agent.run_native("Question")
        self.assertEqual(result["stop_reason"], "output_truncated")
        self.assertEqual(result["tool_calls"], 0)
        self.assertEqual(len(agent.messages), 2)

    def test_missing_id_is_not_replaced_with_fake_id(self):
        agent, _ = self.make({"locator": [reply([call("")])]})
        result = agent.run_native("Question")
        self.assertFalse(result["protocol_valid"])
        self.assertEqual(result["tool_calls"], 0)

    def test_reused_id_rejected(self):
        agent, _ = self.make({"locator": [reply([call("x")]), reply([call("x")])]})
        result = agent.run_native("Question")
        self.assertFalse(result["protocol_valid"])
        self.assertEqual(result["tool_calls"], 1)

    def test_reasoner_timeout_is_not_insufficient(self):
        self.runtime[-1]["max_rounds"] = 1
        agent, _ = self.make({"locator": [reply([call("r"), call("q", "call_reasoner", {})])],
            "selector": [json_reply({"selected_pages": [0]})],
            "reasoner": [Failure("timeout", "synthetic timeout", True)]})
        result = agent.run_native("Question")
        self.assertIsNone(result["semantic_status"])
        self.assertEqual(result["tool_failures"][0]["error"]["code"], "timeout")
        self.assertEqual(result["stop_reason"], "max_rounds")

    def test_real_insufficient_is_separate_from_round_budget(self):
        self.runtime[-1]["max_rounds"] = 1
        agent, _ = self.make({"locator": [reply([call("r"), call("q", "call_reasoner", {})])],
            "selector": [json_reply({"selected_pages": [0]})],
            "reasoner": [json_reply({"status": "insufficient", "missing": [{
                "missing_claim": "Need dates", "expected_type": "text", "search_hint": "date"}]})]})
        result = agent.run_native("Question")
        self.assertEqual(result["execution_status"], "incomplete")
        self.assertEqual(result["semantic_status"], "insufficient")
        self.assertEqual(result["tool_failures"], [])

    def test_empty_reasoner_answer_rejected(self):
        plan = self.good_plan()
        plan["reasoner"] = [json_reply({"status": "answerable", "answer": ""})]
        self.runtime[-1]["max_rounds"] = 2
        agent, _ = self.make(plan)
        result = agent.run_native("Question")
        self.assertIsNone(result["raw_answer"])
        self.assertEqual(result["tool_failures"][0]["error"]["code"], "invalid_reasoner_answer")

    def test_plain_locator_answer_is_not_accepted(self):
        agent, _ = self.make({"locator": [reply(text="I think the answer is 42")]})
        result = agent.run_native("Question")
        self.assertEqual(result["stop_reason"], "premature_final")
        self.assertIsNone(result["raw_answer"])

    def test_max_rounds_does_not_fabricate_final_turn(self):
        self.runtime[-1]["max_rounds"] = 1
        agent, t = self.make({"locator": [reply([call("r")])]})
        result = agent.run_native("Question")
        self.assertEqual(result["stop_reason"], "max_rounds")
        self.assertFalse(result["last_tool_observation_consumed_by_locator"])
        self.assertEqual(len(t.sent), 1)

    def test_controller_fallback_is_logged_not_faked_as_tool_call(self):
        self.runtime[-1].update(max_rounds=1, controller_fallback=True)
        plan = self.good_plan()
        plan["locator"] = plan["locator"][:1]
        agent, _ = self.make(plan)
        result = agent.run_native("Question")
        self.assertEqual(result["controller_fallback_calls"], 1)
        self.assertEqual(result["finalized_by"], "controller")
        self.assertEqual([m["tool_call_id"] for m in agent.messages if m["role"] == "tool"], ["r0"])
        self.assertIn("controller_fallback_start", self.audit.path.read_text())

    def test_missing_image_is_explicit(self):
        Path(self.document.page_image_path(0)).unlink()
        self.runtime[-1]["max_rounds"] = 1
        agent, _ = self.make({"locator": [reply([call("r")])]})
        result = agent.run_native("Question")
        self.assertEqual(result["total_image_parts_in_history"], 0)
        self.assertIn("IMAGE_UNAVAILABLE", json.dumps(agent.messages))

    def test_unknown_tool_yields_paired_error(self):
        self.runtime[-1]["max_rounds"] = 1
        agent, _ = self.make({"locator": [reply([call("x", "missing_tool", {})])]})
        result = agent.run_native("Question")
        self.assertEqual(result["tool_failures"][0]["error"]["code"], "unknown_tool")
        self.assertTrue(validate_history(agent.messages))

    def test_legacy_preserves_original_user_observation(self):
        from run import legacy_run
        plan = self.good_plan()
        plan["locator"].append(reply(text="Evidence located"))
        _, t = self.make(plan, native=False)
        result = legacy_run(self.runtime, self.document, "Question", self.audit)
        self.assertEqual(result["raw_answer"], "42")
        request = [b for role, b in t.sent if role == "locator"][1]["messages"]
        self.assertTrue(all(m["role"] in {"user", "system"} for m in request))
        self.assertTrue(any(isinstance(m.get("content"), list) for m in request))

    def test_unanswered_unanswerable_gt_is_still_zero(self):
        _, t = self.make({})
        result = {"execution_status": "error", "raw_answer": None}
        score(result, {"answer": "Not answerable"}, HERE / "baseline", t, self.audit)
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["scoring_status"], "not_answered")
        self.assertEqual(t.sent, [])

    def test_official_scoring_only_offline(self):
        _, t = self.make({"extractor": [reply(text="Extracted answer: 42\nAnswer format: Int")]})
        result = {"execution_status": "completed", "raw_answer": "There are 42 objects.", "evidence_pages": [0]}
        case = {"question": "Count objects", "answer": "42", "answer_format": "Int", "evidence_pages": "[1]"}
        score(result, case, HERE / "baseline", t, self.audit)
        self.assertEqual(result["score"], 1.0)
        self.assertNotIn('"answer": "42"', json.dumps(t.sent))
        self.assertTrue(result["evidence_page_hit"])

class ProtocolTests(unittest.TestCase):
    def test_orphans_duplicates_and_pending(self):
        with self.assertRaises(ValueError):
            validate_history([{"role": "tool", "tool_call_id": "a"}])
        messages = [{"role": "assistant", "tool_calls": [call("a")]}]
        with self.assertRaises(ValueError):
            validate_history(messages)
        self.assertTrue(validate_history(messages, allow_pending=True))
        messages.append({"role": "tool", "tool_call_id": "a"})
        self.assertTrue(validate_history(messages))
        with self.assertRaises(ValueError):
            validate_history(messages + [messages[-1]])

    def test_reasoning_adapter_drops_response_only_fields(self):
        m = assistant_history({"content": "", "reasoning_content": "kept", "refusal": None,
                               "tool_calls": [call("a")]})
        self.assertEqual(m["reasoning"], "kept")
        self.assertNotIn("refusal", m)

    def test_asset_replay_round_trip_and_tamper(self):
        with tempfile.TemporaryDirectory(dir=HERE / "_test_work") as d:
            path = Path(d) / "image.png"
            path.write_bytes(b"synthetic image bytes")
            assets = Assets(Path(d) / "assets")
            raw = image_part(path)
            packed = assets.pack(raw)
            self.assertEqual(assets.unpack(packed), raw)
            next(assets.directory.iterdir()).write_bytes(b"modified")
            with self.assertRaises(ValueError):
                assets.unpack(packed)

    def test_transport_retries_and_explicit_parameters(self):
        import httpx
        with tempfile.TemporaryDirectory(dir=HERE / "_test_work") as d:
            audit = Audit(Path(d) / "events.jsonl")
            seen = []
            def handler(request):
                seen.append(json.loads(request.content))
                return httpx.Response(429, text="busy") if len(seen) == 1 else httpx.Response(200, json=reply(text="done"))
            t = Transport(read_json(HERE / "baseline/settings.json"), audit,
                          httpx.Client(transport=httpx.MockTransport(handler), trust_env=False))
            r = t.complete("locator", [{"role": "user", "content": "Q"}], tools=[{"type": "function"}], attempts=2)
            self.assertEqual(r["choices"][0]["message"]["content"], "done")
            self.assertEqual(len(seen), 2)
            self.assertTrue(seen[0]["chat_template_kwargs"]["enable_thinking"])
            self.assertEqual(audit.requests, 2)
            self.assertEqual(audit.failures, 1)
            self.assertNotIn("Authorization", audit.path.read_text())
            t.close()

    def test_context_limit_is_not_retried(self):
        import httpx
        with tempfile.TemporaryDirectory(dir=HERE / "_test_work") as d:
            audit = Audit(Path(d) / "events.jsonl")
            t = Transport(read_json(HERE / "baseline/settings.json"), audit,
                httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(400, text="maximum context length exceeded"))))
            with self.assertRaises(Failure) as caught:
                t.complete("locator", [{"role": "user", "content": "Q"}], attempts=3)
            self.assertEqual(caught.exception.code, "context_limit")
            self.assertEqual(audit.requests, 1)
            t.close()

    def test_template_tool_vs_user_reasoning_boundary(self):
        from jinja2 import Environment
        env = Environment()
        env.globals["raise_exception"] = lambda s: (_ for _ in ()).throw(ValueError(s))
        template = env.from_string((HERE / "baseline/model_metadata/chat_template.jinja").read_text())
        messages = [{"role": "user", "content": "Question"},
            {"role": "assistant", "content": "", "reasoning_content": "UNIQUE_REASONING_MARKER"},
            {"role": "tool", "tool_call_id": "x", "content": "observation"}]
        good = template.render(messages=messages, tools=[], add_generation_prompt=True)
        messages[-1]["role"] = "user"
        bad = template.render(messages=messages, tools=[], add_generation_prompt=True)
        self.assertIn("UNIQUE_REASONING_MARKER", good)
        self.assertNotIn("UNIQUE_REASONING_MARKER", bad)

    def test_source_repository_unchanged(self):
        self.assertEqual(verify(HERE / "baseline")["original_repository_changed"], [])

if __name__ == "__main__":
    (HERE / "_test_work").mkdir(exist_ok=True)
    unittest.main(verbosity=2)
