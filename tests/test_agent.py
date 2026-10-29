"""Standalone Agent contract and subprocess-resume tests. No GPU or paid API."""
from __future__ import annotations
import ast
import copy
import http.server
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import httpx
from PIL import Image
from agent.client import Client, Failure
from agent.config import defaults, load, validate
from agent.documents import Documents, select_cases
from agent.evaluation import prepare, verify, aggregate, needs_work, result_path
from agent.logging import Audit, read, replay, sha
from agent.loop import Agent, validate_history
from agent.launcher import gpu_ports, server_command

def reply(message, finish="stop"):
    return {"choices": [{"message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": 25, "completion_tokens": 5, "total_tokens": 30}}

def tool(name, args, ident):
    return {"role": "assistant", "content": None, "reasoning": "Inspect the evidence.",
            "tool_calls": [{"id": ident, "type": "function",
                            "function": {"name": name, "arguments": json.dumps(args)}}]}

def role_of(body):
    if body.get("tools"):
        return "locator"
    system = body["messages"][0]["content"]
    if system.startswith("Answer the question"):
        return "reasoner"
    return "extractor"

class Fixture:
    def __init__(self):
        self.requests = []
        self.lock = threading.Lock()
        self.fail_extract = False
        self.hold_extract = False
        self.extract_started = threading.Event()
        self.release = threading.Event()
        fixture = self
        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args): pass
            def send(self, status, value):
                data = json.dumps(value).encode()
                try:
                    self.send_response(status); self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data))); self.end_headers()
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError): pass
            def do_GET(self):
                self.send(200, {"data": [{"id": "longdoc-qwen35"}]})
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                role = role_of(body)
                with fixture.lock:
                    fixture.requests.append((role, body))
                if role == "locator":
                    step = sum(m["role"] == "tool" for m in body["messages"])
                    name, args = [("read_page", {"pages": [0]}), ("call_reasoner", {"pages": [0]})][min(step, 1)]
                    self.send(200, reply(tool(name, args, "call_" + str(step)), "tool_calls"))
                elif role == "reasoner":
                    self.send(200, reply({"role": "assistant", "content": '{"status":"answerable","answer":"red"}'}))
                else:
                    fixture.extract_started.set()
                    if fixture.hold_extract:
                        fixture.release.wait(20)
                    if fixture.fail_extract:
                        self.send(500, {"error": "synthetic extraction failure"})
                    else:
                        self.send(200, reply({"role": "assistant", "content": "Extracted answer: red\nAnswer format: String"}))
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
    def close(self):
        self.release.set(); self.server.shutdown(); self.server.server_close(); self.thread.join(3)

class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="standalone_agent_")
        self.root = Path(self.tmp.name)
        self.data = self.root / "data"; self.data.mkdir()
        images = self.data / "images_1024/doc"; images.mkdir(parents=True)
        crops = self.data / "visual_crops/doc"; crops.mkdir(parents=True)
        for i in range(2):
            Image.new("RGB", (64, 64), ("red", "white")[i]).save(images / f"page_{i}.png")
        Image.new("RGB", (32, 32), "red").save(crops / "page_0.png_b1.jpg")
        self.cases = [{"doc_id": "doc.pdf", "question": "What color is item " + str(i) + "?",
                       "answer": "red", "answer_format": "Str", "evidence_pages": "[1]"} for i in range(4)]
        (self.data / "samples.json").write_text(json.dumps(self.cases))
        rows = [{"image_path": "/old/unavailable/doc/page_0.png", "blocks": [
                    {"block_id": 0, "label": "paragraph_title", "content": "Colors"},
                    {"block_id": 1, "label": "image", "content": "OCR_SENTINEL"}]},
                {"image_path": "/old/unavailable/doc/page_1.png", "blocks": [
                    {"block_id": 0, "label": "text", "content": "Details"}]}]
        (self.data / "stage2_blocks_cache.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        (self.data / "crop_captions.jsonl").write_text("")
        model = self.root / "model"; model.mkdir(); (model / "config.json").write_text("{}")
        self.config = defaults()
        self.config.update(data_dir=str(self.data), samples=str(self.data / "samples.json"),
            blocks=str(self.data / "stage2_blocks_cache.jsonl"), images_dir=str(self.data / "images_1024"),
            crop_root=str(self.data / "visual_crops"), captions=str(self.data / "crop_captions.jsonl"),
            model_path=str(model), max_rounds=6, request_attempts=1)
        self.documents = Documents(self.config, ["doc.pdf"])
    def tearDown(self):
        self.tmp.cleanup()
    def run_agent(self, handler):
        audit = Audit(self.root / "unit/trace.jsonl")
        client = Client(self.config, audit, httpx.MockTransport(handler))
        try:
            result = Agent(self.documents["doc.pdf"], client, audit, self.config).run("What color?")
            return result, audit
        finally:
            client.close()

class Contracts(Base):
    def test_locator_image_only_history_and_direct_reasoner(self):
        requests = []
        def handler(request):
            body = json.loads(request.content); role = role_of(body); requests.append((role, body))
            if role == "locator":
                count = sum(m["role"] == "tool" for m in body["messages"])
                name = "read_page" if count == 0 else "call_reasoner"
                return httpx.Response(200, json=reply(tool(name, {"pages": [0]}, "id" + str(count)), "tool_calls"))
            self.assertEqual(role, "reasoner")
            return httpx.Response(200, json=reply({"role": "assistant", "content": '{"status":"answerable","answer":"red"}'}))
        result, audit = self.run_agent(handler)
        self.assertEqual(result["raw_answer"], "red")
        self.assertTrue(result["protocol_valid"])
        self.assertNotIn("selector", audit.requests)
        self.assertEqual(audit.requests["locator"], 2)
        history = [body for role, body in requests if role == "locator"][-1]["messages"]
        validate_history(history)
        observation = next(m for m in history if m["role"] == "tool")
        self.assertEqual(observation["tool_call_id"], "id0")
        self.assertEqual([p["type"] for p in observation["content"]], ["text", "image_url"])
        self.assertEqual(observation["content"][0]["text"], "page_index=0")
        self.assertNotIn("OCR_SENTINEL", json.dumps(history))
        reasoner_body = next(body for role, body in requests if role == "reasoner")
        self.assertIn("OCR_SENTINEL", json.dumps(reasoner_body))
        self.assertEqual(sum(p.get("type") == "image_url" for p in reasoner_body["messages"][1]["content"]), 2)
        self.assertEqual(next(m for m in history if m["role"] == "assistant")["reasoning"], "Inspect the evidence.")
        self.assertEqual(requests[0][1]["chat_template_kwargs"], {"enable_thinking": True})
        self.assertEqual(reasoner_body["chat_template_kwargs"], {"enable_thinking": False})

    def test_budget_and_technical_error_are_distinct(self):
        self.config["max_rounds"] = 1
        result, _ = self.run_agent(lambda req: httpx.Response(200, json=reply(tool("read_page", {"pages": [0]}, "x"), "tool_calls")))
        self.assertEqual((result["execution_status"], result["stop_reason"]), ("incomplete", "max_rounds"))
        result, _ = self.run_agent(lambda req: httpx.Response(400, json={"error": "maximum context length exceeded"}))
        self.assertEqual((result["execution_status"], result["stop_reason"]), ("incomplete", "context_limit"))

    def test_output_truncation_not_accepted_as_answer(self):
        result, _ = self.run_agent(lambda req: httpx.Response(200, json=reply({"role":"assistant","content":"red"}, "length")))
        self.assertEqual(result["stop_reason"], "output_truncated")
        self.assertIsNone(result["raw_answer"])

    def test_invalid_reasoner_type_stops_with_real_cause(self):
        def handler(req):
            body = json.loads(req.content)
            if role_of(body) == "locator":
                count = sum(m["role"] == "tool" for m in body["messages"])
                return httpx.Response(200, json=reply(tool("read_page" if count == 0 else "call_reasoner",
                    {"pages": [0]}, "id" + str(count)), "tool_calls"))
            return httpx.Response(200, json=reply({"role": "assistant", "content": '{"status":"answerable","answer":["2005","2006"]}'}))
        result, audit = self.run_agent(handler)
        self.assertEqual(result["stop_reason"], "invalid_reasoner_answer")
        self.assertEqual(audit.requests["reasoner"], 1)
        self.assertNotIn("selector", audit.requests)

    def test_json_retry_is_explicit_and_fully_costed(self):
        self.config["json_attempts"] = 2
        audit = Audit(self.root / "json/trace.jsonl"); attempts = []
        def handler(req):
            attempts.append(json.loads(req.content))
            value = '{"status":"answerable","answer":"red"' if len(attempts) == 1 else '{"status":"answerable","answer":"red"}'
            return httpx.Response(200, json=reply({"role": "assistant", "content": value}))
        client = Client(self.config, audit, httpx.MockTransport(handler))
        try:
            self.assertEqual(client.json("reasoner", [{"role": "user", "content": "reason"}]),
                             {"status": "answerable", "answer": "red"})
        finally:
            client.close()
        self.assertEqual(audit.requests["reasoner"], 2)
        self.assertEqual(len(attempts[1]["messages"]), 3)

    def test_replay_restores_real_images_and_checks_hash(self):
        audit=Audit(self.root/"replay/trace.jsonl")
        from agent.logging import image_part
        body={"messages":[{"role":"tool","tool_call_id":"x","content":[image_part(self.documents["doc.pdf"].pages[0].image)]}]}
        audit.emit("llm_request",body=body)
        target=self.root/"request.json";replay(audit.path,1,target)
        self.assertEqual(read(target),body)
        image=next(audit.assets.directory.iterdir());image.write_bytes(b"changed")
        with self.assertRaises(ValueError):replay(audit.path,1,target)

    def test_snapshot_independence_and_tamper_detection(self):
        root=self.root/"snapshot_run";cases=select_cases(self.config["samples"],"0")
        run=prepare(root,self.config,cases);verify(root,run,self.config)
        self.assertTrue((root/"snapshot/agent/loop.py").is_file())
        (root/"snapshot/agent/loop.py").write_text("# changed")
        with self.assertRaises(ValueError):verify(root,run,self.config)

    def test_no_dependency_on_old_agent(self):
        forbidden={"citeagent_v2","frozen_legacy"}
        for p in (ROOT/"agent").rglob("*.py"):
            tree=ast.parse(p.read_text())
            for node in ast.walk(tree):
                if isinstance(node,ast.Import):
                    self.assertFalse(any(n.name.split(".")[0] in forbidden for n in node.names),str(p))
                elif isinstance(node,ast.ImportFrom):
                    self.assertNotIn((node.module or "").split(".")[0],forbidden,str(p))

    def test_selection_validation_and_server_defaults(self):
        for value in ["1,1","-1,2","1,,2"]:
            with self.assertRaises(ValueError):gpu_ports(value,None,True)
        self.assertEqual(gpu_ports("1,4,5",None,True),([1,4,5],[8071,8072,8073]))
        cmd=server_command(self.config,8071,4)
        for flag,value in [("--tensor-parallel-size","1"),("--dtype","bfloat16"),("--cpu-offload-gb","0"),("--max-num-seqs","4")]:
            self.assertEqual(cmd[cmd.index(flag)+1],value)
        self.assertIn("--enable-prefix-caching",cmd)



    def test_identical_ocr_duplicates_allowed_conflicts_rejected(self):
        p=Path(self.config["blocks"]);rows=p.read_text().splitlines()
        p.write_text(p.read_text()+rows[0]+"\n")
        document=Documents(self.config,["doc.pdf"])["doc.pdf"]
        self.assertEqual(len(document.pages[0].blocks),2)
        changed=json.loads(rows[0]);changed["blocks"][0]["content"]="conflicting"
        p.write_text(p.read_text()+json.dumps(changed)+"\n")
        with self.assertRaises(ValueError):Documents(self.config,["doc.pdf"])

    def test_scientific_doc_id_keeps_dots_and_captions(self):
        doc="2310.05634v2"
        for group in ["images_1024","visual_crops"]:
            (self.data/group/"doc").rename(self.data/group/doc)
        p=Path(self.config["blocks"]);p.write_text(p.read_text().replace("/doc/","/"+doc+"/"))
        Path(self.config["captions"]).write_text(json.dumps({"doc_id":doc,"page":0,"caption":"CAPTION_SENTINEL"})+"\n")
        store=Documents(self.config,[doc+".pdf"])
        self.assertIn("CAPTION_SENTINEL",store[doc].captions[0])
        self.assertIs(store[doc],store[doc+".pdf"])

    def test_data_directory_config_derives_input_paths(self):
        path=self.root/"paths.json";path.write_text(json.dumps({"data_dir":str(self.data)}))
        config=load(path)
        self.assertEqual(config["blocks"],str(self.data/"stage2_blocks_cache.jsonl"))
        self.assertEqual(config["images_dir"],str(self.data/"images_1024"))

    def test_parallel_ids_resolved_when_reasoner_finishes(self):
        def handler(req):
            body = json.loads(req.content)
            if role_of(body) == "locator":
                count = sum(m["role"] == "tool" for m in body["messages"])
                message = tool("read_page", {"pages": [0]}, "read") if count == 0 else tool("call_reasoner", {"pages": [0]}, "reason")
                if count:
                    message["tool_calls"].extend(tool("search", {"query": "extra"}, "skip")["tool_calls"])
                return httpx.Response(200, json=reply(message, "tool_calls"))
            return httpx.Response(200, json=reply({"role": "assistant", "content": '{"status":"answerable","answer":"red"}'}))
        result, audit = self.run_agent(handler)
        self.assertEqual(result["execution_status"], "completed")
        events = [json.loads(line) for line in audit.path.read_text().splitlines()]
        history = next(e for e in reversed(events) if e["event"] == "history_checkpoint")["messages"]
        validate_history(history)
        self.assertEqual(history[-1]["tool_call_id"], "skip")

    def test_repeated_signals_do_not_interrupt_process_cleanup(self):
        code = """import subprocess,sys,time,signal
from agent.launcher import interrupt_handler,cleanup_guard,stop_groups
child=subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"],start_new_session=True)
print(child.pid,flush=True)
try:
    with interrupt_handler():
        time.sleep(60)
except KeyboardInterrupt:
    pass
finally:
    with cleanup_guard():
        print("cleanup",flush=True)
        time.sleep(.4)
        stop_groups([child],signal.SIGTERM,.5)
    print("done",flush=True)
"""
        proc=subprocess.Popen([sys.executable,"-u","-c",code],cwd=ROOT,stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE,text=True)
        child_pid=None
        try:
            child_pid=int(proc.stdout.readline())
            proc.send_signal(signal.SIGINT)
            self.assertEqual(proc.stdout.readline().strip(),"cleanup")
            proc.send_signal(signal.SIGINT);proc.send_signal(signal.SIGHUP)
            self.assertEqual(proc.wait(timeout=6),0,proc.stderr.read())
            self.assertEqual(proc.stdout.readline().strip(),"done")
            with self.assertRaises(ProcessLookupError):os.kill(child_pid,0)
        finally:
            if proc.poll() is None:proc.kill();proc.wait()
            if child_pid:
                try:os.killpg(child_pid,signal.SIGKILL)
                except ProcessLookupError:pass
            proc.stdout.close();proc.stderr.close()


    def test_multiple_page_read_returns_only_indices_and_full_images(self):
        from agent.evidence import Evidence
        from agent.tools import Tools
        audit = Audit(self.root / "read/trace.jsonl")
        client = Client(self.config, audit, httpx.MockTransport(lambda req: self.fail("read_page called a model")))
        evidence = Evidence(self.documents["doc.pdf"])
        try:
            message, outcome = Tools("question", evidence, client, audit, self.config).execute(
                tool("read_page", {"pages": [1, 0]}, "read")["tool_calls"][0])
        finally:
            client.close()
        self.assertTrue(outcome["ok"])
        self.assertEqual(message["role"], "tool")
        self.assertEqual(message["tool_call_id"], "read")
        self.assertEqual([p["type"] for p in message["content"]], ["text", "image_url", "text", "image_url"])
        self.assertEqual([p["text"] for p in message["content"] if p["type"] == "text"],
                         ["page_index=1", "page_index=0"])
        encoded = json.dumps(message)
        self.assertNotIn("OCR_SENTINEL", encoded)
        self.assertNotIn("Colors", encoded)
        self.assertNotIn("crop", encoded)
        self.assertEqual(evidence.read_pages, {0, 1})
        self.assertEqual(evidence.pages, set())

    def test_reasoner_rejects_missing_empty_unread_and_invalid_pages(self):
        from agent.evidence import Evidence
        from agent.tools import Tools
        audit = Audit(self.root / "invalid/trace.jsonl")
        client = Client(self.config, audit, httpx.MockTransport(lambda req: self.fail("Invalid evidence reached model")))
        evidence = Evidence(self.documents["doc.pdf"])
        evidence.read_pages.add(0); evidence.select([0])
        tools = Tools("question", evidence, client, audit, self.config)
        try:
            for args in ({}, {"pages": []}, {"pages": [1]}, {"pages": [999]}, {"pages": ["0"]},
                         {"pages": [True]}, {"pages": [0], "extra": 1}):
                message, outcome = tools.execute(tool("call_reasoner", args, "invalid")["tool_calls"][0])
                self.assertFalse(outcome["ok"], args)
                self.assertTrue(outcome["error"]["retryable"])
                self.assertEqual(evidence.pages, {0})
                self.assertEqual(audit.requests["reasoner"], 0)
                self.assertEqual(message["tool_call_id"], "invalid")
        finally:
            client.close()

    def test_each_reasoner_call_uses_exact_replacement_page_set(self):
        requests = []
        reasoner_calls = []
        def handler(req):
            body = json.loads(req.content); role = role_of(body); requests.append((role, body))
            if role == "locator":
                step = sum(m["role"] == "tool" for m in body["messages"])
                name, pages = [("read_page", [0, 1]), ("call_reasoner", [0]), ("call_reasoner", [1])][step]
                return httpx.Response(200, json=reply(tool(name, {"pages": pages}, "id"+str(step)), "tool_calls"))
            self.assertEqual(role, "reasoner"); reasoner_calls.append(body)
            result = {"status":"insufficient","missing":[{"missing_claim":"Need detail page","search_hint":"details"}]} if len(reasoner_calls)==1 else {"status":"answerable","answer":"white"}
            return httpx.Response(200, json=reply({"role":"assistant","content":json.dumps(result)}))
        result, audit = self.run_agent(handler)
        self.assertEqual(result["raw_answer"], "white")
        self.assertEqual(result["evidence_pages"], [1])
        self.assertEqual(result["read_pages"], [0, 1])
        self.assertEqual(result["evidence_version"], 2)
        self.assertIn("OCR_SENTINEL", json.dumps(reasoner_calls[0]))
        self.assertNotIn("OCR_SENTINEL", json.dumps(reasoner_calls[1]))
        self.assertNotIn("page_index=0", json.dumps(reasoner_calls[1]))
        self.assertEqual(sum(p["type"]=="image_url" for p in reasoner_calls[1]["messages"][1]["content"]), 1)
        history = [body for role, body in requests if role=="locator"][-1]["messages"]
        self.assertNotIn("OCR_SENTINEL", json.dumps(history))
        self.assertEqual(sum(p["type"]=="image_url" for m in history if m["role"]=="tool"
                             for p in (m["content"] if isinstance(m["content"],list) else [])), 2)
        self.assertNotIn("selector", audit.requests)

    def test_unread_evidence_can_be_corrected_by_locator(self):
        calls = []
        def handler(req):
            body=json.loads(req.content);role=role_of(body);calls.append((role,body))
            if role=="locator":
                step=sum(m["role"]=="tool" for m in body["messages"])
                name=["call_reasoner","read_page","call_reasoner"][step]
                return httpx.Response(200,json=reply(tool(name,{"pages":[0]},"id"+str(step)),"tool_calls"))
            return httpx.Response(200,json=reply({"role":"assistant","content":'{"status":"answerable","answer":"red"}'}))
        result,audit=self.run_agent(handler)
        self.assertEqual(result["execution_status"],"completed")
        self.assertEqual(audit.requests["reasoner"],1)
        second=[body for role,body in calls if role=="locator"][1]
        self.assertIn("invalid_evidence_pages",json.dumps(second))

    def test_schema_has_no_selector_and_requires_explicit_pages(self):
        from agent.tools import schemas
        from agent.config import ROLES
        definitions={s["function"]["name"]:s["function"] for s in schemas(3)}
        self.assertEqual(set(definitions),{"search","read_page","call_reasoner"})
        self.assertEqual(definitions["call_reasoner"]["parameters"]["required"],["pages"])
        self.assertNotIn("selector",ROLES)
        self.assertFalse((ROOT/"agent/selector.py").exists())
        self.assertFalse((ROOT/"agent/prompts/selector.txt").exists())

    def test_old_protocol_output_cannot_resume_as_new_mode(self):
        output=self.root/"old_protocol"
        run=prepare(output,self.config,select_cases(self.config["samples"],"0"))
        run["schema"]=1
        with self.assertRaisesRegex(ValueError,"different Agent protocol"):
            verify(output,run,self.config)

    def test_root_has_no_scripts_and_compat_runs_from_another_directory(self):
        self.assertFalse([p.name for p in ROOT.iterdir() if p.suffix in {".py",".sh"}])
        command=[sys.executable,str(ROOT/"compat/test_vllm_cache.py"),"--help"]
        result=subprocess.run(command,cwd=self.root,capture_output=True,text=True,timeout=15)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)


    def test_premature_locator_answer_routes_to_independent_reasoner(self):
        requests = []
        def handler(req):
            body = json.loads(req.content); role = role_of(body); requests.append((role, body))
            if role == "locator":
                turns = sum(1 for r, _ in requests if r == "locator")
                if turns == 1:
                    return httpx.Response(200, json=reply(tool("read_page", {"pages": [0]}, "read"), "tool_calls"))
                if turns == 2:
                    return httpx.Response(200, json=reply({"role":"assistant","content":"WRONG_LOCATOR_DRAFT"}))
                self.assertEqual(body["tool_choice"], {"type":"function","function":{"name":"call_reasoner"}})
                # vLLM named tool calls may legitimately have finish_reason=stop.
                return httpx.Response(200, json=reply(tool("call_reasoner", {"pages":[0]}, "model_selected_pages"), "stop"))
            self.assertEqual(role, "reasoner")
            self.assertNotIn("WRONG_LOCATOR_DRAFT", json.dumps(body))
            self.assertIn("OCR_SENTINEL", json.dumps(body))
            return httpx.Response(200, json=reply({"role":"assistant","content":'{"status":"answerable","answer":"red"}'}))
        result, audit = self.run_agent(handler)
        self.assertEqual(result["raw_answer"], "red")
        self.assertEqual(result["stop_reason"], "reasoner_answerable")
        self.assertEqual(result["protocol_corrections"], 1)
        self.assertEqual(result["protocol_correction_rounds"], 1)
        self.assertEqual(result["locator_non_tool_responses"][0]["content"], "WRONG_LOCATOR_DRAFT")
        self.assertEqual(result["controller_fallback_calls"], 0)
        self.assertEqual(audit.requests["locator"], 3)
        self.assertEqual(audit.requests["reasoner"], 1)
        self.assertEqual(audit.costs()["http_requests"], 4)
        self.assertEqual(requests[0][1]["tool_choice"], "required")
        events=[json.loads(line) for line in audit.path.read_text().splitlines()]
        correction=next(e for e in events if e["event"]=="protocol_correction")
        self.assertEqual(correction["source"], "controller")
        history=next(e for e in reversed(events) if e["event"]=="history_checkpoint")["messages"]
        validate_history(history)

    def test_plain_answer_never_accepted_when_server_ignores_tool_requirement(self):
        result, audit = self.run_agent(lambda req: httpx.Response(200, json=reply(
            {"role":"assistant","content":"red"})))
        self.assertEqual(result["execution_status"], "incomplete")
        self.assertEqual(result["stop_reason"], "premature_final")
        self.assertIsNone(result["raw_answer"])
        self.assertEqual(result["protocol_corrections"], 2)
        self.assertEqual(result["protocol_correction_rounds"], 2)
        self.assertEqual(audit.requests["locator"], 3)
        self.assertEqual(audit.requests["reasoner"], 0)

    def test_protocol_correction_respects_last_round_budget(self):
        self.config["max_rounds"] = 1
        result, audit = self.run_agent(lambda req: httpx.Response(200, json=reply(
            {"role":"assistant","content":"red"})))
        self.assertEqual(result["stop_reason"], "premature_final")
        self.assertEqual(result["protocol_corrections"], 0)
        self.assertEqual(audit.requests["locator"], 1)

    def test_protocol_correction_can_be_disabled_for_comparison(self):
        self.config["max_protocol_corrections"] = 0
        self.config["locator_tool_choice"] = "auto"
        seen = []
        def handler(req):
            seen.append(json.loads(req.content))
            return httpx.Response(200, json=reply({"role":"assistant","content":"red"}))
        result, audit = self.run_agent(handler)
        self.assertEqual(seen[0]["tool_choice"], "auto")
        self.assertEqual(result["protocol_corrections"], 0)
        self.assertIsNone(result["raw_answer"])
        self.assertEqual(audit.requests["locator"], 1)

    def test_correction_without_read_pages_requires_evidence_first(self):
        requests = []
        def handler(req):
            body=json.loads(req.content);role=role_of(body);requests.append((role,body))
            if role=="locator":
                turn=sum(r=="locator" for r,b in requests)
                if turn==1:
                    return httpx.Response(200,json=reply({"role":"assistant","content":"I know the answer"}))
                if turn==2:
                    self.assertEqual(body["tool_choice"],"required")
                    return httpx.Response(200,json=reply(tool("read_page",{"pages":[0]},"read"),"tool_calls"))
                return httpx.Response(200,json=reply(tool("call_reasoner",{"pages":[0]},"reason"),"tool_calls"))
            return httpx.Response(200,json=reply({"role":"assistant","content":'{"status":"answerable","answer":"red"}'}))
        result,audit=self.run_agent(handler)
        self.assertEqual(result["stop_reason"],"reasoner_answerable")
        self.assertEqual(result["protocol_corrections"],1)
        self.assertEqual(audit.requests["reasoner"],1)

    def test_correction_configuration_validated(self):
        for key, value in [("locator_tool_choice","none"),("max_protocol_corrections",-1),
                           ("max_protocol_corrections",True),("max_protocol_corrections",5)]:
            config=copy.deepcopy(self.config);config[key]=value
            with self.assertRaises(ValueError):validate(config)

class Integration(Base):
    def setUp(self):
        super().setUp();self.fixture=Fixture();self.processes=[]
        for spec in self.config["roles"].values():spec["url"]=self.fixture.url
        self.config_path=self.root/"config.json";self.config_path.write_text(json.dumps(self.config))
    def tearDown(self):
        self.fixture.release.set()
        for process in self.processes:
            if process.poll() is None:process.kill();process.wait()
        self.fixture.close();super().tearDown()
    def command(self,extra=()):
        return [sys.executable,"-m","agent","eval","--config",str(self.config_path),
                "--no-serve","--output-dir",str(self.root/"output"),*extra]
    def execute(self,extra=()):
        proc=subprocess.run(self.command(extra),cwd=ROOT,capture_output=True,text=True,timeout=45)
        self.assertEqual(proc.returncode,0,proc.stdout[-3000:]+proc.stderr[-3000:])
        return proc
    def test_full_eval_resume_and_gpu_count_change(self):
        self.execute(["--gpu-ids","1,4","--concurrency-per-gpu","2"])
        summary=read(self.root/"output/summary.json");self.assertEqual(summary["scored"],4)
        self.assertEqual(summary["accuracy"],1.0)
        before=len(self.fixture.requests)
        self.execute(["--gpu-ids","5"])
        self.assertEqual(len(self.fixture.requests),before)
        self.assertTrue(read(self.root/"output/run_status.json")["status"]=="finished")

    def test_failed_extraction_retry_reuses_qa(self):
        self.fixture.fail_extract=True
        self.execute(["--indices","0"])
        row=read(result_path(self.root/"output",0))
        self.assertEqual((row["execution_status"],row["scoring_status"]),("completed","error"))
        before=sum(r=="locator" for r,b in self.fixture.requests)
        self.fixture.fail_extract=False;self.execute(["--retry-failed"])
        self.assertEqual(sum(r=="locator" for r,b in self.fixture.requests),before)
        self.assertEqual(read(result_path(self.root/"output",0))["score"],1.0)

    def test_interrupt_preserves_qa_and_resumes_extraction(self):
        self.fixture.hold_extract=True
        log=self.root/"interrupt.log"
        with log.open("w") as stream:
            process=subprocess.Popen(self.command(["--indices","0"]),cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            self.processes.append(process)
            self.assertTrue(self.fixture.extract_started.wait(25),log.read_text())
            self.assertEqual(read(result_path(self.root/"output",0))["phase"],"agent_saved")
            process.send_signal(signal.SIGINT)
            self.assertEqual(process.wait(timeout=25),130,log.read_text())
        row=read(result_path(self.root/"output",0));self.assertEqual(row["raw_answer"],"red")
        before=sum(r=="locator" for r,b in self.fixture.requests)
        self.fixture.hold_extract=False;self.fixture.release.set()
        self.execute()
        self.assertEqual(sum(r=="locator" for r,b in self.fixture.requests),before)
        self.assertEqual(read(result_path(self.root/"output",0))["scoring_status"],"scored")


    def test_existing_service_ports_route_to_distinct_replicas(self):
        second=Fixture()
        try:
            self.execute(["--ports",str(self.fixture.server.server_port)+","+str(second.server.server_port)])
            self.assertTrue(self.fixture.requests)
            self.assertTrue(second.requests)
            self.assertEqual(read(self.root/"output/summary.json")["scored"],4)
        finally:second.close()

    def test_package_runs_without_old_code_or_repository(self):
        import shutil
        isolated=self.root/"checkout";isolated.mkdir()
        shutil.copytree(ROOT/"agent",isolated/"agent",ignore=shutil.ignore_patterns("__pycache__"))
        command=[sys.executable,"-m","agent","eval","--config",str(self.config_path),"--no-serve",
                 "--indices","0","--output-dir",str(self.root/"isolated_output")]
        proc=subprocess.run(command,cwd=isolated,capture_output=True,text=True,timeout=30)
        self.assertEqual(proc.returncode,0,proc.stdout+proc.stderr)
        self.assertEqual(read(self.root/"isolated_output/summary.json")["accuracy"],1.0)

    def test_changed_image_is_a_technical_failure_not_missing_evidence(self):
        root=self.root/"output";prepare(root,self.config,select_cases(self.config["samples"],"0"))
        Image.new("RGB",(64,64),"blue").save(self.data/"images_1024/doc/page_0.png")
        self.execute()
        row=read(result_path(root,0))
        self.assertEqual(row["stop_reason"],"input_asset_changed")
        self.assertEqual(row["score"],0.0)
        self.assertFalse(self.fixture.requests)

    def test_current_gt_never_sent_to_any_role(self):
        self.cases[0]["answer"]="UNIQUE_GT_MUST_NOT_LEAK_8472"
        (self.data/"samples.json").write_text(json.dumps(self.cases))
        self.execute(["--indices","0"])
        self.assertNotIn("UNIQUE_GT_MUST_NOT_LEAK_8472",json.dumps(self.fixture.requests))

if __name__=="__main__":
    unittest.main(verbosity=2)
