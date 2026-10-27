#!/usr/bin/env python3
"""Offline invariant and configurable-replica HTTP fixture tests; no GPU/model."""
from __future__ import annotations
import copy
import http.server
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
sys.dont_write_bytecode = True
from support import SOURCE_CORE, DATA, read, atomic, needs_work, settled, result_path, sha

SCRIPT = Path(__file__).resolve().parent / "run_native_4gpu.py"

class ResumeRules(unittest.TestCase):
    def test_resume_agent_vs_score(self):
        self.assertTrue(needs_work(None, "all"))
        self.assertFalse(needs_work(None, "score"))
        qa = {"execution_status":"completed","raw_answer":"42","scoring_status":"pending"}
        self.assertTrue(needs_work(qa,"all"))
        self.assertFalse(needs_work(qa,"agent"))
        done = {**qa, "scoring_status":"scored"}
        self.assertFalse(needs_work(done,"all",True))
        self.assertTrue(settled(done,"all"))
        error = {"execution_status":"error","scoring_status":"not_answered"}
        self.assertFalse(needs_work(error,"all"))
        self.assertTrue(needs_work(error,"all",True))
        self.assertFalse(needs_work(error,"score",True))

    def test_semantic_change_rejected_runtime_timeout_allowed(self):
        from run_native_4gpu import parser, parameters, DEFAULTS
        previous = {"parameters":copy.deepcopy(DEFAULTS)}
        args = parser().parse_args(["--timeout","700"])
        self.assertEqual(parameters(args,previous)["timeout"],700)
        args = parser().parse_args(["--no-thinking"])
        with self.assertRaises(ValueError):
            parameters(args,previous)

    def test_configurable_replicas_and_matching_ports(self):
        from run_native_4gpu import parser, gpu_ports
        for arguments in [["--gpu-ids=1,1"], ["--gpu-ids=-1,2"], [],
                          ["--gpu-ids=1,4,5", "--ports=8071,8072"],
                          ["--gpu-ids=1,2", "--ports=8071,8071"],
                          ["--gpu-ids=1,2", "--ports=0,8072"],
                          ["--gpu-ids=1,,4"], ["--gpu-ids=1,4,"]]:
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    gpu_ports(parser().parse_args(arguments))
        for ids in [[0], [1,2], [1,4,5], [1,2,4,6], [0,1,2,3,4,5,6,7]]:
            args = parser().parse_args(["--gpu-ids", ",".join(map(str, ids))])
            self.assertEqual(gpu_ports(args), (ids, list(range(8071,8071+len(ids)))))
        args = parser().parse_args(["--gpu-ids", "1, 4, 5", "--ports", "9071,9072,9073"])
        self.assertEqual(gpu_ports(args), ([1,4,5], [9071,9072,9073]))
        args = parser().parse_args(["--no-serve", "--ports", "8071,8072"])
        self.assertEqual(gpu_ports(args, False), ([], [8071,8072]))
        self.assertEqual(gpu_ports(parser().parse_args([]), False, 2), ([], [8071,8072]))

    def test_default_caches_enabled_no_offload_tp1(self):
        from run_native_4gpu import parser,parameters,server_command
        args=parser().parse_args(["--gpu-ids","1,2,4,6"])
        command=server_command(args,parameters(args),8071)
        self.assertIn("--enable-prefix-caching",command)
        self.assertEqual(command[command.index("--mm-processor-cache-gb")+1],"1")
        for flag,value in [("--tensor-parallel-size","1"),("--cpu-offload-gb","0"),("--dtype","bfloat16")]:
            self.assertEqual(command[command.index(flag)+1],value)

    def test_prefix_caching_opt_in_and_per_request_usage_details(self):
        from run_native_4gpu import parser,parameters,server_command
        args=parser().parse_args(["--gpu-ids","0,1","--enable-prefix-caching"])
        command=server_command(args,parameters(args),8071)
        self.assertIn("--enable-prefix-caching",command)
        self.assertNotIn("--no-enable-prefix-caching",command)
        self.assertIn("--enable-prompt-tokens-details",command)

    def test_per_gpu_concurrency_alias_and_explicit_cache_opt_out(self):
        from run_native_4gpu import parser,parameters,server_command
        for option in ["--concurrency", "--concurrency-per-gpu"]:
            args=parser().parse_args(["--gpu-ids","0,1",option,"4"])
            command=server_command(args,parameters(args),8071)
            self.assertEqual(args.concurrency,4)
            self.assertEqual(command[command.index("--max-num-seqs")+1],"4")
        args=parser().parse_args(["--gpu-ids","0,1","--no-enable-prefix-caching","--mm-processor-cache-gb","0"])
        command=server_command(args,parameters(args),8071)
        self.assertIn("--no-enable-prefix-caching",command)
        self.assertNotIn("--enable-prefix-caching",command)
        self.assertEqual(command[command.index("--mm-processor-cache-gb")+1],"0.0")

class ShutdownTests(unittest.TestCase):
    def test_repeated_interrupt_and_hangup_do_not_abort_server_cleanup(self):
        program = "\n".join([
            "import sys,os,signal,time,subprocess,json",
            "sys.dont_write_bytecode=True",
            "sys.path.insert(0, " + repr(str(SCRIPT.parent)) + ")",
            "from run_native_4gpu import install_shutdown_handlers,cleanup_guard,stop_groups",
            "install_shutdown_handlers()",
            "child=subprocess.Popen([sys.executable,'-c',\"import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); print('ready',flush=True); time.sleep(60)\"],stdout=subprocess.PIPE,text=True,start_new_session=True)",
            "assert child.stdout.readline().strip()=='ready'",
            "print(child.pid,flush=True)",
            "try:",
            "    time.sleep(60)",
            "except KeyboardInterrupt:",
            "    pass",
            "finally:",
            "    with cleanup_guard():",
            "        print('cleanup',flush=True)",
            "        time.sleep(.3)",
            "        stop_groups([child],grace=.1)",
            "print('done',flush=True)",
        ])
        process=subprocess.Popen([sys.executable,"-u","-c",program],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=True)
        child_pid=None
        try:
            child_pid=int(process.stdout.readline().strip())
            os.kill(process.pid,signal.SIGINT)
            self.assertEqual(process.stdout.readline().strip(),"cleanup")
            os.kill(process.pid,signal.SIGINT)
            os.kill(process.pid,signal.SIGHUP)
            self.assertEqual(process.wait(timeout=10),0,process.stderr.read())
            self.assertEqual(process.stdout.readline().strip(),"done")
            with self.assertRaises(ProcessLookupError):os.kill(child_pid,0)
        finally:
            if process.poll() is None:
                process.kill();process.wait()
            if child_pid:
                try:os.killpg(child_pid,signal.SIGKILL)
                except ProcessLookupError:pass
            process.stdout.close();process.stderr.close()

    def test_worker_cleanup_error_does_not_skip_servers(self):
        from unittest.mock import patch
        from run_native_4gpu import cleanup_children
        workers,servers=[object()],[object()]
        with patch("run_native_4gpu.stop_groups",side_effect=[RuntimeError("synthetic worker failure"),None]) as stop:
            errors=cleanup_children(workers,servers)
        self.assertEqual(stop.call_count,2)
        self.assertIs(stop.call_args_list[1].args[0],servers)
        self.assertEqual(errors[0]["phase"],"workers")

class Fixture:
    def __init__(self):
        self.lock=threading.Lock()
        self.requests=[]
        self.hold_question=None
        self.release=threading.Event()
        self.fail_question=None
        self.failures_left=0
        fixture=self
        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def send_json(self,code,value):
                payload=json.dumps(value).encode()
                try:
                    self.send_response(code);self.send_header("Content-Type","application/json")
                    self.send_header("Content-Length",str(len(payload)));self.end_headers()
                    self.wfile.write(payload)
                except (BrokenPipeError,ConnectionResetError):pass
            def do_GET(self):
                self.send_json(200,{"data":[{"id":"citeagent-qwen35","fixture_only":True}]})
            def do_POST(self):
                body=json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                messages=body["messages"]
                if body.get("tools"):
                    role="locator"
                elif messages[0]["role"]=="user":
                    role="extractor"
                elif "全局证据选择器" in messages[0]["content"]:
                    role="selector"
                else:role="reasoner"
                with fixture.lock:
                    fixture.requests.append((role,body))
                    count=len(fixture.requests)
                if role=="extractor":
                    question=messages[-1]["content"]
                    if fixture.hold_question and fixture.hold_question in question:
                        fixture.release.wait(30)
                    with fixture.lock:
                        fail=bool(fixture.fail_question and fixture.fail_question in question and fixture.failures_left>0)
                        if fail:fixture.failures_left-=1
                    if fail:
                        self.send_json(500,{"error":"synthetic extraction failure"});return
                message={"role":"assistant","content":"","reasoning":"SYNTHETIC_REASONING"}
                finish="stop"
                if role=="locator":
                    calls=[tc["function"]["name"] for m in messages for tc in m.get("tool_calls",[])]
                    if not calls:name,args="search",{"query":"system"}
                    elif "read_page" not in calls:name,args="read_page",{"pages":[0]}
                    else:name,args="call_reasoner",{}
                    message["tool_calls"]=[{"id":f"fixture-{count}","type":"function",
                        "function":{"name":name,"arguments":json.dumps(args)}}]
                    finish="tool_calls"
                elif role=="selector":message["content"]=json.dumps({"selected_pages":[0]})
                elif role=="reasoner":message["content"]=json.dumps({"status":"answerable","answer":"42"})
                else:message["content"]="Extracted answer: 42\nAnswer format: Int"
                self.send_json(200,{"id":f"fixture-{count}","model":"SYNTHETIC_TEST_ONLY",
                    "choices":[{"finish_reason":finish,"message":message}],
                    "usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12}})
        self.servers=[http.server.ThreadingHTTPServer(("127.0.0.1",0),Handler) for _ in range(4)]
        for server in self.servers:
            server.daemon_threads=True
            threading.Thread(target=server.serve_forever,daemon=True).start()
        self.ports=",".join(str(s.server_port) for s in self.servers)
    def close(self):
        self.release.set()
        for server in self.servers:
            server.shutdown();server.server_close()

class ReplicaIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (SOURCE_CORE/"full_eval/_test_work").mkdir(exist_ok=True)
        cls.fixture=Fixture()
        cls.cases=read(DATA/"samples.json")
    @classmethod
    def tearDownClass(cls):
        cls.fixture.close()
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=SOURCE_CORE/"full_eval/_test_work",prefix="integration_")
        self.root=Path(self.tmp.name)
        self.fixture.hold_question=None;self.fixture.release.clear()
        self.fixture.fail_question=None;self.fixture.failures_left=0
        self.processes=[]
    def tearDown(self):
        self.fixture.release.set()
        for process in self.processes:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:process.wait(timeout=20)
                except subprocess.TimeoutExpired:process.kill();process.wait()
        self.tmp.cleanup()
    def command(self,name="run",extra=(),replicas=4):
        ports=",".join(str(s.server_port) for s in self.fixture.servers[:replicas])
        return [sys.executable,str(SCRIPT),"--no-serve","--ports",ports,
            "--output-dir",str(self.root/name),"--indices","281,282,283,284",
            "--max-rounds","4","--timeout","10","--wall-seconds","60",
            "--progress-interval","0.1",*extra]
    def execute(self,command):
        log=self.root/f"launcher_{len(list(self.root.glob('launcher_*.log')))}.log"
        with log.open("w") as f:
            process=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT)
            self.processes.append(process)
            code=process.wait(timeout=90)
        self.assertEqual(code,0,log.read_text()[-6000:])
        return log
    def test_four_workers_progress_merge_and_completed_resume(self):
        start=len(self.fixture.requests)
        self.execute(self.command())
        root=self.root/"run"
        summary=read(root/"summary.json")
        self.assertEqual(summary["total"],4)
        self.assertEqual(summary["processed"],4)
        self.assertEqual(summary["scored"],4)
        self.assertEqual(summary["agent_errors"],0)
        self.assertTrue(summary["run_finished"])
        self.assertTrue(read(root/"progress.json")["run_finished"])
        self.assertEqual([s["processed"] for s in summary["shards"]],[1,1,1,1])
        for case in read(root/"snapshot/cases.json"):
            r=read(result_path(root,case["_sample_index"]))
            self.assertTrue(r["protocol_valid"])
            self.assertEqual(r["raw_answer"],"42")
            events=[json.loads(s) for s in Path(r["latest_trace"]).read_text().splitlines()]
            req=[e for e in events if e["event"]=="llm_request" and e.get("role")=="locator"][-1]
            messages=req["body"]["messages"]
            self.assertEqual(sum(m["role"]=="user" for m in messages),1)
            self.assertTrue(any(p.get("type")=="image_url" for m in messages if isinstance(m.get("content"),list) for p in m["content"]))
        for asset in (root/"assets").iterdir():
            self.assertEqual(sha(asset),asset.name.split(".")[0])
        count=len(self.fixture.requests)
        self.assertGreater(count,start)
        self.execute(self.command())
        self.assertEqual(len(self.fixture.requests),count)
        # Snapshot is authoritative; launcher does not edit the source Agent.
        self.assertEqual(sha(root/"snapshot/core/agent.py"),sha(SOURCE_CORE/"agent.py"))

    def test_one_two_three_replicas_progress_and_resume(self):
        for replicas, ids in [(1,"0"), (2,"1,2"), (3,"1,4,5")]:
            with self.subTest(replicas=replicas):
                name=f"replicas_{replicas}"
                self.execute(self.command(name, replicas=replicas, extra=["--gpu-ids",ids]))
                root=self.root/name
                summary=read(root/"summary.json")
                self.assertEqual(summary["replica_count"],replicas)
                self.assertEqual(summary["processed"],4)
                self.assertEqual(summary["scored"],4)
                self.assertEqual(summary["agent_errors"],0)
                cases=read(root/"snapshot/cases.json")
                expected=[sum(c["_sample_index"]%replicas==i for c in cases) for i in range(replicas)]
                self.assertEqual([s["processed"] for s in summary["shards"]],expected)
                self.assertEqual(read(root/"runtime.json")["gpu_ids"],list(map(int,ids.split(","))))
                self.assertTrue(read(root/"progress.json")["run_finished"])
                count=len(self.fixture.requests)
                self.execute(self.command(name, replicas=replicas))
                self.assertEqual(len(self.fixture.requests),count)

    def test_two_replicas_two_concurrent_questions_each(self):
        self.execute(self.command(replicas=2,extra=["--gpu-ids","0,1","--concurrency-per-gpu","2"]))
        root=self.root/"run"
        runtime=read(root/"runtime.json")
        self.assertEqual(runtime["concurrency_per_gpu"],2)
        self.assertEqual(runtime["max_num_seqs"],2)
        self.assertTrue(runtime["enable_prefix_caching"])
        self.assertEqual(runtime["mm_processor_cache_gb"],1)
        assigned=[i for ids in runtime["assignments"].values() for i in ids]
        self.assertEqual(sorted(assigned),[281,282,283,284])
        self.assertEqual(len(assigned),len(set(assigned)))
        self.assertEqual(len(list((root/"workers").glob("shard*_slot*.log"))),4)
        self.assertEqual(read(root/"summary.json")["scored"],4)

    def test_resume_with_different_gpu_count_keeps_qa_and_snapshot(self):
        self.execute(self.command(replicas=2, extra=["--stage","agent"]))
        root=self.root/"run"
        before={str(p):sha(p) for p in (root/"snapshot").rglob("*") if p.is_file() and not p.is_symlink()}
        config_hash=sha(root/"run_config.json")
        locator_before=sum(role=="locator" for role,_ in self.fixture.requests)
        self.execute(self.command(replicas=3, extra=["--gpu-ids","1,4,5"]))
        self.assertEqual(sum(role=="locator" for role,_ in self.fixture.requests),locator_before)
        summary=read(root/"summary.json")
        self.assertEqual(summary["replica_count"],3)
        self.assertEqual(summary["scored"],4)
        self.assertEqual([s["processed"] for s in summary["shards"]],[1,1,2])
        self.assertEqual(sha(root/"run_config.json"),config_hash)
        self.assertEqual(before,{str(p):sha(p) for p in (root/"snapshot").rglob("*") if p.is_file() and not p.is_symlink()})
        self.execute(self.command(replicas=3, extra=["--merge-only"]))
        self.assertEqual(read(root/"summary.json")["replica_count"],3)

    def test_extractor_failure_retries_without_rerunning_agent(self):
        self.fixture.fail_question=self.cases[281]["question"]
        self.fixture.failures_left=2
        self.execute(self.command())
        root=self.root/"run"
        first=read(result_path(root,281))
        self.assertEqual(first["execution_status"],"completed")
        self.assertEqual(first["scoring_status"],"error")
        locator_before=sum(role=="locator" for role,_ in self.fixture.requests)
        self.execute(self.command(extra=["--retry-failed"]))
        self.assertEqual(read(result_path(root,281))["scoring_status"],"scored")
        self.assertEqual(sum(role=="locator" for role,_ in self.fixture.requests),locator_before)

    def test_ctrl_c_preserves_qa_and_resume_only_extracts(self):
        self.fixture.hold_question=self.cases[281]["question"]
        log=self.root/"interrupt.log"
        stream=log.open("w")
        process=subprocess.Popen(self.command(),stdout=stream,stderr=subprocess.STDOUT)
        self.processes.append(process)
        root=self.root/"run"
        deadline=time.monotonic()+60
        while time.monotonic()<deadline:
            path=result_path(root,281)
            if path.exists() and read(path).get("phase")=="agent_saved":break
            if process.poll() is not None:self.fail(log.read_text()[-6000:])
            time.sleep(.1)
        else:self.fail("QA checkpoint not produced")
        process.send_signal(signal.SIGINT)
        self.assertEqual(process.wait(timeout=30),130)
        stream.close()
        self.fixture.release.set()
        self.fixture.hold_question=None
        r=read(result_path(root,281))
        self.assertEqual(r["raw_answer"],"42")
        self.assertEqual(r["scoring_status"],"pending")
        original_trace=r["latest_trace"]
        self.execute(self.command())
        r=read(result_path(root,281))
        self.assertEqual(r["scoring_status"],"scored")
        self.assertNotEqual(r["latest_trace"],original_trace)
        events=[json.loads(x) for x in Path(r["latest_trace"]).read_text().splitlines()]
        self.assertFalse(any(e.get("role")=="locator" and e["event"]=="llm_request" for e in events))
        self.assertTrue(read(root/"summary.json")["run_finished"])

if __name__=="__main__":
    unittest.main(verbosity=2)
