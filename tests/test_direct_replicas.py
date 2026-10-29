"""Direct-QA multi-replica tests: mock services, real evaluator processes, no GPUs."""
import argparse
import contextlib
import http.server
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
DIRECT=ROOT/"evaluations/direct_qa"
sys.path.insert(0,str(DIRECT))
import run_replicas as runner

REAL_POPEN=subprocess.Popen

class Service:
    def __init__(self):
        self.bodies=[];self.reject_images_above=None;self.prompt_tokens_by_images={}
        lock=threading.Lock();bodies=self.bodies;service=self
        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def send(self,value,status=200):
                data=json.dumps(value).encode()
                self.send_response(status);self.send_header("Content-Type","application/json")
                self.send_header("Content-Length",str(len(data)));self.end_headers();self.wfile.write(data)
            def do_GET(self):self.send({"data":[{"id":"qwen35-mmlongbench"}]})
            def do_POST(self):
                body=json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                with lock:bodies.append(body)
                images=sum(v.get("type")=="image_url" for m in body["messages"]
                           for v in (m["content"] if isinstance(m.get("content"),list) else []))
                if images and service.reject_images_above is not None and images > service.reject_images_above:
                    self.send({"error":"maximum context length exceeded"},400);return
                prompt_tokens=service.prompt_tokens_by_images.get(images,20)
                answer="red" if images else "Extracted answer: red\nAnswer format: String"
                self.send({"choices":[{"message":{"role":"assistant","content":answer},"finish_reason":"stop"}],
                           "usage":{"prompt_tokens":prompt_tokens,"completion_tokens":5,"total_tokens":prompt_tokens+5}})
        self.server=http.server.ThreadingHTTPServer(("127.0.0.1",0),Handler)
        self.server.daemon_threads=True
        self.port=self.server.server_port
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
    def close(self):
        self.server.shutdown();self.server.server_close();self.thread.join(3)

class DirectReplicas(unittest.TestCase):
    def setUp(self):
        from PIL import Image
        self.tmp=tempfile.TemporaryDirectory(prefix="direct_replicas_")
        self.root=Path(self.tmp.name);self.model=self.root/"model";self.model.mkdir()
        (self.model/"config.json").write_text(json.dumps({"text_config":{"max_position_embeddings":262144}}))
        self.images=self.root/"images";doc=self.images/"doc";doc.mkdir(parents=True)
        for i in range(2):Image.new("RGB",(64,64),["red","white"][i]).save(doc/f"page_{i}.png")
        self.samples=[{"doc_id":"doc.pdf","question":f"Question {i}: what color?","answer":"red","answer_format":"Str",
                       "evidence_pages":"[1]","evidence_sources":"['Figure']","doc_type":"Academic paper"} for i in range(8)]
        self.input=self.root/"samples.json";self.input.write_text(json.dumps(self.samples))
        self.output=self.root/"out";self.services=[]
    def tearDown(self):
        for service in self.services:service.close()
        self.tmp.cleanup()
    def args(self,gpus="0,1,2,3"):
        return ["--gpu-ids",gpus,"--model-path",str(self.model),"--input",str(self.input),
                "--images-dir",str(self.images),"--output-dir",str(self.output),
                "--concurrency","8","--max-num-seqs","8","--enable-prefix-caching","--progress-interval",".05"]
    def execute(self,args):
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            return runner.main(args)

    def test_four_gpu_dry_run_has_four_tp1_shards_and_native_context(self):
        out=io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(runner.main(self.args("1,4,5,7")+["--dry-run"]),0)
        text=out.getvalue()
        self.assertEqual(text.count("--num-shards 4"),4)
        self.assertIn("max_model_len=262144",text)
        for i in range(4):self.assertIn("--shard-index "+str(i),text)
        self.assertFalse(self.output.exists())

    def test_one_and_three_gpu_plans_and_invalid_ports(self):
        for gpus,n in [("6",1),("1,4,7",3)]:
            out=io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(runner.main(self.args(gpus)+["--dry-run"]),0)
            self.assertEqual(out.getvalue().count("--num-shards "+str(n)),n)
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):runner.main(self.args()+["--ports","8047,8048"])
            with self.assertRaises(SystemExit):runner.main(self.args("1,1")+["--dry-run"])

    def test_full_four_replica_run_merge_resume_and_environment(self):
        self.services=[Service() for _ in range(4)]
        ports=",".join(str(s.port) for s in self.services)
        commands=self.args()+["--ports",ports]
        launches=[]
        class Dummy:
            pid=987654321
            def poll(self):return None
        def popen(command,**kwargs):
            if command[0]=="bash":
                launches.append(kwargs["env"]);return Dummy()
            return REAL_POPEN(command,**kwargs)
        original_stop=runner.stop_processes
        def stop(processes):
            original_stop([p for p in processes if isinstance(p,REAL_POPEN)])
        with patch.object(runner.subprocess,"Popen",side_effect=popen),\
             patch.object(runner,"port_available",return_value=True),\
             patch.object(runner,"stop_processes",side_effect=stop):
            self.assertEqual(self.execute(commands),0)
            before=sum(len(s.bodies) for s in self.services)
            self.assertEqual(self.execute(commands),0)
            self.assertEqual(sum(len(s.bodies) for s in self.services),before)
        self.assertEqual(before,16)
        self.assertEqual({env["GPU_IDS"] for env in launches},{"0","1","2","3"})
        for env in launches:
            self.assertEqual(env["TENSOR_PARALLEL_SIZE"],"1")
            self.assertEqual(env["MAX_NUM_SEQS"],"8")
            self.assertEqual(env["ENABLE_PREFIX_CACHING"],"1")
            self.assertEqual(env["MAX_MODEL_LEN"],"262144")
            self.assertEqual(env["CPU_OFFLOAD_GB"],"0")
            self.assertEqual(env["PYTHON_BIN"],sys.executable)
        summary=json.loads((self.output/"merged.summary.json").read_text())
        self.assertEqual(summary["total"],8);self.assertEqual(summary["complete"],8)
        self.assertEqual(summary["accuracy_percent"],100)
        self.assertEqual(json.loads((self.output/"progress.json").read_text())["processed"],8)
        for i in range(4):
            rows=json.loads((self.output/f"shard{i}.json").read_text())
            self.assertEqual([r["_sample_index"] for r in rows],[i,i+4])
            self.assertTrue(self.services[i].bodies)
        for service in self.services:
            generations=[body for body in service.bodies if isinstance(body["messages"][0]["content"],list)]
            self.assertEqual(len(generations),2)
            for body in generations:
                self.assertNotIn("tools",body)
                self.assertFalse(body["chat_template_kwargs"]["enable_thinking"])
                images=sum(v.get("type")=="image_url" for m in body["messages"]
                           for v in (m["content"] if isinstance(m.get("content"),list) else []))
                self.assertEqual(images,2)
        self.assertEqual(self.execute(["--merge-only","--output-dir",str(self.output)]),0)

    def run_one(self, extra=(), reject=None, prompt_tokens=None):
        service=Service();self.services.append(service)
        service.reject_images_above=reject
        service.prompt_tokens_by_images=prompt_tokens or {}
        command=self.args("0")+["--ports",str(service.port),"--limit","1"]+list(extra)
        class Dummy:
            pid=987654321
            def poll(self):return None
        def popen(command,**kwargs):
            return Dummy() if command[0]=="bash" else REAL_POPEN(command,**kwargs)
        original_stop=runner.stop_processes
        with patch.object(runner.subprocess,"Popen",side_effect=popen), \
             patch.object(runner,"port_available",return_value=True), \
             patch.object(runner,"stop_processes",side_effect=lambda ps:original_stop([p for p in ps if isinstance(p,REAL_POPEN)])):
            self.assertEqual(self.execute(command),0)
            row=json.loads((self.output/"shard0.json").read_text())[0]
            before=len(service.bodies)
            self.assertEqual(self.execute(command),0)
            self.assertEqual(len(service.bodies),before)
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                self.execute(command+["--input-token-limit","1900"])
        self.assertEqual(self.execute(["--merge-only","--output-dir",str(self.output)]),0)
        return row,service.bodies

    def test_prefix_budget_preserves_only_contiguous_whole_pages(self):
        e=runner.evaluation;paths=tuple(Path(str(i)) for i in range(8))
        with patch.object(e,"page_visual_tokens",return_value=(60000,)*8):
            kept,meta=e.prefix_pages(paths,"Question?",262144,8192,260000)
        self.assertEqual(kept,paths[:4]);self.assertEqual(meta["kept_page_indices"],[0,1,2,3])
        self.assertEqual(meta["input_token_limit"],253952)
        self.assertEqual(meta["original_pages"],8);self.assertEqual(meta["dropped_pages"],4)
        self.assertLessEqual(meta["estimated_prompt_tokens_with_reserve"],253952)
        with patch.object(e,"page_visual_tokens",return_value=(60000,)*8):
            with self.assertRaises(ValueError):e.prefix_pages(paths,"Question?",262144,8192,1000)

    def test_end_to_end_whole_page_truncation(self):
        from PIL import Image
        for path in (self.images/"doc").glob("*.png"):
            Image.new("RGB",(1024,1024),"red").save(path)
        row,bodies=self.run_one(["--input-token-limit","2000"])
        meta=row["input_truncation"]
        self.assertTrue(meta["applied"])
        self.assertEqual((meta["original_pages"],meta["kept_pages"],meta["dropped_pages"]),(2,1,1))
        self.assertEqual(meta["kept_page_indices"],[0]);self.assertEqual(meta["actual_prompt_tokens"],20)
        self.assertEqual(row["page_count"],1);self.assertEqual(row["original_page_count"],2)
        images=[v for v in bodies[0]["messages"][0]["content"] if v["type"]=="image_url"]
        self.assertEqual(len(images),1)
        self.assertEqual(json.loads((self.output/"merged.summary.json").read_text())["truncated_inputs"],1)

    def test_context_overflow_trims_tail_and_retries(self):
        row,bodies=self.run_one(reject=1)
        meta=row["input_truncation"]
        self.assertEqual(meta["context_retries"],1);self.assertEqual(meta["kept_page_indices"],[0])
        self.assertEqual(row["status"],"complete");self.assertEqual(len(bodies),3)
        self.assertEqual(row["estimated_visual_tokens"],runner.evaluation.native_page_info(tuple((self.images/"doc").glob("page_0.png")))[0])

    def test_actual_usage_above_input_cap_discards_and_retries(self):
        row,bodies=self.run_one(["--input-token-limit","2000"],prompt_tokens={2:2500,1:1000})
        meta=row["input_truncation"]
        self.assertEqual(meta["context_retries"],1);self.assertEqual(meta["actual_prompt_tokens"],1000)
        self.assertEqual(meta["discarded_generations"][0]["usage"]["prompt_tokens"],2500)
        self.assertEqual(len(bodies),3)

    def test_serve_dry_run_uses_current_python_without_conda_name(self):
        env=os.environ.copy()
        env.pop("CONDA_DEFAULT_ENV",None);env.pop("CONDA_PREFIX",None)
        env.update(DRY_RUN="1",PYTHON_BIN=sys.executable,MODEL_PATH=str(self.model),GPU_IDS="7",
                   TENSOR_PARALLEL_SIZE="1",MAX_MODEL_LEN="262144",MAX_NUM_SEQS="8",
                   ENABLE_PREFIX_CACHING="1",MM_PROCESSOR_CACHE_GB="1")
        p=subprocess.run(["bash",str(DIRECT/"serve_qwen35_mmlongbench.sh")],
                         env=env,capture_output=True,text=True,timeout=15)
        self.assertEqual(p.returncode,0,p.stdout+p.stderr)
        self.assertIn("vllm.entrypoints.openai.api_server",p.stdout)
        self.assertIn("--enable-prefix-caching",p.stdout)
        self.assertIn("--mamba-cache-mode align",p.stdout)

    def test_fresh_clone_scoring_falls_back_to_bundled_files(self):
        isolated=self.root/"checkout";folder=isolated/"evaluations/direct_qa";folder.mkdir(parents=True)
        (isolated/".gitignore").write_text("")
        (isolated/"agent/metrics").mkdir(parents=True);(isolated/"agent/prompts").mkdir()
        shutil.copy2(ROOT/"agent/metrics/mmlongbench.py",isolated/"agent/metrics/mmlongbench.py")
        shutil.copy2(ROOT/"agent/prompts/extract.txt",isolated/"agent/prompts/extract.txt")
        shutil.copy2(DIRECT/"evaluate_qwen35_mmlongbench.py",folder/"evaluate_qwen35_mmlongbench.py")
        command=[sys.executable,str(folder/"evaluate_qwen35_mmlongbench.py"),"--input",str(self.input),
                 "--images-dir",str(self.images),"--native-resolution","--max-pixels","1048576","--dry-run"]
        p=subprocess.run(command,cwd=isolated,capture_output=True,text=True,timeout=15)
        self.assertEqual(p.returncode,0,p.stdout+p.stderr)

if __name__=="__main__":unittest.main(verbosity=2)
