"""Document routing, scheduling, resume and merge using in-process fake HTTP."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluations/direct_qa"))
import run_replicas as runner
from partitioning import shard_assignments
evaluation = runner.evaluation


class DocumentRouting(unittest.TestCase):
    def setUp(self):
        from PIL import Image
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.images = self.root / "images"
        self.output = self.root / "out"
        self.output.mkdir()
        self.samples = []
        for q in range(4):
            for doc in range(3):
                self.samples.append({"doc_id": f"doc{doc}.pdf", "question": f"d{doc}_q{q}",
                    "answer": "red", "answer_format": "Str", "evidence_pages": "[1]",
                    "evidence_sources": "['Figure']", "doc_type": "Academic paper"})
        for doc in range(3):
            folder = self.images / f"doc{doc}"
            folder.mkdir(parents=True)
            Image.new("RGB", (64, 64), "red").save(folder / "page_0.png")
        self.input = self.root / "samples.json"
        self.input.write_text(json.dumps(self.samples))
        self.calls = []
        self.payloads = []
        self.lock = threading.Lock()
        test = self

        class Session:
            trust_env = False
            def post(self, url, json, **kwargs):
                generation = isinstance(json["messages"][0]["content"], list)
                question = (json["messages"][0]["content"][-1]["text"] if generation
                            else json["messages"][1]["content"])
                with test.lock:
                    test.calls.append((url, generation, question))
                    test.payloads.append(json)
                text = "red" if generation else "Extracted answer: red\nAnswer format: Str"
                return NS(ok=True, status_code=200, json=lambda: {
                    "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 20, "completion_tokens": 5}})
            def close(self):
                pass
        self.Session = Session

    def tearDown(self):
        self.tmp.cleanup()

    def evaluate(self, shard, limit=0, strategy="document", extra=()):
        args = ["--input", str(self.input), "--images-dir", str(self.images),
                "--output", str(self.output / f"shard{shard}.json"),
                "--base-url", f"http://replica{shard}/v1", "--num-shards", "2",
                "--shard-index", str(shard), "--shard-strategy", strategy,
                "--concurrency", "8", "--limit", str(limit)]
        with patch.object(evaluation.requests, "Session", self.Session), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return evaluation.main(args + list(extra))

    def test_rolling_pipeline_sampling_resume_and_seed_mismatch(self):
        extra = ["--document-window", "4", "--warmup-concurrency", "2",
                 "--extraction-concurrency", "2", "--seed", "17"]
        self.assertEqual(self.evaluate(0, extra=extra), 0)
        self.assertEqual(len(self.payloads), 16)
        for body in self.payloads:
            self.assertEqual(body["temperature"], 0.0)
            self.assertEqual(body["seed"], 17)
        metadata = json.loads((self.output / "shard0.meta.json").read_text())
        self.assertEqual(metadata["generation"]["seed"], 17)
        self.assertEqual(metadata["extraction"]["temperature"], 0.0)
        self.assertEqual(metadata["document_scheduler"]["document_window"], 4)
        self.assertEqual(self.evaluate(0, extra=extra), 0)
        self.assertEqual(len(self.payloads), 16)
        with self.assertRaises(SystemExit):
            self.evaluate(0, extra=extra + ["--seed", "18"])

    def test_rolling_predict_then_score_without_images(self):
        extra = ["--document-window", "4"]
        self.assertEqual(self.evaluate(0, extra=extra + ["--stage", "predict"]), 0)
        self.assertEqual(len(self.payloads), 8)
        for doc in self.images.iterdir():
            for image in doc.iterdir():
                image.unlink()
        self.assertEqual(self.evaluate(0, extra=extra + ["--stage", "score"]), 0)
        self.assertEqual(len(self.payloads), 16)
        self.assertTrue(all(row["status"] == "complete" for row in
                            json.loads((self.output / "shard0.json").read_text())))

    def test_rolling_failed_extraction_resume_keeps_generated_answer(self):
        base = self.Session
        failed = threading.Event()
        class FailOneExtraction(base):
            def post(self, url, json, **kwargs):
                response = super().post(url, json, **kwargs)
                if not isinstance(json["messages"][0]["content"], list) and not failed.is_set():
                    failed.set()
                    return NS(ok=False, status_code=400, text="test extraction failure")
                return response
        self.Session = FailOneExtraction
        extra = ["--document-window", "4", "--extraction-concurrency", "1"]
        self.assertEqual(self.evaluate(0, extra=extra), 1)
        self.assertEqual(sum(gen for _, gen, _ in self.calls), 8)
        rows = json.loads((self.output / "shard0.json").read_text())
        self.assertTrue(all(row.get("response") for row in rows))
        self.assertEqual(sum(row["status"] == "error" for row in rows), 1)
        self.Session = base
        self.assertEqual(self.evaluate(0, extra=extra), 0)
        self.assertEqual(sum(gen for _, gen, _ in self.calls), 8)

    def test_complete_input_mapping_stays_stable_and_never_splits_pdf(self):
        owners = shard_assignments(self.samples, 2, "document")
        self.assertEqual(owners, shard_assignments(self.samples, 2, "document"))
        for doc in range(3):
            self.assertEqual(len({owners[i] for i, s in enumerate(self.samples)
                                  if s["doc_id"] == f"doc{doc}.pdf"}), 1)
        self.assertEqual(shard_assignments(self.samples, 2), [i % 2 for i in range(12)])

    def test_warm_first_question_drain_pdf_and_keep_original_indices(self):
        self.assertEqual(self.evaluate(0), 0)
        calls = [call for call in self.calls if call[0].startswith("http://replica0")]
        # First PDF finishes its cold answer + extraction before any sibling;
        # the entire first PDF drains before the next PDF is submitted.
        self.assertTrue(calls[0][1])
        self.assertIn("d0_q0", calls[0][2])
        self.assertFalse(calls[1][1])
        self.assertIn("d0_q0", calls[1][2])
        docs = ["d0" if "d0_q" in q else "d2" for _, _, q in calls]
        self.assertEqual(docs, ["d0"] * 8 + ["d2"] * 8)
        rows = json.loads((self.output / "shard0.json").read_text())
        self.assertEqual([r["_sample_index"] for r in rows], [0, 2, 3, 5, 6, 8, 9, 11])

    def test_limit_extension_resume_merge_and_progress(self):
        for shard in range(2):
            self.assertEqual(self.evaluate(shard, limit=6), 0)
        self.assertEqual(sum(gen for _, gen, _ in self.calls), 6)
        for shard in range(2):
            self.assertEqual(self.evaluate(shard), 0)
            self.assertEqual(self.evaluate(shard), 0)
        self.assertEqual(sum(gen for _, gen, _ in self.calls), 12)
        args = NS(input=self.input, gpu_ids="0,1", output_dir=self.output, shard_strategy="document")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(runner.merge_results(args, self.samples, self.samples), 0)
        rows = json.loads((self.output / "merged.json").read_text())
        self.assertEqual([r["_sample_index"] for r in rows], list(range(12)))
        self.assertTrue(all(r["status"] == "complete" for r in rows))
        with contextlib.redirect_stderr(io.StringIO()):
            progress = runner.Progress(args, 12)
            snapshot = progress.update()
            progress.close()
        self.assertEqual([s["total"] for s in snapshot["shards"]], [8, 4])
        self.assertEqual(snapshot["complete"], 12)

    def test_shard_strategy_change_and_misrouted_rows_rejected(self):
        self.assertEqual(self.evaluate(0), 0)
        with self.assertRaises(SystemExit):
            self.evaluate(0, strategy="sample")
        rows = json.loads((self.output / "shard0.json").read_text())
        rows[0]["_sample_index"] = 1  # This question belongs to replica 1.
        (self.output / "shard0.json").write_text(json.dumps(rows))
        args = NS(input=self.input, gpu_ids="0,1", output_dir=self.output, shard_strategy="document")
        with self.assertRaises(ValueError):
            runner.merge_results(args, self.samples, self.samples)

    def test_runner_document_plan_and_empty_shard_progress(self):
        model = self.root / "model"
        model.mkdir()
        (model / "config.json").write_text(json.dumps({"max_position_embeddings": 262144}))
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(runner.main([
                "--gpu-ids", "0,1,2,3", "--model-path", str(model),
                "--input", str(self.input), "--images-dir", str(self.images),
                "--output-dir", str(self.root / "new_run"), "--shard-strategy", "document",
                "--mm-processor-cache-gb", "4", "--dry-run"]), 0)
        self.assertIn("图像预处理缓存=4.0", output.getvalue())
        self.assertEqual(output.getvalue().count("--shard-strategy document"), 4)
        self.assertEqual(output.getvalue().count("--document-warmup"), 4)
        args = NS(input=self.input, gpu_ids="0,1,2,3", output_dir=self.output,
                  shard_strategy="document")
        with contextlib.redirect_stderr(io.StringIO()):
            progress = runner.Progress(args, 12)
            snapshot = progress.update()
            progress.close()
        self.assertEqual([s["total"] for s in snapshot["shards"]], [4, 4, 4, 0])

    def test_runner_rolling_workers_sampling_environment_and_resume_without_sockets(self):
        model = self.root / "model"
        model.mkdir()
        (model / "config.json").write_text(json.dumps({"max_position_embeddings": 262144}))
        launches = []
        base = self.Session
        class Session(base):
            def get(self, *args, **kwargs):
                return NS(ok=True, json=lambda: {"data": [{"id": "qwen35-mmlongbench"}]})
        class Process:
            pid = 987654321
            def __init__(self, returncode=None):
                self.returncode = returncode
            def poll(self):
                return self.returncode
        def launch(command, **kwargs):
            if command[0] == "bash":
                launches.append(kwargs["env"])
                return Process()
            return Process(evaluation.main(command[2:]))
        args = ["--gpu-ids", "0,1", "--model-path", str(model), "--input", str(self.input),
                "--images-dir", str(self.images), "--output-dir", str(self.output),
                "--shard-strategy", "document", "--document-window", "4",
                "--concurrency", "8", "--seed", "17", "--batch-invariant"]
        with patch.object(evaluation.requests, "Session", Session), \
             patch.object(runner.subprocess, "Popen", side_effect=launch), \
             patch.object(runner, "port_available", return_value=True), \
             patch.object(runner, "stop_processes"), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(runner.main(args), 0)
            count = len(self.payloads)
            self.assertEqual(runner.main(args + ["--document-window", "1"]), 0)
            self.assertEqual(len(self.payloads), count)
            with self.assertRaises(SystemExit):
                runner.main(args + ["--seed", "18"])
        self.assertEqual(count, 24)
        self.assertTrue(all(env["MODEL_SEED"] == "17" and env["VLLM_BATCH_INVARIANT"] == "1"
                            for env in launches))
        self.assertTrue(all(body["seed"] == 17 and body["temperature"] == 0 for body in self.payloads))
        merged = json.loads((self.output / "merged.json").read_text())
        self.assertTrue(all(row["status"] == "complete" for row in merged))
        self.assertEqual(len(merged), len(self.samples))


if __name__ == "__main__":
    unittest.main(verbosity=2)
