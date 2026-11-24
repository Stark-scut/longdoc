"""CPU-only document scheduling, image references and worker resume checks."""
from __future__ import annotations
import contextlib
import copy
import io
import json
import multiprocessing
from pathlib import Path
import shutil
import time
import unittest
from unittest.mock import patch
import httpx
from PIL import Image
from test_agent import Base, reply, role_of, tool
from agent import logging as logs
from agent.cli import main
from agent.client import Client
from agent.documents import Documents, select_cases
from agent.evaluation import Interrupted, aggregate, needs_work, prepare, result_path, worker
from agent.launcher import work_plan
from agent.logging import Assets, Audit, atomic, image_part, read, replay
from agent.scheduling import DocumentQueue, partition_documents


def consume_queue(path, slot, output):
    seen = []
    with DocumentQueue(path).work(slot) as indices:
        for index in indices:
            seen.append(index)
            time.sleep(.001 * (index % 3))
    atomic(output, seen)


class Performance(Base):
    def queue_cases(self, counts):
        cases = []
        for doc, count in counts.items():
            for _ in range(count):
                cases.append({"doc_id": doc + ".pdf", "_sample_index": len(cases)})
        return cases

    def test_document_affinity_balance_and_stable_resume(self):
        cases = self.queue_cases({"a": 9, "b": 8, "c": 5, "d": 4, "e": 2})
        pending = list(range(len(cases)))
        parts = partition_documents(cases, pending, 2)
        owner = {i: replica for replica, indices in enumerate(parts) for i in indices}
        for doc in {c["doc_id"] for c in cases}:
            self.assertEqual(len({owner[c["_sample_index"]] for c in cases if c["doc_id"] == doc}), 1)
        self.assertLessEqual(abs(len(parts[0]) - len(parts[1])), 2)
        resumed = partition_documents(cases, pending[::3], 2)
        self.assertEqual({i: replica for replica, part in enumerate(resumed) for i in part},
                         {i: owner[i] for i in pending[::3]})
        reordered = partition_documents(list(reversed(cases)), pending, 2)
        self.assertEqual({i: replica for replica, part in enumerate(reordered) for i in part}, owner)

    def test_sample_strategy_keeps_previous_round_robin_assignment(self):
        cases = self.queue_cases({"a": 12})
        parts, assignments = work_plan(cases, list(range(12)), 2, 3, "sample")
        self.assertEqual(parts, [list(range(0, 12, 2)), list(range(1, 12, 2))])
        self.assertEqual(assignments["0:0"], [0, 6])
        self.assertEqual(assignments["1:2"], [5, 11])

    def test_rolling_window_and_slow_tail_do_not_block_new_pdfs(self):
        cases = self.queue_cases({"a": 3, "b": 3, "c": 3})
        queue = DocumentQueue.create(self.root / "queue.json", cases, list(range(9)), 2)
        claimed = [queue.claim(i)[1] for i in range(4)]
        self.assertEqual(claimed, [0, 3, 1, 4])
        self.assertEqual(queue.claim(4), ("task", 2))
        self.assertEqual(queue.claim(5), ("task", 5))
        with self.assertRaises(ValueError):
            queue.claim(0)
        queue.finish(4)
        # A's slow questions remain in flight, but a free worker starts C now.
        self.assertEqual(queue.claim(4), ("task", 6))
        queue.finish(4)
        with queue.work(4) as indices:
            rest = list(indices)
        self.assertEqual(rest, [7, 8])
        # Exhausted workers can exit while the slow questions finish elsewhere.
        self.assertEqual(queue.claim(4), ("done", None))
        for slot in (0, 1, 2, 3, 5):
            queue.finish(slot)
        state = read(queue.path)
        self.assertEqual(state["completed"], 9)
        self.assertFalse(state["inflight"])

    def test_multiple_processes_claim_every_question_exactly_once(self):
        cases = self.queue_cases({str(i): 11 for i in range(8)})
        queue = DocumentQueue.create(self.root / "queue.json", cases, list(range(88)), 3)
        ctx = multiprocessing.get_context("spawn")
        workers = [ctx.Process(target=consume_queue,
                              args=(str(queue.path), slot, str(self.root / f"seen{slot}.json")))
                   for slot in range(4)]
        try:
            for process in workers:
                process.start()
            for process in workers:
                process.join(20)
                self.assertEqual(process.exitcode, 0)
            seen = [index for slot in range(4) for index in read(self.root / f"seen{slot}.json")]
            self.assertEqual(sorted(seen), list(range(88)))
            self.assertEqual(read(queue.path)["completed"], 88)
            self.assertFalse(read(queue.path)["inflight"])
        finally:
            for process in workers:
                if process.is_alive():
                    process.kill()
                if process.pid:
                    process.join(5)

    def test_reference_replay_matches_payload_without_image_copies(self):
        original = image_part(self.data / "images_1024/doc/page_0.png")
        body = {"messages": [{"role": "user", "content": [copy.deepcopy(original)]}]}
        audit = Audit(self.root / "reference/trace.jsonl", asset_mode="reference")
        with patch("agent.logging.base64.b64decode", side_effect=AssertionError("unnecessary decode")):
            audit.emit("llm_request", body=body)
            audit.emit("llm_request", body=body)
        files = list(audit.assets.directory.iterdir())
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0].suffix, ".json")
        self.assertEqual(read(files[0])["path"], str((self.data / "images_1024/doc/page_0.png").resolve()))
        target = self.root / "restored.json"
        replay(audit.path, 1, target)
        self.assertEqual(read(target), body)
        Image.new("RGB", (64, 64), "blue").save(self.data / "images_1024/doc/page_0.png")
        with self.assertRaises(ValueError):
            replay(audit.path, 1, target)

    def test_reference_mode_can_replay_legacy_copies_and_untracked_data(self):
        assets = Assets(self.root / "assets", "reference")
        # External plain data URLs lack source paths and keep portable copies.
        original = str(image_part(self.data / "images_1024/doc/page_0.png")["image_url"]["url"])
        packed = assets.pack(original)
        self.assertEqual(assets.unpack(packed), original)
        self.assertTrue((assets.directory / packed[8:]).is_file())

    def test_image_cache_is_bounded_and_invalidates_on_source_changes(self):
        logs._images.clear(); logs._image_bytes = 0
        with patch("agent.logging._IMAGE_CACHE_BYTES", 600):
            first_path = self.data / "images_1024/doc/page_0.png"
            first = image_part(first_path)
            self.assertIs(image_part(first_path)["image_url"]["url"], first["image_url"]["url"])
            image_part(self.data / "images_1024/doc/page_1.png")
            self.assertLessEqual(logs._image_bytes, 600)
            Image.new("RGB", (64, 64), "blue").save(first_path)
            self.assertNotEqual(first, image_part(first_path))
            self.assertLessEqual(logs._image_bytes, 600)

    def test_lazy_document_cache_preserves_evidence_and_search(self):
        shutil.copytree(self.data / "images_1024/doc", self.data / "images_1024/other")
        shutil.copytree(self.data / "visual_crops/doc", self.data / "visual_crops/other")
        path = self.data / "stage2_blocks_cache.jsonl"
        rows = path.read_text()
        path.write_text(rows + rows.replace("/doc/", "/other/"))
        docs = Documents(self.config, ["doc", "other"], max_cached=1)
        self.assertFalse(docs.documents)
        first = docs["doc"]
        self.assertEqual(first.render([0]), self.documents["doc"].render([0]))
        self.assertEqual(first.search("Colors", 2), self.documents["doc"].search("Colors", 2))
        self.assertIs(docs["doc"], first)
        docs["other"]
        self.assertEqual(list(docs.documents), ["other"])
        self.assertIsNot(docs["doc"], first)
        self.assertEqual(docs["doc"].outline(), first.outline())

    def worker_run(self, root, cases, pending, handler, run_id):
        queue = DocumentQueue.create(root / f"queues/{run_id}.json", cases, pending, 2)
        atomic(root / "runtime.json", {"run_id": run_id, "local_servers": False, "map_replicas": False,
               "ports": [8071], "stage": "all", "retry_failed": True, "asset_mode": "reference",
               "assignments": {"0:0": pending[::2], "0:1": pending[1::2]},
               "queues": {"0": str(queue.path.relative_to(root))}, "replica_indices": {"0": pending}})
        def client(config, audit):
            return Client(config, audit, httpx.MockTransport(handler))
        with patch("agent.evaluation.Client", side_effect=client), patch("agent.evaluation.signal.signal"), \
                contextlib.redirect_stdout(io.StringIO()):
            return worker(root, 0, 0)

    def handler(self, bodies, interrupt_extract=False):
        def handle(request):
            body = json.loads(request.content); bodies.append(body)
            role = role_of(body)
            if role == "locator":
                turn = sum(m["role"] == "tool" for m in body["messages"])
                name = "read_page" if turn == 0 else "call_reasoner"
                return httpx.Response(200, json=reply(tool(name, {"pages": [0]}, f"call{turn}"), "tool_calls"))
            if role == "reasoner":
                return httpx.Response(200, json=reply({"role": "assistant", "content":
                                                     '{"status":"answerable","answer":"red"}'}))
            if interrupt_extract:
                raise Interrupted()
            return httpx.Response(200, json=reply({"role": "assistant", "content": "Extracted answer: red\nAnswer format: String"}))
        return handle

    def test_worker_drains_shared_queue_and_preserves_wire_payload_and_results(self):
        root = self.root / "output"
        cases = select_cases(self.config["samples"])
        prepare(root, self.config, cases)
        bodies = []
        # One worker drains four questions, including the other slot's initial list.
        self.assertEqual(self.worker_run(root, cases, list(range(4)), self.handler(bodies), "first"), 0)
        summary = aggregate(root)
        self.assertEqual(summary["scored"], 4)
        self.assertEqual(summary["accuracy"], 1)
        self.assertTrue(all(b["temperature"] == 0 and b["seed"] == 42 for b in bodies))
        self.assertEqual({role_of(b) for b in bodies}, {"locator", "reasoner", "extractor"})
        self.assertTrue(all(p.suffix == ".json" for p in (root / "assets").iterdir()))
        row = read(result_path(root, 0))
        events = [json.loads(s) for s in Path(row["agent_trace"]).read_text().splitlines()]
        event = next(e for e in events if e["event"] == "llm_request" and "asset://" in json.dumps(e))
        replay(row["agent_trace"], event["sequence"], self.root / "replayed.json")
        self.assertIn(read(self.root / "replayed.json"), bodies)
        before = len(bodies)
        self.assertEqual(self.worker_run(root, cases, list(range(4)), self.handler(bodies), "resume"), 0)
        self.assertEqual(len(bodies), before)

    def test_interrupted_extraction_reuses_saved_answer_on_resume(self):
        root = self.root / "output"; cases = select_cases(self.config["samples"], "0")
        prepare(root, self.config, cases)
        bodies = []
        self.assertEqual(self.worker_run(root, cases, [0], self.handler(bodies, True), "first"), 130)
        row = read(result_path(root, 0))
        self.assertEqual(row["raw_answer"], "red")
        self.assertTrue(needs_work(row, "all", True))
        before = len([b for b in bodies if role_of(b) == "locator"])
        self.assertEqual(self.worker_run(root, cases, [0], self.handler(bodies), "resume"), 0)
        self.assertEqual(len([b for b in bodies if role_of(b) == "locator"]), before)
        self.assertEqual(read(result_path(root, 0))["score"], 1)

    def test_dry_run_reports_plan_and_new_settings_without_writes(self):
        path = self.root / "config.json"; atomic(path, self.config)
        output = self.root / "dry_run"
        with contextlib.redirect_stdout(io.StringIO()) as stream:
            code = main(["eval", "--config", str(path), "--gpu-ids", "0,1,2,3", "--dry-run",
                         "--output-dir", str(output), "--concurrency-per-gpu", "16",
                         "--max-num-seqs", "16", "--mm-processor-cache-gb", "3"])
        self.assertEqual(code, 0)
        plan = json.loads(stream.getvalue())
        self.assertEqual(plan["shard_strategy"], "document")
        self.assertEqual(plan["asset_mode"], "reference")
        self.assertEqual(plan["config"]["server"]["mm_processor_cache_gb"], 3)
        self.assertEqual(plan["concurrency_per_gpu"], 16)
        self.assertEqual(sum(r["documents"] for r in plan["replicas"]), 1)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
