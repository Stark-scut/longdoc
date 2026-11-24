"""Event-driven tests for cold-prefix gating, shared request limits and resume."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import json
import tempfile
import threading
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluations/direct_qa"))
from document_scheduler import PageContentCache, run_document_pipeline
from compare_reproducibility import compare


def rows(documents=3, questions=2):
    return [{"doc_id": str(doc), "_sample_index": doc * questions + q, "q": q}
            for doc in range(documents) for q in range(questions)]


class PipelineTests(unittest.TestCase):
    def start_pipeline(self, records, generate, extract=None, **options):
        errors, completed = [], []
        stopping = threading.Event()
        config = dict(concurrency=4, document_window=2, warmup_concurrency=2,
                      extraction_concurrency=1, token_budget=0,
                      document_cost=lambda doc, group: 1, warmup=True,
                      stopping=stopping, completed=lambda row: completed.append(row["_sample_index"]))
        config.update(options)

        def run():
            try:
                with ThreadPoolExecutor(max_workers=config["concurrency"]) as pool:
                    run_document_pipeline(records, pool, generate, extract, **config)
            except BaseException as exc:
                errors.append(exc)
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        return thread, completed, errors, stopping

    def test_parallel_warmups_unlock_siblings_and_next_pdf_before_extraction(self):
        started = [threading.Event(), threading.Event()]
        sibling, third, release = threading.Event(), threading.Event(), threading.Event()
        counts = {"total": 0, "cold": 0, "extract": 0}
        peaks = dict(counts)
        lock = threading.Lock()

        def enter(kind):
            with lock:
                counts["total"] += 1
                counts[kind] += 1
                for key, value in counts.items():
                    peaks[key] = max(peaks[key], value)

        def leave(kind):
            with lock:
                counts["total"] -= 1
                counts[kind] -= 1

        def generate(row):
            cold = row["q"] == 0
            enter("cold" if cold else "other")
            try:
                doc = int(row["doc_id"])
                if cold and doc < 2:
                    started[doc].set()
                    self.assertTrue(started[1 - doc].wait(3), "warmups were serialized")
                if not cold:
                    sibling.set()
                if doc == 2:
                    third.set()
                row["response"] = "answer"
            finally:
                leave("cold" if cold else "other")

        def extract(row):
            enter("extract")
            try:
                self.assertTrue(release.wait(5), "extraction gate was not released")
            finally:
                leave("extract")

        counts["other"] = peaks["other"] = 0
        thread, completed, errors, _ = self.start_pipeline(rows(), generate, extract)
        try:
            self.assertTrue(sibling.wait(3), "siblings waited for extraction")
            self.assertTrue(third.wait(3), "next PDF waited for extraction to drain")
        finally:
            release.set()
            thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(sorted(completed), list(range(6)))
        self.assertEqual(peaks["cold"], 2)
        self.assertLessEqual(peaks["total"], 4)
        self.assertEqual(peaks["extract"], 1)

    def test_failed_warmup_next_question_warms_before_siblings(self):
        retry, release, sibling = threading.Event(), threading.Event(), threading.Event()

        def generate(row):
            if row["q"] == 0:
                row["status"] = "error"
                return
            if row["q"] == 1:
                retry.set()
                self.assertTrue(release.wait(5))
            else:
                sibling.set()
            row["response"] = "answer"

        thread, completed, errors, _ = self.start_pipeline(rows(1, 3), generate)
        try:
            self.assertTrue(retry.wait(3))
            self.assertFalse(sibling.is_set())
        finally:
            release.set()
            thread.join(5)
        self.assertEqual(errors, [])
        self.assertTrue(sibling.is_set())
        self.assertEqual(sorted(completed), [0, 1, 2])

    def test_token_admission_and_oversized_document_runs_alone(self):
        for first_cost in (8, 20):
            with self.subTest(first_cost=first_cost):
                first, second, release = threading.Event(), threading.Event(), threading.Event()
                def generate(row):
                    if row["doc_id"] == "0":
                        first.set()
                        self.assertTrue(release.wait(5))
                    else:
                        second.set()
                    row["response"] = "answer"
                thread, completed, errors, _ = self.start_pipeline(
                    rows(2, 1), generate, document_window=4, token_budget=10,
                    document_cost=lambda doc, group: first_cost if doc == "0" else 8)
                try:
                    self.assertTrue(first.wait(3))
                    self.assertFalse(second.is_set())
                finally:
                    release.set()
                    thread.join(5)
                self.assertEqual(errors, [])
                self.assertTrue(second.is_set())
                self.assertEqual(sorted(completed), [0, 1])

    def test_resume_extracts_saved_response_without_rewarming_it(self):
        records = rows(2, 2)
        records[0]["response"] = "saved"
        generated, extracted = [], []
        def generate(row):
            generated.append(row["_sample_index"])
            row["response"] = "new"
        def extract(row):
            extracted.append(row["_sample_index"])
        thread, completed, errors, _ = self.start_pipeline(records, generate, extract, concurrency=1)
        thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(sorted(generated), [1, 2, 3])
        self.assertEqual(sorted(extracted), [0, 1, 2, 3])
        self.assertEqual(sorted(completed), [0, 1, 2, 3])

    def test_interrupt_stops_new_work_and_preserves_saved_responses(self):
        records = rows(1, 2)
        records[0]["response"] = "saved"
        def extract(row):
            raise KeyboardInterrupt()
        thread, completed, errors, stopping = self.start_pipeline(
            records, lambda row: self.fail("should not start generation"), extract, concurrency=1)
        thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertIsInstance(errors[0], KeyboardInterrupt)
        self.assertTrue(stopping.is_set())
        self.assertEqual(records[0]["response"], "saved")
        self.assertNotIn("response", records[1])


class PageCacheTests(unittest.TestCase):
    def test_lru_byte_budget_and_oversize_entries(self):
        cache, calls = PageContentCache(2, 8), []
        def build(key):
            calls.append(key)
            return [{"text": key}]
        cache.get("aaaa", build)
        cache.get("bbbb", build)
        cache.get("aaaa", build)  # Retain A; evict B when C is admitted.
        cache.get("cccc", build)
        cache.get("bbbb", build)
        self.assertEqual(calls, ["aaaa", "bbbb", "cccc", "bbbb"])
        cache.get("too-large", build)
        cache.get("too-large", build)
        self.assertEqual(calls[-2:], ["too-large", "too-large"])
        self.assertLessEqual(cache.bytes, 8)
        self.assertLessEqual(len(cache.entries), 2)

    def test_concurrent_cache_misses_build_once_and_failure_can_retry(self):
        cache, calls = PageContentCache(4, 1024), []
        started, release = threading.Event(), threading.Event()
        def build(key):
            calls.append(key)
            started.set()
            self.assertTrue(release.wait(3))
            return [{"text": "encoded pages"}]
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(cache.get, "doc", build) for _ in range(4)]
            try:
                self.assertTrue(started.wait(3))
            finally:
                release.set()
            self.assertTrue(all(f.result() == [{"text": "encoded pages"}] for f in futures))
        self.assertEqual(calls, ["doc"])
        with self.assertRaises(OSError):
            cache.get("bad", lambda key: (_ for _ in ()).throw(OSError("missing page")))
        self.assertEqual(cache.get("bad", build), [{"text": "encoded pages"}])


class ReproducibilityTests(unittest.TestCase):
    def test_compare_checks_text_extraction_scores_and_complete_matching_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / "a.json", Path(directory) / "b.json"
            row = {"_sample_index": 0, "doc_id": "a.pdf", "question": "color?",
                   "answer": "red", "answer_format": "Str", "status": "complete",
                   "response": "red", "extracted_res": "red", "pred": "red", "score": 1.0}
            a.write_text(json.dumps([row]))
            b.write_text(json.dumps([row]))
            self.assertTrue(compare(a, b)["identical"])
            b.write_text(json.dumps([{**row, "response": "The color is red."}]))
            result = compare(a, b)
            self.assertEqual(result["differences"]["response"], [0])
            self.assertEqual(result["agreement"]["score"], 1.0)
            for changed in ({"status": "predicted"}, {"question": "different"}, {"_sample_index": 1}):
                b.write_text(json.dumps([{**row, **changed}]))
                with self.assertRaises(ValueError):
                    compare(a, b)


if __name__ == "__main__":
    unittest.main(verbosity=2)
