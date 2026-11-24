"""Bounded document scheduling and client image cache; no model/scoring logic."""

from collections import OrderedDict, deque
from concurrent.futures import FIRST_COMPLETED, Future, wait
import threading


class PageContentCache:
    """LRU bounded by document count and encoded bytes, with per-key single flight."""

    def __init__(self, max_documents=1, max_bytes=512 * 1024**2):
        self.max_documents, self.max_bytes = max_documents, max_bytes
        self.entries, self.loading = OrderedDict(), {}
        self.bytes = 0
        self.lock = threading.Lock()

    def get(self, key, build):
        with self.lock:
            if key in self.entries:
                self.entries.move_to_end(key)
                return self.entries[key][0]
            future = self.loading.get(key)
            owner = future is None
            if owner:
                future = self.loading[key] = Future()
        if not owner:
            return future.result()
        try:
            value = build(key)
            size = sum(len(item.get("text", "").encode("utf-8")) +
                       len(item.get("image_url", {}).get("url", "")) for item in value)
            with self.lock:
                if size <= self.max_bytes and self.max_documents > 0:
                    while self.entries and (len(self.entries) >= self.max_documents or
                                            self.bytes + size > self.max_bytes):
                        _, (_, removed) = self.entries.popitem(last=False)
                        self.bytes -= removed
                    self.entries[key] = (value, size)
                    self.bytes += size
                self.loading.pop(key)
                future.set_result(value)
            return value
        except BaseException as exc:
            with self.lock:
                self.loading.pop(key, None)
                future.set_exception(exc)
            raise


def run_document_pipeline(records, pool, generate, extract, *, concurrency,
                          document_window, warmup_concurrency, extraction_concurrency,
                          token_budget, document_cost, warmup, stopping, completed):
    """Each future owns at most one request; generation + extraction share the cap.

    A document is released after its generation jobs finish, independently of
    extraction. A failed warmup does not unlock siblings: the next question tries
    to warm the document. Cached responses on resume go straight to extraction.
    The token budget is an admission estimate, not a reservation in vLLM's cache.
    """
    groups, extraction_queue = OrderedDict(), deque()
    for row in records:
        if row.get("response"):
            if extract is None:
                completed(row)
            else:
                extraction_queue.append(row)
        else:
            groups.setdefault(row["doc_id"], deque()).append(row)
    pending = deque((doc, rows, document_cost(doc, rows)) for doc, rows in groups.items())
    active, futures = OrderedDict(), {}
    cold_running = extraction_running = 0

    def admit():
        used = sum(state["cost"] for state in active.values())
        while pending and len(active) < document_window:
            doc, rows, cost = pending[0]
            # Always admit one document, including one larger than the budget.
            if active and token_budget and used + cost > token_budget:
                break
            pending.popleft()
            active[doc] = {"rows": rows, "cost": cost, "running": 0,
                           "ready": not warmup, "warming": False}
            used += cost

    try:
        while pending or active or extraction_queue or futures:
            if stopping.is_set():
                break
            admit()
            while len(futures) < concurrency:
                if extraction_queue and extraction_running < extraction_concurrency:
                    row = extraction_queue.popleft()
                    futures[pool.submit(extract, row)] = ("extract", None, row, False)
                    extraction_running += 1
                    continue
                cold = next((doc for doc, state in active.items()
                             if state["rows"] and not state["ready"] and not state["warming"]), None)
                if cold is not None and cold_running < warmup_concurrency:
                    doc, is_cold = cold, True
                else:
                    doc = next((doc for doc, state in active.items()
                                if state["ready"] and state["rows"]), None)
                    is_cold = False
                if doc is None:
                    break
                state = active[doc]
                row = state["rows"].popleft()
                state["running"] += 1
                if is_cold:
                    state["warming"] = True
                    cold_running += 1
                futures[pool.submit(generate, row)] = ("generate", doc, row, is_cold)
            if not futures:
                if stopping.is_set():
                    break
                raise RuntimeError("Document scheduler made no progress")
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            # Stable handling when several completions are observed together.
            for future in sorted(done, key=lambda f: futures[f][2]["_sample_index"]):
                phase, doc, row, is_cold = futures.pop(future)
                future.result()
                if phase == "extract":
                    extraction_running -= 1
                    completed(row)
                    continue
                state = active[doc]
                state["running"] -= 1
                if is_cold:
                    cold_running -= 1
                    state["warming"] = False
                    state["ready"] = bool(row.get("response"))
                if row.get("response") and extract is not None:
                    extraction_queue.append(row)
                else:
                    completed(row)
                if not state["rows"] and state["running"] == 0:
                    del active[doc]
    except BaseException:
        stopping.set()
        raise
    finally:
        for future in futures:
            future.cancel()
