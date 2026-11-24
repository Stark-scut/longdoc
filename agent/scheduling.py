"""Document affinity and per-replica work queues shared by local workers."""
from __future__ import annotations
from collections import OrderedDict
from contextlib import contextmanager
import copy
import fcntl
from pathlib import Path
from .documents import doc_key
from .logging import atomic, read


def partition_documents(cases, pending, replicas):
    """Balance question counts, keeping every PDF on one replica across resumes."""
    groups = {}
    for case in cases:
        groups.setdefault(doc_key(case["doc_id"]), []).append(case["_sample_index"])
    pending = set(pending)
    loads = [0] * replicas
    partitions = [[] for _ in range(replicas)]
    for doc, indices in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])):
        replica = min(range(replicas), key=lambda i: (loads[i], i))
        loads[replica] += len(indices)
        partitions[replica].extend(i for i in indices if i in pending)
    return partitions


class DocumentQueue:
    """Short flock transactions; no coordinator service or GPU-side changes.

    A window bounds PDFs with questions left to claim. Workers share those
    questions; already assigned tail questions never block admitting a new PDF.
    Queues are recreated per launch;
    saved per-sample results remain the authority when resuming after a crash.
    """
    def __init__(self, path):
        self.path = Path(path)

    @classmethod
    def create(cls, path, cases, indices, window):
        if window < 1:
            raise ValueError("document-window must be positive")
        by_index = {c["_sample_index"]: c for c in cases}
        groups = OrderedDict()
        for index in indices:
            groups.setdefault(doc_key(by_index[index]["doc_id"]), []).append(index)
        atomic(path, {"window": window, "waiting": list(groups), "active": [],
                      "pending": groups, "inflight": {}, "completed": 0})
        return cls(path)

    @contextmanager
    def transaction(self):
        with self.path.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = read(self.path)
            previous = copy.deepcopy(state)
            yield state
            if state != previous:
                atomic(self.path, state)

    def claim(self, slot):
        slot = str(slot)
        with self.transaction() as state:
            if slot in state["inflight"]:
                raise ValueError("Worker already has an unfinished question")
            while state["waiting"] and len(state["active"]) < state["window"]:
                state["active"].append(state["waiting"].pop(0))
            for doc in list(state["active"]):
                if state["pending"][doc]:
                    index = state["pending"][doc].pop(0)
                    state["inflight"][slot] = {"doc": doc, "index": index}
                    # Round robin within the active window gives every PDF a turn.
                    state["active"].remove(doc)
                    if state["pending"][doc]:
                        state["active"].append(doc)
                    return "task", index
            return "done", None

    def finish(self, slot):
        with self.transaction() as state:
            state["inflight"].pop(str(slot))
            state["completed"] += 1

    @contextmanager
    def work(self, slot):
        def indices():
            while True:
                status, index = self.claim(slot)
                if status == "done":
                    return
                try:
                    yield index
                finally:
                    self.finish(slot)
        iterator = indices()
        try:
            yield iterator
        finally:
            iterator.close()
