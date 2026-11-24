"""Per-question, append-only page evidence explicitly stored by Locator."""
from dataclasses import dataclass
from .client import Failure

@dataclass(frozen=True)
class EvidenceSnapshot:
    pages: tuple[int, ...]
    version: int

class EvidenceMemory:
    def __init__(self, document):
        self.document, self.read_pages = document, set()
        self._pages, self.version = set(), 0

    @property
    def pages(self):
        return frozenset(self._pages)

    def store(self, pages):
        if (not isinstance(pages, list) or not pages or
            any(type(p) is not int or p not in self.document.pages or p not in self.read_pages for p in pages)):
            raise Failure("invalid_evidence_pages",
                          "store requires a nonempty list of successfully read 0-based page indices", True)
        # Validate the whole request before mutation; a bad batch stores nothing.
        requested = set(pages)
        added = requested - self._pages
        already_stored = requested & self._pages
        if added:
            self._pages.update(added)
            self.version += 1
        return {"added_pages": sorted(added), "already_stored_pages": sorted(already_stored),
                "memory_pages": sorted(self._pages), "memory_version": self.version}

    def snapshot(self):
        return EvidenceSnapshot(tuple(sorted(self._pages)), self.version)

    def render(self, snapshot=None):
        snapshot = self.snapshot() if snapshot is None else snapshot
        return self.document.render(snapshot.pages)
