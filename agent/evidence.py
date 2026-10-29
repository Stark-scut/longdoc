"""Locator-selected evidence, replaced explicitly on every Reasoner invocation."""
from .client import Failure

class Evidence:
    def __init__(self, document):
        self.document, self.read_pages, self.pages, self.version = document, set(), set(), 0

    def select(self, pages):
        if (not isinstance(pages, list) or not pages or
            any(type(p) is not int or p not in self.read_pages for p in pages)):
            raise Failure("invalid_evidence_pages",
                          "pages must be a nonempty list of successfully read 0-based page indices", True)
        selected = set(pages)
        if selected != self.pages:
            self.pages = selected
            self.version += 1
        return sorted(self.pages)

    def render(self):
        return self.document.render(sorted(self.pages))
