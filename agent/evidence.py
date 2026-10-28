"""Evidence admission and versioning, independent of conversation history."""
from .client import Failure

class Evidence:
    def __init__(self, document):
        self.document, self.read_pages, self.pages, self.version = document, set(), set(), 0

    def admit(self, pages):
        if not isinstance(pages, list) or any(type(p) is not int or p not in self.read_pages for p in pages):
            raise Failure("invalid_selection", "selected_pages must reference successfully read physical page indices")
        previous = set(self.pages)
        self.pages.update(pages)
        if self.pages != previous:
            self.version += 1
        return sorted(self.pages)

    def render(self):
        return self.document.render(sorted(self.pages))
