"""Page-image-only reads and direct Locator-to-Reasoner evidence selection."""
import json
from .client import Failure
from .reasoner import reason

def schemas(max_pages):
    def function(name, description, properties, required):
        return {"type": "function", "function": {"name": name, "description": description,
                "parameters": {"type": "object", "properties": properties, "required": required,
                               "additionalProperties": False}}}
    pages = {"type": "array", "items": {"type": "integer", "minimum": 0}, "minItems": 1}
    return [
        function("search", "Search OCR and available captions; return short snippets and 0-based page indices.",
                 {"query": {"type": "string"}, "top_k": {"type": "integer", "minimum": 1}}, ["query"]),
        function("read_page", "Read whole page images with page indices only. No OCR text, titles or crops are returned.",
                 {"pages": {**pages, "maxItems": max_pages}}, ["pages"]),
        function("call_reasoner",
                 "Answer using exactly the pages you specify. Each page must have been successfully read. "
                 "Reasoner receives their full images, OCR and crops. This list replaces previous evidence.",
                 {"pages": pages}, ["pages"]),
    ]

class Tools:
    def __init__(self, question, evidence, client, audit, config):
        self.question, self.evidence, self.client, self.audit, self.config = question, evidence, client, audit, config
        self.last_reason = None
        self.count = 0

    def execute(self, call):
        self.count += 1
        name = call["function"]["name"]
        try:
            try:
                args = json.loads(call["function"].get("arguments") or "{}")
            except (ValueError, TypeError) as exc:
                raise Failure("invalid_arguments", "Tool arguments must be JSON", True) from exc
            if not isinstance(args, dict):
                raise Failure("invalid_arguments", "Tool arguments must be an object", True)
            parts = []
            if name == "search":
                if set(args) - {"query", "top_k"} or not isinstance(args.get("query"), str) or not args["query"].strip():
                    raise Failure("invalid_arguments", "search requires a nonempty query", True)
                top = args.get("top_k", min(5, len(self.evidence.document.pages)))
                if type(top) is not int or not 1 <= top <= len(self.evidence.document.pages):
                    raise Failure("invalid_arguments", "top_k must be within the document page count", True)
                result = self.evidence.document.search(args["query"], top)
            elif name == "read_page":
                pages = args.get("pages")
                if (set(args) != {"pages"} or not isinstance(pages, list) or
                    not 1 <= len(pages) <= self.config["read_page_max"] or
                    any(type(p) is not int or p not in self.evidence.document.pages for p in pages)):
                    raise Failure("invalid_arguments", "Invalid pages: use 0-based indices within read_page_max", True)
                pages = list(dict.fromkeys(pages))
                parts = self.evidence.document.render_locator(pages)
                self.evidence.read_pages.update(pages)
                result = {"pages": pages}
            elif name == "call_reasoner":
                if set(args) != {"pages"}:
                    raise Failure("invalid_arguments", "call_reasoner requires exactly one argument: pages", True)
                self.evidence.select(args["pages"])
                result = reason(self.question, self.evidence, self.client)
                self.last_reason = result
            else:
                raise Failure("unknown_tool", name, True)
            outcome = {"ok": True, "tool": name, "result": result,
                       "evidence_pages": sorted(self.evidence.pages), "evidence_version": self.evidence.version}
        except Failure as exc:
            outcome = {"ok": False, "tool": name, "error": exc.as_dict(),
                       "evidence_pages": sorted(self.evidence.pages), "evidence_version": self.evidence.version}
            parts = []
        # Do not serialize the internal page record/outcome into successful read_page observations.
        content = parts if name == "read_page" and outcome["ok"] else [
            {"type": "text", "text": json.dumps(outcome, ensure_ascii=False)}]
        message = {"role": "tool", "tool_call_id": call["id"], "content": content}
        self.audit.emit("tool_result", name=name, tool_call_id=call["id"], outcome=outcome, message=message)
        return message, outcome
