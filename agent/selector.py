"""Page-level evidence selection; result reuse only when its inputs agree."""
from pathlib import Path
from .logging import fingerprint

PROMPT = (Path(__file__).parent / "prompts/selector.txt").read_text()

class Selector:
    def __init__(self, question, evidence, client):
        self.question, self.evidence, self.client = question, evidence, client
        self.signature = None

    def select(self, missing):
        candidate_pages = sorted(self.evidence.read_pages)
        signature = fingerprint({"pages": candidate_pages, "missing": missing,
                                 "evidence": sorted(self.evidence.pages)})
        if signature == self.signature:
            return {"selected_pages": sorted(self.evidence.pages), "skipped": "unchanged_state"}
        if not candidate_pages:
            return {"selected_pages": [], "skipped": "no_read_pages"}
        parts = [{"type": "text", "text": "Question: " + self.question +
                  "\nCurrent evidence pages: " + str(sorted(self.evidence.pages)) +
                  "\nMissing claims: " + str(missing)}]
        parts.extend(self.evidence.document.render(candidate_pages))
        value = self.client.json("selector", [{"role": "system", "content": PROMPT},
                                               {"role": "user", "content": parts}])
        self.evidence.admit(value.get("selected_pages"))
        # Our own admission must not trigger an identical second selection.
        self.signature = fingerprint({"pages": candidate_pages, "missing": missing,
                                      "evidence": sorted(self.evidence.pages)})
        return {"selected_pages": value["selected_pages"], "evidence_pages": sorted(self.evidence.pages)}
