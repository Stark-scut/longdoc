"""Read MMLongBench inputs without changing source paths, OCR or images."""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from .client import Failure
from .logging import image_part

def doc_key(name):
    return Path(str(name)).name.removesuffix(".pdf")

def tokens(text):
    return re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", text.lower())

@dataclass
class Page:
    index: int
    image: Path
    blocks: list
    crops: dict

    def locator_parts(self):
        if not self.image.is_file():
            raise Failure("image_unavailable", str(self.image))
        return [{"type": "text", "text": f"page_index={self.index}"}, image_part(self.image)]

    def parts(self):
        # Full evidence belongs to Reasoner; Locator uses locator_parts instead.
        if not self.image.is_file():
            raise Failure("image_unavailable", str(self.image))
        parts = [{"type": "text", "text": f"page_index={self.index} full page image:"},
                 image_part(self.image)]
        for block in self.blocks:
            block_id = block.get("block_id")
            label = str(block.get("label", "text"))
            text = str(block.get("content") or "")
            parts.append({"type": "text", "text": f"page_index={self.index} block_id={block_id} label={label}\n{text}"})
            crop = self.crops.get(str(block_id))
            if crop and crop.is_file():
                parts.extend([{"type": "text", "text": f"page_index={self.index} crop block_id={block_id}:"},
                              image_part(crop)])
        return parts

class Document:
    def __init__(self, doc_id, pages, captions=None):
        self.doc_id, self.pages = doc_id, pages
        self.captions = captions or {}
        self.order = sorted(pages)
        self.texts = {i: "\n".join(str(b.get("content") or "") for b in pages[i].blocks) for i in self.order}
        self.counts = {i: Counter(tokens(self.texts[i] + " " + self.captions.get(i, ""))) for i in self.order}
        self.df = Counter(term for counts in self.counts.values() for term in counts)
        self.avglen = sum(sum(v.values()) for v in self.counts.values()) / max(1, len(pages))

    def outline(self):
        lines = []
        for index in self.order:
            titles = [str(b.get("content") or "").strip() for b in self.pages[index].blocks
                      if b.get("label") in {"doc_title", "paragraph_title", "title", "section_title"}]
            if titles:
                lines.append(f"[p{index}] " + " | ".join(titles))
        return "\n".join(lines) or "(No OCR headings available; use search.)"

    def search(self, query, top_k):
        query_terms = set(tokens(query))
        scores = {}
        for index, counts in self.counts.items():
            length = sum(counts.values())
            score = 0.0
            for term in query_terms:
                tf = counts.get(term, 0)
                if tf:
                    idf = math.log(1 + (len(self.pages) - self.df[term] + .5) / (self.df[term] + .5))
                    score += idf * tf * 2.5 / (tf + 1.5 * (.25 + .75 * length / max(1, self.avglen)))
            scores[index] = score
        result = []
        for index in sorted(self.order, key=lambda i: (-scores[i], i))[:top_k]:
            text = self.texts[index]
            locations = [text.lower().find(term) for term in query_terms if term in text.lower()]
            start = max(0, min(locations) - 80) if locations else 0
            result.append({"page": index, "score": round(scores[index], 5),
                           "snippet": text[start:start + 360]})
        return result

    def render_locator(self, indices):
        return [part for index in indices for part in self.pages[index].locator_parts()]

    def render(self, indices):
        parts = []
        for index in indices:
            parts.extend(self.pages[index].parts())
        return parts

class Documents:
    def __init__(self, config, doc_ids):
        self.config = config
        wanted = {doc_key(name) for name in doc_ids}
        records = {name: {} for name in wanted}
        with Path(config["blocks"]).open() as f:
            for line in f:
                row = json.loads(line)
                image = Path(row["image_path"])
                doc = image.parent.name
                if doc not in records:
                    continue
                match = re.fullmatch(r"page_(\d+)\.[A-Za-z]+", image.name)
                if match:
                    index = int(match.group(1))
                    if index in records[doc] and records[doc][index] != row["blocks"]:
                        raise ValueError(f"Conflicting OCR page: {doc}/{index}")
                    records[doc][index] = row["blocks"]
        captions = {name: {} for name in wanted}
        caption_file = Path(config["captions"])
        if caption_file.is_file():
            with caption_file.open() as f:
                for line in f:
                    row = json.loads(line)
                    doc = doc_key(row["doc_id"])
                    if doc in captions:
                        page = int(row["page"])
                        captions[doc][page] = captions[doc].get(page, "") + " " + row.get("caption", "")
        self.documents = {}
        for doc in sorted(wanted):
            images = list((Path(config["images_dir"]) / doc).glob("page_*.png"))
            if not images:
                raise ValueError("Missing page images: " + doc)
            crops = Path(config["crop_root"]) / doc
            pages = {}
            for image in images:
                index = int(image.stem.split("_")[-1])
                blocks = records[doc].get(index, [])
                crop_paths = {}
                for block in blocks:
                    block_id = str(block.get("block_id"))
                    matches = sorted(crops.glob(image.name + "_b" + block_id + ".*"))
                    if matches:
                        crop_paths[block_id] = matches[0]
                pages[index] = Page(index, image, blocks, crop_paths)
            extra = set(records[doc]) - set(pages)
            if extra:
                raise ValueError(f"OCR pages without image: {doc}/{sorted(extra)}")
            self.documents[doc] = Document(doc, pages, captions[doc])

    def __getitem__(self, doc_id):
        return self.documents[doc_key(doc_id)]

def select_cases(path, indices=None, limit=0):
    samples = json.loads(Path(path).read_text())
    selected = list(range(len(samples))) if indices is None else [int(v) for v in indices.split(",")]
    if limit:
        selected = selected[:limit]
    if not selected or len(set(selected)) != len(selected) or any(i < 0 or i >= len(samples) for i in selected):
        raise ValueError("Empty, duplicate or invalid sample indices")
    return [dict(samples[i], _sample_index=i) for i in selected]

def validate_images(document, max_pixels):
    from PIL import Image
    for page in document.pages.values():
        with Image.open(page.image) as image:
            width, height = image.size
        area = max(32, round(width / 32) * 32) * max(32, round(height / 32) * 32)
        if area > max_pixels:
            raise ValueError(f"max_pixels would resize {page.image}; need {area}")
