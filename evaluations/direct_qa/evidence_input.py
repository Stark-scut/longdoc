"""Gold-page OCR/crop inputs, using the same Page.parts renderer as Reasoner."""
import hashlib
import json
from pathlib import Path
import re
import sys
import threading

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from agent.documents import Page


class EvidenceInput:
    def __init__(self, page_sets, blocks, crop_root):
        self.blocks_path = Path(blocks).resolve()
        self.crop_root = Path(crop_root).resolve()
        if not self.crop_root.is_dir():
            raise ValueError(f"Missing crop root: {self.crop_root}")
        wanted = {(path.parent.name, int(path.stem.split('_')[-1])): path
                  for paths in page_sets for path in paths}
        records = {}
        digest = hashlib.sha256()
        with self.blocks_path.open('rb') as stream:
            for line in stream:
                digest.update(line)
                if not line.strip():
                    continue
                row = json.loads(line)
                image = Path(row['image_path'])
                match = re.fullmatch(r'page_(\d+)\.[A-Za-z]+', image.name)
                if not match:
                    continue
                key = (image.parent.name, int(match[1]))
                if key not in wanted:
                    continue
                if key in records and records[key] != row['blocks']:
                    raise ValueError(f'Conflicting OCR page: {key}')
                records[key] = row['blocks']
        missing = sorted(set(wanted) - records.keys())
        if missing:
            raise ValueError(f'Missing OCR records for selected evidence pages: {missing[:10]}')
        self.pages = {}
        for (doc, index), path in wanted.items():
            crops = {}
            for block in records[(doc, index)]:
                block_id = str(block.get('block_id'))
                matches = sorted(p for p in (self.crop_root / doc).glob(path.name + '_b' + block_id + '.*')
                                 if p.is_file())
                if matches:
                    crops[block_id] = matches[0]
            self.pages[path] = Page(index, path, records[(doc, index)], crops)
        self.lock = threading.Lock()
        images = sorted({p for paths in page_sets for p in self.image_paths(paths)})
        inventory = [[str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns] for p in images]
        self.metadata = {'mode': 'gold-pages-ocr-crops-v1', 'blocks': str(self.blocks_path),
                         'blocks_sha256': digest.hexdigest(), 'crop_root': str(self.crop_root),
                         'image_inventory_sha256': hashlib.sha256(json.dumps(inventory).encode()).hexdigest()}

    def image_paths(self, paths):
        images = []
        for path in paths:
            page = self.pages[path]
            images.append(path)
            images.extend(page.crops[str(block.get('block_id'))] for block in page.blocks
                          if str(block.get('block_id')) in page.crops)
        return tuple(images)

    def content(self, paths):
        # Page.parts uses Agent's image cache; serialize construction across evaluator threads.
        with self.lock:
            for path in self.image_paths(paths):
                if not path.is_file():
                    raise ValueError(f'Selected evidence image disappeared: {path}')
            return [part for path in paths for part in self.pages[path].parts()]

    def describe(self, paths):
        images = self.image_paths(paths)
        return {'mode': self.metadata['mode'], 'page_indices': [self.pages[p].index for p in paths],
                'page_count': len(paths), 'image_count': len(images), 'crop_count': len(images) - len(paths),
                'ocr_block_count': sum(len(self.pages[p].blocks) for p in paths),
                'images': [str(p.resolve()) for p in images]}

    def text_bytes(self, paths):
        return sum(len(f"page_index={self.pages[p].index} block_id={b.get('block_id')} "
                       f"label={b.get('label', 'text')}\n{b.get('content') or ''}".encode('utf-8'))
                   for p in paths for b in self.pages[p].blocks)
