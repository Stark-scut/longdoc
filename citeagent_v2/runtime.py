"""Load the frozen legacy copy. Each mode runs in its own process."""
from __future__ import annotations
import os
from pathlib import Path
import sys
from common import HERE, digest, read_json

if (HERE / "vendor").exists():
    sys.path.insert(0, str(HERE / "vendor"))

def load(baseline):
    baseline = Path(baseline).resolve()
    manifest = read_json(baseline / "manifest.json")
    for name, expected in manifest["files"].items():
        if digest(baseline / name) != expected:
            raise ValueError("Frozen baseline checksum changed: " + name)
    source = read_json(HERE / "source_manifest.json")
    for name, expected in source["copied_source_sha256"].items():
        if digest(HERE / "frozen_legacy" / name) != expected:
            raise ValueError("Frozen legacy source changed: " + name)
    settings = read_json(baseline / "settings.json")
    os.environ["CITEAGENT_DATA_DIR"] = str(baseline / "data")
    from frozen_legacy import config, data_prep, llm, locator, memory, pipeline, reasoner
    for name, value in settings["legacy_effective"].items():
        if not name.endswith("API_KEY"):
            setattr(config, name, value)
    config.DATA_DIR = str(baseline / "data")
    config.JSONL_PATH = str(baseline / "data/blocks.jsonl")
    config.SAMPLES_PATH = str(baseline / "cases.json")
    config.IMAGES_DIR = str(baseline / "images")
    config.CROP_ROOT = str(baseline / "crops")
    config.CROP_CAPTIONS_PATH = str(baseline / "data/captions.jsonl")
    config.NEW_IMAGE_PREFIX = config.IMAGES_DIR + "/"
    def frozen_image_path(path):
        parts = Path(path).parts
        return str(baseline / "images" / parts[-2] / parts[-1])
    data_prep._fix_image_path = frozen_image_path
    tool = next(t for t in locator.TOOLS if t["function"]["name"] == "read_page")
    tool["function"]["parameters"]["properties"]["pages"].update(
        maxItems=config.READ_PAGE_MAX,
        description=f"0-indexed physical page numbers; at most {config.READ_PAGE_MAX} pages")
    return config, data_prep, llm, locator, memory, pipeline, reasoner, settings

def configure_transport(runtime, transport, native):
    from transport import install
    _, _, llm, locator, _, _, reasoner, _ = runtime
    install(llm, reasoner, locator, transport, native)

def implementation_hash():
    from common import json_hash
    return json_hash({str(p.relative_to(HERE)): digest(p) for p in sorted(HERE.glob("*.py"))})
