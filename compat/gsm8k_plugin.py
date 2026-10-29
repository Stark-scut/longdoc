#!/usr/bin/env python3
"""Compatibility entry; implementation: training/grpo/rewards.py."""
import runpy
import sys
from pathlib import Path
_target = Path(__file__).resolve().parent.parent / 'training/grpo/rewards.py'
sys.path.insert(0, str(_target.parent))
if __name__ == "__main__":
    runpy.run_path(str(_target), run_name="__main__")
else:
    globals().update({k: v for k, v in runpy.run_path(str(_target)).items() if k not in {"__name__", "__file__", "__spec__"}})
