#!/usr/bin/env python3
"""Gold evidence pages + OCR + crops, using the original e2e prompt and scoring."""
import sys

from run_answerable import main


if __name__ == "__main__":
    raise SystemExit(main(["--gold-evidence-ocr-crops", *sys.argv[1:]]))
