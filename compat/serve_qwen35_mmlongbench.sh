#!/bin/bash
# Compatibility entry; use evaluations/direct_qa/serve_qwen35_mmlongbench.sh directly.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$ROOT/evaluations/direct_qa/serve_qwen35_mmlongbench.sh" "$@"
