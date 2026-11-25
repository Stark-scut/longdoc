#!/usr/bin/env bash
# Evaluate answerable questions using two A100 generation replicas and a ColEmbed service.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
export GPU_IDS="${GPU_IDS:-0,1}"
export OUTPUT_DIR="${OUTPUT_DIR:-output/agent/a100_2gpu_tp1_colembed_answerable_c${CONCURRENCY:-4}_mm3}"

exec bash "$SCRIPT_DIR/run_a100.sh" \
    --answerable-only \
    --colembed-url "${COLEMBED_URL:-http://127.0.0.1:8091}" \
    --colembed-index-dir "${COLEMBED_INDEX_DIR:-$REPO_DIR/output/agent/indexes/nemotron_colembed_vl_4b_v2_vllm}" \
    "$@"
