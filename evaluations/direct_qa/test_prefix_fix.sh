#!/usr/bin/env bash
# A100 prefix-cache backport smoke test. Extra CLI arguments override defaults.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="$(cd -- "$REPO_ROOT/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$WORKSPACE_ROOT/.venv_grpo/bin/python}"

export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_NET="${NCCL_NET:-Socket}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export PYTHONUNBUFFERED=1

"$PYTHON_BIN" "$SCRIPT_DIR/patch_vllm_mamba_align.py" check
"$PYTHON_BIN" "$SCRIPT_DIR/patch_vllm_mamba_align.py" apply
exec "$PYTHON_BIN" "$SCRIPT_DIR/run_replicas.py" \
  --gpu-ids "${GPU_IDS:-0,1,2,3}" \
  --model-path "${MODEL_PATH:-$WORKSPACE_ROOT/../models/Qwen3.5-9B}" \
  --input "$WORKSPACE_ROOT/aaai_commit/data/mmlongbench/samples.json" \
  --images-dir "$WORKSPACE_ROOT/aaai_commit/data/mmlongbench/images_1024" \
  --max-model-len 262144 --truncate-long-documents --input-token-limit 260000 \
  --concurrency 16 --max-num-seqs 16 --enable-prefix-caching \
  --shard-strategy document --document-warmup \
  --document-window 4 --warmup-concurrency 2 --extraction-concurrency 2 \
  --document-token-budget 350000 --page-content-cache-mb 512 \
  --temperature 0 --extract-temperature 0 --seed 0 \
  --mm-processor-cache-gb 4 --cpu-offload-gb 0 \
  --timeout 180 --limit 32 \
  --output-dir "${OUTPUT_DIR:-$REPO_ROOT/output/mmlongbench/qwen35_a100_doc_prefix_mm4_rolling_seed0}" \
  "$@"
