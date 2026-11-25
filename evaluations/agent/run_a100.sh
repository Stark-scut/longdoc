#!/usr/bin/env bash
# Four A100s by default, with one BF16 Qwen3.5-9B service per GPU.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"

export PYTHON="${PYTHON:-$WORKSPACE_DIR/.venv_grpo/bin/python}"
export DATA_DIR="${DATA_DIR:-$WORKSPACE_DIR/data/mmlongbench}"
export MODEL_PATH="${MODEL_PATH:-$WORKSPACE_DIR/../models/Qwen3.5-9B}"
export GPU_IDS="${GPU_IDS:-0,1,2,3}"
export CONCURRENCY="${CONCURRENCY:-4}"
export OUTPUT_DIR="${OUTPUT_DIR:-output/agent/a100_4gpu_tp1_status_c${CONCURRENCY}_mm3}"

# Match the A100 Direct QA setup and bypass FlashInfer sampling JIT with older nvcc.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

exec bash "$SCRIPT_DIR/run_3090.sh" --tensor-parallel-size 1 "$@"
