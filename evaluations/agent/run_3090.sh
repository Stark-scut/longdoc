#!/usr/bin/env bash
# Eight 24GB RTX 3090s: four BF16 Qwen3.5-9B services, TP=2 each.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_DIR"

# Use the active inference environment; PYTHON can also be an absolute path.
if [[ -z "${PYTHON:-}" ]]; then
    if command -v python >/dev/null 2>&1; then
        PYTHON=python
    else
        PYTHON=python3
    fi
fi
if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "Cannot find Python. Activate the vLLM environment or set PYTHON=/path/to/python." >&2
    exit 1
fi

# Avoid the server's previously failing RoCE/IB initialization. Keep GPU P2P enabled.
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_NET="${NCCL_NET:-Socket}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1

# Consecutive pairs in GPU_IDS form one service; reorder IDs for the machine's topology.
# Additional CLI arguments override defaults, e.g. --limit 8 or --dry-run.
exec "$PYTHON" -m agent eval \
    --config configs/agent/native.json \
    --model-path "${MODEL_PATH:-/media/data5/leizh/models/Qwen3.5-9B}" \
    --data-dir "${DATA_DIR:-/media/data5/leizh/longdoc/aaai_commit/data/mmlongbench}" \
    --gpu-ids "${GPU_IDS:-0,1,2,3,4,5,6,7}" \
    --tensor-parallel-size 2 \
    --concurrency-per-replica "${CONCURRENCY:-4}" \
    --max-num-seqs "${CONCURRENCY:-4}" \
    --max-num-batched-tokens 4096 --max-model-len 65536 \
    --gpu-memory-utilization 0.90 --enforce-eager \
    --enable-prefix-caching --mm-processor-cache-gb 3 \
    --shard-strategy document --document-window 4 --asset-mode reference \
    --output-dir "${OUTPUT_DIR:-output/agent/3090_8gpu_tp2_memory_c${CONCURRENCY:-4}_mm3}" \
    "$@"
