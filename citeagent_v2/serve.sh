#!/bin/bash
set -euo pipefail
: "${GPU_ID:?请显式指定一张空闲 GPU，例如 GPU_ID=0 bash serve.sh}"
MODEL_PATH="${MODEL_PATH:-/media/data1/leizh/models/Qwen3.5-9B}"
PYTHON_BIN="${PYTHON_BIN:-/media/data1/leizh/anaconda3/envs/qwen35/bin/python}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1
unset NPROC_PER_NODE NNODES NODE_RANK RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE
if [[ ! "$GPU_ID" =~ ^[0-9]+$ ]]; then
    echo "GPU_ID 必须是一张卡的物理编号" >&2
    exit 1
fi
# One GPU, BF16, no weight offload, no experimental prefix caching.
exec "$PYTHON_BIN" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" --served-model-name citeagent-qwen35 \
    --host 127.0.0.1 --port "${PORT:-8067}" \
    --tensor-parallel-size 1 --dtype bfloat16 --cpu-offload-gb 0 \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.95}" \
    --max-model-len "${MAX_MODEL_LEN:-65536}" \
    --max-num-seqs 1 --max-num-batched-tokens 2048 \
    --enable-chunked-prefill --no-enable-prefix-caching --enforce-eager \
    --limit-mm-per-prompt '{"image":{"count":128,"width":1024,"height":1024},"video":0}' \
    --mm-processor-kwargs '{"min_pixels":1024,"max_pixels":1048576}' \
    --mm-processor-cache-gb 1 \
    --chat-template-content-format openai \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
    --default-chat-template-kwargs '{"enable_thinking":true}'
