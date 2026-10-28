#!/bin/bash
set -euo pipefail

# 单卡 read_page Agent 服务：默认 GPU 1，TP=1，BF16，不进行 CPU 权重 offload。
# GPU_ID=4 bash serve_qwen35_read_page_agent_single_gpu.sh
if [[ "${CONDA_DEFAULT_ENV:-}" != qwen35 ]]; then
    echo "ERROR: 请先 conda activate qwen35" >&2
    exit 1
fi
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
MODEL="${MODEL_PATH:-/media/data1/leizh/models/Qwen3.5-9B}"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-${GPU_IDS:-1}}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PATH="$CONDA_PREFIX/bin:$PATH"
export LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1
unset NPROC_PER_NODE NNODES NODE_RANK RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE
if [[ ! "$CUDA_VISIBLE_DEVICES" =~ ^[0-9]+$ || ! -f "$MODEL/config.json" ]]; then
    echo "ERROR: GPU_ID 必须是一张空闲卡的编号（例如 1），MODEL_PATH 下需要 config.json" >&2
    exit 1
fi

CMD=("$CONDA_PREFIX/bin/vllm" serve "$MODEL"
    --served-model-name qwen35-read-page-agent
    --host 127.0.0.1 --port "${PORT:-8057}"
    --tensor-parallel-size 1 --dtype bfloat16
    --cpu-offload-gb "${CPU_OFFLOAD_GB:-0}"
    --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.95}"
    --max-model-len "${MAX_MODEL_LEN:-65536}"
    --max-num-seqs 1 --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-512}"
    --enable-chunked-prefill --enforce-eager
    --limit-mm-per-prompt "{\"image\":{\"count\":${MAX_IMAGES:-32},\"width\":1024,\"height\":1024},\"video\":0}"
    --mm-processor-kwargs '{"min_pixels":1024,"max_pixels":1048576}'
    --mm-encoder-tp-mode data --mm-processor-cache-gb 0
    --chat-template-content-format openai
    --default-chat-template-kwargs '{"enable_thinking":true}'
    --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder
)
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf '%q ' "${CMD[@]}"
    printf '\n'
    exit 0
fi

"$CONDA_PREFIX/bin/python" - "${PORT:-8057}" "${VLLM_GPU_MEMORY_UTILIZATION:-0.95}" <<'PY'
import os
import socket
import sys
import torch
with socket.socket() as sock:
    sock.bind(('127.0.0.1', int(sys.argv[1])))
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit('ERROR: 无法访问 GPU_ID 指定的单张卡')
utilization = float(sys.argv[2])
if not 0 < utilization < 1:
    raise SystemExit('ERROR: VLLM_GPU_MEMORY_UTILIZATION 必须在 (0,1) 内')
for logical, physical in enumerate(os.environ['CUDA_VISIBLE_DEVICES'].split(',')):
    with torch.cuda.device(logical):
        free, total = torch.cuda.mem_get_info()
    print(f'GPU {physical}: free={free / 2**30:.2f} GiB / total={total / 2**30:.2f} GiB', flush=True)
    if free < utilization * total + 0.25 * 2**30:
        raise SystemExit(f'ERROR: GPU {physical} 空闲显存不足，请指定其他 GPU_ID')
PY

LOCAL_CACHE_ROOT="$(mktemp -d "/tmp/qwen35_read_page_${UID}_XXXXXXXX")"
export TRITON_CACHE_DIR="$LOCAL_CACHE_ROOT/triton"
export TORCHINDUCTOR_CACHE_DIR="$LOCAL_CACHE_ROOT/torchinductor"
export VLLM_CACHE_ROOT="$LOCAL_CACHE_ROOT/vllm"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$VLLM_CACHE_ROOT"
LOGDIR="$WORKDIR/output/read_page_agent/server_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$LOGDIR"
printf '%q ' "${CMD[@]}" > "$LOGDIR/command.txt"
printf '\n' >> "$LOGDIR/command.txt"
echo "GPU=$CUDA_VISIBLE_DEVICES；TP=1；BF16；CPU offload=${CPU_OFFLOAD_GB:-0} GiB；thinking=true；tool parser=qwen3_coder"
echo "API=http://127.0.0.1:${PORT:-8057}/v1；模型名=qwen35-read-page-agent"
echo "保留 images_1024 页图分辨率；最多 ${MAX_IMAGES:-32} 张历史图片；日志：$LOGDIR/server.log"
"${CMD[@]}" 2>&1 | tee "$LOGDIR/server.log"
