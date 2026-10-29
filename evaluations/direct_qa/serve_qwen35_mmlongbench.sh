#!/bin/bash
set -euo pipefail

# MMLongBench 全文页图直接问答：默认 GPU 1,2，TP=2；也支持单卡 TP=1。
# 示例：GPU_IDS=4,6 bash serve_qwen35_mmlongbench.sh
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
MODEL="${MODEL_PATH:-/media/data1/leizh/models/Qwen3.5-9B}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS:-1,2}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
if [[ -n "${CONDA_PREFIX:-}" ]]; then export PATH="$CONDA_PREFIX/bin:$PATH"; fi
export LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}"
if [[ -n "${CONDA_PREFIX:-}" ]]; then export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"; fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1
unset NPROC_PER_NODE NNODES NODE_RANK RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE

IFS=, read -r -a GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
TP="${TENSOR_PARALLEL_SIZE:-${#GPU_LIST[@]}}"
if [[ "$TP" != "${#GPU_LIST[@]}" || ! -f "$MODEL/config.json" ]]; then
    echo "ERROR: TENSOR_PARALLEL_SIZE 必须等于 GPU_IDS 中的卡数，且模型需要 config.json" >&2
    exit 1
fi
MAX_PAGES="${MAX_PAGES:-512}"
MAX_PIXELS="${MAX_PIXELS:-262144}"
MIN_PIXELS="${MIN_PIXELS:-65536}"
GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.75}"
# 这是本脚本参数名，不是 vLLM 原生环境变量；CLI 已显式传参，避免 vLLM 的未知变量警告。
unset VLLM_GPU_MEMORY_UTILIZATION
MM_LIMIT="{\"image\": $MAX_PAGES, \"video\": 0}"
if [[ "${PROFILE_IMAGE_EDGE:-0}" != 0 ]]; then
    MM_LIMIT="{\"image\": {\"count\": $MAX_PAGES, \"width\": $PROFILE_IMAGE_EDGE, \"height\": $PROFILE_IMAGE_EDGE}, \"video\": 0}"
fi
# 归档最长 468 页；约 128K 上下文、每页最多约 256 个视觉 token。
# 实际文本/视觉 token 超限时仍可能失败，不会自动删除页面。
CMD=("$PYTHON_BIN" -m vllm.entrypoints.openai.api_server --model "$MODEL"
    --served-model-name qwen35-mmlongbench
    --host 127.0.0.1
    --port "${PORT:-8037}"
    --tensor-parallel-size "$TP"
    --dtype "${DTYPE:-bfloat16}"
    --cpu-offload-gb "${CPU_OFFLOAD_GB:-0}"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --max-model-len "${MAX_MODEL_LEN:-131072}"
    --max-num-seqs "${MAX_NUM_SEQS:-8}"
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-2048}"
    --enable-chunked-prefill
    --disable-custom-all-reduce
    --limit-mm-per-prompt "$MM_LIMIT"
    --mm-processor-kwargs "{\"min_pixels\": $MIN_PIXELS, \"max_pixels\": $MAX_PIXELS}"
    --mm-encoder-tp-mode data
    --mm-processor-cache-gb "${MM_PROCESSOR_CACHE_GB:-0}"
    --chat-template-content-format openai
    --default-chat-template-kwargs '{"enable_thinking": false}'
    --reasoning-parser qwen3
)
if [[ "${ENFORCE_EAGER:-1}" == 1 ]]; then
    CMD+=(--enforce-eager)
elif [[ "${ENFORCE_EAGER:-1}" == 0 ]]; then
    CMD+=(--max-cudagraph-capture-size "${MAX_CUDAGRAPH_CAPTURE_SIZE:-${MAX_NUM_SEQS:-8}}")
else
    echo "ERROR: ENFORCE_EAGER 只能是 0 或 1" >&2
    exit 1
fi
if [[ "${ENABLE_PREFIX_CACHING:-0}" == 1 ]]; then
    # 本地 vLLM 的 Qwen3.5 只实现了 align 模式，不支持 all；此功能仍属实验性。
    CMD+=(--enable-prefix-caching --mamba-cache-mode align)
elif [[ "${ENABLE_PREFIX_CACHING:-0}" != 0 ]]; then
    echo "ERROR: ENABLE_PREFIX_CACHING 只能是 0 或 1" >&2
    exit 1
fi
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf '%q ' "${CMD[@]}"
    printf '\n'
    exit 0
fi

# 编译缓存放在本机 /tmp，避开 /home NFS 的 Stale file handle 问题。
LOCAL_CACHE_ROOT="$(mktemp -d "/tmp/qwen35_mmlb_${UID}_XXXXXXXX")"
export TRITON_CACHE_DIR="$LOCAL_CACHE_ROOT/triton"
export TORCHINDUCTOR_CACHE_DIR="$LOCAL_CACHE_ROOT/torchinductor"
export VLLM_CACHE_ROOT="$LOCAL_CACHE_ROOT/vllm"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$VLLM_CACHE_ROOT"
LOGDIR="${SERVER_LOGDIR:-$WORKDIR/output/mmlongbench/server_$(date +%Y%m%d_%H%M%S)_$$}"
mkdir -p "$LOGDIR"
echo "GPU=$CUDA_VISIBLE_DEVICES；模型=qwen35-mmlongbench；API=http://127.0.0.1:${PORT:-8037}/v1"
echo "每请求最多 $MAX_PAGES 页图，每页最多 $MAX_PIXELS 像素；日志：$LOGDIR/server.log"
echo "服务端每轮最多调度 ${MAX_NUM_SEQS:-8} 条序列（MAX_NUM_SEQS）；默认 thinking=false"
echo "TP=$TP；精度=${DTYPE:-bfloat16}；CPU 权重 offload=${CPU_OFFLOAD_GB:-0} GiB"
echo "图像预处理缓存=${MM_PROCESSOR_CACHE_GB:-0} GiB/进程；prefix caching=${ENABLE_PREFIX_CACHING:-0}；enforce eager=${ENFORCE_EAGER:-1}"
echo "本地编译缓存：$LOCAL_CACHE_ROOT"
printf '%q ' "${CMD[@]}" > "$LOGDIR/command.txt"
printf '\n' >> "$LOGDIR/command.txt"

"$PYTHON_BIN" - "${PORT:-8037}" "$GPU_MEMORY_UTILIZATION" "$TP" <<'PY'
import os
import socket
import sys
import torch

with socket.socket() as sock:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('127.0.0.1', int(sys.argv[1])))
if not torch.cuda.is_available() or torch.cuda.device_count() != int(sys.argv[3]):
    raise SystemExit('ERROR: 当前环境无法访问 GPU_IDS 指定的全部 CUDA GPU')
utilization = float(sys.argv[2])
if not 0 < utilization < 1:
    raise SystemExit('ERROR: VLLM_GPU_MEMORY_UTILIZATION 必须在 (0, 1) 内')
for logical_id, physical_id in enumerate(os.environ['CUDA_VISIBLE_DEVICES'].split(',')):
    with torch.cuda.device(logical_id):
        free, total = torch.cuda.mem_get_info()
    print(f'GPU {physical_id} -> cuda:{logical_id}: free={free / 2**30:.2f} GiB', flush=True)
    if free < utilization * total + 0.25 * 2**30:
        raise SystemExit(f'ERROR: GPU {physical_id} 空闲显存不足，请换用空闲 GPU_IDS')
PY

"${CMD[@]}" 2>&1 | tee "$LOGDIR/server.log"
