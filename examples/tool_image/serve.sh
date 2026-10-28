#!/bin/bash
set -euo pipefail

# 终端一：conda activate qwen35 && bash serve_qwen35_tool_image.sh
# 可覆盖 GPU_IDS=4,6 PORT=8027；前台运行，Ctrl-C 停止服务。
if [[ "${CONDA_DEFAULT_ENV:-}" != qwen35 ]]; then
    echo "ERROR: 请先 conda activate qwen35" >&2
    exit 1
fi
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
MODEL="${MODEL_PATH:-/media/data1/leizh/models/Qwen3.5-9B}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS:-1,2}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PATH="$CONDA_PREFIX/bin:$PATH"
export LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONUNBUFFERED=1
unset NPROC_PER_NODE NNODES NODE_RANK RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE

IFS=, read -r -a GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
if [[ "${#GPU_LIST[@]}" != 2 || ! -f "$MODEL/config.json" ]]; then
    echo "ERROR: 需要指定两张空闲 GPU，且 MODEL_PATH 下必须有 config.json" >&2
    exit 1
fi

# 按 Qwen 官方示例启用 thinking 和自动工具调用，并保留视觉编码器。
# 只允许一张小图片，限制视觉预处理规模，降低 3090 初始化显存峰值。
CMD=("$CONDA_PREFIX/bin/vllm" serve "$MODEL"
    --served-model-name qwen35-tool-image
    --host 127.0.0.1
    --port "${PORT:-8027}"
    --tensor-parallel-size 2
    --dtype bfloat16
    --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.75}"
    --max-model-len "${MAX_MODEL_LEN:-8192}"
    --max-num-seqs 2
    --max-num-batched-tokens 2048
    --enforce-eager
    --disable-custom-all-reduce
    --limit-mm-per-prompt '{"image": 1, "video": 0}'
    --mm-processor-kwargs '{"min_pixels": 65536, "max_pixels": 262144}'
    --mm-processor-cache-gb 0
    --chat-template-content-format openai
    --default-chat-template-kwargs '{"enable_thinking": true}'
    --reasoning-parser qwen3
    --enable-auto-tool-choice
    --tool-call-parser qwen3_coder
)
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf '%q ' "${CMD[@]}"
    printf '\n'
    exit 0
fi

# 避开之前出现 Stale file handle 的 NFS 缓存。
LOCAL_CACHE_ROOT="$(mktemp -d "/tmp/qwen35_tool_image_${UID}_XXXXXXXX")"
export TRITON_CACHE_DIR="$LOCAL_CACHE_ROOT/triton"
export TORCHINDUCTOR_CACHE_DIR="$LOCAL_CACHE_ROOT/torchinductor"
export VLLM_CACHE_ROOT="$LOCAL_CACHE_ROOT/vllm"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$VLLM_CACHE_ROOT"
LOGDIR="$WORKDIR/output/tool_image_server_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$LOGDIR"
echo "GPU=$CUDA_VISIBLE_DEVICES；thinking=true；tool parser=qwen3_coder"
echo "API：http://127.0.0.1:${PORT:-8027}/v1；模型名：qwen35-tool-image"
echo "日志：$LOGDIR/server.log；本地缓存：$LOCAL_CACHE_ROOT"

"$CONDA_PREFIX/bin/python" - "${PORT:-8027}" "${VLLM_GPU_MEMORY_UTILIZATION:-0.75}" <<'PY'
import socket
import sys
import torch

with socket.socket() as sock:
    sock.bind(('127.0.0.1', int(sys.argv[1])))
if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
    raise SystemExit('ERROR: 当前环境需要能访问指定的两张 CUDA GPU')
utilization = float(sys.argv[2])
if not 0 < utilization < 1:
    raise SystemExit('ERROR: VLLM_GPU_MEMORY_UTILIZATION 必须在 (0, 1) 内')
for logical_id, physical_id in enumerate(__import__('os').environ['CUDA_VISIBLE_DEVICES'].split(',')):
    with torch.cuda.device(logical_id):
        free, total = torch.cuda.mem_get_info()
    props = torch.cuda.get_device_properties(logical_id)
    print(f'GPU {physical_id} -> cuda:{logical_id}: UUID={props.uuid}, free={free / 2**30:.2f} GiB', flush=True)
    if free < utilization * total + 2**30:
        raise SystemExit(f'ERROR: GPU {physical_id} 空闲显存不足，请换用空闲 GPU_IDS')
PY

# pipefail 保留服务启动错误；前台进程组可接收 Ctrl-C。
"${CMD[@]}" 2>&1 | tee "$LOGDIR/server.log"
