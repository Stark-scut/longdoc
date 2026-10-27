#!/bin/bash
set -euo pipefail

# 请先在终端执行 conda activate qwen35
if [[ "${CONDA_DEFAULT_ENV:-}" != "qwen35" ]]; then
    echo "ERROR: 请先执行 conda activate qwen35"
    exit 1
fi

# 两步冒烟测试：GPU 1,2 运行独立 rollout，GPU 4,6,7 更新 LoRA。
# 使用同步 GRPO；每次更新后由 ms-swift 同步权重到 rollout 服务。
# 可选：DRY_RUN=1 bash test_grpo.sh，只打印启动命令，不使用 GPU。
WORKDIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${MODEL_PATH:-/media/data1/leizh/models/Qwen3.5-9B}"
PLUGIN="$WORKDIR/gsm8k_plugin.py"
OUTPUT="${OUTPUT_DIR:-$WORKDIR/output/grpo_smoke_$(date +%Y%m%d_%H%M%S)_$$}"
ROLLOUT_GPUS=1,2
TRAIN_GPUS=4,6,7
ROLLOUT_PORT="${ROLLOUT_PORT:-8017}"
SYNC_PORT="${SYNC_PORT:-51271}"
TRAIN_MASTER_PORT="${MASTER_PORT:-29571}"
ROLLOUT_STARTUP_TIMEOUT="${ROLLOUT_STARTUP_TIMEOUT:-900}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.70}"
cd "$WORKDIR"

if [[ ! -f "$MODEL/config.json" || ! -f "$PLUGIN" ]]; then
    echo "ERROR: 模型 config.json 或奖励插件不存在：$MODEL / $PLUGIN" >&2
    exit 1
fi
if [[ ! -x "$CONDA_PREFIX/bin/swift" ]]; then
    echo "ERROR: 当前 conda 环境没有 swift 命令" >&2
    exit 1
fi
if ! command -v setsid >/dev/null; then
    echo "ERROR: 需要 setsid 命令来管理 rollout 和训练的子进程" >&2
    exit 1
fi

# ===== CUDA 动态库 =====
export LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$CONDA_PREFIX/bin:$PATH"
# 固定 CUDA 枚举顺序，并在下方用 UUID 校验 nvidia-smi 编号与 CUDA 的对应关系。
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# 两条命令分别设置可见 GPU 和进程数，防止 rollout 继承训练的分布式环境。
unset NPROC_PER_NODE NNODES NODE_RANK RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE
unset MASTER_ADDR MASTER_PORT
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

# ===== FlashInfer =====
unset VLLM_USE_FLASHINFER_SAMPLER

# 三张训练卡使用 ZeRO-3 CPU offload，限制通信桶和 GPU 参数缓存的峰值。
# rollout 不占训练卡，无需 colocate 模式的休眠和手动 offload 开关。
DEEPSPEED_CONFIG='{
  "bf16": {"enabled": "auto"},
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {"device": "cpu", "pin_memory": true},
    "offload_param": {"device": "cpu", "pin_memory": true},
    "overlap_comm": false,
    "contiguous_gradients": true,
    "reduce_bucket_size": 5000000,
    "stage3_prefetch_bucket_size": 0,
    "stage3_param_persistence_threshold": 10000,
    "stage3_max_live_parameters": 100000000,
    "stage3_max_reuse_distance": 100000000,
    "stage3_gather_16bit_weights_on_model_save": true
  },
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": "auto",
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "steps_per_print": 2000,
  "wall_clock_breakdown": false
}'

# 独立 vLLM 服务：一个 TP=2、DP=1 的推理实例，只使用物理 GPU 1,2。
# 为 NCCL 权重同步留出余量；GSM8K 只需文本，关闭多模态输入。
ROLLOUT_CMD=(env CUDA_VISIBLE_DEVICES="$ROLLOUT_GPUS"
    "$CONDA_PREFIX/bin/swift" rollout
    --model "$MODEL"
    --torch_dtype bfloat16
    --host 127.0.0.1
    --port "$ROLLOUT_PORT"
    --enable_thinking false
    --vllm_use_async_engine false
    --vllm_tensor_parallel_size 2
    --vllm_data_parallel_size 1
    --vllm_gpu_memory_utilization "$VLLM_GPU_MEMORY_UTILIZATION"
    --vllm_max_model_len 512
    --vllm_max_num_seqs 4
    --vllm_enforce_eager true
    --vllm_enable_lora true
    --vllm_max_lora_rank 8
    --vllm_limit_mm_per_prompt '{"image": 0, "video": 0}'
    --vllm_mm_processor_cache_gb 0
    --vllm_engine_kwargs '{"language_model_only": true, "max_num_batched_tokens": 512}'
)

SYSTEM_PROMPT='You are a helpful math assistant. Solve the problem step by step and put your final answer within \boxed{}.'

TRAIN_CMD=(env CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" NPROC_PER_NODE=3 \
  NNODES=1 NODE_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT="$TRAIN_MASTER_PORT" \
  "$CONDA_PREFIX/bin/swift" rlhf \
  --rlhf_type grpo \
  --model "$MODEL" \
  --dataset 'modelscope/gsm8k#64' \
  --external_plugins "$PLUGIN" \
  --reward_funcs gsm8k_accuracy gsm8k_format \
  --columns '{"answer": "solution"}' \
  --system "$SYSTEM_PROMPT" \
  --enable_thinking false \
  --tuner_type lora \
  --lora_rank 8 \
  --lora_alpha 32 \
  --target_modules all-linear \
  --freeze_vit true \
  --torch_dtype bfloat16 \
  --use_vllm true \
  --vllm_mode server \
  --vllm_server_host 127.0.0.1 \
  --vllm_server_port "$ROLLOUT_PORT" \
  --vllm_server_group_port "$SYNC_PORT" \
  --vllm_server_timeout 240 \
  --async_generate false \
  --move_model_batches 32 \
  --max_length 256 \
  --max_completion_length 128 \
  --num_generations 3 \
  --temperature 1.0 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --gradient_checkpointing true \
  --deepspeed "$DEEPSPEED_CONFIG" \
  --optim adamw_torch \
  --learning_rate 1e-5 \
  --beta 0.0 \
  --max_steps 2 \
  --save_steps 2 \
  --save_total_limit 1 \
  --save_only_model true \
  --logging_steps 1 \
  --log_completions true \
  --report_to none \
  --output_dir "$OUTPUT")

echo "rollout GPU=$ROLLOUT_GPUS (TP=2); 训练 GPU=$TRAIN_GPUS (3 个训练进程)"
echo "输出目录：$OUTPUT"
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf '%q ' "${ROLLOUT_CMD[@]}"
    printf '\n'
    printf '%q ' "${TRAIN_CMD[@]}"
    printf '\n'
    exit 0
fi

# /home 位于 NFS，Triton 动态库曾报 Stale file handle。
# 每次测试使用全新的本地缓存，rollout 和全部训练 worker 继承这些路径。
# 保留目录便于排查；不要在训练期间清理缓存文件。
LOCAL_CACHE_ROOT="$(mktemp -d "/tmp/grpo_cache_${UID}_XXXXXXXX")"
export TRITON_CACHE_DIR="$LOCAL_CACHE_ROOT/triton"
export TORCHINDUCTOR_CACHE_DIR="$LOCAL_CACHE_ROOT/torchinductor"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
echo "本地编译缓存：$LOCAL_CACHE_ROOT（首次编译可能较慢）"

# 检查五张 GPU 和端口，避免连到其他任务的服务或被自动改到其他端口。
CUDA_VISIBLE_DEVICES="$ROLLOUT_GPUS,$TRAIN_GPUS" "$CONDA_PREFIX/bin/python" - \
    "$ROLLOUT_PORT" "$SYNC_PORT" "$TRAIN_MASTER_PORT" "$ROLLOUT_STARTUP_TIMEOUT" \
    "$VLLM_GPU_MEMORY_UTILIZATION" <<'PY'
import csv
import os
import socket
import subprocess
import sys
from importlib.metadata import version
import torch

ports = [int(value) for value in sys.argv[1:4]]
if len(set(ports)) != 3 or any(not 1 <= port <= 65535 for port in ports):
    raise SystemExit("ERROR: rollout、权重同步和训练端口必须合法且互不相同")
if int(sys.argv[4]) <= 0:
    raise SystemExit("ERROR: ROLLOUT_STARTUP_TIMEOUT 必须为正整数（秒）")
utilization = float(sys.argv[5])
if not 0 < utilization <= 1:
    raise SystemExit("ERROR: VLLM_GPU_MEMORY_UTILIZATION 必须在 (0, 1] 内")
for port in ports:
    with socket.socket() as sock:
        try:
            sock.bind(('127.0.0.1', port))
        except OSError as exc:
            raise SystemExit(f"ERROR: 端口 {port} 不可用，请修改对应端口变量：{exc}")
for package in ("ms-swift", "torch", "vllm", "transformers", "deepspeed"):
    print(f"{package}: {version(package)}", flush=True)
if not torch.cuda.is_available() or torch.cuda.device_count() != 5:
    raise SystemExit("ERROR: 当前环境必须能访问 CUDA_VISIBLE_DEVICES 指定的五张 GPU")
snapshot = subprocess.check_output([
    'nvidia-smi', '--query-gpu=index,uuid,pci.bus_id,memory.used,memory.free',
    '--format=csv,noheader,nounits',
], text=True)
print('nvidia-smi 快照（index, UUID, PCI bus, used MiB, free MiB）：\n' + snapshot, flush=True)
gpu_info = {row[0].strip(): [item.strip() for item in row[1:]]
            for row in csv.reader(snapshot.splitlines())}
def normalize_uuid(value):
    return str(value).lower().removeprefix('gpu-')

for logical_id, physical_id in enumerate(os.environ["CUDA_VISIBLE_DEVICES"].split(",")):
    with torch.cuda.device(logical_id):
        free, total = torch.cuda.mem_get_info()
        accelerator_free, _ = torch.accelerator.get_memory_info(logical_id)
    properties = torch.cuda.get_device_properties(logical_id)
    actual_uuid = str(properties.uuid)
    expected_uuid = gpu_info[physical_id][0]
    print(f"GPU {physical_id} -> cuda:{logical_id}: "
          f"{properties.name}, UUID={actual_uuid}, "
          f"free={free / 2**30:.2f} GiB / total={total / 2**30:.2f} GiB, "
          f"accelerator_free={accelerator_free / 2**30:.2f} GiB", flush=True)
    if normalize_uuid(actual_uuid) != normalize_uuid(expected_uuid):
        raise SystemExit(f"ERROR: CUDA GPU 映射与 nvidia-smi 不一致：GPU {physical_id} "
                         f"期望 {expected_uuid}，实际 {actual_uuid}，停止启动")
    if abs(free - accelerator_free) > 512 * 2**20:
        raise SystemExit("ERROR: torch.cuda 与 torch.accelerator 报告的空闲显存不一致，"
                         "请先排查 PyTorch 的设备映射/显存查询")
    if logical_id < 2 and min(free, accelerator_free) < utilization * total + 2**30:
        raise SystemExit(f"ERROR: rollout GPU {physical_id} 空闲显存不足，"
                         f"至少需要 vLLM 预算 {utilization * total / 2**30:.2f} GiB "
                         "+ 1 GiB 初始化余量；请检查占用进程")
PY

mkdir -p "$OUTPUT"
printf 'TRITON_CACHE_DIR=%s\nTORCHINDUCTOR_CACHE_DIR=%s\n' \
    "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" > "$OUTPUT/cache_paths.log"
echo "rollout 日志：$OUTPUT/rollout.log"
echo "训练日志：$OUTPUT/train.log"
echo "启动期间显存记录：$OUTPUT/gpu_memory.log"

ROLLOUT_PID=""
TRAIN_PID=""
GPU_MONITOR_PID=""
cleanup() {
    local status=$?
    trap - EXIT INT TERM
    if [[ -n "$GPU_MONITOR_PID" ]]; then
        kill -TERM "$GPU_MONITOR_PID" 2>/dev/null || true
        wait "$GPU_MONITOR_PID" 2>/dev/null || true
    fi
    # 仅结束本脚本启动的两个进程组，包括 vLLM/torchrun 的 worker。
    local pid
    for pid in "$TRAIN_PID" "$ROLLOUT_PID"; do
        if [[ -n "$pid" ]]; then
            kill -TERM -- "-$pid" 2>/dev/null || true
        fi
    done
    if [[ -n "$TRAIN_PID" || -n "$ROLLOUT_PID" ]]; then
        sleep 2
    fi
    for pid in "$TRAIN_PID" "$ROLLOUT_PID"; do
        if [[ -n "$pid" ]]; then
            kill -KILL -- "-$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
        fi
    done
    return "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# 记录外部显存占用的变化；nvidia-smi 监控进程不占 CUDA 显存。
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory \
    --format=csv > "$OUTPUT/gpu_processes_before.log" 2>&1 || true
nvidia-smi --query-gpu=timestamp,index,uuid,pci.bus_id,memory.used,memory.free \
    --format=csv --loop=2 > "$OUTPUT/gpu_memory.log" 2>&1 &
GPU_MONITOR_PID=$!

setsid "${ROLLOUT_CMD[@]}" > "$OUTPUT/rollout.log" 2>&1 &
ROLLOUT_PID=$!
echo "等待 rollout 服务就绪（最多 ${ROLLOUT_STARTUP_TIMEOUT}s）……"
if ! "$CONDA_PREFIX/bin/python" - "$ROLLOUT_PID" "$ROLLOUT_PORT" "$ROLLOUT_STARTUP_TIMEOUT" <<'PY'
import json
import os
import sys
import time
import urllib.error
import urllib.request

pid, port, timeout = map(int, sys.argv[1:])
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
deadline = time.monotonic() + timeout
while time.monotonic() < deadline:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        raise SystemExit("ERROR: rollout 进程已退出")
    try:
        with opener.open(f'http://127.0.0.1:{port}/health/', timeout=2) as response:
            if json.load(response).get('status') == 'ok':
                print("rollout 服务已就绪", flush=True)
                break
    except (urllib.error.URLError, OSError, ValueError):
        pass
    time.sleep(2)
else:
    raise SystemExit("ERROR: 等待 rollout 服务超时")
PY
then
    echo "rollout 日志中的首批错误：" >&2
    if command -v rg >/dev/null; then
        rg -n -m 5 'ValueError:|OutOfMemoryError:|CUDA out of memory|RuntimeError:' \
            "$OUTPUT/rollout.log" || true
    fi
    tail -n 80 "$OUTPUT/rollout.log"
    echo "当前 GPU 与占用进程：" >&2
    nvidia-smi || true
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory \
        --format=csv > "$OUTPUT/gpu_processes_failed.log" 2>&1 || true
    exit 1
fi

# 后台启动使 shell 在等待训练时也能处理 Ctrl-C 并清理两个进程组。
setsid "${TRAIN_CMD[@]}" > >(tee "$OUTPUT/train.log") 2>&1 &
TRAIN_PID=$!
if wait "$TRAIN_PID"; then
    echo "两步 GRPO 测试完成，输出目录：$OUTPUT"
else
    status=$?
    echo "ERROR: 训练失败（exit=$status），请查看 train.log 和 rollout.log" >&2
    exit "$status"
fi
