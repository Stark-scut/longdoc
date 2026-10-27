#!/bin/bash
set -euo pipefail

# 兼容原入口：现在同样默认单卡 TP=1，使用 GPU_ID 指定一张空闲卡。
WORKDIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$WORKDIR/serve_qwen35_read_page_agent_single_gpu.sh" "$@"
