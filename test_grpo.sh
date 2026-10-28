#!/bin/bash
# Compatibility entry; use training/grpo/smoke.sh directly.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$ROOT/training/grpo/smoke.sh" "$@"
