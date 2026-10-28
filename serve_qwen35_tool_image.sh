#!/bin/bash
# Compatibility entry; use examples/tool_image/serve.sh directly.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$ROOT/examples/tool_image/serve.sh" "$@"
