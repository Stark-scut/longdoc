#!/bin/bash
# Compatibility entry; use examples/read_page_agent/serve.sh directly.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$ROOT/examples/read_page_agent/serve.sh" "$@"
