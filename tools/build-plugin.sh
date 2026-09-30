#!/bin/bash
# Build and install the plugin fork (plugin/) into $GSQ_VENV (default .venv) from source:
# CUDA 13.0 toolchain (tools/setup-cuda-toolchain.sh), sm86 only, 2 jobs.
# Editable install, so Python changes under plugin/ apply without rebuilding.
# Run it under tools/capped.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/tools/cuda-env.sh"
VENV=${GSQ_VENV:-$ROOT/.venv}
export PATH=$VENV/bin:$PATH          # ninja from the venv
export MAX_JOBS=${MAX_JOBS:-2}
cd "$ROOT"
uv pip install --python "$VENV/bin/python" --no-build-isolation --no-deps --link-mode=copy -e plugin -v 2>&1
