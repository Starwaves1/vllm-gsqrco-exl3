#!/bin/bash
# Build and install a plugin package into $GSQ_VENV (default .venv) from source:
# CUDA 13.0 toolchain (tools/setup-cuda-toolchain.sh), sm86 only, 2 jobs.
# Editable install, so Python changes apply without rebuilding.
# GSQ_PLUGIN picks the package: plugin (GGUF, default) or plugin-exl3 (EXL3; builds _C_exl3
# unless VLLM_EXL3_BUILD=0).
# Run it under tools/capped.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/tools/cuda-env.sh"
VENV=${GSQ_VENV:-$ROOT/.venv}
PLUGIN=${GSQ_PLUGIN:-plugin}
[ "$PLUGIN" = plugin-exl3 ] && export VLLM_EXL3_BUILD=${VLLM_EXL3_BUILD:-1}
export PATH=$VENV/bin:$PATH          # ninja from the venv
export MAX_JOBS=${MAX_JOBS:-2}
cd "$ROOT"
uv pip install --python "$VENV/bin/python" --no-build-isolation --no-deps --link-mode=copy -e "$PLUGIN" -v 2>&1
