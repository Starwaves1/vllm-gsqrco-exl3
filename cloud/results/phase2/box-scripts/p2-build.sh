#!/bin/bash
set -euxo pipefail
cd /workspace/gsq-vllm
source tools/cuda-env.sh
export PATH=/workspace/gsq-vllm/.venv/bin:$PATH
cd plugin
rm -rf build
time VLLM_GGUF_BUILD_LCPP=1 MAX_JOBS=24 python setup.py build_ext --inplace
ls -la vllm_gguf_plugin/_C_gguf.abi3.so
echo BUILD_DONE
