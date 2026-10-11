#!/bin/bash
. /workspace/kit-env.sh
VENV=/workspace/kit-work/venv-cu13; PY=$VENV/bin/python
export CUDA_HOME=/workspace/kit/build/cu130/nvidia/cu13
export PATH=$CUDA_HOME/bin:$VENV/bin:$PATH TORCH_CUDA_ARCH_LIST="7.5;8.6" MAX_JOBS=4
command -v g++-13 >/dev/null && export CC=gcc-13 CXX=g++-13 NVCC_CCBIN=g++-13
export NVCC_APPEND_FLAGS="-allow-unsupported-compiler"
nvcc --version | tail -2; g++-13 --version | head -1; $PY -c "import torch;print(torch.__version__, torch.version.cuda)"
cd /workspace/exl3-sm75/plugin-exl3
date -Is
VLLM_EXL3_BUILD=1 nice -n 10 $PY setup.py build_ext --inplace
echo "=== exit $?"; date -Is
