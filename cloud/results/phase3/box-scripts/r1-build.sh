#!/bin/bash
# R1: in-place Route L build of wt-r1 (no GPU)
cd /workspace/wt-r1
( source /workspace/gsq-vllm/tools/cuda-env.sh; export PATH=/workspace/gsq-vllm/.venv/bin:$PATH; cd plugin
  VLLM_GGUF_BUILD_LCPP=1 MAX_JOBS=16 python setup.py build_ext --inplace ) > /workspace/logs/opt/r1/build.log 2>&1
echo "build rc=$?"; grep -E "error|ptxas.*iq3_mma" /workspace/logs/opt/r1/build.log | head -30
