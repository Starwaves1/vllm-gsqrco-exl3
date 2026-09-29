#!/bin/bash
# R2: in-place Route L build of wt-r2 (no GPU). Prints ptxas lines of the packed kernels.
L=/workspace/logs/opt/r2; mkdir -p $L
WT=${WT:-/workspace/wt-r2}; cd $WT
( source tools/cuda-env.sh; export PATH=/workspace/gsq-vllm/.venv/bin:$PATH; cd plugin
  VLLM_GGUF_BUILD_LCPP=1 MAX_JOBS=12 python setup.py build_ext --inplace ) > $L/build-$(basename $WT).log 2>&1
echo "build rc=$?"; grep -E "error" $L/build-$(basename $WT).log | head -30
source tools/cuda-env.sh; cuobjdump -res-usage plugin/vllm_gguf_plugin/_C_gguf.abi3.so 2>/dev/null | grep -A1 -E "iq3_packed_mmq" \
  | grep -oE "(Function [^ ]*|REG:[0-9]+|STACK:[0-9]+|SHARED:[0-9]+|LOCAL:[0-9]+)" | paste - - - - - | cut -c1-200 | head -30
