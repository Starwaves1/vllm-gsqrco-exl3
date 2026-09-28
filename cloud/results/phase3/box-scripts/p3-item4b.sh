#!/bin/bash
# item 4b: one product per run of adjacent same-type shards (fused layers); ggml_dequantize
# output left uninitialised (every kernel writes all of it). Build, parity, speed, profile.
source /workspace/p3/p3-lib.sh
date -u +"start %FT%TZ"
( source tools/cuda-env.sh; export PATH=/workspace/gsq-vllm/.venv/bin:$PATH; cd plugin
  VLLM_GGUF_BUILD_LCPP=1 MAX_JOBS=24 python setup.py build_ext --inplace ) > $L/item4b-build.log 2>&1
echo "build rc=$?"
parity item4b
tools/pytest tests/gpu/test_kernel_parity.py -q -s -k "same_type_run or mixed_shard" 2>&1 | grep -E "^blk|passed|failed" > $L/item4b-runs.log; cat $L/item4b-runs.log
speed item4b
/workspace/p3/p3-profile.sh item4b gsq > $L/item4b-profile.log 2>&1; tail -1 $L/item4b-profile.log
date -u +"end %FT%TZ"; echo ITEM4B_DONE
