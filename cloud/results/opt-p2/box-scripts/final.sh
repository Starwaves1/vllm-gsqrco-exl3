#!/bin/bash
# opt-p2, once at the end (one gpuq job): CPU guards + routing table, GPU guards on the Route L
# ops, compute-sanitizer memcheck + initcheck (no caching allocator, so initcheck sees every
# scratch read) on: MMQ at 1..9 rows (the read tail is now zeroed by the quantize kernel), IQ1_M
# MMVQ and its 8-row chunks, x_q8 sharing, the owned kernels.
#   final.sh TAG
TAG=$1; WT=/workspace/wt-p2-$TAG
source /workspace/wt-p2-$TAG/cloud/results/opt-p2/box-scripts/lib.sh
exec > >(tee $L/$TAG-final.log) 2>&1
date -u +"start %FT%TZ"; wait_build
tools/pytest tests/cpu/test_lcpp_guards.py tests/cpu/test_lcpp_routing.py -q > $L/$TAG-cpu-guards.log 2>&1; echo "cpu guards: $(tail -1 $L/$TAG-cpu-guards.log)"
tools/pytest tests/gpu/test_kernel_guards.py -q -rs -k "lcpp or first_call" > $L/$TAG-guards.log 2>&1; echo "gpu guards rc=$?: $(tail -1 $L/$TAG-guards.log)"
T=tests/gpu/test_kernel_parity.py
ids=$(tools/pytest $T --collect-only -q 2>/dev/null | grep -E "^$T::(test_lcpp_mmq\[(IQ3_S|Q4_K|Q2_K|IQ4_XS)-(1|5|8|9)-bfloat16\]|test_lcpp_mmvq\[IQ1_M-(1|4|8)-bfloat16\]|test_lcpp_iq1_m_chunks|test_lcpp_x_q8\[(IQ3_S|Q4_K|IQ1_M)-(4|8)\]|test_quantize_x_q8_1_mixed_route|test_lcpp_iq3\[.*-(1-bfloat16-real|4-bfloat16-row_tail|8-float16-k_tail|6-bfloat16-odd_rows)\])")
echo "$ids" > $L/$TAG-sanitizer-ids.txt; echo "sanitizer cases: $(echo "$ids" | wc -l)"
for tool in memcheck initcheck; do
  PYTORCH_NO_CUDA_MEMORY_CACHING=1 /usr/local/cuda/bin/compute-sanitizer --tool $tool --error-exitcode 99 --print-limit 20 \
    tools/pytest -q $ids > $L/$TAG-sanitizer-$tool.log 2>&1
  echo "$tool rc=$?: $(grep -E "passed|failed" $L/$TAG-sanitizer-$tool.log | tail -1) / $(tail -1 $L/$TAG-sanitizer-$tool.log)"
done
( cd plugin/vllm_gguf_plugin/csrc/lcpp && grep -E "^[0-9a-f]{64}  " VENDORED.md | sha256sum -c --quiet && echo "vendored files unchanged" )
date -u +"end %FT%TZ"
