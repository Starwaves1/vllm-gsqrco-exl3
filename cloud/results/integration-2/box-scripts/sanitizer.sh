#!/bin/bash
# integration-2, one gpuq job: compute-sanitizer memcheck + initcheck (torch caching allocator off)
# on every owned op and the Route L paths around them: the 1..8-row kernels (dp4a / mma IQ3,
# Q4_K/IQ2_S) at 1 and 8 rows, packed IQ3 (decode kernel 1/8/9/32, tiled 1/8/33/128, routed
# 1/8/9/33/128), mma_k at 1/9/33/64 (its op limit), x_q8 sharing, IQ1_M chunks, MMQ with the
# quantize-zeroed tail, and the fused-layer tests at 8/9/32 rows (ids in sanitizer-ids.txt).
source /workspace/wt-int2/cloud/results/integration-2/box-scripts/lib.sh
date -u +"start %FT%TZ"
T=tests/gpu/test_kernel_parity.py
RE="^$T::(test_lcpp_iq3\[.*-(1|8)-bfloat16-(real|row_tail)\]|test_lcpp_iq3_packed\[.*-(1|8|9|32)-bfloat16-(real|many_tiles)\]|test_lcpp_iq3_packed_tiled\[.*-(1|8|33|128)-bfloat16-(real|rows_208)\]|test_routing_packed_whole_tensor\[.*-(1|8|9|33|128)\]|test_lcpp_mma_k\[.*-(1|9|33|64)-bfloat16-(real|row_tail|big_tail)\]|test_routing_whole_tensor\[(Q4_K|IQ4_XS|IQ2_S|IQ1_M)-(9|32)-True\]|test_lcpp_x_q8\[.*-(1|8)\]|test_quantize_x_q8_1_mixed_route|test_lcpp_iq1_m_chunks\[.*\]|test_lcpp_mmq\[(IQ3_S|Q4_K)-(1|5|8|9|128)-bfloat16\]|test_lcpp_packed_layer\[(1|8|128)-.*\]|test_lcpp_mixed_shard_layer\[(8|9|32)-.*\]|test_lcpp_same_type_run\[(8|9|32)-qkv_widest-.*\])$"
ids=$(tools/pytest $T --collect-only -q 2>/dev/null | grep -E "$RE")
echo "$ids" > $L/sanitizer-ids.txt; echo "sanitizer cases: $(echo "$ids" | wc -l)"
for tool in memcheck initcheck; do
  PYTORCH_NO_CUDA_MEMORY_CACHING=1 /usr/local/cuda/bin/compute-sanitizer --tool $tool --error-exitcode 99 --print-limit 20 \
    tools/pytest -q $ids > $L/sanitizer-$tool.log 2>&1
  echo "$tool rc=$?"; grep -E "passed|failed" $L/sanitizer-$tool.log | tail -1; tail -1 $L/sanitizer-$tool.log
done
date -u +"end %FT%TZ"
