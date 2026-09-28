#!/bin/bash
# phase 2 queue: wait for the lcpp guard run, then compute-sanitizer subsets, then serve + parity
source /workspace/box-env.sh
cd /workspace/gsq-vllm
until grep -q "^rc=" /workspace/logs/p2/guards-lcpp.log; do sleep 10; done
S=/usr/local/cuda/bin/compute-sanitizer
T=tests/gpu/test_kernel_parity.py
ids=()
for t in IQ3_S IQ2_XS Q2_K Q4_K Q6_K; do for n in 1 5 7 9 128; do ids+=("$T::test_lcpp_mmq[$t-$n-bfloat16]"); done; ids+=("$T::test_lcpp_mmvq[$t-4-bfloat16]" "$T::test_lcpp_mmvq[$t-8-bfloat16]"); done
ids+=("$T::test_lcpp_mmq_odd_rows[IQ3_S-5]" "$T::test_lcpp_mmq_odd_rows[Q4_K-128]" "$T::test_lcpp_graph_replay[IQ3_S-mmq-5]" "$T::test_lcpp_graph_replay[IQ3_S-mmvq-4]")
VLLM_GGUF_LCPP=1 $S --tool memcheck --error-exitcode 99 --print-limit 20 tools/pytest -q "${ids[@]}" > /workspace/logs/p2/sanitizer-parity.log 2>&1
echo "rc=$?" >> /workspace/logs/p2/sanitizer-parity.log
GSQ_COMPUTE_SANITIZER=$S tools/pytest tests/gpu/test_kernel_guards.py -q -rA \
  -k "(lcpp and (row_too_big or k_mismatch or graph_replay or x_rowstride or w_narrow_view)) or (row_too_big and IQ3_S-mmvq and not lcpp)" \
  > /workspace/logs/p2/sanitizer-guards.log 2>&1
echo "rc=$?" >> /workspace/logs/p2/sanitizer-guards.log
/workspace/p2-serve-parity.sh > /workspace/logs/p2/serve-parity.log 2>&1
echo QUEUE_DONE >> /workspace/logs/p2/serve-parity.log
