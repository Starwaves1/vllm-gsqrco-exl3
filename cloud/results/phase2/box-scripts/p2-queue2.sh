#!/bin/bash
# phase 2 queue 2: after serve+parity: sanitizer reruns (no caching allocator), microbench,
# decode-step profile, speed pass
source /workspace/box-env.sh
cd /workspace/gsq-vllm
until grep -q QUEUE_DONE /workspace/logs/p2/serve-parity.log 2>/dev/null; do sleep 20; done
S=/usr/local/cuda/bin/compute-sanitizer
T=tests/gpu/test_kernel_parity.py
ids=()
for t in IQ3_S IQ2_XS Q2_K Q4_K Q6_K; do for n in 1 5 7 9 128; do ids+=("$T::test_lcpp_mmq[$t-$n-bfloat16]"); done; ids+=("$T::test_lcpp_mmvq[$t-4-bfloat16]" "$T::test_lcpp_mmvq[$t-8-bfloat16]"); done
ids+=("$T::test_lcpp_mmq_odd_rows[IQ3_S-5]" "$T::test_lcpp_mmq_odd_rows[Q4_K-128]" "$T::test_lcpp_graph_replay[IQ3_S-mmq-5]" "$T::test_lcpp_graph_replay[IQ3_S-mmvq-4]")
for tool in memcheck initcheck; do
  PYTORCH_NO_CUDA_MEMORY_CACHING=1 VLLM_GGUF_LCPP=1 $S --tool $tool --error-exitcode 99 --print-limit 20 tools/pytest -q "${ids[@]}" > /workspace/logs/p2/sanitizer-parity-$tool.log 2>&1
  echo "rc=$?" >> /workspace/logs/p2/sanitizer-parity-$tool.log
done
GSQ_COMPUTE_SANITIZER=$S tools/pytest tests/gpu/test_kernel_guards.py -q -rA \
  -k "(lcpp and (row_too_big or k_mismatch or graph_replay or x_rowstride or w_narrow_view)) or (row_too_big and IQ3_S-mmvq and not lcpp)" \
  > /workspace/logs/p2/sanitizer-guards-2.log 2>&1
echo "rc=$?" >> /workspace/logs/p2/sanitizer-guards-2.log
.venv/bin/python bench/micro/gemm.py --out /workspace/logs/p2/micro.tsv > /workspace/logs/p2/micro.log 2>&1
echo "rc=$?" >> /workspace/logs/p2/micro.log
/workspace/p2-profile.sh > /workspace/logs/p2/profile.log 2>&1
.venv/bin/python /workspace/p2trace.py > /workspace/logs/p2/profile-summary.txt 2>&1
/workspace/p2-speed.sh > /workspace/logs/p2/speed.log 2>&1
echo QUEUE2_DONE >> /workspace/logs/p2/speed.log
