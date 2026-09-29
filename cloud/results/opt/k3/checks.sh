#!/bin/bash
# K3 final checks (once): full kernel parity (Route L), GPU guards (lcpp), CPU guards, memcheck + initcheck on the mma_k cases
source /workspace/box-env.sh
export VLLM_GGUF_LCPP=1 GSQ_ALLOW_GPU=1 PYTHONPATH=/workspace/wt-k3/plugin:/workspace/wt-k3/tools
L=/workspace/logs/opt/k3; cd /workspace/wt-k3; PT=/workspace/gsq-vllm/tools/pytest
$PT tests/gpu/test_kernel_parity.py -q -s -k "lcpp or routing" > $L/final-parity.log 2>&1; echo "parity rc=$?"; tail -2 $L/final-parity.log
grep -E "vs MMQ rel" $L/final-parity.log > $L/final-mma_k-vs-mmq.txt; wc -l < $L/final-mma_k-vs-mmq.txt
$PT tests/gpu/test_kernel_guards.py -q -k lcpp > $L/final-guards.log 2>&1; echo "guards rc=$?"; tail -1 $L/final-guards.log
$PT tests/cpu/test_lcpp_guards.py -q > $L/final-cpu-guards.log 2>&1; echo "cpu guards rc=$?"; tail -1 $L/final-cpu-guards.log
S=/usr/local/cuda/bin/compute-sanitizer; T=tests/gpu/test_kernel_parity.py; ids=()
for t in Q4_K IQ4_XS IQ2_S; do
  for c in "9-bfloat16-real" "9-bfloat16-row_tail" "17-float16-row_tail" "33-bfloat16-row_tail" "64-bfloat16-row_tail" "33-bfloat16-k_tail" "16-float32-k_tail" "64-bfloat16-down" "32-float32-real" "16-bfloat16-no_pieces"; do
    n=${c%%-*}; rest=${c#*-}; ids+=("$T::test_lcpp_mma_k[$t-$n-${rest%%-*}-${rest#*-}]"); done
  for r in 17408 5120; do ids+=("$T::test_lcpp_mma_k_whole_tensor[$t-$r-64]"); done
done
for tool in memcheck initcheck; do
  PYTORCH_NO_CUDA_MEMORY_CACHING=1 $S --tool $tool --error-exitcode 99 --print-limit 20 $PT -q "${ids[@]}" > $L/final-sanitizer-$tool.log 2>&1
  echo "$tool rc=$?"; tail -1 $L/final-sanitizer-$tool.log; grep -E "ERROR SUMMARY" $L/final-sanitizer-$tool.log | tail -1
done
