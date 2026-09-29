#!/bin/bash
# item 5 review fixes (__align__(16) on the IQ3 kernel's staged q8_1 tile; same_type_run
# bit-exact at n=8): rebuild in place, CPU guards, full kernel parity with Route L on,
# memcheck + initcheck on the IQ3 kernel cases (no graph replay: capture is unsupported
# under compute-sanitizer). One gpuq job.
cd /workspace/gsq-vllm; source /workspace/box-env.sh  # GSQ_ALLOW_GPU=1, GSQ_GGUF
L=/workspace/logs/p3/item5-review; mkdir -p $L
( source tools/cuda-env.sh; export PATH=/workspace/gsq-vllm/.venv/bin:$PATH; cd plugin
  VLLM_GGUF_BUILD_LCPP=1 MAX_JOBS=24 python setup.py build_ext --inplace ) > $L/build.log 2>&1
echo "build rc=$?"
tools/pytest tests/cpu/test_lcpp_guards.py -q > $L/cpu-guards.log 2>&1; tail -1 $L/cpu-guards.log
VLLM_GGUF_LCPP=1 tools/pytest tests/gpu/test_kernel_parity.py -q > $L/parity.log 2>&1; echo "rc=$?" >> $L/parity.log; tail -2 $L/parity.log
T=tests/gpu/test_kernel_parity.py; ids=()
for t in IQ3_S IQ3_XXS; do
  for c in 1-bfloat16-real 4-bfloat16-row_tail 8-float16-k_tail 5-float32-row_tail 8-float32-real 3-bfloat16-k_tail; do
    ids+=("$T::test_lcpp_iq3[$t-$c]")
  done
done
for tool in memcheck initcheck; do
  PYTORCH_NO_CUDA_MEMORY_CACHING=1 /usr/local/cuda/bin/compute-sanitizer --tool $tool --error-exitcode 99 --print-limit 20 \
    tools/pytest -q "${ids[@]}" > $L/sanitizer-$tool.log 2>&1
  echo "$tool rc=$?"; grep -E "passed|failed" $L/sanitizer-$tool.log | tail -1; tail -1 $L/sanitizer-$tool.log
done
