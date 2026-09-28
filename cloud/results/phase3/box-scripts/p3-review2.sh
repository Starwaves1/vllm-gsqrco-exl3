#!/bin/bash
# review 2 fixes: rebuild, parity with Route L on and off, CPU guards, initcheck on dequant + run tests
source /workspace/p3/p3-lib.sh
( source tools/cuda-env.sh; export PATH=/workspace/gsq-vllm/.venv/bin:$PATH; cd plugin
  VLLM_GGUF_BUILD_LCPP=1 MAX_JOBS=24 python setup.py build_ext --inplace ) > $L/review2-build.log 2>&1; echo "build rc=$?"
tools/pytest tests/cpu/test_lcpp_guards.py -q 2>&1 | tail -1
parity review2
VLLM_GGUF_LCPP=0 tools/pytest tests/gpu/test_kernel_parity.py -q > $L/review2-parity-stock.log 2>&1; tail -1 $L/review2-parity-stock.log
tools/pytest tests/gpu/test_kernel_parity.py -q -s -k same_type_run 2>&1 | grep -E "^blk|passed|failed"
PYTORCH_NO_CUDA_MEMORY_CACHING=1 /usr/local/cuda/bin/compute-sanitizer --tool initcheck --error-exitcode 99 tools/pytest -q tests/gpu/test_kernel_parity.py -k "test_dequantize and bfloat16" > $L/review2-initcheck.log 2>&1
echo "initcheck rc=$?"; grep -E "ERROR SUMMARY|passed" $L/review2-initcheck.log | tail -2
echo REVIEW2_DONE
