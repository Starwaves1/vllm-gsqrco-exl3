#!/bin/bash
# item 4: owned fp32/fp16/bf16 -> q8_1 quantizer in the shim (no input cast). Build, CPU guard
# tests, kernel parity (incl. the bytes-vs-vendored quantizer test), GPU guards, memcheck and
# initcheck on a subset, decode speed, profile, microbench.
source /workspace/p3/p3-lib.sh
date -u +"start %FT%TZ"
( source tools/cuda-env.sh; export PATH=/workspace/gsq-vllm/.venv/bin:$PATH; cd plugin
  VLLM_GGUF_BUILD_LCPP=1 MAX_JOBS=24 python setup.py build_ext --inplace ) > $L/item4-build.log 2>&1
echo "build rc=$?"; ls -la plugin/vllm_gguf_plugin/_C_gguf.abi3.so
tools/pytest tests/cpu/test_lcpp_guards.py -q > $L/item4-cpu-guards.log 2>&1; tail -1 $L/item4-cpu-guards.log
parity item4
tools/pytest tests/gpu/test_kernel_guards.py -q -k lcpp > $L/item4-guards.log 2>&1; tail -1 $L/item4-guards.log
S=/usr/local/cuda/bin/compute-sanitizer; T=tests/gpu/test_kernel_parity.py; ids=()
for t in IQ3_S Q2_K Q4_K; do
  ids+=("$T::test_lcpp_mmq[$t-1-bfloat16]" "$T::test_lcpp_mmq[$t-9-float16]" "$T::test_lcpp_mmvq[$t-4-bfloat16]" "$T::test_lcpp_mmvq[$t-1-float16]")
  ids+=("$T::test_lcpp_quantize_vs_vendored[$t-mmq-9-rowstride]" "$T::test_lcpp_quantize_vs_vendored[$t-q8_1-4-bfloat16]")
done
for tool in memcheck initcheck; do
  PYTORCH_NO_CUDA_MEMORY_CACHING=1 $S --tool $tool --error-exitcode 99 --print-limit 20 tools/pytest -q "${ids[@]}" > $L/item4-sanitizer-$tool.log 2>&1
  echo "$tool rc=$?"; tail -1 $L/item4-sanitizer-$tool.log
done
speed item4
/workspace/p3/p3-profile.sh item4 gsq > $L/item4-profile.log 2>&1; tail -1 $L/item4-profile.log
.venv/bin/python bench/micro/gemm.py --out /workspace/runs/p3-item4-micro.tsv > $L/item4-micro.log 2>&1; echo "micro rc=$?"
date -u +"end %FT%TZ"; echo ITEM4_DONE
