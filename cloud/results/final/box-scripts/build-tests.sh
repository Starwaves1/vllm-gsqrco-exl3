#!/bin/bash
# one gpuq job: in-place Route L build of wt-final, then the pack tests (CPU + GPU) and the GPU
# pack memory / time / bytes comparison vs the pre-fix pack (packmem.py).
source /workspace/wt-final/cloud/results/final/box-scripts/lib.sh
date -u +"start %FT%TZ"; git log --oneline -1
( source tools/cuda-env.sh; export PATH=$GSQ_VENV/bin:$PATH; cd plugin
  VLLM_GGUF_BUILD_LCPP=1 MAX_JOBS=24 python setup.py build_ext --inplace ) > $L/build.log 2>&1; echo "build rc=$?"
tools/pytest tests/cpu/test_iq3_pack.py tests/cpu/test_lcpp_routing.py tests/cpu/test_lcpp_guards.py -q > $L/cpu.log 2>&1; echo "cpu rc=$?: $(tail -1 $L/cpu.log)"
tools/pytest tests/gpu/test_kernel_parity.py -q -rs -k "pack or packed or iq3" > $L/parity-pack.log 2>&1; echo "parity pack rc=$?: $(tail -1 $L/parity-pack.log)"
$GSQ_VENV/bin/python -W ignore $S/packmem.py > $L/packmem-cuda.txt 2>&1; echo "packmem cuda rc=$?"; tail -8 $L/packmem-cuda.txt
date -u +"end %FT%TZ"
