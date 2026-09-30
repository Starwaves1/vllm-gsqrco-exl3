#!/bin/bash
# one gpuq job: in-place Route L build of wt-final, then the pack tests (CPU + GPU) and the GPU
# pack memory / time / bytes comparison vs the pre-fix pack (packmem.py).
source /workspace/wt-final/cloud/results/final/box-scripts/lib.sh
date -u +"start %FT%TZ"; git log --oneline -1
( cd plugin && VLLM_GGUF_BUILD_LCPP=1 MAX_JOBS=24 $GSQ_VENV/bin/python setup.py build_ext --inplace ) > $L/build.log 2>&1; echo "build rc=$?"
tools/pytest tests/cpu/test_iq3_pack.py tests/cpu/test_lcpp_routing.py tests/cpu/test_lcpp_guards.py -q > $L/cpu.log 2>&1; echo "cpu rc=$?: $(tail -1 $L/cpu.log)"
tools/pytest tests/gpu/test_kernel_parity.py -q -rs -k "pack or packed or iq3" > $L/parity-pack.log 2>&1; echo "parity pack rc=$?: $(tail -1 $L/parity-pack.log)"
$GSQ_VENV/bin/python $S/packmem.py cuda > $L/packmem-cuda.txt 2>&1; echo "packmem cuda rc=$?"; tail -8 $L/packmem-cuda.txt
for v in old new; do $GSQ_VENV/bin/python $S/packmem.py cpu $v >> $L/packmem-cpu.txt 2>&1; done; echo "packmem cpu:"; cat $L/packmem-cpu.txt
date -u +"end %FT%TZ"
