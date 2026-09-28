#!/bin/bash
# phase-1 steps 3-5 on the box. Step 3 reuses the smoke server; then it is stopped.
set -uo pipefail
source /workspace/box-env.sh
cd /workspace/gsq-vllm
R=/workspace/runs; S=$R/p1-smoke
STEP=${1:-all}
if [ $STEP = all ] || [ $STEP = fit ]; then
  echo "=== step 3 fit $(date -u +%FT%TZ)"
  GSQ_SERVER_LOG=$S/server.log tools/pytest tests/gpu/test_fit_200k.py -v -s --junitxml=$R/p1-fit.xml > $R/p1-fit.log 2>&1; echo "fit rc=$?"
  nvidia-smi --query-gpu=memory.used --format=csv > $R/p1-fit-vram.csv
  grep -E "GPU KV cache size|Maximum concurrency|num_gpu_blocks|Available KV cache memory|Model loading took|Loading weights took|memory profiling|CUDA graph" $S/server.log > $R/p1-fit-kvlines.txt
fi
if [ $STEP = all ] || [ $STEP = stop ]; then
  P=$(cat $S/serve.pid); kill -INT $P; for i in $(seq 60); do kill -0 $P 2>/dev/null || break; sleep 2; done; kill -9 $P 2>/dev/null
  sleep 5; pkill -9 -f "VLLM::EngineCore"; sleep 3; box_clean_shm
  nvidia-smi --query-gpu=memory.used --format=csv,noheader; echo "server stopped"
fi
if [ $STEP = all ] || [ $STEP = kernels ]; then
  echo "=== step 4 kernels $(date -u +%FT%TZ)"
  tools/pytest tests/gpu/test_kernel_parity.py -v -s -rA --junitxml=$R/p1-kernel-parity.xml > $R/p1-kernel-parity.log 2>&1; echo "parity rc=$?"
  tools/pytest tests/gpu/test_kernel_guards.py -v -s -rA --junitxml=$R/p1-kernel-guards.xml > $R/p1-kernel-guards.log 2>&1; echo "guards rc=$?"
fi
if [ $STEP = all ] || [ $STEP = speed ]; then
  echo "=== step 5 speed $(date -u +%FT%TZ)"
  box_clean_shm || exit 1
  nvidia-smi --query-gpu=power.limit,clocks.max.sm --format=csv > $R/p1-speed-power.csv
  OUT=$R/p1-speed-gsq bench/speed/run.sh gsq --start > $R/p1-speed.log 2>&1; echo "speed rc=$?"
fi
echo "P1_REST_DONE $(date -u +%FT%TZ)"
