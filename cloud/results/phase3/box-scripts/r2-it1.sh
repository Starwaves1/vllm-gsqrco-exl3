#!/bin/bash
# R2 it1 (one gpuq job): v2 (wt-r2d) tile-width sweep, then v1 (wt-r2c) vs v2 at the default tiles.
S=/workspace/wt-r2d/cloud/results/phase3/box-scripts
WT=/workspace/wt-r2d bash $S/r2-micro.sh it1-v2sweep --types IQ3_S --shapes 17408x5120,5120x17408 \
  --tokens 16,32,64,128,256,2048 --variants new --tn 16,32,64,128
WT=/workspace/wt-r2c bash $S/r2-micro.sh it1-v1 --types IQ3_S,IQ3_XXS --shapes 17408x5120 --tokens 32,128,2048 --variants new
WT=/workspace/wt-r2d bash $S/r2-micro.sh it1-v2 --types IQ3_S,IQ3_XXS --shapes 17408x5120 --tokens 32,128,2048 --variants new
# stream-K fixup price at decode sizes: 68 CTAs = one whole 256-row tile each at 17408 rows (no split)
WT=/workspace/wt-r2d bash $S/r2-micro.sh it1-g --types IQ3_S --shapes 17408x5120,10240x5120 --tokens 16,32 --variants new --g 40,68,82
# v3 (wt-r2e): mma addend from shared memory (no re-materialized moves), accumulators reset per tile
WT=/workspace/wt-r2e bash $S/r2-micro.sh it1-v3 --types IQ3_S,IQ3_XXS --shapes 17408x5120,5120x17408 --tokens 16,32,64,128,2048 --variants new
# v4 (wt-r2f): scales staged [slice][column] (one 8-byte load per column pair), shift-only copy offsets
WT=/workspace/wt-r2f bash $S/r2-micro.sh it1-v4 --types IQ3_S,IQ3_XXS --shapes 17408x5120,5120x17408 --tokens 16,32,64,128,2048 --variants new
# v5 (wt-r2b): v4 + store epilogue without per-element branches
WT=/workspace/wt-r2b bash $S/r2-micro.sh it1-v5 --types IQ3_S,IQ3_XXS --shapes 17408x5120,5120x17408 --tokens 16,32,64,128,2048 --variants new
# v7 (wt-r2g): v5 + the two half blocks as a real loop (half the hot loop's code)
WT=/workspace/wt-r2g bash $S/r2-micro.sh it1-v7 --types IQ3_S,IQ3_XXS --shapes 17408x5120,5120x17408 --tokens 16,32,64,128,2048 --variants new
# v8 (wt-r2h): v5 + the four slice pairs as a real loop (a quarter of the hot loop's code)
WT=/workspace/wt-r2h bash $S/r2-micro.sh it1-v8 --types IQ3_S,IQ3_XXS --shapes 17408x5120,5120x17408 --tokens 16,32,64,128,2048 --variants new
# ncu stall reasons (v5, wt-r2b): tiled n=2048 (C128) and n=32 (C32), MMQ n=2048
source /workspace/box-env.sh
export VLLM_GGUF_LCPP=1 GSQ_ALLOW_GPU=1 WT=/workspace/wt-r2b PYTHONPATH=/workspace/wt-r2b/plugin:/workspace/wt-r2b/tools:/workspace/wt-r2b/tests/gpu
for WTP in wt-r2b wt-r2h; do for n in 2048 32; do
  export WT=/workspace/$WTP PYTHONPATH=/workspace/$WTP/plugin:/workspace/$WTP/tools:/workspace/$WTP/tests/gpu
  timeout 600 /usr/local/cuda/bin/ncu -k regex:"iq3_packed_mmq|mul_mat_q" --launch-skip 4 --launch-count 2 \
    --section WarpStateStats --section SchedulerStats --section InstructionStats --section LaunchStats --section Occupancy \
    --section ComputeWorkloadAnalysis --section MemoryWorkloadAnalysis \
    /workspace/gsq-vllm/.venv/bin/python $S/r2_prof.py IQ3_S 17408 5120 $n > /workspace/logs/opt/r2/it1-ncu-$WTP-$n.log 2>&1
  echo "ncu $n rc=$?" >> /workspace/logs/opt/r2/it1-ncu-$WTP-$n.log
done; done
