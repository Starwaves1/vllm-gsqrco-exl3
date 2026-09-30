#!/bin/bash
# R2 it2 (one gpuq job, wt-r2c = v9: v5 + a fixup kernel without per-element divisions, + DEV ablations)
S=/workspace/wt-r2c/cloud/results/phase3/box-scripts
export WT=/workspace/wt-r2c
bash $S/r2-micro.sh it2-abl --types IQ3_S --shapes 17408x5120 --tokens 32,128,2048 --variants new --abl 1,2,4,8,16,7,31
bash $S/r2-micro.sh it2-g --types IQ3_S --shapes 17408x5120,5120x17408 --tokens 16,32,128 --variants new --g 68,82
bash $S/r2-micro.sh it2-all --types IQ3_S,IQ3_XXS --shapes 17408x5120,5120x17408,10240x5120 --tokens 8,16,32,64,128,256,2048 --variants new,r1,mmq,unpack_mmq
# v10 (wt-r2e): v9 + C16 / C32 at 2 CTAs per SM (<= 128 registers): tile widths 17 = C16x2, 33 = C32x2
WT=/workspace/wt-r2e bash $S/r2-micro.sh it2-v10 --types IQ3_S --shapes 17408x5120,5120x17408 --tokens 8,16,32,64,128,2048 --variants new --tn 16,17,32,33
