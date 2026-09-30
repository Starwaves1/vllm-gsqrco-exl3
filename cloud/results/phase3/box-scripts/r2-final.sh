#!/bin/bash
# R2 final (one gpuq job, wt-r2): microbench of the routed kernels vs MMQ on all shapes, then the
# final tests (r2-final-tests.sh).
S=/workspace/wt-r2/cloud/results/phase3/box-scripts
WT=/workspace/wt-r2 bash $S/r2-micro.sh final --types IQ3_S,IQ3_XXS --shapes 17408x5120,5120x17408,10240x5120 \
  --tokens 1,4,8,9,16,24,32,48,64,128,129,256,512,2048 --variants new,r1,mmq
WT=/workspace/wt-r2 bash $S/r2-final-tests.sh
