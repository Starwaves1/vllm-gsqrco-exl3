#!/bin/bash
# R2 it3 (one gpuq job): v11 (tiles 256 x 16 / 32 / 64, tile width and whole-wave / split schedule from
# a unit-cost model) with one 128-value block per pipeline step (wt-r2d) vs two (wt-r2f); then the
# packed parity tests on wt-r2d.
S=/workspace/wt-r2d/cloud/results/phase3/box-scripts
for T in d f; do
  WT=/workspace/wt-r2$T bash $S/r2-micro.sh it3-$T --types IQ3_S,IQ3_XXS --shapes 17408x5120,5120x17408,10240x5120 \
    --tokens 8,9,12,16,24,32,48,64,128,129,256,512,2048 --variants new,r1
done
WT=/workspace/wt-r2d bash $S/r2-micro.sh it3-tn --types IQ3_S --shapes 17408x5120,5120x17408,10240x5120 \
  --tokens 16,32,64,128,256,2048 --variants new --tn 16,32,64
# v12 (wt-r2h): v11 + weights through one shared-memory stage (cp.async, ~1.5 units ahead)
WT=/workspace/wt-r2h bash $S/r2-micro.sh it3-h --types IQ3_S,IQ3_XXS --shapes 17408x5120,5120x17408,10240x5120 \
  --tokens 9,12,16,24,32,48,64,128 --variants new
WT=/workspace/wt-r2d bash $S/r2-tests.sh it3 "tiled or routing_packed or packed_layer"
