#!/bin/bash
# R2 after review (one gpuq job, wt-r2): the packed tests, GPU guards -k packed, CPU guards, and the
# routed op's microbench at the shapes / row counts the plan logic touches.
S=/workspace/wt-r2/cloud/results/phase3/box-scripts
WT=/workspace/wt-r2 bash $S/r2-tests.sh verify "packed or pack_roundtrip or tiled"
WT=/workspace/wt-r2 bash $S/r2-micro.sh verify --types IQ3_S --shapes 17408x5120,5120x17408 --tokens 16,32,128,129,2048 --variants new
