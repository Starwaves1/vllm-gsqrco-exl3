#!/bin/bash
# routing sweep: every weight shape of this model per type; mma_k (stream-K, and v4 fixed-split schedule "tiled")
# vs vendored MMQ in the same process; ragged MTP sizes included
L=/workspace/logs/opt/k3; R=${1:-/workspace/wt-k3x}; T=${2:-route}; N=9,12,16,17,20,24,28,32,33,48,64
$L/ab.sh $T-q4k $R Q4_K $N ";tiled" 1024x5120,2048x5120,5120x6144,6144x5120,10240x5120,12288x5120,17408x5120,5120x17408,34816x5120,248320x5120
$L/ab.sh $T-iq4 $R IQ4_XS $N ";tiled" 1024x5120,2048x5120,5120x6144,6144x5120,10240x5120,12288x5120,17408x5120,5120x17408,34816x5120
$L/ab.sh $T-iq2 $R IQ2_S $N ";tiled" 1024x5120,6144x5120,17408x5120,5120x17408,34816x5120
