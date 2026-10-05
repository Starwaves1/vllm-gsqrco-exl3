#!/bin/bash
# concat probe: CUDA memory snapshot (alloc stacks) at the fp8 draft head's entry, modes 2hf / 2hfc / 2hc, load only
export WT=/workspace/wt-exl3-dbg2 EXL3_OPT_TAG=-dbg2
cd $WT || exit 1
source $WT/cloud/results/exl3-opt/box-scripts/lib.sh
require_mr_build
rc=0
for m in "$@"; do
  D=$R/snap-$m; rm -rf $D; mkdir -p $D
  # the bf16 head (no f) never enters the fp8 method: snapshot then comes from the KV-cache profile instead
  EXL3_DBG_SNAP=$D/snap.pkl serve_mr $m $D && { echo "$m: server came up (unexpected with f)"; }
  stopall
  grep -E "EXL3DBG|Model loading took|OutOfMemory|Loading drafter" $D/server.log | cut -c1-300
  [ -s $D/snap.pkl ] || { echo "$m: no snapshot"; rc=1; }
done
exit $rc
