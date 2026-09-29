#!/bin/bash
# integration-1, one gpuq job = one server: the definition-of-done ladder, bench/speed/run.sh gsq
# unmodified (production's run_benchmarks.sh single twice: decode c=1,2,4,8; salted prefill
# 8k/64k/180k; MTP acceptance; per-phase clock logs). No profiler configured.
source /workspace/wt-int/cloud/results/integration-1/box-scripts/lib.sh
R=/workspace/runs/int1-ladder; rm -rf $R; mkdir -p $R
date -u +"start %FT%TZ"; stopall
serve
OUT=$R bench/speed/run.sh gsq > $L/ladder.log 2>&1; echo "speed rc=$?"
stopall
grep -E "^ROW" $R/summary.txt | tail -20
date -u +"end %FT%TZ"
