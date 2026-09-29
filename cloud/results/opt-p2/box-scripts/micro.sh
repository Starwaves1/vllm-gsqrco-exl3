#!/bin/bash
# opt-p2 microbench job: build the stage in place, run MICRO (a python file in box-scripts).
#   micro.sh TAG MICRO
TAG=$1; WT=/workspace/wt-p2-$TAG
source /workspace/wt-p2-$TAG/cloud/results/opt-p2/box-scripts/lib.sh
exec > >(tee $L/$TAG-micro.log) 2>&1
date -u +"start %FT%TZ"; wait_build
$GSQ_VENV/bin/python $S/$2
date -u +"end %FT%TZ"
