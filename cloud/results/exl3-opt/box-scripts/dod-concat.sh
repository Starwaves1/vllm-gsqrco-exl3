#!/bin/bash
# EXL3 DoD, concat (EXL3_MR_CONCAT): what it buys, measured as an A/B ladder 2hf vs 2hfc at fixed k=3 on production's argv
# minus --enable-cumem-allocator. On production's argv 2hfc cannot load: under the cumem allocator the weights load into
# vLLM's private memory pool, and the parts freed by the concatenation stay reserved there until the load ends (snapshot
# probe: 22.47 GiB reserved at the fp8 draft head's entry against 17.68 for 2hf, the same 14.62 GiB allocated), so the
# head's quantization OOMs. Exits non-zero on any failure.
#   gpuq submit exl3dod-concat -- bash /workspace/wt-exl3-dod2/cloud/results/exl3-opt/box-scripts/dod-concat.sh
WT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
export WT EXL3_OPT_TAG=${EXL3_OPT_TAG:--dod-nocumem} EXL3_OPT_IGNORE_DEPS=1 EXL3_ARGV_DROP=--enable-cumem-allocator
cd "$WT" || exit 1
exec bash "$WT/cloud/results/exl3-opt/box-scripts/run-job.sh" 12-mr-ladder 2hf 2hfc
