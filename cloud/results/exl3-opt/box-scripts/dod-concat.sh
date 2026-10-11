#!/bin/bash
# EXL3 DoD, concat (EXL3_MR_CONCAT): A/B ladder 2hf vs 2hfc at fixed k=3 on production's argv (12-mr-ladder). Before the
# draft head's chunked quantization 2hfc could not load there: under --enable-cumem-allocator the weights load into vLLM's
# private memory pool, the parts freed by the concatenation stay reserved in it until the load ends (snapshot probe,
# concat-probe/: 22.47 GiB reserved at the fp8 head's entry against 17.68 for 2hf, both 14.62 GiB allocated), and the
# head's whole-matrix fp32 temporaries OOMed. Exits non-zero on any failure.
#   gpuq submit exl3dod-concat -- bash /workspace/wt-exl3-dod2/cloud/results/exl3-opt/box-scripts/dod-concat.sh
WT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
export WT EXL3_OPT_TAG=${EXL3_OPT_TAG:--dod} EXL3_OPT_IGNORE_DEPS=1
cd "$WT" || exit 1
exec bash "$WT/cloud/results/exl3-opt/box-scripts/run-job.sh" 12-mr-ladder 2hf 2hfc
