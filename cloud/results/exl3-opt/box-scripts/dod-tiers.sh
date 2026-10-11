#!/bin/bash
# EXL3 DoD, tiers: 21-tier.sh for erlidev's 3.00 / 3.50 / 4.00 / 4.50 bpw, one at a time (each non-shipped tier is
# deleted after its run), mode 2hf at fixed k=3. Exits non-zero if any tier failed (no failure is swallowed).
#   gpuq submit exl3dod-tiers -- bash /workspace/wt-exl3-dod/cloud/results/exl3-opt/box-scripts/dod-tiers.sh [TIER...]
WT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
export WT EXL3_OPT_TAG=${EXL3_OPT_TAG:--dod} EXL3_OPT_IGNORE_DEPS=1
cd "$WT" || exit 1
tiers=("$@"); [ $# -gt 0 ] || tiers=(SC_3.00bpw_H4_V4 SC_3.50bpw_H4_V6 SC_4.00bpw_H5_V6 SC_4.50bpw_H5_V6)
rc=0
for tier in "${tiers[@]}"; do
  bash "$WT/cloud/results/exl3-opt/box-scripts/run-job.sh" 21-tier "$tier" 2hf || { echo "tier $tier: rc=$?"; rc=1; }
done
exit $rc
