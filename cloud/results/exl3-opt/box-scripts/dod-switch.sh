#!/bin/bash
# EXL3 DoD, single-stack switch test: one checkout (this one) and one venv (venv-main) serve GSQ-RCO (scripts/serve-gsq.sh,
# the GGUF plugin) -> EXL3 (scripts/serve-exl3.sh, mode 2hf) -> GSQ -> EXL3, 10-minute torture legs, both at a fixed MTP k=3
# (lib.sh rewrites the argv for both). Both serve scripts come from this checkout, so GSQ_HF_CONFIG and GSQ_ROOT agree and
# the KV-tier namespace rule (scripts/env.sh, gsq_assert_hf_config) holds unchanged. The 2026-10-04 run started serve-gsq.sh
# from another checkout (wt-torture) under this one's exported GSQ_HF_CONFIG, and the rule refused it.
#   gpuq submit exl3dod-switch -- bash /workspace/wt-exl3-dod/cloud/results/exl3-opt/box-scripts/dod-switch.sh
WT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
export WT EXL3_OPT_TAG=${EXL3_OPT_TAG:--dod}
cd "$WT" || exit 1
source "$WT/cloud/results/exl3-opt/box-scripts/lib.sh"
mode_env 2hf
# the GGUF plugin from a built checkout of the same source (plugin/ is identical at 32ae6ec and here; this one is unbuilt)
export PYTHONPATH=$WT/plugin-exl3:${GSQ_PLUGIN_DIR:-/workspace/wt-gsq-32ae6ec/plugin}:$WT/tools
export GSQ_KV_TIER_MAX_BYTES=8000000000 VLLM_GGUF_LCPP=1
rm -rf "$GSQ_KV_TIER_ROOT"; box_clean_shm || true
exec bench/torture.sh switch --a scripts/serve-gsq.sh --b scripts/serve-exl3.sh --rounds 2 --minutes 10 --skip plog,echo
