#!/bin/bash
# R3-53 the 50-plp-repro requests on the fixed plugin ($R3_PLUGIN_WT) with vLLM's prompt logprobs
# computed in bounded row passes (patches/prompt-logprobs-chunked.patch on /workspace/venv-r3-plp,
# a hardlinked overlay copy of venv-main; a proposal for the qwen38/main overlay, production
# untouched). Servers: GSQ and stock W4A16 (stock vLLM main dies at 512 prompt tokens without it).
# Output: /workspace/logs/r3/53-plp-vllmpatch/summary.txt + 50-plp-repro-vllmpatch/. GPU time ~15 min.
#   bash 53-plp-vllmpatch.sh [--plan]
source "$(dirname "$0")/lib.sh"
r3_init 53-plp-vllmpatch "$@"
if [ $R3_PLAN = 1 ]; then sed -n '2,7p' "$0"; exit 0; fi
r3_step overlay
bash "$R3_S/overlay-venv.sh" /workspace/venv-r3-plp "$R3_S/../patches/prompt-logprobs-chunked.patch" > "$L/overlay.log" 2>&1 \
  || { cat "$L/overlay.log"; r3_die "overlay venv"; }
r3_step repro
( R3_TAG=vllmpatch GSQ_VENV_OVERRIDE=/workspace/venv-r3-plp bash "$R3_S/50-plp-repro.sh" ) > "$L/repro.log" 2>&1
rc=$?
r3_summary "R3-53 repro with the vLLM prompt-logprobs patch, plugin $R3_PLUGIN_WT ($(git -C "$R3_PLUGIN_WT" rev-parse --short HEAD)): rc=$rc" \
  "$(cut -c1-300 "$R3_LOGS/50-plp-repro-vllmpatch/summary.txt" 2>/dev/null)"
cat "$L/summary.txt"
