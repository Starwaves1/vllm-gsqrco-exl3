#!/bin/bash
# R3-54 isolate the echo / short-prompt prompt_logprobs NaN (50: GSQ's final hidden states hold NaN
# at some prompt positions of a 5- or 40-token prompt, not at 200; W4A16 fine; 52: none at all with
# --enforce-eager). One server per variant, production's main argv plus:
#   graphs     nothing (the repro)
#   nographs   --compilation-config cudagraph_mode NONE (torch.compile kept)
#   lcpp0      VLLM_GGUF_LCPP=0 (the plugin's stock kernels, graphs on)
#   cap4       max_cudagraph_capture_size 4 (prefills above 4 tokens run without graphs)
#   noconn     no --kv-transfer-config (graphs on)
# Requests: r3plp.py's short cases (echo on 5 / 40 / 200-token prompts, prompt_logprobs=1 on the
# 5-token prompt, logprobs only), NaN probe on every LogitsProcessor call (pyhook R3_NANCHECK).
# Output: /workspace/logs/r3/54-nan-matrix/{summary.txt, <variant>/}. GPU time ~25 min.
#   bash 54-nan-matrix.sh [--plan]
source "$(dirname "$0")/lib.sh"
r3_init 54-nan-matrix "$@"
if [ $R3_PLAN = 1 ]; then sed -n '2,14p' "$0"; exit 0; fi
export R3_EXTRA_PYTHONPATH=$R3_S/pyhook
r3_env
r3_preflight
CC_PROD=$(grep -A1 -x -- "--compilation-config" "$GSQ_PROD_ARGV" | tail -1)
run() {  # tag
  mkdir -p "$L/$1"
  export R3_NANCHECK=$L/$1/nan.jsonl
  r3_serve "$1"
  "$PY" "$R3_S/r3plp.py" --long "" --out "$R/results.jsonl" || r3_die r3plp
}
v_graphs() { run graphs; }
v_nographs() { R3_MUT=("set|--compilation-config|${CC_PROD%\}},\"cudagraph_mode\":\"NONE\"}"); run nographs; }
v_lcpp0() { export VLLM_GGUF_LCPP=0; run lcpp0; }
v_cap4() { R3_MUT=("set|--compilation-config|${CC_PROD/\"max_cudagraph_capture_size\":48/\"max_cudagraph_capture_size\":4}"); run cap4; }
v_noconn() { R3_MUT=("drop|--kv-transfer-config"); run noconn; }
r3_summary "R3-54 NaN matrix (box, $(date -u +%F)), plugin $(git -C "$R3_PLUGIN_WT" rev-parse --short HEAD); prod compilation-config $CC_PROD"
for v in graphs nographs lcpp0 cap4 noconn; do
  r3_step "$v"; r3_variant "$v" "v_$v"
  if [ -f "$L/$v/results.jsonl" ]; then
    r3_summary "$v: $("$PY" -c "
import json,sys
for l in open('$L/$v/results.jsonl'):
    r=json.loads(l); print(r['case']+'='+str(r['status']), end=' ')")"
    [ -f "$L/$v/nan.jsonl" ] && r3_summary "   nan probe: $(grep -c '"logits_nan": [1-9]' "$L/$v/nan.jsonl") calls with NaN logits; first: $(grep -m1 '"hidden_nan": [1-9]' "$L/$v/nan.jsonl" | cut -c1-200)"
    r3_summary "   server compile: $(grep -E "cudagraph_mode|CUDAGraphMode|capture" "$L/$v/server.log" | head -2 | cut -c1-200 | tr '\n' ' ')"
  fi
done
cat "$L/summary.txt"
r3_finish
