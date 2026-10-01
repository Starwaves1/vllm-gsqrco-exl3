#!/bin/bash
# R3-55 the full 50-plp-repro request set on the fixed plugin ($R3_PLUGIN_WT) with both vLLM-side
# proposals on an overlay copy of venv-main (/workspace/venv-r3-plp2; production untouched):
#   patches/prompt-logprobs-chunked.patch        prompt logprobs in 64 MiB passes (the OOM)
#   patches/prompt-logprobs-after-drafter.patch  copy the target hidden states before the drafter
#                                                runs when prompt logprobs are wanted (the NaN: the
#                                                drafter's CUDA graphs share the global graph pool)
# CUDA graphs on, production's main argv. Servers: GSQ and stock W4A16. Also compares echo's prompt
# logprobs with the eager run of 52 (same prompt, T=0).
# Output: /workspace/logs/r3/55-plp-both-patches/summary.txt + 50-plp-repro-both/. GPU time ~15 min.
#   bash 55-plp-both-patches.sh [--plan]
source "$(dirname "$0")/lib.sh"
r3_init 55-plp-both-patches "$@"
if [ $R3_PLAN = 1 ]; then sed -n '2,11p' "$0"; exit 0; fi
P=$R3_S/../patches
r3_step overlay
bash "$R3_S/overlay-venv.sh" /workspace/venv-r3-plp2 "$P/prompt-logprobs-chunked.patch" "$P/prompt-logprobs-after-drafter.patch" \
  > "$L/overlay.log" 2>&1 || { cat "$L/overlay.log"; r3_die "overlay venv"; }
grep -c "rows_per_pass\|hidden_states = hidden_states.clone()" \
  /workspace/venv-r3-plp2/lib/python3.12/site-packages/vllm/v1/worker/gpu_model_runner.py | grep -qx 2 \
  || r3_die "overlay is missing one of the two patches"
r3_step repro
( R3_TAG=both GSQ_VENV_OVERRIDE=/workspace/venv-r3-plp2 bash "$R3_S/50-plp-repro.sh" ) > "$L/repro.log" 2>&1
rc=$?
r3_summary "R3-55 repro with both vLLM patches, plugin $R3_PLUGIN_WT ($(git -C "$R3_PLUGIN_WT" rev-parse --short HEAD)): rc=$rc" \
  "$(cut -c1-300 "$R3_LOGS/50-plp-repro-both/summary.txt" 2>/dev/null)"
E=$R3_LOGS/52-nan-trace/srv/echo5.json
if [ -f "$E" ] && [ -f "$R3_LOGS/50-plp-repro-both/gsq/results.jsonl" ]; then
  r3_summary "eager (52) echo5 prompt logprobs: $("$PY" -c "import json; c=json.load(open('$E'))['choices'][0]; print(c['logprobs']['token_logprobs'][:5])")" \
    "graphs+patch echo-T0 head:          $(grep '"echo-T0"' "$R3_LOGS/50-plp-repro-both/gsq/results.jsonl" | "$PY" -c "import json,sys; print(json.loads(sys.stdin.read())['token_logprobs_head'][:5])")"
fi
cat "$L/summary.txt"
