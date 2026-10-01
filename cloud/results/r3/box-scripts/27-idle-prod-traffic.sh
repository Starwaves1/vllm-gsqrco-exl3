#!/bin/bash
# R3-27 production-shaped traffic vs job 20's raw completions: does the API path (chat template,
# qwen3 reasoning parser, qwen3_coder tool parser, streaming deltas) add per-step GPU idle?
# One server, production's exact main argv (both parsers are in it). c=2, ~96k-token contexts, k=5:
#   a-raw     /v1/completions on token ids, ignore_eos (job 20's c2 load; the control)
#   b-chat    streaming chat completions, reasoning on (template default), default sampling
#   c-tools   b + the two agent tools in the request (tool_choice auto)
#   d-turns   c in 2,048-token turns: each turn appends the reply + a short user message, so turns
#             after the first are prefix-cache hits (production's ~90 %); 120 s window
# Per window: ms/step and GPU idle as in job 20 (drafts/s + nvidia-smi util), the same over
# "clean" seconds only (running == 2, no prompt tokens: turn boundaries excluded), and API-server /
# EngineCore CPU % (/proc utime+stime over the window).
# Output: /workspace/logs/r3/27-idle-prod-traffic/{summary.txt, srv/}. GPU time ~25 min.
#   bash 27-idle-prod-traffic.sh [--plan]
source "$(dirname "$0")/lib.sh"
r3_init 27-idle-prod-traffic "$@"
if [ $R3_PLAN = 1 ]; then sed -n '2,15p' "$0"; exit 0; fi
r3_env
r3_preflight
for f in --reasoning-parser --tool-call-parser; do
  grep -qx -- "$f" "$GSQ_PROD_ARGV" || r3_die "production argv lacks $f"
done
r3_serve srv
export R3_PIDS="api:$API_PID,engine:$ENGINE_PID"
"${LOAD[@]}" warm || r3_die warm
r3_step a-raw;   "${LOAD[@]}" steady --conc 2 --tokens 96000 --max-tokens 12000 --window 90 --k 5 --pname c2 --tag a-raw --out "$R" || r3_die a-raw
r3_step b-chat;  "${LOAD[@]}" steady --chat --temperature default --conc 2 --tokens 96000 --max-tokens 12000 --window 90 --k 5 --pname chat --tag b-chat --out "$R" || r3_die b-chat
r3_step c-tools; "${LOAD[@]}" steady --chat --tools --temperature default --conc 2 --tokens 96000 --max-tokens 12000 --window 90 --k 5 --pname chat --tag c-tools --out "$R" || r3_die c-tools
r3_step d-turns; "${LOAD[@]}" steady --chat --tools --turns 40 --temperature default --conc 2 --tokens 96000 --max-tokens 2048 --window 120 --k 5 --pname chat --tag d-turns --allow-problems --out "$R" || r3_die d-turns
r3_stop
r3_summary "R3-27 prod-shaped traffic (box, $(date -u +%F)), production's main argv, c=2 x 96k, k=5" "$(cat "$R/lines.txt")" "" \
  "job 20 control: c2 57.3 ms/step, idle 6.3; production n=2: 63.6 ms/step, idle 13.2 (util 79 %)" \
  "decision: b/c/d idle (clean) >= ~10 ms -> the API path is the host cost (vLLM API-server side, proposal); ~6 ms like a-raw -> not the parsers"
cat "$L/summary.txt"
