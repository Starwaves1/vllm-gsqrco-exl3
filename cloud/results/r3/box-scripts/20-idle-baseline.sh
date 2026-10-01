#!/bin/bash
# R3-20 idle baseline: production's exact main argv on the box, then steady decode measured the way
# the 2026-10-01 production profile measured it (/metrics drafts per second + nvidia-smi util at 1 Hz).
#   1. serve (prod argv; CPU tier 13 GiB as on every box run, fs tier capped at 7 GB (box disk))
#   2. warm-up, then fill the CPU tier past its 0.85 write-back watermark (2 x 90k distinct prompts,
#      prefill only) so the tiering manager's per-step write-back scan runs as in production
#   3. c2s: c=2 at short context (2 x 4k), 60 s window: same n and k as c2, ~25x less context, so
#      c2 - c2s isolates any per-step cost that grows with context (production's total context sat
#      at ~200k at every running count, so its data cannot tell "fixed" from "grows with context")
#   3b. c=2: two distinct ~96k-token prompts (prod n=2 median context 204k), 90 s window after both
#      decode (k=5 by the schedule)
#   4. c=8: eight distinct 18k-token prompts (144k -> ~200k during the window; prod n=8 median 195k),
#      90 s window (k=3)
# Output: /workspace/logs/r3/20-idle-baseline/{summary.txt, srv/*-steady.json, metrics, nvsmi}
# GPU time: ~22 min (load 4, fill 4, c2s 1.2, c=2 prefill 4 + 1.6, c=8 prefill 4 + 1.6).
#   bash 20-idle-baseline.sh [--plan]
source "$(dirname "$0")/lib.sh"
r3_init 20-idle-baseline "$@"
if [ $R3_PLAN = 1 ]; then
  sed -n '2,17p' "$0"; echo "argv mutations: none (production's main argv)"; exit 0
fi
r3_env
r3_preflight
r3_serve srv
r3_step warm;  "${LOAD[@]}" warm || r3_die "warm-up"
r3_step fill;  "${LOAD[@]}" fill --n 2 --tokens 90000 --conc 2 || r3_die "fill"
r3_step c2s;   "${LOAD[@]}" steady --conc 2 --tokens 4000 --max-tokens 9000 --window 60 --k 5 --tag c2s --out "$R" || r3_die "c2s window"
r3_step c2;    "${LOAD[@]}" steady --conc 2 --tokens 96000 --max-tokens 12000 --window 90 --k 5 --tag c2 --out "$R" || r3_die "c2 window"
r3_step c8;    "${LOAD[@]}" steady --conc 8 --tokens 18000 --max-tokens 9000 --window 90 --k 3 --tag c8 --out "$R" || r3_die "c8 window"
curl -s "$GSQ_URL/metrics" > "$R/metrics-final.txt"
r3_stop
r3_summary "R3-20 idle baseline (box, production's main argv, $(date -u +%F))" "$(cat "$R/lines.txt")" "" \
  "reference: $R3_PROD_REF" \
  "decision: box idle (c2/c8 'idle' column) within ~3 ms of production's 13-15 ms -> the gap reproduces here: run 21/22. Box idle <= ~6 ms -> the box does not reproduce production's idle: run 26 (CPU contention) and ask Garrett for a prod-side check."
cat "$L/summary.txt"
