#!/bin/bash
# R3-40 decode ladder on production's main argv (the box re-benchmark on main, and the check after
# every kept change). Two methods, one server each:
#   prodbench  bench/speed/run.sh gsq --start, decode only (production's run_benchmarks.sh single,
#              8 real prompts x 1024 tokens, c=1/2/4/8, pass 2 kept), as in REPORT.md
#   steady     r3load steady windows at short context (2k-token prompts), c=1/2/4/8, T=0, 40 s each:
#              ms/step straight from /metrics drafts (k from production's schedule: 5,5,5,3)
# Env: R3_PLUGIN_WT (plugin build under test), GSQ_VENV_OVERRIDE (a patched copy of venv-main),
# R3_TAG (label for the logs dir, default "main").
# Output: /workspace/logs/r3/40-ladder-$R3_TAG/{summary.txt, prodbench/, steady/}
# GPU time: ~35 min.
#   bash 40-ladder.sh [--plan]
source "$(dirname "$0")/lib.sh"
R3_TAG=${R3_TAG:-main}
TAG=$R3_TAG
r3_init 40-ladder "$@"
if [ $R3_PLAN = 1 ]; then sed -n '2,13p' "$0"; echo "plugin: $R3_PLUGIN_WT venv: ${GSQ_VENV_OVERRIDE:-/workspace/venv-main}"; exit 0; fi
r3_env
r3_preflight

r3_step prodbench
mkdir -p "$L/prodbench"
( OUT=$L/prodbench GSQ_PREFILL=" " GSQ_RUNS=$L "$R3_WT/bench/speed/run.sh" gsq --start ) > "$L/prodbench/run.log" 2>&1 \
  || { tail -30 "$L/prodbench/run.log"; r3_die "prodbench"; }
r3_stop
grep -E "^ROW" "$L/prodbench/summary.txt" > "$L/prodbench/rows.txt" || true

steady_ladder() {
  r3_serve steady
  "${LOAD[@]}" warm || r3_die warm
  for c in 1 2 4 8; do
    k=5; [ $c -ge 5 ] && k=3
    "${LOAD[@]}" steady --conc $c --tokens 2000 --max-tokens 4000 --window 40 --k $k --pname lad$c \
      --tag "c$c" --out "$R" || r3_die "c$c window"
  done
}
r3_step steady
r3_variant steady steady_ladder

r3_summary "R3-40 decode ladder [$TAG] (box, $(date -u +%F)), production's main argv; plugin $(git -C "$R3_PLUGIN_WT" rev-parse --short HEAD 2>/dev/null) venv ${GSQ_VENV_OVERRIDE:-/workspace/venv-main}" \
  "--- production run_benchmarks.sh single, pass 2 (tok/s; ms/step = C x 1000 / tok/s x tok/step) ---" \
  "$(grep -E "pass 2|conc|MTP|clocks" "$L/prodbench/rows.txt" 2>/dev/null | tail -20)" "" \
  "--- steady windows, short context (ms/step from /metrics) ---" "$(cat "$L/steady/lines.txt" 2>/dev/null)" "" \
  "0.27.1 reference (REPORT.md, k=3): 27.9 / 31.6 / 35.7 / 44.2 ms/step c=1/2/4/8"
cat "$L/summary.txt"
r3_finish
