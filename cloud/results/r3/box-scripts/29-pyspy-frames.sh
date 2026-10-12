#!/bin/bash
# R3-29 the box side of prod-sample-20261001.md 3b: the EngineCore main thread by step segment
# (drain wait at the first draft pass, draft-loop waits, target forward launch split, drafter host
# work, sampler, runner prep, scheduler) in ms/step, same py-spy recipe as production (250 Hz,
# --nonblocking, no native frames, 30 s), plus a 100 Hz --idle record for off-CPU ticks.
# Production's main argv; windows: c=8 x 18k (k=3, near production's 8-9 running) and c=2 x 96k (k=5).
# Output: /workspace/logs/r3/29-pyspy-frames/{summary.txt, srv/pyspy-*.raw, segments-*.txt}. GPU ~20 min.
#   bash 29-pyspy-frames.sh [--plan]
source "$(dirname "$0")/lib.sh"
r3_init 29-pyspy-frames "$@"
PYSPY=${R3_PYSPY:-/workspace/r3-tools/bin/py-spy}
if [ $R3_PLAN = 1 ]; then sed -n '2,8p' "$0"; exit 0; fi
export R3_EXTRA_PYTHONPATH=$R3_S/pyhook R3_PTRACE_ANY=1
r3_env
r3_preflight
[ -x "$PYSPY" ] || r3_die "no py-spy at $PYSPY"
r3_serve srv
export R ENGINE_PID PYSPY
"${LOAD[@]}" warm || r3_die warm
r3_step c8; "${LOAD[@]}" steady --conc 8 --tokens 18000 --max-tokens 9000 --window 40 --k 3 --prewarm --tag c8 --out "$R" --hook "bash $R3_S/29-hook.sh c8" || r3_die c8
r3_step c2; "${LOAD[@]}" steady --conc 2 --tokens 96000 --max-tokens 12000 --window 40 --k 5 --prewarm --tag c2 --out "$R" --hook "bash $R3_S/29-hook.sh c2" || r3_die c2
r3_stop
r3_summary "R3-29 py-spy step segments (box, $(date -u +%F)), production's main argv, same recipe as prod-sample-20261001.md" "$(cat "$R/lines.txt")" ""
for t in c8 c2; do
  ms=$("$PY" -c "import json;print(json.load(open('$R/$t-steady.json'))['ms_per_engine_step'])")
  "$PY" "$R3_S/r3analyze.py" segments "$R/pyspy-$t.raw" --ms-per-step "$ms" --rate 250 --seconds 30 --json "$R/segments-$t.json" > "$R/segments-$t.txt"
  "$PY" "$R3_S/r3analyze.py" segments "$R/pyspy-$t-idle.raw" --ms-per-step "$ms" --rate 100 --seconds 20 > "$R/segments-$t-idle.txt"
  r3_summary "--- $t, on-CPU samples (250 Hz, no --idle) ---" "$(cat "$R/segments-$t.txt")" "--- $t, with --idle (100 Hz): where off-CPU ticks sit ---" "$(cat "$R/segments-$t-idle.txt")" ""
done
r3_summary "production (S3, 72-89 ms/step at ~8-9 running): drain wait 46 %, forward launch 39 % (eager GDN 23 %), drafter host 5.3 %, runner prep 4.4 %, sampler 3.0 %, scheduler 0.6 %; main thread off-CPU 31 % of ticks"
cat "$L/summary.txt"
