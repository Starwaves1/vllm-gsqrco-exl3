#!/bin/bash
# R3-21 idle A/B: the 20-idle-baseline c=2 window (2 x 96k prompts, k=5, 90 s) under three argv
# variants, one server each, nothing else changed:
#   noconn   --kv-transfer-config removed (no OffloadingConnector at all: no per-step
#            build_connector_meta / tiering on_schedule_end / write-back scan / worker hooks)
#   nofs     the fs secondary tier removed, CPU tier kept (GSQ_KV_FS_TIER=0, as production's start
#            script does when the drive is missing); CPU tier filled past the write-back watermark
#   async    --async-scheduling instead of --no-async-scheduling, connector + fs tier kept, CPU tier
#            filled (TEST ONLY: production runs sync scheduling; nothing is kept); adds a c=8 window
# Compare each 'c2' line with 20-idle-baseline's c2 line (same prompts, same window rules).
# A variant that fails is reported and the others still run; the job then exits non-zero.
# Output: /workspace/logs/r3/21-idle-connector-off/{summary.txt, <variant>/...}
# GPU time: ~40 min (noconn ~10, nofs ~13, async ~17).
#   bash 21-idle-connector-off.sh [--plan]
source "$(dirname "$0")/lib.sh"
r3_init 21-idle-connector-off "$@"
if [ $R3_PLAN = 1 ]; then
  sed -n '2,15p' "$0"
  echo "argv mutations: noconn: drop --kv-transfer-config | nofs: GSQ_KV_FS_TIER=0 | async: swap --no-async-scheduling -> --async-scheduling"
  exit 0
fi
r3_env
r3_preflight

c2() { "${LOAD[@]}" steady --conc 2 --tokens 96000 --max-tokens 12000 --window 90 --k 5 --tag c2 --out "$R" || r3_die "c2 window"; }

v_noconn() {
  R3_MUT=("drop|--kv-transfer-config")
  r3_serve noconn
  "${LOAD[@]}" warm || r3_die warm
  c2
}
v_nofs() {
  export GSQ_KV_FS_TIER=0
  r3_serve nofs
  "${LOAD[@]}" warm || r3_die warm
  "${LOAD[@]}" fill --n 2 --tokens 90000 --conc 2 || r3_die fill
  c2
}
v_async() {
  R3_MUT=("swap|--no-async-scheduling|--async-scheduling")
  r3_serve async
  "${LOAD[@]}" warm || r3_die warm
  "${LOAD[@]}" fill --n 2 --tokens 90000 --conc 2 || r3_die fill
  c2
  "${LOAD[@]}" steady --conc 8 --tokens 18000 --max-tokens 9000 --window 90 --k 3 --tag c8 --out "$R" || r3_die "c8 window"
}

r3_summary "R3-21 idle A/B (box, $(date -u +%F)); baseline = 20-idle-baseline c2 line"
for v in noconn nofs async; do
  r3_step "variant $v"
  r3_variant "$v" "v_$v"
  [ -f "$L/$v/lines.txt" ] && r3_summary "$(sed "s/^/$v /" "$L/$v/lines.txt")"
done
B=$R3_LOGS/20-idle-baseline/srv/lines.txt
[ -f "$B" ] && r3_summary "baseline (20) $(grep '^c2' "$B")"
r3_summary "" "decision: idle (or pooled ms/step) drops >= 8 ms with noconn -> the connector is the cost; nofs vs noconn splits fs tier vs CPU tier/connector core -> profile its hooks (22 instr 'KV connector' block). < 3 ms change -> not the connector -> 22's engine-loop / runner breakdown and py-spy flame. async: the ms/step it saves is the overlappable host time (test only, propose)."
cat "$L/summary.txt"
r3_finish
