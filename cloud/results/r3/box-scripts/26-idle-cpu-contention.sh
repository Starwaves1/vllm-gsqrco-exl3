#!/bin/bash
# R3-26 (conditional: run when 20 shows the box idle well below production's) host-CPU sensitivity of
# the decode step. Production's host is a 6-core / 12-thread Ryzen 5 9600X held at 3.9 GHz
# (cpufreq max 3.9 GHz, EPP "power") that also runs a qemu VM averaging ~4 busy threads; the box is a
# 32-core Threadripper 3970X. Two servers, production's main argv, 20's c=2 x 96k window:
#   pin6       server pinned (taskset) to 6 physical cores + their SMT siblings
#   pin6-load  same, plus 4 busy-loop threads pinned to the same 12 logical CPUs (~ the VM's load)
# If pin6-load's idle approaches production's 13-15 ms, production's gap is host contention: the
# fix is on the host (Garrett: VM placement / CPU pinning / boost), not in vLLM or the plugin.
# Output: /workspace/logs/r3/26-idle-cpu-contention/{summary.txt, pin6/, pin6-load/}
# GPU time: ~25 min.
#   bash 26-idle-cpu-contention.sh [--plan]
source "$(dirname "$0")/lib.sh"
r3_init 26-idle-cpu-contention "$@"
# 6 physical cores (first socket/die) and their SMT siblings, from lscpu's CPU,CORE table
CPUS=$(lscpu -p=CPU,CORE | grep -v '^#' | awk -F, '$2 < 6 {print $1}' | paste -sd, -)
if [ $R3_PLAN = 1 ]; then sed -n '2,13p' "$0"; echo "cpu set: $CPUS"; exit 0; fi
[ "$(echo "$CPUS" | tr ',' '\n' | wc -l)" = 12 ] || r3_die "expected 12 logical CPUs on cores 0-5, got '$CPUS'"
r3_env
r3_preflight
BURN=()
burn_stop() { [ ${#BURN[@]} -gt 0 ] && kill "${BURN[@]}" 2>/dev/null; BURN=(); }
c2() { "${LOAD[@]}" steady --conc 2 --tokens 96000 --max-tokens 12000 --window 90 --k 5 --tag c2 --out "$R" || r3_die "c2 window"; }
v_pin6() {
  R3_SERVE_PREFIX=(taskset -c "$CPUS")
  r3_serve pin6
  "${LOAD[@]}" warm || r3_die warm
  "${LOAD[@]}" fill --n 2 --tokens 90000 --conc 2 || r3_die fill
  c2
}
v_pin6_load() {
  R3_SERVE_PREFIX=(taskset -c "$CPUS")
  r3_serve pin6-load
  "${LOAD[@]}" warm || r3_die warm
  "${LOAD[@]}" fill --n 2 --tokens 90000 --conc 2 || r3_die fill
  for _ in 1 2 3 4; do taskset -c "$CPUS" python3 -c 'while True: pass' & BURN+=($!); done
  trap 'burn_stop' EXIT
  c2
  burn_stop
}
r3_summary "R3-26 host CPU contention (box, $(date -u +%F)); cpu set $CPUS"
r3_step pin6;      r3_variant pin6 v_pin6
[ -f "$L/pin6/lines.txt" ] && r3_summary "$(sed 's/^/pin6 /' "$L/pin6/lines.txt")"
r3_step pin6-load; r3_variant pin6-load v_pin6_load
[ -f "$L/pin6-load/lines.txt" ] && r3_summary "$(sed 's/^/pin6-load /' "$L/pin6-load/lines.txt")"
B=$R3_LOGS/20-idle-baseline/srv/lines.txt
[ -f "$B" ] && r3_summary "unpinned (20) $(grep '^c2' "$B")"
r3_summary "reference: $R3_PROD_REF"
cat "$L/summary.txt"
r3_finish
