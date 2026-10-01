#!/bin/bash
# gpuq entry point for the EXL3 optimization jobs: run one, record its status, enforce the one
# real dependency (12 and 13 measure exl3_gemm_mr in the served model, so they need 10's parity).
#   gpuq submit exl3opt-NAME -- bash /workspace/wt-exl3-opt/cloud/results/exl3-opt/box-scripts/run-job.sh NAME
# NAME: 10-mr-parity | 11-mr-micro | 12-mr-ladder | 13-profile | 14-fit  (then the job's args)
# Status: /workspace/logs/exl3-opt/status/NAME (ok | fail rc=N | skipped: ...); log run-job.log.
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
L=/workspace/logs/exl3-opt${EXL3_OPT_TAG:-}; ST=$L/status; mkdir -p "$ST"
name=${1:?job name}; shift
log() { echo "$(date -u +%FT%TZ) $name: $*" | tee -a "$L/run-job.log"; }
declare -A NEEDS=([12-mr-ladder]=10-mr-parity [13-profile]=10-mr-parity [14-fit]=10-mr-parity)
dep=${NEEDS[$name]:-}
if [ -n "$dep" ] && [ "${EXL3_OPT_IGNORE_DEPS:-0}" != 1 ] && [ "$(cat "$ST/$dep" 2>/dev/null)" != ok ]; then
  echo "skipped: needs $dep, whose status is '$(cat "$ST/$dep" 2>/dev/null || echo none)'" > "$ST/$name"
  log "$(cat "$ST/$name")"; exit 0
fi
log start
case $name in
  1[0-9]-*) bash "$S/$name.sh" "$@" > /dev/null 2>&1 ;;   # the job scripts tee to $L/NAME.log themselves
  *) log "unknown job"; exit 2 ;;
esac
rc=$?
if [ $rc = 0 ]; then echo ok > "$ST/$name"; else echo "fail rc=$rc" > "$ST/$name"; fi
log "$(cat "$ST/$name")"
exit $rc
