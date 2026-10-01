#!/bin/bash
# gpuq entry point for the EXL3 phase-1 chain: run one job, record its exit status, enforce the
# real dependencies. A failed job never stops the queue; a job whose predecessor failed is skipped
# with a log line (status "skipped"), every other job runs anyway.
#   gpuq submit exl3-NAME -- bash /workspace/wt-exl3/cloud/results/exl3/box-scripts/run-job.sh NAME
# NAME: postsoak | model | 01-kernel-parity | 02-draft-head | 03-smoke | 04-parity-ref |
#       05-parity-vllm | 06-ladder | 07-fit
# Status: /workspace/logs/exl3/status/NAME (ok | fail rc=N | skipped: ...); log run-job.log.
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
L=/workspace/logs/exl3; ST=$L/status; mkdir -p "$ST"
name=${1:?job name}
log() { echo "$(date -u +%FT%TZ) $name: $*" | tee -a "$L/run-job.log"; }
declare -A NEEDS=([03-smoke]=02-draft-head [05-parity-vllm]=04-parity-ref [06-ladder]=02-draft-head)
dep=${NEEDS[$name]:-}
if [ -n "$dep" ] && [ "$(cat "$ST/$dep" 2>/dev/null)" != ok ]; then
  echo "skipped: needs $dep, whose status is '$(cat "$ST/$dep" 2>/dev/null || echo none)'" > "$ST/$name"
  log "$(cat "$ST/$name")"; exit 0
fi
log start
case $name in
  postsoak) CONFIRM_DELETE_SOAK_KV=1 "$S/00-prep.sh" postsoak > "$L/00-postsoak.log" 2>&1 ;;
  model)
    { "$S/00-prep.sh" model && {
        # the A/B checkpoint only if 15 GB stay free after it (parity dumps, fs KV tier, logs)
        MIN_FREE_GB=15 "$S/00-prep.sh" model --alt || echo "NOTE: turboderp 3.50bpw (--alt) skipped: not enough disk"; }
    } > "$L/00-model.log" 2>&1 ;;
  0[1-7]-*) bash "$S/$name.sh" > /dev/null 2>&1 ;;   # the job scripts tee to $L/NAME.log themselves
  *) log "unknown job"; exit 2 ;;
esac
rc=$?
if [ $rc = 0 ]; then echo ok > "$ST/$name"; else echo "fail rc=$rc" > "$ST/$name"; fi
log "$(cat "$ST/$name")"
exit $rc
