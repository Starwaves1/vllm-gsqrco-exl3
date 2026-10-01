#!/bin/bash
# EXL3 opt, job 20 (~80 min): generation-corruption check on the EXL3 defaults under production's main
# argv (bench/corruption_check.py: 38 prompts = production's 8 bench prompts + 30 fixed ones, chat with
# reasoning on, streaming, token ids; c=1/2/4/8 x T=0 and T=0.7 = 304 requests per config).
# Configs: ref = --enforce-eager (T=0 only, c=8; the reference for token ids and per-prompt
# baselines), asis = production's argv (which has --no-async-scheduling), asyncon = + --async-scheduling
# (vLLM main's default), nospec = without --speculative-config. Configs: the job's arguments (default
# "ref asis asyncon nospec"). Corruption rates per config in summary.txt.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
job_log 20-corruption
require_idle_gpu
require_mr_build
O=$R/20-corruption; mkdir -p "$O"
modes=("${MODES_ARGS[@]}"); [ ${#modes[@]} = 0 ] && modes=(ref asis asyncon nospec)
PROMPTS=/workspace/deploy/bench/prompts_real.jsonl
rc=0
for cfg in "${modes[@]}"; do
  D=$O/$cfg; rm -rf "$D"; mkdir -p "$D"
  echo "=== $cfg $(date -u +%FT%TZ)"
  extra=(); unset GSQ_NO_MTP; runargs=(--conc 1,2,4,8 --temps 0,0.7)
  case $cfg in
    ref) extra=(--enforce-eager); runargs=(--conc 8 --temps 0) ;;
    asis) ;;
    asyncon) extra=(--async-scheduling) ;;
    nospec) export GSQ_NO_MTP=1 ;;
    *) echo "unknown config $cfg"; rc=1; continue ;;
  esac
  if serve_mr 2h "$D" "${extra[@]}"; then
    "$GSQ_VENV/bin/python" bench/corruption_check.py run --url "$GSQ_URL/v1" --label "$cfg" --out "$O" \
      --prompts "$PROMPTS" "${runargs[@]}" 2>&1 | tee "$D/run.log" || rc=1
  else
    rc=1
  fi
  stopall; unset GSQ_NO_MTP
  gzip -kf "$D/server.log"
  keep "$D" "20-corruption/$cfg" "$D/run.log" "$D/load.txt" "$D/argv.txt" "$D/server.log.gz"
done
runs=(); for cfg in asis asyncon nospec; do [ -s "$O/$cfg.jsonl" ] && runs+=("$O/$cfg.jsonl"); done
[ -s "$O/ref.jsonl" ] && [ ${#runs[@]} -gt 0 ] && "$GSQ_VENV/bin/python" bench/corruption_check.py compare \
  --ref "$O/ref.jsonl" "${runs[@]}" | tee "$O/summary.txt"
gzip -kf "$O"/*.jsonl
keep "$O" 20-corruption "$O/summary.txt" "$O"/*.jsonl.gz
exit $rc
