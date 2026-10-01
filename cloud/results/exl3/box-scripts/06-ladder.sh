#!/bin/bash
# EXL3 phase 1, job 06 (~60-90 min): speed ladder on production's main argv with MTP, on
# erlidev's Swift SC_3.50bpw_H4_V6 (--alt: turboderp's 3.50bpw, see lib.sh).
# bench/speed/run.sh exl3 --start: production's run_benchmarks.sh `single` (real-prompt cohorts,
# c=1/2/4/8, default sampling and greedy, two passes, keep pass 2), the salted prefill ladder
# 8k/64k/180k, MTP acceptance per position from /metrics, GPU clocks/power every second per phase.
# Judge on ms/step (EXL3.md), next to the W4A16 and GGUF rows of requested-workloads/01.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
job_log 06-ladder
require_idle_gpu
[ -f "$EXL3_MODEL/mtp_draft_head.safetensors" ] || die "run 02-draft-head first"
O=$R/06-ladder; rm -rf "$O"
rm -rf "$GSQ_KV_TIER_ROOT"; box_clean_shm || true
OUT=$O bench/speed/run.sh exl3 --start || { echo "ladder failed"; tail -40 "$O/server.log" 2>/dev/null; }
rm -rf "$GSQ_KV_TIER_ROOT"; box_clean_shm || true
cat "$O/summary.txt" 2>/dev/null || true
[ -f "$O/server.log" ] && gzip -kf "$O/server.log"
keep "$O" 06-ladder "$O/summary.txt" "$O"/clocks-*.csv "$O"/*.txt "$O/server.log.gz"
[ -s "$O/summary.txt" ] || die "no ladder summary"
