#!/bin/bash
# R3-62 what fixed k=3 costs against the per-batch-size schedule (for the redeploy config).
# Production's own bench (bench/speed/run.sh gsq: run_benchmarks.sh single, 8 real prompts x 1024
# tokens, c=1/2/4/8, greedy and default sampling, pass 2 kept), GSQ-RCO, production's main argv except
# --speculative-config:
#   fixed3  {"num_speculative_tokens":3} on venv-main
#   sched   production's [[1,4,5],[5,8,3],[9,16,2]] on venv-main + conv1d-accepted-bound.patch
#           (the fix only changes which accepted counts the conv kernel rejects; same speed)
# At c=1/2/4 the schedule runs k=5, at c=8 k=3, so c=8 is a same-config control.
# Output: /workspace/logs/r3/62-k-cost/. GPU time ~40 min.
#   bash 62-k-cost.sh [--plan]
source "$(dirname "$0")/lib.sh"
r3_init 62-k-cost "$@"
if [ "$R3_PLAN" = 1 ]; then sed -n '2,10p' "$0"; exit 0; fi
r3_env
r3_preflight
FIXV=/workspace/venv-r3-convfix
bash "$R3_S/overlay-venv.sh" $FIXV "$R3_S/../patches/conv1d-accepted-bound.patch" > "$L/overlay.log" 2>&1 \
  || { cat "$L/overlay.log"; r3_die "overlay venv"; }
bench() {  # name venv speculative-config
  export GSQ_VENV_OVERRIDE=$2; r3_env
  local argv=$L/$1-argv.txt
  awk -v spec="$3" 'p {print spec; p=0; next} {print} $0=="--speculative-config" {p=1}' "$GSQ_PROD_ARGV" > "$argv"
  grep -qxF "$3" "$argv" || r3_die "speculative-config not replaced"
  mkdir -p "$L/$1"
  ( export GSQ_PROD_ARGV=$argv; OUT=$L/$1 GSQ_PREFILL=" " GSQ_RUNS=$L "$R3_WT/bench/speed/run.sh" gsq --start ) \
    > "$L/$1/run.log" 2>&1 || { tail -30 "$L/$1/run.log"; r3_die "prodbench $1"; }
}
v_fixed3() { bench fixed3 /workspace/venv-main '{"method":"mtp","num_speculative_tokens":3,"draft_sample_method":"probabilistic"}'; }
v_sched()  { bench sched $FIXV '{"method":"mtp","num_speculative_tokens":5,"draft_sample_method":"probabilistic","num_speculative_tokens_per_batch_size":[[1,4,5],[5,8,3],[9,16,2]]}'; }
r3_summary "R3-62 fixed k=3 vs the per-batch-size schedule (box, $(date -u +%F)), production's bench, pass 2"
for v in ${R3_62_ONLY:-fixed3 sched}; do
  r3_step "$v"; r3_variant "$v" "v_$v"
  r3_summary "--- $v ---" "$(awk '/pass 2/ {p=1; next} /^#/ {p=0} p' "$L/$v/summary.txt" 2>/dev/null)" \
    "$(grep -E '^ROW (MTP|clocks.*pass2)' "$L/$v/summary.txt" 2>/dev/null)"
done
cat "$L/summary.txt"
r3_finish
