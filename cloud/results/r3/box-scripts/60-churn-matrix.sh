#!/bin/bash
# R3-60 what in the engine corrupts running decodes when short requests join (58: a neighbour
# sending plain 4-token or even 1-token completions corrupts 10-22 of 30 answers at c=4, T=0, on GSQ
# AND on stock W4A16; 0/30 without the neighbour). One server per variant, production's main argv
# except the named change; load = 58's: 30 greedy non-streamed chat answers at c=4 beside a client
# sending 1-token completions back to back.
#   base       production argv (GSQ)
#   nospec     no --speculative-config
#   noprefix   --no-enable-prefix-caching (and mamba cache mode none)
#   mambanone  --mamba-cache-mode none (prefix caching stays on for attention)
#   noconn     no --kv-transfer-config
#   eager      --enforce-eager
#   k3fixed    num_speculative_tokens 3, no per-batch schedule
# Output: /workspace/logs/r3/60-churn-matrix/. GPU time ~45 min. R3_60_ONLY picks variants.
#   bash 60-churn-matrix.sh [--plan]
source "$(dirname "$0")/lib.sh"
r3_init 60-churn-matrix "$@"
if [ "$R3_PLAN" = 1 ]; then sed -n '2,16p' "$0"; exit 0; fi
r3_env
r3_preflight
run() {  # tag
  r3_serve "$1"
  "${LOAD[@]}" warm || r3_die warm
  "$PY" "$R3_S/r3tok.py" run --nonstream --max-tokens 600 --conc 4 --temps "" --side plain1 --side-conc 4 \
    --out "$L/runs" --tag "$1" || r3_die "r3tok $1"
}
v_base()      { run base; }
v_nospec()    { R3_MUT=("drop|--speculative-config"); run nospec; }
v_noprefix()  { R3_MUT=("swap|--enable-prefix-caching|--no-enable-prefix-caching" "set|--mamba-cache-mode|none"); run noprefix; }
v_mambanone() { R3_MUT=("set|--mamba-cache-mode|none"); run mambanone; }
v_noconn()    { R3_MUT=("drop|--kv-transfer-config"); run noconn; }
v_eager()     { R3_MUT=("flag|--enforce-eager"); run eager; }
v_k3fixed()   { R3_MUT=("set|--speculative-config|{\"method\":\"mtp\",\"num_speculative_tokens\":3,\"draft_sample_method\":\"probabilistic\"}"); run k3fixed; }
report() {
  GSQ_ALLOW_GPU=1 PYTHONPATH=$R3_S "$PY" "$R3_S/r3tok.py" report "$L/runs" --ref none --details 20 > "$L/report.txt" 2>&1
  { cat "$L/report.txt"; for f in "${R3_FAILED[@]}"; do echo "variant $f: FAILED"; done; } > "$L/summary.txt"
}
for v in ${R3_60_ONLY:-base nospec noprefix mambanone noconn eager k3fixed}; do
  r3_step "$v"; r3_variant "$v" "v_$v"; report
done
cat "$L/summary.txt"
r3_finish
