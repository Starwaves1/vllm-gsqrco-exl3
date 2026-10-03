#!/bin/bash
# R3-61 GPU confirmation of the incident mechanism (incident-root-cause.md) and of the fix.
# PR #50021's conv1d check (in venv-main's overlay) zeroes a request's GDN conv output when the
# previous step accepted more tokens than this step verifies, which the per-batch-size MTP schedule
# causes whenever K drops. Load = job 60's: 30 greedy chat answers beside a 1-token side client.
#   kstep        schedule [[1,1,5],[2,2,3],[3,16,2]]: at c=1 the side client flips K 5<->3, at c=2
#                3<->2, every few steps                            -> expect corruption
#   kstep-k3     same traffic, fixed k=3, no schedule              -> expect clean
#   kstep-fix    kstep on venv-main + patches/conv1d-accepted-bound.patch -> expect clean
#   base         job 60's base cell again (production schedule, c=4 + side client, venv-main) -> ~23/30
#   mainconv     base on venv-main + patches/conv1d-upstream-main.patch (upstream main's kernel,
#                no #50021 check)                                  -> expect clean
#   fix-base / fix-eager / fix-mambanone / fix-noconn: job 60's corrupting variants (production
#                schedule, c=4 + side client) on the patched venv -> expect clean
#   fix-w4a16    the same for production's W4A16 checkpoint        -> expect clean
#   fix-c9       production schedule, no side client, c=8/9/12 T=0 and T=1 (job 28b's regime)
# First: the CPU kernel test on venv-main's and the patched causal_conv1d.py.
# Output: /workspace/logs/r3/61-kchange-fix/. GPU time ~75 min. R3_61_ONLY picks variants.
#   bash 61-kchange-fix.sh [--plan]
source "$(dirname "$0")/lib.sh"
r3_init 61-kchange-fix "$@"
if [ "$R3_PLAN" = 1 ]; then sed -n '2,19p' "$0"; exit 0; fi
r3_env
r3_preflight
FIXV=/workspace/venv-r3-convfix
bash "$R3_S/overlay-venv.sh" $FIXV "$R3_S/../patches/conv1d-accepted-bound.patch" > "$L/overlay.log" 2>&1 \
  || { cat "$L/overlay.log"; r3_die "overlay venv"; }
MAINV=/workspace/venv-r3-mainconv
bash "$R3_S/overlay-venv.sh" $MAINV "$R3_S/../patches/conv1d-upstream-main.patch" >> "$L/overlay.log" 2>&1 \
  || { cat "$L/overlay.log"; r3_die "overlay venv (main conv)"; }
CONV=lib/python3.12/site-packages/vllm/model_executor/layers/mamba/ops/causal_conv1d.py
r3_step kernel-test
sha256sum /workspace/venv-main/$CONV $FIXV/$CONV $MAINV/$CONV | tee "$L/conv-sha256.txt"
CUDA_VISIBLE_DEVICES= TRITON_INTERPRET=1 "$PY" "$R3_S/r3conv_kchange.py" /workspace/venv-main/$CONV $FIXV/$CONV $MAINV/$CONV \
  > "$L/kernel-test.txt" 2>&1; echo "kernel test rc=$? (1 = some case wrong, expected for venv-main)" >> "$L/kernel-test.txt"
grep -v -i warn "$L/kernel-test.txt"

SPEC='{"method":"mtp","num_speculative_tokens":5,"draft_sample_method":"probabilistic","num_speculative_tokens_per_batch_size":'
KSTEP=("set|--speculative-config|${SPEC}[[1,1,5],[2,2,3],[3,16,2]]}")
K3=("set|--speculative-config|{\"method\":\"mtp\",\"num_speculative_tokens\":3,\"draft_sample_method\":\"probabilistic\"}")
side() {  # tag conc
  "$PY" "$R3_S/r3tok.py" run --nonstream --max-tokens 600 --temps "" --side plain1 --side-conc "$2" \
    --out "$L/runs" --tag "$1" || r3_die "r3tok $1"
}
kstep() {  # tag
  r3_serve "$1"; "${LOAD[@]}" warm || r3_die warm
  side "$1-c1" 1; side "$1-c2" 2
}
fixed() { export GSQ_VENV_OVERRIDE=$FIXV; r3_env; }
v_kstep()     { R3_MUT=("${KSTEP[@]}"); kstep kstep; }
v_kstep_k3()  { R3_MUT=("${K3[@]}"); kstep kstep-k3; }
v_kstep_fix() { fixed; R3_MUT=("${KSTEP[@]}"); kstep kstep-fix; }
c4() { r3_serve "$1"; "${LOAD[@]}" warm || r3_die warm; side "$1" 4; }
v_base()          { c4 base; }
v_mainconv()      { export GSQ_VENV_OVERRIDE=$MAINV; r3_env; c4 mainconv; }
v_fix_base()      { fixed; c4 fix-base; }
v_fix_eager()     { fixed; R3_MUT=("flag|--enforce-eager"); c4 fix-eager; }
v_fix_mambanone() { fixed; R3_MUT=("set|--mamba-cache-mode|none"); c4 fix-mambanone; }
v_fix_noconn()    { fixed; R3_MUT=("drop|--kv-transfer-config"); c4 fix-noconn; }
v_fix_w4a16()     { fixed; export R3_MODEL_KIND=baseline; c4 fix-w4a16; }
v_fix_c9() {
  fixed; r3_serve fix-c9; "${LOAD[@]}" warm || r3_die warm
  "$PY" "$R3_S/r3tok.py" run --nonstream --max-tokens 600 --per-conc 4 --conc 8,9,12 --temps 0,1.0 \
    --out "$L/runs" --tag fix-c9 || r3_die "r3tok fix-c9"
}
report() {
  "$PY" "$R3_S/r3tok.py" report "$L/runs" --ref none --details 12 > "$L/report.txt" 2>&1
  { grep -v -i warn "$L/kernel-test.txt"; echo; cat "$L/report.txt"
    for f in "${R3_FAILED[@]}"; do echo "variant $f: FAILED"; done; } > "$L/summary.txt"
}
for v in ${R3_61_ONLY:-base fix_base kstep kstep_k3 kstep_fix mainconv fix_w4a16 fix_eager fix_mambanone fix_noconn fix_c9}; do
  r3_step "$v"; r3_variant "$v" "v_$v"; report
done
cat "$L/summary.txt"
r3_finish
