#!/bin/bash
# R3-24 MTP k-schedule test at 9 running (TEST ONLY, per Garrett's rule: numbers and a proposal,
# nothing is kept; production's argv is not touched). Two servers, identical except the schedule:
#   prod   [[1,4,5],[5,8,3],[9,16,2]]   (production: k=2 at 9 running)
#   prop   [[1,4,5],[5,9,3],[10,16,2]]  (k=3 at 9 running)
# Workload per server: 9 distinct 20k-token chat prompts (~180k total; production's 9-running
# context median 191k), all started together; window 1 at the model's default sampling
# (generation_config, seeded), window 2 at T=0 on the same prompts (prefix-cache hits, no re-prefill).
# Reported: ms/step, tok/s, tokens/step/seq, acceptance per position, GPU util.
# Output: /workspace/logs/r3/24-k-schedule-test/{summary.txt, prod/, prop/}
# GPU time: ~25 min (per server: load 4, prefill 180k ~3.5, two 90 s windows ~4).
#   bash 24-k-schedule-test.sh [--plan]
source "$(dirname "$0")/lib.sh"
r3_init 24-k-schedule-test "$@"
SPEC_PROD='{"method":"mtp","num_speculative_tokens":5,"draft_sample_method":"probabilistic","num_speculative_tokens_per_batch_size":[[1,4,5],[5,8,3],[9,16,2]]}'
SPEC_PROP='{"method":"mtp","num_speculative_tokens":5,"draft_sample_method":"probabilistic","num_speculative_tokens_per_batch_size":[[1,4,5],[5,9,3],[10,16,2]]}'
if [ $R3_PLAN = 1 ]; then sed -n '2,14p' "$0"; echo "prod: $SPEC_PROD"; echo "prop: $SPEC_PROP"; exit 0; fi
r3_env
r3_preflight
grep -qxF "$SPEC_PROD" "$GSQ_PROD_ARGV" || r3_die "production argv's --speculative-config is not $SPEC_PROD"

run() {  # name spec k
  R3_MUT=("set|--speculative-config|$2")
  r3_serve "$1"
  "${LOAD[@]}" warm || r3_die warm
  "${LOAD[@]}" steady --conc 9 --tokens 20000 --max-tokens 5000 --window 90 --k "$3" --temperature default \
    --pname k9 --tag c9-tdef --out "$R" || r3_die "c9 T=default window"
  "${LOAD[@]}" steady --conc 9 --tokens 20000 --max-tokens 5000 --window 90 --k "$3" --temperature 0 \
    --pname k9 --tag c9-t0 --out "$R" || r3_die "c9 T=0 window"
}
v_prod() { run prod "$SPEC_PROD" 2; }
v_prop() { run prop "$SPEC_PROP" 3; }
r3_summary "R3-24 k schedule at 9 running (TEST ONLY; box, $(date -u +%F))"
for v in prod prop; do
  r3_step "schedule $v"
  r3_variant "$v" "v_$v"
  [ -f "$L/$v/lines.txt" ] && r3_summary "$(sed "s/^/$v /" "$L/$v/lines.txt")"
done
"$PY" - "$L" >> "$L/summary.txt" <<'EOF' || echo "comparison failed" >> "$L/summary.txt"
import json, sys, os
L = sys.argv[1]
for t in ("c9-tdef", "c9-t0"):
    try:
        a = json.load(open(os.path.join(L, "prod", f"{t}-steady.json")))
        b = json.load(open(os.path.join(L, "prop", f"{t}-steady.json")))
    except FileNotFoundError:
        continue
    print(f"{t}: ms/step {a['ms_per_step_pooled']:.1f} -> {b['ms_per_step_pooled']:.1f} "
          f"({100 * (b['ms_per_step_pooled'] / a['ms_per_step_pooled'] - 1):+.1f}%), gen tok/s {a['gen_tok_s']:.0f} -> {b['gen_tok_s']:.0f} "
          f"({100 * (b['gen_tok_s'] / a['gen_tok_s'] - 1):+.1f}%), tok/step/seq {a['tok_per_step_per_seq']:.2f} -> {b['tok_per_step_per_seq']:.2f}; "
          f"acceptance per pos prod {a['accepted_per_pos']} prop {b['accepted_per_pos']}")
EOF
r3_summary "production reference: n=9 k=2 80.5 ms/step (decode-only pooled), 22.7 tok/step (2.53 per seq); n=8 k=3 2.00/3 accepted (3.00 per seq)" \
  "proposal rule: keep production's schedule unless prop gives >= +5% gen tok/s at T=default with no ms/step regression beyond the added rows; any change is Garrett's call."
cat "$L/summary.txt"
r3_finish
