#!/bin/bash
# R3-25 --long-prefill-token-threshold A/B (TEST ONLY: numbers and a proposal, nothing kept).
# Four servers, production's main argv except the threshold: 128 (production), 256, 512, and 128 with
# --long-prefill-token-threshold-adaptive (cap floored at max_num_batched_tokens / requests: 512 at c=4).
# Mixed load at c=4: three decoders (40k-token prompts, k=5, max 9000 tokens) plus one lane sending
# back-to-back prefill-only requests cycling 600 / 1400 fresh tokens / a long-agent "turn" (60k
# cached prefix + 1800 fresh tokens, production's median long turn) / 4000 fresh tokens, for 120 s.
# Reported per threshold: decoder ms/step with prefills interleaved (drafts/s, as production's
# "with short prefills" column), TTFT per lane kind, lane prefill tok/s, total gen tok/s.
# Output: /workspace/logs/r3/25-prefill-chunk-test/{summary.txt, t128/, t256/, t512/, t128a/}
# GPU time: ~40 min (per server: load 4, turn prefix 60k ~1.2, decoders 120k ~2, window 2 + slack).
#   bash 25-prefill-chunk-test.sh [--plan]
source "$(dirname "$0")/lib.sh"
r3_init 25-prefill-chunk-test "$@"
if [ $R3_PLAN = 1 ]; then sed -n '2,14p' "$0"; exit 0; fi
r3_env
r3_preflight
grep -qx -- "--long-prefill-token-threshold" "$GSQ_PROD_ARGV" || r3_die "production argv has no --long-prefill-token-threshold"

run() {  # threshold [adaptive]
  R3_MUT=("set|--long-prefill-token-threshold|$1")
  [ "${2:-}" = a ] && R3_MUT+=("flag|--long-prefill-token-threshold-adaptive")
  r3_serve "t$1${2:-}"
  "${LOAD[@]}" warm || r3_die warm
  "${LOAD[@]}" mixed --decoders 3 --dec-tokens 40000 --max-tokens 9000 --lane 600,1400,turn,4000 \
    --turn-prefix 60000 --turn-new 1800 --window 120 --tag "mix-t$1${2:-}" --out "$R" || r3_die "mixed window"
}
v_128() { run 128; }
v_256() { run 256; }
v_512() { run 512; }
v_128a() { run 128 a; }
r3_summary "R3-25 long-prefill-token-threshold at c=4 mixed (TEST ONLY; box, $(date -u +%F))"
for t in 128 256 512 128a; do
  r3_step "threshold $t"
  r3_variant "t$t" "v_$t"
  [ -f "$L/t$t/lines.txt" ] && r3_summary "$(cat "$L/t$t/lines.txt")"
done
r3_summary "production reference: decode +10-13 ms/step at 4-8 running when short prefills interleave (81.6 vs 71.4 at n=4); long turns 726 tok/s" \
  "proposal rule: a threshold that cuts turn TTFT >= 20% while adding <= 3 ms to decoder ms/step is worth proposing; otherwise keep 128. Garrett's call."
cat "$L/summary.txt"
r3_finish
