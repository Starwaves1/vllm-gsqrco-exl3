#!/bin/bash
# R3-28 generation corruption matrix (production reports EOS mid-reasoning, repeated fragments, a CJK
# character inside a year, U+FFFD / dropped characters in streamed chat). One server per
# configuration, production's exact main argv except the one change named:
#   prod     as-is (note: production runs --no-async-scheduling; k schedule [[1,4,5],[5,8,3],[9,16,2]])
#   async    --async-scheduling
#   eager    --enforce-eager (the reference for token ids and finish reasons)
#   nospec   no --speculative-config
#   k3       fixed k=3 (no per-batch schedule)
#   w4a16    production's previous W4A16 model, stock vLLM path, same argv (ours vs vLLM)
#   prodnan  prod + pyhook R3_NANCHECK: NaN/inf in any LogitsProcessor input/output, target and MTP
#            draft head (NaN draft probs at T > 0 would turn rejection sampling into garbage tokens)
# Load per server (r3tok.py): 30 English prompts (em dashes, curly quotes, numbers, URLs), streaming
# chat, reasoning on, max 400 tokens, T=0 and T=0.7, each at c=1/2/4/8 (240 requests), then a pass
# at c=4 T=0 beside one client sending echo+prompt_logprobs requests (30). Corruption flags: see
# r3tok.py. Report: r3tok.py report across all servers (ids/finish compared with eager's T=0 c=1).
# R3_28_ONLY="prod async" runs a subset. Output: /workspace/logs/r3/28-token-corruption/
# GPU time: ~2 h (eager is slow).
#   bash 28-token-corruption.sh [--plan]
source "$(dirname "$0")/lib.sh"
r3_init 28-token-corruption "$@"
if [ $R3_PLAN = 1 ]; then sed -n '2,22p' "$0"; exit 0; fi
r3_env
r3_preflight
SPEC_K3='{"method":"mtp","num_speculative_tokens":3,"draft_sample_method":"probabilistic"}'
grep -qx -- "--no-async-scheduling" "$GSQ_PROD_ARGV" || r3_die "production argv has no --no-async-scheduling (update the variants)"
run() {  # tag
  r3_serve "$1"
  "${LOAD[@]}" warm || r3_die warm
  "$PY" "$R3_S/r3tok.py" run --out "$L/runs" --tag "$1" --plp-pass || r3_die "r3tok $1"
}
v_prod()   { run prod; }
v_async()  { R3_MUT=("swap|--no-async-scheduling|--async-scheduling"); run async; }
v_eager()  { R3_MUT=("flag|--enforce-eager"); run eager; }
v_prodnan() {  # prod + a NaN/inf probe on every LogitsProcessor call (target AND the MTP draft head)
  export R3_EXTRA_PYTHONPATH=$R3_S/pyhook R3_NANCHECK=$L/runs/nan-prodnan.jsonl.txt R3_NANCHECK_NAN_ONLY=1
  r3_env
  run prodnan
}
v_nospec() { R3_MUT=("drop|--speculative-config"); run nospec; }
v_k3()     { R3_MUT=("set|--speculative-config|$SPEC_K3"); run k3; }
v_w4a16()  { export R3_MODEL_KIND=baseline; run w4a16; }
report() {
  "$PY" "$R3_S/r3tok.py" report "$L/runs" --ref eager > "$L/report.txt" 2>&1
  { cat "$L/report.txt"; for f in "${R3_FAILED[@]}"; do echo "variant $f: FAILED (see run.log)"; done; } > "$L/summary.txt"
}
for v in ${R3_28_ONLY:-prod async prodnan eager nospec k3 w4a16}; do
  r3_step "$v"; r3_variant "$v" "v_$v"
  report
done
cat "$L/summary.txt"
r3_finish
