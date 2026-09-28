#!/bin/bash
# Speed bench: production's own bench script, unmodified, plus a salted prefill ladder.
#
#   GSQ_ALLOW_GPU=1 bench/speed/run.sh gsq|baseline [--start]
#
# 1. Production's bench/run_benchmarks.sh (deploy repo, pinned by sha256 below) in
#    `single` mode: real-prompt cohorts at concurrency 1, 2, 4, 8, default sampling and
#    greedy, with MTP tokens/step from /metrics. It is copied verbatim into the run dir
#    with a `venv` symlink to this repo's .venv, because it runs `venv/bin/vllm` relative
#    to its repo and must not execute production's venv. Two passes, as its header says;
#    the second is the one to keep.
# 2. Prefill ladder 8k / 64k / 180k (c=1; 8k and 64k also c=2) with a fresh --seed per
#    invocation, so no prefix-cache or KV-tier entry can hit (the production script's
#    random prompts use seed 0 every run).
# 3. MTP acceptance per position from /metrics (vllm:spec_decode_*), before/after.
#
# The server must mirror production (serve-gsq.sh / serve-baseline.sh: MTP k=3, fp8 KV).
# --start launches that server here and stops it at the end; otherwise one must already
# answer on GSQ_URL. Results: $GSQ_RUNS/<ts>-speed-<kind>/ (summary.txt = ROW lines).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../scripts/env.sh"
KIND=${1:?usage: run.sh gsq|baseline [--start]}; shift
START=0; [ "${1:-}" = --start ] && START=1
case $KIND in
  gsq) TOKENIZER=$GSQ_HF_CONFIG ;;
  baseline) TOKENIZER=$GSQ_BASELINE_MODEL ;;
  *) gsq_die "kind must be gsq or baseline" ;;
esac
gsq_check_port
gsq_require_gpu

PROD_BENCH_SHA256=c0c56bbb8408a9a0e8ae4d3043e08dc73c8d79311f4f1a6695c352e7ea288705   # deploy repo 2138d1ae
PROD_PROMPTS_SHA256=27da4fd8b2dcc133b98cec54bc337c34f5acb1b6372cc2fea1d8a92da9c78650
SRC=${GSQ_DEPLOY_REPO:-$GSQ_PROD_REPO}/bench
for f in run_benchmarks.sh:$PROD_BENCH_SHA256 prompts_real.jsonl:$PROD_PROMPTS_SHA256; do
  n=${f%%:*}; want=${f#*:}
  [ "$(sha256sum "$SRC/$n" | cut -d' ' -f1)" = "$want" ] || gsq_die "$SRC/$n differs from the pinned production copy"
done

OUT=${OUT:-$GSQ_RUNS/$(date +%Y%m%d-%H%M%S)-speed-$KIND}
STAGE=$OUT/prodbench            # REPO for the production script: bench/ + venv symlink
mkdir -p "$STAGE/bench"
cp "$SRC/run_benchmarks.sh" "$SRC/prompts_real.jsonl" "$STAGE/bench/"
ln -sfn "$GSQ_VENV" "$STAGE/venv"
SUM=$OUT/summary.txt

SERVER_PID=
if [ $START = 1 ]; then
  GSQ_LOG=$OUT/server.log "$GSQ_ROOT/scripts/serve-$KIND.sh" > /dev/null 2>&1 &
  SERVER_PID=$!
  trap '[ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null; wait 2>/dev/null' EXIT
  gsq_wait_health 2400 "$SERVER_PID" || gsq_die "server did not come up; see $OUT/server.log"
fi
curl -sf -o /dev/null "$GSQ_URL/health" || gsq_die "no server on $GSQ_URL"

M() { curl -s "$GSQ_URL/metrics" -H "Authorization: Bearer $GSQ_API_KEY"; }
M | grep -E '^vllm:spec_decode' > "$OUT/spec_before.prom" || true

export VLLM_API_KEY=$GSQ_API_KEY HOST=127.0.0.1 PORT=$GSQ_PORT MODEL=$TOKENIZER
export OPENAI_API_KEY=$GSQ_API_KEY   # what vllm bench serve sends (the production script maps it too)
{
  echo "# kind=$KIND url=$GSQ_URL tokenizer=$TOKENIZER $(date -u +%FT%TZ)"
  for pass in 1 2; do
    echo "# production run_benchmarks.sh single, pass $pass$([ $pass = 1 ] && echo ' (warm-up, discard)')"
    OUT=$OUT/prod-pass$pass bash "$STAGE/bench/run_benchmarks.sh" single | grep -E '^(ROW|#)'
  done
} | tee -a "$SUM"

B=("$GSQ_VENV/bin/vllm" bench serve --host 127.0.0.1 --port "$GSQ_PORT" --model "$TOKENIZER" --served-model-name qwen3.8-27b)
num() { awk "/$1/ {print \$$2}" "$3"; }
pf() { local len=$1 c=$2 n=$3 seed=$((RANDOM * 32768 + RANDOM)) log=$OUT/prefill_${1}_c$2.log
  "${B[@]}" --dataset-name random --random-input-len "$len" --random-output-len 1 \
    --num-prompts "$n" --max-concurrency "$c" --seed "$seed" > "$log" 2>&1
  local in dur; in=$(num "Total input tokens" 4 "$log"); dur=$(num "Benchmark duration" 4 "$log")
  echo "ROW prefill len=$len conc=$c seed=$seed | $(python3 -c "print(f'{$in/$dur:.0f}')") tok/s | meanTTFT=$(num "Mean TTFT" 4 "$log") ms"
}
{
  echo "# salted prefill ladder"
  pf 8192 1 8; pf 8192 2 8
  pf 65536 1 2; pf 65536 2 4
  pf 180000 1 2
} | tee -a "$SUM"

M | grep -E '^vllm:spec_decode' > "$OUT/spec_after.prom" || true
python3 - "$OUT/spec_before.prom" "$OUT/spec_after.prom" <<'PY' | tee -a "$SUM"
import re, sys
def load(p):
    d = {}
    for line in open(p):
        m = re.match(r'(vllm:spec_decode_\w+?)(?:_total)?(\{[^}]*\})?\s+([0-9.eE+-]+)$', line.strip())
        if m:
            pos = re.search(r'position="(\d+)"', m.group(2) or "")
            d[(m.group(1), pos.group(1) if pos else None)] = d.get((m.group(1), pos.group(1) if pos else None), 0) + float(m.group(3))
    return d
a, b = load(sys.argv[1]), load(sys.argv[2])
g = lambda k, p=None: b.get((k, p), 0) - a.get((k, p), 0)
drafts, dt, acc = g("vllm:spec_decode_num_drafts"), g("vllm:spec_decode_num_draft_tokens"), g("vllm:spec_decode_num_accepted_tokens")
if drafts:
    per = [g("vllm:spec_decode_num_accepted_tokens_per_pos", str(i)) / drafts for i in range(8) if ("vllm:spec_decode_num_accepted_tokens_per_pos", str(i)) in b]
    print(f"ROW MTP whole run | acceptance rate {acc / dt:.4f} | mean accepted length {1 + acc / drafts:.3f} | per position {' '.join(f'{x:.3f}' for x in per)}")
else:
    print("ROW MTP | no drafts recorded (speculative decoding off?)")
PY
echo "results: $OUT"
