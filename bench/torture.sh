#!/bin/bash
# This repo's wrapper around the standalone torture harness (bench/torture/torture, TORTURE.md):
# it serves this repo's models under production's vLLM-main argv (env/prod-main-serve-argv.txt:
# MTP k=5 schedule, 16 seqs, fp8 KV, prefix caching, KV offload tiers, CUDA graphs) from .venv-main,
# with the repo's guards (GSQ_ALLOW_GPU=1; never ports 18080/18081; refuse while production is live)
# and results under $GSQ_RUNS. The serve script is any of ours: scripts/serve-gsq.sh, scripts/serve-exl3.sh, ...;
# run every serve script from this checkout (env.sh derives GSQ_HF_CONFIG from it: the KV-tier namespace rule).
#
#   GSQ_ALLOW_GPU=1 bench/torture.sh run --serve SCRIPT [--port 18090] [--hours 12 | --minutes 20] [torture options]
#   GSQ_ALLOW_GPU=1 bench/torture.sh switch --a SCRIPT --b SCRIPT [--rounds 3] [--minutes 10] [--port 18090] [torture options]
#   bench/torture.sh --plan [--hours 12 | --minutes 20] [--seed N]
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
T=$ROOT/bench/torture/torture
PY=${TORTURE_PYTHON:-python3}   # the harness is standard library only
[ "${1:-}" = --plan ] && exec "$PY" "$T" run "$@"
export GSQ_VENV=${GSQ_VENV:-$ROOT/.venv-main}
export GSQ_PROD_ARGV=${GSQ_PROD_ARGV:-$ROOT/env/prod-main-serve-argv.txt}
export VLLM_USE_V2_MODEL_RUNNER=${VLLM_USE_V2_MODEL_RUNNER:-0}   # production's start script default
source "$ROOT/scripts/env.sh"

MODE=${1:-}; shift || true
ARGS=()
while [ $# -gt 0 ]; do
  case $1 in
    --serve) ARGS+=(--cmd "$(realpath "$2")"); shift ;;
    --a|--b) ARGS+=("$1" "$(realpath "$2")"); shift ;;
    --port) export GSQ_PORT=$2 GSQ_URL=http://127.0.0.1:$2; shift ;;
    *) ARGS+=("$1") ;;
  esac; shift
done
case $MODE in
  run) SUB=serve ;;
  switch) SUB=switch; ARGS+=(--tier-root "$GSQ_KV_TIER_ROOT") ;;
  *) gsq_die "usage: bench/torture.sh run|switch ... (see the header)" ;;
esac
gsq_check_port
gsq_require_gpu
gsq_refuse_if_prod_live GSQ_ALLOW_BESIDE_PROD
grep -qx -- --enforce-eager "$GSQ_PROD_ARGV" && gsq_die "the argv has --enforce-eager; the torture must run with CUDA graphs"
echo "torture via $GSQ_PROD_ARGV, venv $GSQ_VENV"
exec "$PY" "$T" "$SUB" --port "$GSQ_PORT" --api-key "$GSQ_API_KEY" \
  --out "$GSQ_RUNS/$(date +%Y%m%d-%H%M%S)-torture-$MODE" "${ARGS[@]}"
