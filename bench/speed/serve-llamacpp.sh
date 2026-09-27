#!/bin/bash
# llama-server b11211 on the same GGUF, for the MTP-acceptance and speed reference only.
# Flags = the deploy repo's single-user/start_swift_gguf.sh (k=3 MTP, q8_0 KV for model
# and drafter, FA, model sampling, production template), but port GSQ_LLAMA_PORT (18091)
# and localhost. Gated like the vLLM servers.
#   GSQ_ALLOW_GPU=1 bench/speed/serve-llamacpp.sh [--dry-run] [extra llama-server args]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../scripts/env.sh"
PORT=${GSQ_LLAMA_PORT:-18091}
case $PORT in 18080|18081) gsq_die "port $PORT belongs to production" ;; esac
LLAMA_SERVER=${LLAMA_SERVER:-${LLAMA_DIR:-$HOME/llama.cpp-b11211}/build/bin/llama-server}
DRY=0; [ "${1:-}" = --dry-run ] && { DRY=1; shift; }
ARGS=(-m "$GSQ_GGUF" --alias qwen3.8-27b --host 127.0.0.1 --port "$PORT"
  -ngl 99 -c "${CTX:-200000}" -np 1 -fa on -ctk q8_0 -ctv q8_0
  --spec-type draft-mtp --spec-draft-n-max 3 -ctkd q8_0 -ctvd q8_0
  --jinja --chat-template-file "$GSQ_HF_CONFIG/chat_template.jinja"
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0
  --metrics --api-key "$GSQ_API_KEY" "$@")
echo "$LLAMA_SERVER ${ARGS[*]}"
[ $DRY = 1 ] && exit 0
gsq_require_gpu
gsq_refuse_if_prod_live GSQ_ALLOW_BESIDE_PROD
exec "$LLAMA_SERVER" "${ARGS[@]}"
