#!/bin/bash
# Serve the EXL3 checkpoint (GSQ_EXL3_MODEL, a plain HF dir with quantization_config.quant_method
# "exl3": vLLM detects the quantization, no flag) with the EXL3 plugin, on production's argv
# (GSQ_PROD_ARGV; env/prod-main-serve-argv.txt for vLLM main) except: interpreter/venv, model,
# port (18090), host (127.0.0.1), fs KV tier root/cap, chat template (the repo copy of
# production's, same sha256, asserted) and VLLM_PLUGINS (all four: both LoRA resolvers, gguf and
# exl3, as a production venv with both packages loads them). MTP uses the checkpoint's own mtp.*
# layer; the vocab-truncated draft head needs tools/exl3_draft_head.py run once on the dir.
# The argv diff against production is printed before launch.
#
#   scripts/serve-exl3.sh --dry-run          print argv + diff, run the non-GPU guards, exit
#   GSQ_ALLOW_GPU=1 scripts/serve-exl3.sh    launch (foreground; GSQ_LOG=<file> to tee)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
DRY=0; [ "${1:-}" = --dry-run ] && DRY=1

gsq_check_port
gsq_assert_tier_root
[ -f "$GSQ_EXL3_MODEL/config.json" ] || [ $DRY = 1 ] || gsq_die "no config.json in GSQ_EXL3_MODEL=$GSQ_EXL3_MODEL"
[ $DRY = 1 ] || grep -q '"quant_method": *"exl3"' "$GSQ_EXL3_MODEL/config.json" || gsq_die "$GSQ_EXL3_MODEL is not an EXL3 checkpoint"
[ "$(sha256sum "$GSQ_HF_CONFIG/chat_template.jinja" | cut -d' ' -f1)" = "$GSQ_PROD_CHAT_TEMPLATE_SHA256" ] \
  || gsq_die "chat template copy differs from production's"

gsq_load_prod_argv
gsq_rewrite_argv "$GSQ_EXL3_MODEL"
gsq_print_argv_diff serve-exl3 "$GSQ_PLUGINS_EXL3"
[ $DRY = 1 ] && { echo "dry run: not launching"; exit 0; }

gsq_require_gpu
gsq_refuse_if_prod_live GSQ_ALLOW_BESIDE_PROD
gsq_exec_server exl3 "$GSQ_PLUGINS_EXL3"
