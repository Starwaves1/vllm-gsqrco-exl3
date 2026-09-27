#!/bin/bash
# Serve the Swift GSQ-RCO IQ3_S-mtp GGUF from the isolated .venv with the GGUF plugin
# enabled, on production's argv except: interpreter/venv, model (the .gguf), HF config
# dir (+ --hf-config-path/--tokenizer), port (18090), host (127.0.0.1), fs KV tier root/cap,
# and VLLM_PLUGINS (gguf on). The chat template is the repo copy of production's (same
# sha256, asserted). The argv diff against production is printed before launch.
#
#   scripts/serve-gsq.sh --dry-run          print argv + diff, run the non-GPU guards, exit
#   GSQ_ALLOW_GPU=1 scripts/serve-gsq.sh    launch (foreground; GSQ_LOG=<file> to tee)
#
# Refuses: ports 18080/18081; no GSQ_ALLOW_GPU=1; production live on this host (override
# GSQ_ALLOW_BESIDE_PROD=1, and only with Garrett's go); an HF config dir other than
# hf-config/<name>; production's fs tier root.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
DRY=0; [ "${1:-}" = --dry-run ] && DRY=1

gsq_check_port
gsq_assert_hf_config
gsq_assert_tier_root
[ -f "$GSQ_GGUF" ] || [ $DRY = 1 ] || gsq_die "GGUF not found: $GSQ_GGUF (set GSQ_GGUF)"
case "$(realpath -m "$GSQ_GGUF")" in *.gguf) ;; *) gsq_die "GSQ_GGUF must be a .gguf file" ;; esac

gsq_load_prod_argv
# --hf-config-path is what the plugin turns into model_config.model (plugin.py
# _get_gguf_config_source); --tokenizer keeps the tokenizer on the same dir.
gsq_rewrite_argv "$GSQ_GGUF" --hf-config-path "$GSQ_HF_CONFIG" --tokenizer "$GSQ_HF_CONFIG"
gsq_print_argv_diff serve-gsq "$GSQ_PLUGINS_ON"
[ $DRY = 1 ] && { echo "dry run: not launching"; exit 0; }

gsq_require_gpu
gsq_refuse_if_prod_live GSQ_ALLOW_BESIDE_PROD
gsq_exec_server gsq "$GSQ_PLUGINS_ON"
