#!/bin/bash
# Serve a W4A16 baseline on production's argv (same flags as serve-gsq.sh, plugin OFF),
# for cloud / gap A-B comparisons only. Model: GSQ_BASELINE_MODEL, default production's
# own weights dir (read-only). Port 18090, host 127.0.0.1, own fs KV tier root.
#
#   scripts/serve-baseline.sh --dry-run
#   GSQ_ALLOW_GPU=1 scripts/serve-baseline.sh
#
# Never on this host while production runs: refuses if :18080/:18081 listen or a
# production-venv vLLM is running, unless GSQ_ALLOW_LOCAL_BASELINE=1.
# The baseline's model dir is its own HF config, so its fs-tier folder name equals
# production's; that's why the tier root must differ from production's (asserted).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
DRY=0; [ "${1:-}" = --dry-run ] && DRY=1

gsq_check_port
gsq_assert_tier_root
[ -f "$GSQ_BASELINE_MODEL/config.json" ] || [ $DRY = 1 ] || gsq_die "no config.json in GSQ_BASELINE_MODEL=$GSQ_BASELINE_MODEL"
# The chat template must be production's; serve it from the repo copy (same bytes).
[ "$(sha256sum "$GSQ_HF_CONFIG/chat_template.jinja" | cut -d' ' -f1)" = "$GSQ_PROD_CHAT_TEMPLATE_SHA256" ] \
  || gsq_die "chat template copy differs from production's"

gsq_load_prod_argv
gsq_rewrite_argv "$GSQ_BASELINE_MODEL"
gsq_print_argv_diff serve-baseline "$GSQ_PLUGINS_OFF"
[ $DRY = 1 ] && { echo "dry run: not launching"; exit 0; }

gsq_refuse_if_prod_live GSQ_ALLOW_LOCAL_BASELINE
gsq_require_gpu
gsq_exec_server baseline "$GSQ_PLUGINS_OFF"
