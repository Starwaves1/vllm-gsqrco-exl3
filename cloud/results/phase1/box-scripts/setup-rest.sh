#!/bin/bash
set -euxo pipefail
W=/workspace G=/workspace/gsq-vllm SP=/workspace/gsq-vllm/.venv/lib/python3.12/site-packages
cd $G
env VLLM_REPO=$W/vllm-src SITE_PACKAGES=$SP STATE_FILE=$W/deploy-state/state BACKUP_ROOT=$W/deploy-state/bk $W/deploy/scripts/deploy-vllm.sh --verify
cmp $W/prod-build_backend.py $SP/build_backend.py
tools/setup-cuda-toolchain.sh
MAX_JOBS=32 tools/build-plugin.sh
date -u +"plugin done %FT%TZ"
uv pip freeze --python .venv/bin/python > $W/logs/box-freeze.txt
diff <(sed "s#@ file://.*#@ file#; s#^-e file://.*#-e file#" env/gsq-freeze.txt) <(sed "s#@ file://.*#@ file#; s#^-e file://.*#-e file#" $W/logs/box-freeze.txt) > $W/logs/freeze-diff.txt && echo FREEZE_IDENTICAL || echo FREEZE_DIFFERS
echo SETUP_DONE
