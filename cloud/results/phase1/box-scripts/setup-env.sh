#!/bin/bash
# box adaptation of cloud/bootstrap.sh steps: venv, vLLM overlay, gguf-py, plugin build, freeze diff
set -euxo pipefail
W=/workspace G=/workspace/gsq-vllm
SP=$G/.venv/lib/python3.12/site-packages
cd $G
uv python install 3.12.13
[ -x .venv/bin/python ] || uv venv --python 3.12.13 .venv
uv pip install --python .venv/bin/python --link-mode=copy --no-deps -r env/prod-freeze.txt
date -u +"venv done %FT%TZ"
[ -d $W/vllm-src/.git ] || git clone --filter=blob:none https://github.com/Starwaves1/vllm.git $W/vllm-src
git -C $W/vllm-src fetch --tags origin
mkdir -p $W/deploy-state
DENV=(env VLLM_REPO=$W/vllm-src SITE_PACKAGES=$SP STATE_FILE=$W/deploy-state/state BACKUP_ROOT=$W/deploy-state/bk)
"${DENV[@]}" $W/deploy/scripts/deploy-vllm.sh --init v0.27.1
"${DENV[@]}" $W/deploy/scripts/deploy-vllm.sh ba05ffababdcf89ada26b5d34845e04901e2ddf3
"${DENV[@]}" $W/deploy/scripts/deploy-vllm.sh --verify
cp $W/prod-build_backend.py $SP/build_backend.py
date -u +"overlay done %FT%TZ"
[ -d $W/llama.cpp/.git ] || git clone --branch b11211 --depth 1 https://github.com/ggml-org/llama.cpp $W/llama.cpp
[ "$(git -C $W/llama.cpp rev-parse HEAD)" = d7fb90e8e2494b2908934d956a3202fd60152ee0 ]
uv pip install --python .venv/bin/python --no-deps --link-mode=copy $W/llama.cpp/gguf-py
tools/setup-cuda-toolchain.sh
MAX_JOBS=32 tools/build-plugin.sh
date -u +"plugin done %FT%TZ"
uv pip freeze --python .venv/bin/python > $W/logs/box-freeze.txt
diff <(sed "s#@ file://.*#@ file#; s#^-e file://.*#-e file#" env/gsq-freeze.txt) <(sed "s#@ file://.*#@ file#; s#^-e file://.*#-e file#" $W/logs/box-freeze.txt) > $W/logs/freeze-diff.txt && echo FREEZE_IDENTICAL || echo FREEZE_DIFFERS
echo SETUP_DONE
