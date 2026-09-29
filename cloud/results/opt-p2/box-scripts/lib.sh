#!/bin/bash
# opt-p2 box helpers: source me with WT set (a frozen copy /workspace/wt-p2-<tag> of the local
# worktree, see stage.sh). Route L on, WT's plugin via PYTHONPATH, shared venv.
source /workspace/box-env.sh
export VLLM_GGUF_LCPP=1 GSQ_VENV=/workspace/gsq-vllm/.venv
export PYTHONPATH=$WT/plugin:$WT/tools
L=/workspace/logs/opt/p2; mkdir -p $L
S=$WT/cloud/results/opt-p2/box-scripts
cd $WT
source scripts/env.sh
stopall() { pkill -INT -f "bin/vllm serve"; for i in $(seq 90); do pgrep -f "VLLM::EngineCore|bin/vllm serve" >/dev/null || break; sleep 2; done
  pkill -9 -f "VLLM::EngineCore"; pkill -9 -f "bin/vllm serve"; sleep 5; box_clean_shm; rm -rf /workspace/kvtier; }
build() { ( source tools/cuda-env.sh; export PATH=$GSQ_VENV/bin:$PATH; cd plugin
  VLLM_GGUF_BUILD_LCPP=1 MAX_JOBS=24 python setup.py build_ext --inplace ) > $L/$TAG-build.log 2>&1
  echo "build rc=$?"; tail -1 $L/$TAG-build.log; }
parity() { local T=tests/gpu/test_kernel_parity.py
  tools/pytest $T -q -rs > $L/$TAG-parity-lcpp.log 2>&1; echo "parity lcpp rc=$?: $(tail -1 $L/$TAG-parity-lcpp.log)"
  env -u VLLM_GGUF_LCPP tools/pytest $T -q -rs > $L/$TAG-parity-stock.log 2>&1; echo "parity stock rc=$?: $(tail -1 $L/$TAG-parity-stock.log)"; }
# production's argv with this worktree's hf-config; extra args appended
serve() { gsq_load_prod_argv
  gsq_rewrite_argv "$GSQ_GGUF" --hf-config-path "$GSQ_HF_CONFIG" --tokenizer "$GSQ_HF_CONFIG" "$@"
  export VLLM_PLUGINS=$GSQ_PLUGINS_ON VLLM_API_KEY=$GSQ_API_KEY
  mkdir -p "$GSQ_KV_TIER_ROOT"
  setsid "${GSQ_ARGV[@]}" > $R/server.log 2>&1 &
  gsq_wait_health 2400 $! || { echo SERVER_FAILED; tail -30 $R/server.log; stopall; exit 1; }
  grep -E "draft head|Loading weights took|Model loading took|GPU KV cache size" $R/server.log | cut -c1-220 | head -6; }
