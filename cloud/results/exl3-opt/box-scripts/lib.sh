# shellcheck shell=bash
# EXL3 optimization jobs (10..13, EXL3-OPT.md): every job script sources this. One gpuq job each,
# through run-job.sh (status files, the 10 -> 12/13 dependency):
#   gpuq submit exl3opt-NN -- bash /workspace/wt-exl3-opt/cloud/results/exl3-opt/box-scripts/run-job.sh NN-name
# Builds on the phase-1 box setup (cloud/results/exl3/box-scripts/lib.sh, sourced from this
# worktree: venv-main, the checkpoint, the draft head written by 02-draft-head) with its own
# worktree, run, log, result, autotune-cache and KV-tier dirs, so nothing of phase 1's is
# written. Idempotent: a job overwrites its own run dir. Logs /workspace/logs/exl3-opt/NN-*.log,
# big outputs /workspace/runs/exl3-opt/NN-*/, small results $WT/cloud/results/exl3-opt/NN-*/.
# The plugin is imported from $WT/plugin-exl3 (PYTHONPATH, checked), with _C_exl3 and
# _C_exl3_mr built in place (box: compile-only build, VLLM_EXL3_BUILD=1 setup.py build_ext
# --inplace with the phase-1 toolchain).
# WT / EXL3_OPT_TAG: another worktree of the branch (an experiment) runs with its own run, log and
# result dirs (suffix TAG, e.g. -h16), e.g. WT=/workspace/wt-exl3-opt-h16 EXL3_OPT_TAG=-h16
export WT=${WT:-/workspace/wt-exl3-opt} EXL3_OPT_TAG=${EXL3_OPT_TAG:-}
# the box clone belongs to another uid than root's git expects: let job_log's rev-parse work
export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0=$WT
source "$WT/cloud/results/exl3/box-scripts/lib.sh"
S=$WT/cloud/results/exl3-opt/box-scripts
# EXL3_OPT_VENV: another vLLM venv (e.g. /workspace/venv-r3-plp, GSQ round 3's prompt-logprobs patch)
export GSQ_VENV=${EXL3_OPT_VENV:-$GSQ_VENV}
export R=/workspace/runs/exl3-opt$EXL3_OPT_TAG$ALT L=/workspace/logs/exl3-opt$EXL3_OPT_TAG$ALT
export RES=$WT/cloud/results/exl3-opt$EXL3_OPT_TAG$ALT
export GSQ_RUNS=$R GSQ_KV_TIER_ROOT=/workspace/kvtier-exl3-opt
# a copy of phase 1's autotune cache: exl3_gemm runs the same tile choices as 01..07, and this
# job's tuning never writes into phase 1's
if [ ! -d "$R/tune-cache" ]; then
  mkdir -p "$R"; cp -r /workspace/runs/exl3/tune-cache "$R/tune-cache" 2>/dev/null || mkdir -p "$R/tune-cache"
fi
export EXLLAMAV3_TUNE_CACHE=$R/tune-cache
# production's main argv, as 03/06 (the erlidev quant misses 200,000 by 0.01 GiB, see 06-ladder.sh)
export GSQ_MAX_MODEL_LEN=${EXL3_MAX_MODEL_LEN:-196608}
MODES_ARGS=("$@")  # a job's positional args: the modes it runs (12, 13, 14)
mkdir -p "$R" "$L"

require_mr_build() {  # the plugin under test is this worktree's, with both extensions built
  local got
  got=$(CUDA_VISIBLE_DEVICES="" "$GSQ_VENV/bin/python" -c 'import sys; sys.path.insert(0, "tools"); import no_gpu
import vllm_exl3_plugin as p, vllm_exl3_plugin.ops as o; print(p.__file__, o.OPS_AVAILABLE, o.MR_AVAILABLE)' 2>&1 | tail -1)
  [[ $got == "$WT/plugin-exl3/vllm_exl3_plugin/__init__.py True True" ]] || die "plugin not from $WT or not built: $got"
}
# serve_mr <mode> <run dir> [extra vLLM args]: scripts/serve-exl3.sh (production's argv) in the
# background with EXL3_MR=<mode's digit> and, for an h suffix (2h), EXL3_EMBED_HOST=1 (token
# embedding in host memory); wait for health; load summary in <run dir>/load.txt. (The g/a
# suffixes of the 2026-10-01 runs, glue and K3/K5 from 1 row, are now the code's constants.)
serve_mr() {
  local mode=$1 mr=${1:0:1} host=0 out=$2 t0; shift 2; mkdir -p "$out"
  [[ $mode == *h* ]] && host=1
  export EXL3_MR=$mr EXL3_EMBED_HOST=$host
  # own compile cache per traced-graph variant: vLLM's cache key does not cover the plugin's
  # apply()/embedding() or its parameter layouts (EXL3_MR=2 stores K4 as int32 4-D), so a graph
  # traced under one variant must never be loaded by another (or by phase 1)
  export VLLM_CACHE_ROOT=$R/vllm-cache-mr$mr-h$host
  scripts/serve-exl3.sh --dry-run "$@" > "$out/argv.txt" 2>&1
  rm -rf "$GSQ_KV_TIER_ROOT"; box_clean_shm || true
  t0=$(date +%s)
  GSQ_LOG=$out/server.log setsid scripts/serve-exl3.sh "$@" > /dev/null 2>&1 < /dev/null &
  SPID=$!
  trap stopall EXIT
  gsq_wait_health 2400 "$SPID" || { echo SERVER_FAILED; tail -80 "$out/server.log"; return 1; }
  { echo "EXL3_MR=$mr EXL3_EMBED_HOST=$host healthy after $(( $(date +%s) - t0 )) s"
    grep -E "Loading weights took|Model loading took|model weights|GPU KV cache size|Maximum concurrency|CUDA graph|init engine|exl3|EXL3" "$out/server.log" | cut -c1-260 | head -30
    echo "VRAM after load: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader)"; } | tee "$out/load.txt"
}
