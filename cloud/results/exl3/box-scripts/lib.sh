# shellcheck shell=bash
# EXL3 phase-1 GPU jobs (01..07): every job script sources this. One gpuq job per script:
#   gpuq submit exl3-01 -- bash /workspace/wt-exl3/cloud/results/exl3/box-scripts/01-kernel-parity.sh
# (a wrapper script, not `bash -c "..."`: gpuq stores its command as "$*"). Idempotent: a job can
# be re-run; it overwrites its own run dir. Logs /workspace/logs/exl3/NN-*.log; big outputs stay in
# /workspace/runs/exl3/NN-*/; small results are copied to $WT/cloud/results/exl3/NN-*/ to commit.
# Needs part A (00-prep.sh venv, overlay, plugins, ref, model) done.
set -euo pipefail
source /workspace/box-env.sh               # GSQ_CPU_TIER_BYTES (13 GiB /dev/shm), LLAMA_DIR, box_clean_shm
export WT=${WT:-/workspace/wt-exl3}
S=$WT/cloud/results/exl3/box-scripts
export GSQ_VENV=/workspace/venv-main GSQ_EXL3_VENV=/workspace/venv-exl3ref
export GSQ_PROD_ARGV=$WT/env/prod-main-serve-argv.txt VLLM_USE_V2_MODEL_RUNNER=0   # production's main launcher env
export GSQ_EXL3_MODEL=/workspace/models/Qwen3.8-27B-exl3-3.50bpw
export EXL3_MODEL=$GSQ_EXL3_MODEL
export R=/workspace/runs/exl3 L=/workspace/logs/exl3 RES=$WT/cloud/results/exl3
export EXL3_REF_DIR=$R/kernel-ref EXLLAMAV3_TUNE_CACHE=$R/tune-cache   # one autotune cache for plugin + reference
export GSQ_RUNS=$R GSQ_ALLOW_GPU=1 VLLM_GGUF_LCPP=1
export PYTHONPATH=$WT/plugin-exl3:$WT/plugin:$WT/tools
# production's 40,960 draft ids (~/qwen38-27b-rtx3090/prepare/draft_vocab_ids.json on ms4, copied by part A)
export DRAFT_IDS=/workspace/ref/draft_vocab_ids.json
DRAFT_IDS_SHA256=b64b6dfcf5441eb995ddf77d3d37b018e91b88c56ad1b4c5774ad8fbfac1c388
# own fs KV tier (never the soak's /workspace/kvtier); off when the small disk has no room for its cap
export GSQ_KV_TIER_ROOT=/workspace/kvtier-exl3 GSQ_KV_TIER_MAX_BYTES=${EXL3_KV_TIER_BYTES:-10000000000}
SAN=/usr/local/cuda/bin/compute-sanitizer
mkdir -p "$R" "$L" "$EXLLAMAV3_TUNE_CACHE"
cd "$WT"
source scripts/env.sh
if [ "$(df --output=avail -B1 /workspace | tail -1)" -lt $((GSQ_KV_TIER_MAX_BYTES + 5 * 1073741824)) ]; then
  export GSQ_KV_FS_TIER=0
fi
SPID=

die() { echo "exl3 job: $*" >&2; exit 1; }
job_log() { exec > >(tee -a "$L/$1.log") 2>&1; echo "=== $1 $(date -u +%FT%TZ) wt $(git -C "$WT" rev-parse --short HEAD) fs-tier=$GSQ_KV_FS_TIER"; }
require_idle_gpu() {  # nothing else on the GPU (gpuq runs one job at a time; this catches strays)
  local apps; apps=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader)
  [ -z "$apps" ] || die "GPU busy: $apps"
  [ -f "$EXL3_MODEL/config.json" ] || die "no checkpoint at $EXL3_MODEL (00-prep.sh model)"
}
keep() {  # keep <run dir> <name> files...: copy small outputs into the repo tree
  local dst=$RES/$2; mkdir -p "$dst"; shift 2
  for f in "$@"; do [ -e "$f" ] && cp -r "$f" "$dst/"; done
}
stopall() {  # our server only (its process group), then its shm and fs-tier files
  if [ -n "$SPID" ]; then
    kill -INT -- "-$SPID" 2>/dev/null || true
    for _ in $(seq 90); do kill -0 "$SPID" 2>/dev/null || break; sleep 2; done
    kill -KILL -- "-$SPID" 2>/dev/null || true
    sleep 3
  fi
  SPID=
  box_clean_shm || true
  rm -rf "$GSQ_KV_TIER_ROOT"
}
serve() {  # serve <run dir>: scripts/serve-exl3.sh (production's argv) in the background, wait for health
  local out=$1 t0; mkdir -p "$out"
  scripts/serve-exl3.sh --dry-run > "$out/argv.txt" 2>&1
  rm -rf "$GSQ_KV_TIER_ROOT"; box_clean_shm || true
  t0=$(date +%s)
  GSQ_LOG=$out/server.log setsid scripts/serve-exl3.sh > /dev/null 2>&1 < /dev/null &
  SPID=$!
  trap stopall EXIT
  gsq_wait_health 2400 "$SPID" || { echo SERVER_FAILED; tail -80 "$out/server.log"; return 1; }
  { echo "healthy after $(( $(date +%s) - t0 )) s"
    grep -E "Loading weights took|Model loading took|model weights|GPU KV cache size|Maximum concurrency|MTP drafter|draft head|Available KV|CUDA graph|init engine|exl3|EXL3" "$out/server.log" | cut -c1-260 | head -40
    echo "VRAM after load: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader)"; } | tee "$out/load.txt"
}
junit_line() {  # junit_line <xml>: "tests=.. failures=.. errors=.. skipped=.."
  python3 - "$1" <<'PY'
import sys, xml.etree.ElementTree as ET
r = ET.parse(sys.argv[1]).getroot(); s = r if r.tag == "testsuite" else r.find("testsuite")
print(" ".join(f"{k}={s.get(k)}" for k in ("tests", "failures", "errors", "skipped")), f"time={float(s.get('time', 0)):.0f}s")
PY
}
