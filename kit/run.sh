#!/bin/bash
# Cross-architecture validation kit for the GSQ-RCO (GGUF) and EXL3 vLLM plugins. Read kit/README.md.
#
#   kit/run.sh bootstrap|tier1|tier2|tier3|all [options]
#
#   --gpu N            nvidia-smi index of the card to use (default 0); nothing else is touched
#   --name PREFIX      results folder prefix (default <hostname>-<card>, e.g. mybox-rtx3070)
#   --allow-capped     record numbers even if the card's power limit is below its default
#   --max-gpu-gb G     most GPU memory the kit may use (default: 85 % of the card)
#   --idle-mib N       the card counts as idle at <= N MiB used and <= 10 % util (default 1024)
#   --wait S           wait up to S seconds for the card to become idle (default 600)
#   --no-gpu           bootstrap only, no GPU present (CPU dry run); --arch X.Y and --cuda 12|13 stand in
#   --skip-build       bootstrap without compiling the plugins
#   --parity-k EXPR    pytest -k filter for the tier 1 parity tests (default: all)
#
# Output: kit/results/<prefix>-<YYYY-MM-DD>/{results.json,summary.md,...}; work files (venv, builds,
# synthetic weights) under $KIT_WORK (default kit/.work), models under $HF_HOME (default kit/.work/hf).
set -uo pipefail
KIT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(dirname "$KIT")
TIER=${1:-}; shift || true
GPU=0 NAME= ALLOW_CAPPED= MAX_GB= IDLE_MIB=1024 WAIT=600 NO_GPU=0 ARCH=8.6 CUDA_FORCE= SKIP_BUILD=0 PARITY_K=
while [ $# -gt 0 ]; do
  case $1 in
    --gpu) GPU=$2; shift ;;
    --name) NAME=$2; shift ;;
    --allow-capped) ALLOW_CAPPED=--allow-capped ;;
    --max-gpu-gb) MAX_GB=$2; shift ;;
    --idle-mib) IDLE_MIB=$2; shift ;;
    --wait) WAIT=$2; shift ;;
    --no-gpu) NO_GPU=1 ;;
    --arch) ARCH=$2; shift ;;
    --cuda) CUDA_FORCE=$2; shift ;;
    --skip-build) SKIP_BUILD=1 ;;
    --parity-k) PARITY_K=$2; shift ;;
    *) echo "unknown option $1 (see the header of $0)"; exit 2 ;;
  esac
  shift
done
case $TIER in bootstrap|tier1|tier2|tier3|all) ;; *) sed -n '2,20p' "$0"; exit 2 ;; esac

WORK=${KIT_WORK:-$KIT/.work}
export HF_HOME=${HF_HOME:-$WORK/hf}
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU   # CUDA's index = nvidia-smi's
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PORT=${KIT_PORT:-18200}
UV=${UV:-$(command -v uv || true)}
mkdir -p "$WORK"
log() { echo "[kit $(date +%H:%M:%S)] $*"; }
die() { log "FATAL: $*"; exit 1; }
[ -n "$UV" ] || die "uv not found: install it (https://docs.astral.sh/uv/) or set UV=/path/to/uv"

# ------------------------------------------------------------------------------ card facts
if [ $NO_GPU = 1 ]; then
  CAP=$ARCH CUDA_MAJOR=${CUDA_FORCE:-13} GPU_NAME="none" MEM_MIB=8192
else
  command -v nvidia-smi >/dev/null || die "nvidia-smi not found (use --no-gpu for a CPU dry run)"
  IFS=, read -r GPU_NAME CAP MEM_MIB < <(nvidia-smi -i "$GPU" --query-gpu=name,compute_cap,memory.total \
    --format=csv,noheader,nounits | sed 's/, */,/g') || die "no GPU with index $GPU"
  DRV_CUDA=$(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9.]*\).*/\1/p' | head -1)
  CUDA_MAJOR=${CUDA_FORCE:-${DRV_CUDA%%.*}}
fi
CCN=${CAP/./}
SLUG=$(echo "$GPU_NAME" | tr 'A-Z' 'a-z' | sed 's/nvidia//; s/geforce//; s/[^a-z0-9]//g')
OUT=$KIT/results/${NAME:-$(hostname -s)-$SLUG}-$(date +%F)
[ "$CUDA_MAJOR" -ge 13 ] 2>/dev/null && CU=13 || CU=12
VENV=$WORK/venv-cu$CU
PY=$VENV/bin/python
# memory cap: a fraction of the card for the tier 1 processes, gpu-memory-utilization for vLLM
MAX_MIB=$(awk -v g="${MAX_GB:-0}" -v t="$MEM_MIB" 'BEGIN { m = g > 0 ? g * 1024 : t * 0.85; print int(m < t ? m : t) }')
export KIT_MEM_FRACTION=$(awk -v m="$MAX_MIB" -v t="$MEM_MIB" 'BEGIN { printf "%.3f", m / t }')
log "card $GPU: $GPU_NAME, sm$CCN, $MEM_MIB MiB; kit memory cap $MAX_MIB MiB ($KIT_MEM_FRACTION); CUDA $CU stack; out $OUT"

pyjson() {  # pyjson FILE KEY.PATH VALUE-JSON: set a nested key in a JSON file
  python3 - "$@" <<'EOF'
import json, sys, os
f, path, val = sys.argv[1], sys.argv[2].split("/"), json.loads(sys.argv[3])
d = json.load(open(f)) if os.path.exists(f) else {}
cur = d
for k in path[:-1]:
    cur = cur.setdefault(k, {})
cur[path[-1]] = val
json.dump(d, open(f, "w"), indent=1)
EOF
}
jstr() { python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$1"; }
step() {  # step NAME STATUS [NOTE]: record a step's outcome in steps.json
  pyjson "$OUT/steps.json" "$1" "{\"status\": \"$2\", \"note\": $(jstr "${3:-}"), \"time\": \"$(date -Is)\"}"
  log "step $1: $2 ${3:-}"
}

# ------------------------------------------------------------------------------ bootstrap
bootstrap() {
  local B=$WORK/bootstrap.json
  rm -f "$B"
  pyjson "$B" cuda "\"$CU\""; pyjson "$B" cc "\"$CAP\""; pyjson "$B" venv "$(jstr "$VENV")"
  [ -x "$PY" ] || "$UV" venv --python 3.12 "$VENV" || die "uv venv failed"
  local F=$WORK/freeze-cu$CU.txt
  # the pinned stack of the repo's vLLM-main venv, minus the two local paths (gguf-py, the editable plugin)
  # and FlashInfer's prebuilt JIT cache (not on PyPI; a workaround for one host's broken FlashInfer JIT)
  grep -vE '^(-e |gguf @|flashinfer-jit-cache)' "$ROOT/env/gsq-main-freeze.txt" > "$F"
  if [ $CU = 13 ]; then
    "$UV" pip install --python "$PY" -r "$F" || die "pip install of the pinned cu130 stack failed"
  else
    # CUDA 12 driver: the vLLM pin ships cu130 wheels only, so torch comes from the cu129 index and vLLM's
    # Python code goes in without its CUDA deps: enough for tier 1 (the plugins' kernels); tiers 2/3 need a
    # driver with CUDA 13 (>= 580)
    grep -vE '^(torch|torchvision|torchaudio|triton|vllm|flashinfer[-a-z]*|nvidia-[-a-z0-9]*|cuda-[-a-z]*)( |=|@)' "$F" > "$F.cu12"
    # one resolution, torch-family packages from the PyTorch cu129 index (else torchcodec pulls a cu13 torch)
    "$UV" pip install --python "$PY" -r "$F.cu12" torch==2.13.0 --torch-backend cu129 || die "pip install of the cu129 stack failed"
    "$UV" pip install --python "$PY" --no-deps "$(grep -E '^vllm @' "$F" | sed 's/^vllm @ //')" || die "vLLM wheel install failed"
  fi
  "$UV" pip install --python "$PY" --no-deps "gguf @ git+https://github.com/ggml-org/llama.cpp@b11211#subdirectory=gguf-py" \
    || die "gguf-py b11211 install failed"
  # vendored sources must be byte-identical to their upstream commits (hashes in each VENDORED.md)
  local v ok=1
  for v in plugin/vllm_gguf_plugin/csrc/lcpp plugin-exl3/vllm_exl3_plugin/csrc/exl3 plugin-exl3/vllm_exl3_plugin/csrc/trellis_serve; do
    # "sha256  ./path" lines, or a "| `file` | `upstream path` | sha256 |" table (trellis_serve)
    if (cd "$ROOT/$v" && sed -nE 's/^([0-9a-f]{64})  (.*)/\1  \2/p; s/^\| `([^`]+)` \| `[^`]*` \| ([0-9a-f]{64}) \|.*/\2  \1/p' VENDORED.md \
        | sha256sum -c --quiet); then
      pyjson "$B" "vendored/${v##*/}" '"ok"'
    else
      pyjson "$B" "vendored/${v##*/}" '"MISMATCH"'; ok=0
    fi
  done
  [ $ok = 1 ] || die "vendored sources differ from their VENDORED.md hashes"
  [ $SKIP_BUILD = 1 ] && { log "bootstrap done (no build)"; return 0; }
  # toolchain: CUDA 13.0 from pip wheels (matches torch's cudart; tools/setup-cuda-toolchain.sh), or a
  # local CUDA 12.x toolkit for the cu129 torch
  if [ $CU = 13 ]; then
    PY=$PY "$ROOT/tools/setup-cuda-toolchain.sh" || die "CUDA 13.0 toolchain setup failed"
    export CUDA_HOME=$ROOT/build/cu130/nvidia/cu13
  else
    export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
    "$CUDA_HOME/bin/nvcc" --version 2>/dev/null | grep -q 'release 12\.' || die "CUDA 12 stack needs a CUDA 12.x toolkit: set CUDA_HOME"
  fi
  export PATH=$CUDA_HOME/bin:$VENV/bin:$PATH TORCH_CUDA_ARCH_LIST=$CAP MAX_JOBS=${MAX_JOBS:-4}
  command -v g++-13 >/dev/null && export CC=gcc-13 CXX=g++-13 NVCC_CCBIN=g++-13
  pyjson "$B" toolchain "$(jstr "$("$CUDA_HOME/bin/nvcc" --version | tail -1); host $(${CXX:-g++} --version | head -1)")"
  # a host compiler newer than this nvcc supports (e.g. GCC 15 with CUDA 13.0): build anyway, and say so
  printf 'int main() { return 0; }\n' > "$WORK/probe.cu"
  if "$CUDA_HOME/bin/nvcc" -c "$WORK/probe.cu" -o "$WORK/probe.o" 2>&1 | grep -q 'unsupported GNU version'; then
    export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:+$NVCC_APPEND_FLAGS }-allow-unsupported-compiler"
    pyjson "$B" toolchain_note '"host compiler newer than nvcc supports: built with -allow-unsupported-compiler"'
  fi
  build() {  # build KEY PKG ENV...: editable install; record ok / failed with the log's last error
    local key=$1 pkg=$2; shift 2
    local L=$WORK/build-$key.log
    log "building $key for sm$CCN (log $L)"
    if env "$@" "$UV" pip install --python "$PY" --no-build-isolation --no-deps --link-mode=copy -e "$ROOT/$pkg" -v > "$L" 2>&1; then
      pyjson "$B" "builds/$key" '{"status": "ok"}'; return 0
    fi
    local err; err=$(grep -m1 -E 'error|Error' "$L" | cut -c1-300)
    pyjson "$B" "builds/$key" "{\"status\": \"failed\", \"note\": $(jstr "$err"), \"log\": $(jstr "$L")}"; return 1
  }
  build gguf_route_l plugin VLLM_GGUF_BUILD_LCPP=1 || build gguf_stock plugin VLLM_GGUF_BUILD_LCPP=0
  if ! build exl3 plugin-exl3 VLLM_EXL3_BUILD=1; then
    build exl3_phase1_only plugin-exl3 VLLM_EXL3_BUILD=1 VLLM_EXL3_MR_BUILD=0 \
      || build exl3_python_only plugin-exl3 VLLM_EXL3_BUILD=0
  fi
  GSQ_VENV=$VENV "$ROOT/tools/pytest" --version >/dev/null 2>&1 || true   # installs pytest into build/pytest
  log "bootstrap done: $(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("builds"))' "$B")"
}

# Route L on only where it was built: with VLLM_GGUF_LCPP=1 and no Route L ops the plugin refuses to import
lcpp_env() { if built gguf_route_l; then export VLLM_GGUF_LCPP=1; else unset VLLM_GGUF_LCPP; fi; }
built() { python3 -c 'import json,sys; b=json.load(open(sys.argv[1])).get("builds",{}); sys.exit(0 if b.get(sys.argv[2],{}).get("status")=="ok" else 1)' "$WORK/bootstrap.json" "$1"; }

# ------------------------------------------------------------------------------ measurement plumbing
CLK=
gate() {  # gate STEP: power-limit / clock / idle check before a measurement; refuses capped or busy cards
  "$PY" "$KIT/report.py" gpucheck --out "$OUT/gpucheck-$1.json" --gpu "$GPU" $ALLOW_CAPPED --idle-mib "$IDLE_MIB" --wait "$WAIT"
  case $? in
    0) return 0 ;;
    3) step "$1" refused "power-capped below the default limit (rerun with --allow-capped)"; collect; exit 3 ;;
    *) step "$1" refused "card not idle after ${WAIT}s"; collect; exit 4 ;;
  esac
}
clocks_on() {
  local q=timestamp,clocks.sm,clocks.mem,power.draw,utilization.gpu,temperature.gpu,clocks_event_reasons.active
  nvidia-smi -i "$GPU" --query-gpu=$q --format=csv,noheader,nounits >/dev/null 2>&1 || q=${q/clocks_event/clocks_throttle}
  nvidia-smi -i "$GPU" --query-gpu=$q --format=csv,noheader,nounits -l 1 >> "$OUT/clocks-$1.csv" 2>/dev/null &
  CLK=$!
}
clocks_off() { [ -n "$CLK" ] && kill "$CLK" 2>/dev/null; wait "$CLK" 2>/dev/null; CLK=; }
collect() {  # results.json + summary.md, then the per-generation agreement over every result folder
  "$PY" "$KIT/report.py" collect "$OUT" >/dev/null && log "results: $OUT/summary.md"
  "$PY" "$KIT/report.py" agree >/dev/null && log "generation: $KIT/results/generation-sm$CCN.md"
}
start_out() {
  mkdir -p "$OUT"
  [ -f "$WORK/bootstrap.json" ] || die "run 'kit/run.sh bootstrap' first"
  [ -x "$PY" ] || die "no venv at $VENV: run 'kit/run.sh bootstrap' first"
  cp "$WORK/bootstrap.json" "$OUT/bootstrap.json"
  "$PY" "$KIT/report.py" env --out "$OUT/env.json" --gpu "$GPU" >/dev/null || die "env record failed"
  # the device CUDA sees must be the card nvidia-smi names (CUDA_DEVICE_ORDER=PCI_BUS_ID)
  local seen; seen=$("$PY" -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null)
  [ "$seen" = "$GPU_NAME" ] || die "CUDA device 0 is '$seen', nvidia-smi index $GPU is '$GPU_NAME': refusing"
}
trap 'clocks_off; [ -n "${SRV:-}" ] && kill "$SRV" 2>/dev/null' EXIT

# ------------------------------------------------------------------------------ tier 1
tier1() {
  start_out
  local T=$OUT/tier1 S=$WORK/synth
  local X3=$S/Swift-1.5-Qwen3.8-27B-exl3-SC_3.50bpw_H4_V6
  mkdir -p "$T" "$S"
  [ -s "$S/swift-27b-synthetic.gguf" ] || nice "$PY" "$KIT/synth.py" gguf "$S/swift-27b-synthetic.gguf" || die "synthetic GGUF failed"
  [ -s "$X3/model.safetensors" ] || nice "$PY" "$KIT/synth.py" exl3 "$X3" || die "synthetic EXL3 checkpoint failed"
  lcpp_env
  export GSQ_ALLOW_GPU=1 GSQ_VENV=$VENV GSQ_GGUF=$S/swift-27b-synthetic.gguf EXL3_MODEL=$X3 \
    PYTHONPATH=$KIT${PYTHONPATH:+:$PYTHONPATH}
  local k=(); [ -n "$PARITY_K" ] && k=(-k "$PARITY_K")
  local rc
  gate tier1-parity-gguf; clocks_on tier1-parity
  timeout "${KIT_PARITY_TIMEOUT:-3600}" "$ROOT/tools/pytest" -p kit_guard "$ROOT/tests/gpu/test_kernel_parity.py" -q -rfE \
    "${k[@]}" --junitxml="$T/parity-gguf.xml" > "$T/parity-gguf.log" 2>&1; rc=$?
  case $rc in 0) step tier1.parity-gguf ok ;; 1) step tier1.parity-gguf failed "test failures, see parity-gguf.log" ;;
    124) step tier1.parity-gguf failed "timeout" ;; *) step tier1.parity-gguf failed "pytest rc $rc" ;; esac
  built gguf_route_l || step tier1.route-l unsupported "Route L did not build for sm$CCN; stock kernels only ($WORK/build-gguf_route_l.log)"
  if built exl3 || built exl3_phase1_only; then
    timeout "${KIT_PARITY_TIMEOUT:-3600}" "$ROOT/tools/pytest" -p kit_guard "$ROOT/tests/gpu/test_exl3_kernels.py" \
      "$ROOT/tests/gpu/test_exl3_mr.py" -q -rfE "${k[@]}" --junitxml="$T/parity-exl3.xml" > "$T/parity-exl3.log" 2>&1; rc=$?
    case $rc in 0) step tier1.parity-exl3 ok ;; 1) step tier1.parity-exl3 failed "test failures, see parity-exl3.log" ;;
      *) step tier1.parity-exl3 failed "pytest rc $rc" ;; esac
  else
    step tier1.parity-exl3 unsupported "EXL3 kernels did not build for sm$CCN (see bootstrap.json)"
  fi
  clocks_off
  gate tier1-micro-gguf; clocks_on tier1-micro
  timeout "${KIT_MICRO_TIMEOUT:-2400}" "$PY" "$KIT/micro_gguf.py" --out "$T" > "$T/gguf_micro.log" 2>&1 \
    && step tier1.micro-gguf ok || step tier1.micro-gguf failed "$(tail -1 "$T/gguf_micro.log")"
  if built exl3; then
    timeout "${KIT_MICRO_TIMEOUT:-2400}" "$PY" "$ROOT/bench/micro/exl3_mr.py" --out "$T/exl3_micro" \
      --rows 1,2,4,6,8,12,16,24,32,48,64,128 > "$T/exl3_micro.log" 2>&1 \
      && step tier1.micro-exl3 ok || step tier1.micro-exl3 failed "$(tail -1 "$T/exl3_micro.log")"
  else
    step tier1.micro-exl3 unsupported "needs the full EXL3 build (multi-row kernel)"
  fi
  clocks_off
  unset GSQ_ALLOW_GPU
  collect
}

# ------------------------------------------------------------------------------ tiers 2 and 3
fetch() {  # fetch REPO REVISION [FILE]: print the local path (file, or snapshot dir without safetensors if FILE=cfg)
  "$PY" - "$@" <<'EOF'
import sys
from huggingface_hub import hf_hub_download, snapshot_download
repo, rev = sys.argv[1], sys.argv[2]
f = sys.argv[3] if len(sys.argv) > 3 else None
if f == "cfg":
    print(snapshot_download(repo, revision=rev, ignore_patterns=["*.safetensors", "*.gguf", "*.bin"]))
elif f:
    print(hf_hub_download(repo, f, revision=rev))
else:
    print(snapshot_download(repo, revision=rev))
EOF
}

SRV=
serve() {  # serve LOG ARGS...: start vLLM in the background and wait for /health
  local log=$1; shift
  curl -s -o /dev/null "http://127.0.0.1:$PORT/" && { log "port $PORT is in use (set KIT_PORT)"; return 1; }
  log "serving: $*"
  "$VENV/bin/vllm" serve "$@" > "$log" 2>&1 &
  SRV=$!
  local t=0
  until curl -sf -o /dev/null "http://127.0.0.1:$PORT/health"; do
    kill -0 "$SRV" 2>/dev/null || { SRV=; return 1; }
    sleep 5; t=$((t + 5))
    [ $t -ge "${KIT_SERVE_TIMEOUT:-1500}" ] && { stop; return 1; }
  done
  return 0
}
stop() { [ -n "$SRV" ] && { kill "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null; }; SRV=; sleep 3; }

# run_model TIER NAME PLUGINS K MODEL_PATH MAXLEN GMU EXTRA...: ladder (pass 2, T=0, c=1/2/4/8) and the
# 200-request corruption check against an eager, no-MTP reference of the same model
run_model() {
  local tier=$1 name=$2 plugins=$3 k=$4 model=$5 maxlen=$6 gmu=$7; shift 7
  local D=$OUT/$tier/$name; mkdir -p "$D"
  local argv=("$model" --served-model-name kit --host 127.0.0.1 --port "$PORT" --gpu-memory-utilization "$gmu"
    --max-model-len "$maxlen" --max-num-seqs 8 --max-num-batched-tokens 2048 --long-prefill-token-threshold 128
    --no-async-scheduling --mamba-ssm-cache-dtype float16 --mamba-cache-mode align --enable-prefix-caching
    --limit-mm-per-prompt '{"image":0,"video":0}' "$@")
  # Turing (sm75): no bf16 -> fp16 model dtype, no fp8 KV, Triton attention (FlashAttention needs sm80 and the
  # prebuilt FlashInfer cache covers sm80+)
  if [ "$CCN" -lt 80 ]; then argv+=(--dtype float16 --attention-backend TRITON_ATTN); else argv+=(--kv-cache-dtype fp8); fi
  local spec=(); [ "$k" -gt 0 ] && spec=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$k}")
  printf '%s\n' "${argv[@]}" "${spec[@]}" > "$D/argv.txt"
  lcpp_env
  export VLLM_PLUGINS=$plugins VLLM_USE_FLASHINFER_SAMPLER=0
  gate "$tier-$name"
  local t0=$SECONDS
  if ! serve "$D/server.log" "${argv[@]}" --compilation-config '{"max_cudagraph_capture_size":32}' "${spec[@]}"; then
    step "$tier.$name" failed "server did not start: $(grep -m1 -E 'Error|error' "$D/server.log" | cut -c1-200)"; return 1
  fi
  pyjson "$D/serve.json" load_s "$((SECONDS - t0))"
  pyjson "$D/serve.json" model "$(jstr "$model")"
  pyjson "$D/serve.json" attention "$(jstr "$(grep -m1 -oE 'Using [A-Z_]+ (attention )?backend[^.]*' "$D/server.log")")"
  pyjson "$D/serve.json" kv_cache "$(jstr "$(grep -m1 -oE 'GPU KV cache size: [0-9,]+ tokens' "$D/server.log")")"
  clocks_on "$tier-$name"
  local ok=1
  "$PY" "$KIT/ladder.py" --url "http://127.0.0.1:$PORT" --model kit --out "$D/ladder.json" --label "$name" \
    --note "$([ "$k" -gt 0 ] && echo "MTP k=$k" || echo "no speculative decoding"), T=0, 256 tokens/request" > "$D/ladder.log" 2>&1 || ok=0
  "$PY" "$ROOT/bench/corruption_check.py" run --url "http://127.0.0.1:$PORT" --model kit --label "$name" --out "$D" \
    --conc 1,2,4,8 --temps 0,0.7 --n-prompts 25 --max-tokens 256 > "$D/corruption-run.log" 2>&1 || ok=0
  clocks_off; stop
  if serve "$D/ref-server.log" "${argv[@]}" --enforce-eager; then
    "$PY" "$ROOT/bench/corruption_check.py" run --url "http://127.0.0.1:$PORT" --model kit --label ref --out "$D" \
      --conc 1 --temps 0 --n-prompts 25 --max-tokens 256 > "$D/ref-run.log" 2>&1 || ok=0
    stop
    "$PY" "$ROOT/bench/corruption_check.py" compare --ref "$D/ref.jsonl" "$D/$name.jsonl" > "$D/corruption.jsonl" 2>&1 || ok=0
  else
    ok=0
  fi
  gzip -f "$D/server.log" "$D/ref-server.log" "$D"/*.jsonl 2>/dev/null; gunzip -f "$D/corruption.jsonl.gz" 2>/dev/null
  [ $ok = 1 ] && step "$tier.$name" ok || step "$tier.$name" failed "see $D/*.log"
}

tier2() {
  start_out
  [ $CU = 13 ] || { step tier2.all unsupported "the pinned vLLM needs a CUDA 13 driver (>= 580)"; collect; return; }
  [ "$MAX_MIB" -ge 6000 ] || { step tier2.all unsupported "needs >= 6 GB of GPU memory"; collect; return; }
  local gmu; gmu=$(awk -v m="$MAX_MIB" -v t="$MEM_MIB" 'BEGIN { g = m / t; printf "%.2f", (g > 0.9 ? 0.9 : g) }')
  local cfg gguf
  cfg=$(fetch Qwen/Qwen3.5-4B 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a cfg) || die "Qwen3.5-4B config download failed"
  gguf=$(fetch unsloth/Qwen3.5-4B-MTP-GGUF 86835bf9949e4d14d6860f7910b1340ad4f271a9 Qwen3.5-4B-Q4_K_M.gguf) || die "GGUF download failed"
  echo "3874209241c9a397e2f62cd3f70f80fd2dfbf0dfccb6838416bdb48a714e8630  $gguf" | sha256sum -c --quiet || die "GGUF sha256 mismatch"
  run_model tier2 gguf-qwen3.5-4b-q4km-k3 lora_filesystem_resolver,gguf 3 "$gguf" 8192 "$gmu" --hf-config-path "$cfg" --tokenizer "$cfg"
  if built exl3 && [ "$CCN" -ge 80 ]; then
    local x3; x3=$(fetch UnstableLlama/Qwen3.5-4B-exl3-4.00bpw d56d49f6b712044fb960d4ed8e84c51cd6c14ad0) || die "EXL3 download failed"
    # no public Qwen3.5-4B EXL3 quant keeps the MTP head (2026-10-04), so this one runs without MTP
    run_model tier2 exl3-qwen3.5-4b-4.00bpw-nomtp lora_filesystem_resolver,gguf,exl3 0 "$x3" 8192 "$gmu"
  else
    step tier2.exl3 unsupported "EXL3 needs its full build and sm80+"
  fi
  collect
}

tier3() {
  start_out
  [ $CU = 13 ] || { step tier3.all unsupported "the pinned vLLM needs a CUDA 13 driver (>= 580)"; collect; return; }
  [ "$MAX_MIB" -ge 20000 ] || { step tier3.all unsupported "needs a 24 GB card"; collect; return; }
  local gmu; gmu=$(awk -v m="$MAX_MIB" -v t="$MEM_MIB" 'BEGIN { g = m / t; printf "%.2f", (g > 0.92 ? 0.92 : g) }')
  local cfg=$ROOT/hf-config/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp gguf
  gguf=$(fetch ukisai/Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF d74895bbe5db4bec1e0024e7cc87d59c02d7631a Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf) \
    || die "27B GGUF download failed"
  echo "9aecf1cd41b2cb2f32a74e0d889e33855ebef43b26f43b43feb5720239e677e5  $gguf" | sha256sum -c --quiet || die "27B GGUF sha256 mismatch"
  run_model tier3 gguf-swift-27b-iq3s-k3 lora_filesystem_resolver,gguf 3 "$gguf" 32768 "$gmu" \
    --hf-config-path "$cfg" --tokenizer "$cfg"
  if built exl3 && [ "$CCN" -ge 80 ]; then
    local x3
    if x3=$(fetch erlidev/Swift-1.5-Qwen3.8-27B-EXL3 041dc382133c0ac4a3fcf6c400d70fbea1584c51); then
      run_model tier3 exl3-swift-27b-3.50bpw-k3 lora_filesystem_resolver,gguf,exl3 3 "$x3" 32768 "$gmu"
    else
      step tier3.exl3 failed "erlidev/Swift-1.5-Qwen3.8-27B-EXL3 download failed (gated? set HF_TOKEN)"
    fi
  fi
  collect
}

case $TIER in
  bootstrap) bootstrap ;;
  tier1) tier1 ;;
  tier2) tier2 ;;
  tier3) tier3 ;;
  all) bootstrap && tier1 && tier2 && { [ "$MAX_MIB" -ge 20000 ] && tier3 || true; } ;;
esac
[ "$TIER" != bootstrap ] && log "done. Send $OUT back: see kit/RETURN.md"
exit 0
