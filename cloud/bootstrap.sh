#!/bin/bash
# Phase B on a rented sm86 box (RTX 3090 / A10 / A6000, driver with CUDA 13 support):
# rebuild production's venv + this repo's plugin build, fetch the models, run the GPU
# tests, parity, speed (GGUF and a W4A16 baseline) and optionally the 24 h soak, then
# pack and upload the results.
#
#   cloud/bootstrap.sh --repo URL_OR_BUNDLE [--commit SHA] [--baseline prod|swift]
#                      [--workdir DIR] [--upload user@host:dir] [--soak]
#                      [--skip-tests] [--skip-parity] [--skip-speed] [--dry-run]
#
# --repo: this repo (gsq-vllm). It is not hosted anywhere yet; the simplest transport is
#   a bundle:  git -C ~/gsq-vllm bundle create /tmp/gsq-vllm.bundle main
#              scp /tmp/gsq-vllm.bundle box:   ->   --repo ~/gsq-vllm.bundle
# --commit: commit of this repo to check out (default: PINNED_COMMIT below; the plugin
#   fork is plugin/ at it). Bump it when plugin/ or the harnesses change.
# --baseline (STATUS open question; default prod):
#   prod  = production's W4A16 (-fast): HF Starw1/Qwen3.8-27B-absolute-heresy-W4A16, whose
#           config.json / quantization_config.json / index are byte-identical to
#           production's (report 03; checked here against production's sha256s).
#   swift = Swift-1.5 W4A16 AutoRound, "-prepared" (int8 lm_head/embed/MTP + draft head,
#           report 02). Not on HF: GSQ_BASELINE_SRC=<rsync source of that dir> is required.
#           200k does not fit at 0.94 with it, so it runs with --max-model-len -1 (auto-fit,
#           ~183k), which still covers the 180k prefill point.
# Downloads are public (the GGUF API answered without auth on 2026-09-27); an HF_TOKEN in
# the environment is used if set.
# Nothing here uses sudo or system packages. uv is installed per-user if missing.
# GPU job queue: cloud/box/gpuq (one job at a time, FIFO) is not installed by this script; copy it to
# /usr/local/bin/gpuq on a shared box and start it with `gpuq daemon start` before submitting jobs
# (`gpuq submit NAME -- bash script.sh`, `gpuq ls`, `gpuq wait ID --max SEC`; header has the rest).
set -euo pipefail

PINNED_COMMIT=${GSQ_COMMIT:-b16541fb4dac8932a5f47d72bf01ee4f96ae54fe} # gsq-vllm main with the Phase B harnesses
DEPLOY_REPO_URL=https://github.com/Starwaves1/qwen38-27b-rtx3090.git
DEPLOY_COMMIT=2138d1ae8d9ba2075ed962ba9adb5e67b91a2fe6   # production's deploy repo (bench script, deploy-vllm.sh)
VLLM_FORK_URL=https://github.com/Starwaves1/vllm.git
VLLM_COMMIT=ba05ffababdcf89ada26b5d34845e04901e2ddf3     # production's deployed vLLM patches
LLAMA_URL=https://github.com/ggml-org/llama.cpp
LLAMA_TAG=b11211 LLAMA_COMMIT=d7fb90e8e2494b2908934d956a3202fd60152ee0
PY_VERSION=3.12.13
PROD_W4A16_REPO=Starw1/Qwen3.8-27B-absolute-heresy-W4A16
PROD_SHA256_CONFIG=85972eb15137d6ea706a6aeda3e33dd8c26605fea29afa6f6478b8dc4fdc30b4
PROD_SHA256_QCONFIG=394dacad1bd8e205d420dbc55ac3d87b4fe98b114479e8b34f2c9b90f3c922f3
PROD_SHA256_INDEX=e6f530d7a178a60cff4431b16b7596b4196e6b08b3b23b117911f1a6275cc0ba

REPO='' COMMIT=$PINNED_COMMIT BASELINE=prod W=$HOME/gsq UPLOAD=${GSQ_UPLOAD_DEST:-} SOAK=0 DRY=0
SKIP_TESTS=0 SKIP_PARITY=0 SKIP_SPEED=0
while [ $# -gt 0 ]; do
  case $1 in
    --repo) REPO=$2; shift ;;
    --commit) COMMIT=$2; shift ;;
    --baseline) BASELINE=$2; shift ;;
    --workdir) W=$2; shift ;;
    --upload) UPLOAD=$2; shift ;;
    --soak) SOAK=1 ;;
    --skip-tests) SKIP_TESTS=1 ;;
    --skip-parity) SKIP_PARITY=1 ;;
    --skip-speed) SKIP_SPEED=1 ;;
    --dry-run) DRY=1 ;;
    -h|--help) sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac; shift
done
die() { echo "bootstrap: $*" >&2; exit 2; }
case $BASELINE in prod|swift) ;; *) die "--baseline prod|swift" ;; esac
[ -n "$REPO" ] || die "--repo is required (URL or git bundle of gsq-vllm)"
[ -n "$COMMIT" ] || die "--commit is required"
[ "$BASELINE" = swift ] && [ -z "${GSQ_BASELINE_SRC:-}" ] && die "--baseline swift needs GSQ_BASELINE_SRC=<rsync source of Swift-1.5-Qwen3.8-27B-W4A16-AutoRound-prepared>"

G=$W/gsq-vllm M=$W/models RES=$W/results
STEP=0
step() { STEP=$((STEP + 1)); echo; echo "=== [$STEP] $*"; }
run() { if [ $DRY = 1 ]; then printf '  +'; printf ' %q' "$@"; echo; else "$@"; fi; }
sha_ok() { [ $DRY = 1 ] && return 0; [ "$(sha256sum "$1" | cut -d' ' -f1)" = "$2" ] || die "sha256 mismatch: $1"; }

step "preflight: GPU, driver, disk, shared memory"
if [ $DRY = 0 ]; then
  nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader
  cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
  [ "$cap" = 8.6 ] || die "need an sm86 GPU (3090/A10/A6000); found compute capability $cap"
  mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
  [ "$mem" -ge 23000 ] || die "need >= 24 GB VRAM (found $mem MiB)"
  nvidia-smi | grep -qE 'CUDA Version: 1[3-9]' || die "driver must support CUDA 13 (torch 2.13.0+cu130)"
  mkdir -p "$W"
  free_gb=$(df -BG --output=avail "$W" | tail -1 | tr -dc 0-9)
  [ "$free_gb" -ge 150 ] || die "need >= 150 GB free in $W (models 12 + 16 GB, venv 8 GB, KV fs tier, llama.cpp)"
  shm_gb=$(df -BG --output=avail /dev/shm | tail -1 | tr -dc 0-9)
  if [ "$shm_gb" -lt 26 ]; then
    export GSQ_CPU_TIER_BYTES=$(( (shm_gb - 2) * 1073741824 ))
    echo "WARNING: /dev/shm has ${shm_gb} GB; CPU KV tier reduced to $GSQ_CPU_TIER_BYTES bytes (shown in the argv diff)"
  fi
  command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH=$HOME/.local/bin:$PATH; }
fi

step "clone gsq-vllm at $COMMIT"
[ -d "$G/.git" ] || run git clone "$REPO" "$G"
run git -C "$G" checkout --detach "$COMMIT"
[ $DRY = 1 ] || [ "$(git -C "$G" rev-parse HEAD)" = "$(git -C "$G" rev-parse "$COMMIT^{commit}")" ] || die "checkout failed"

step "venv = production's freeze (PyPI), Python $PY_VERSION"
run uv python install "$PY_VERSION"
[ -x "$G/.venv/bin/python" ] || run uv venv --python "$PY_VERSION" "$G/.venv"
run uv pip install --python "$G/.venv/bin/python" --link-mode=copy --no-deps -r "$G/env/prod-freeze.txt"

step "overlay production's vLLM patches ($VLLM_COMMIT) with production's deploy tool"
[ -d "$W/deploy/.git" ] || run git clone "$DEPLOY_REPO_URL" "$W/deploy"
run git -C "$W/deploy" checkout --detach "$DEPLOY_COMMIT"
[ -d "$W/vllm-src/.git" ] || run git clone --filter=blob:none "$VLLM_FORK_URL" "$W/vllm-src"
run git -C "$W/vllm-src" fetch --tags origin
SP=$G/.venv/lib/python3.12/site-packages
DENV=(env "VLLM_REPO=$W/vllm-src" "SITE_PACKAGES=$SP" "STATE_FILE=$W/deploy-state/state" "BACKUP_ROOT=$W/deploy-state/bk")
run mkdir -p "$W/deploy-state"
run "${DENV[@]}" "$W/deploy/scripts/deploy-vllm.sh" --init v0.27.1
run "${DENV[@]}" "$W/deploy/scripts/deploy-vllm.sh" "$VLLM_COMMIT"
run "${DENV[@]}" "$W/deploy/scripts/deploy-vllm.sh" --verify

step "llama.cpp $LLAMA_TAG (gguf-py for the plugin, reference for parity/MTP)"
[ -d "$W/llama.cpp/.git" ] || run git clone --branch "$LLAMA_TAG" --depth 1 "$LLAMA_URL" "$W/llama.cpp"
[ $DRY = 1 ] || [ "$(git -C "$W/llama.cpp" rev-parse HEAD)" = "$LLAMA_COMMIT" ] || die "llama.cpp is not $LLAMA_TAG"
run uv pip install --python "$G/.venv/bin/python" --no-deps --link-mode=copy "$W/llama.cpp/gguf-py"

step "plugin: CUDA 13.0 toolchain + sm86 build (tools/build-plugin.sh)"
run "$G/tools/setup-cuda-toolchain.sh"
run env MAX_JOBS="$(nproc)" "$G/tools/build-plugin.sh"
if [ $DRY = 0 ]; then
  # same package set as the workstation's env/gsq-freeze.txt (file:// paths differ)
  diff <(sed 's#@ file://.*#@ file#; s#^-e file://.*#-e file#' "$G/env/gsq-freeze.txt") \
       <(uv pip freeze --python "$G/.venv/bin/python" | sed 's#@ file://.*#@ file#; s#^-e file://.*#-e file#') \
    || die "venv differs from env/gsq-freeze.txt"
fi

step "models: GGUF + baseline ($BASELINE)"
export HF_HUB_ENABLE_HF_TRANSFER=0
# constants (model repo/file/sha); the clone's copy, or this script's own in a dry run
if [ -f "$G/scripts/env.sh" ]; then source "$G/scripts/env.sh"; else source "$(dirname "$0")/../scripts/env.sh"; fi
run "$G/.venv/bin/hf" download "$GSQ_GGUF_REPO" "$GSQ_GGUF_FILE" --local-dir "$M"
sha_ok "$M/$GSQ_GGUF_FILE" "$GSQ_GGUF_SHA256"
if [ "$BASELINE" = prod ]; then
  BDIR=$M/Qwen3.8-27B-W4A16-AutoRound-fast
  run "$G/.venv/bin/hf" download "$PROD_W4A16_REPO" --local-dir "$BDIR"
  sha_ok "$BDIR/config.json" "$PROD_SHA256_CONFIG"
  sha_ok "$BDIR/quantization_config.json" "$PROD_SHA256_QCONFIG"
  sha_ok "$BDIR/model.safetensors.index.json" "$PROD_SHA256_INDEX"
else
  BDIR=$M/Swift-1.5-Qwen3.8-27B-W4A16-AutoRound-prepared
  run rsync -a --info=progress2 "$GSQ_BASELINE_SRC/" "$BDIR/"
  export GSQ_BASELINE_MAX_MODEL_LEN=-1
fi

# Harness environment for everything below.
export GSQ_GGUF=$M/$GSQ_GGUF_FILE GSQ_BASELINE_MODEL=$BDIR GSQ_DEPLOY_REPO=$W/deploy \
       GSQ_PROD_REPO=$W/deploy GSQ_KV_TIER_ROOT=$W/kvtier GSQ_RUNS=$RES \
       LLAMA_DIR=$W/llama.cpp GSQ_ALLOW_GPU=1
run mkdir -p "$RES"

if [ $SKIP_PARITY = 0 ] || [ $SKIP_TESTS = 0 ]; then
  step "llama.cpp build (CUDA sm86) + parity tool"
  run "$G/bench/parity/build.sh" --build-llama
fi

if [ $SKIP_TESTS = 0 ]; then
  step "llama.cpp MTP acceptance reference (llama-server :18091)"
  if [ $DRY = 0 ]; then
    "$G/bench/speed/serve-llamacpp.sh" > "$RES/llama-server.log" 2>&1 & LS=$!
    until curl -sf -o /dev/null http://127.0.0.1:18091/health; do kill -0 $LS || die "llama-server died"; sleep 5; done
    "$G/.venv/bin/python" "$G/bench/speed/mtp_acceptance.py" --engine llama --url http://127.0.0.1:18091 \
      --out "$RES/mtp-llamacpp.json" || true
    kill $LS; wait $LS 2>/dev/null || true
    export GSQ_LLAMACPP_MTP=$RES/mtp-llamacpp.json
  fi
  step "GPU tests (tests/gpu: kernels, guards, 200k fit, MTP acceptance)"
  run "$G/tools/pytest" "$G/tests/gpu" -v -s --junitxml="$RES/tests-gpu.xml" -p no:cacheprovider || true
fi

if [ $SKIP_PARITY = 0 ]; then
  step "logit parity vs llama.cpp (bf16 KV reference, then production's fp8 KV for information)"
  run env OUT="$RES/parity-bf16" "$G/bench/parity/run.sh" --kv-cache-dtype auto || true
  run env OUT="$RES/parity-fp8" "$G/bench/parity/run.sh" --kv-cache-dtype fp8 || true
fi

if [ $SKIP_SPEED = 0 ]; then
  step "speed: GGUF, then baseline ($BASELINE)"
  run env OUT="$RES/speed-gsq" "$G/bench/speed/run.sh" gsq --start || true
  run env OUT="$RES/speed-baseline-$BASELINE" GSQ_MAX_MODEL_LEN="${GSQ_BASELINE_MAX_MODEL_LEN:-}" \
    "$G/bench/speed/run.sh" baseline --start || true
fi

if [ $SOAK = 1 ]; then
  step "24 h soak (concurrency 2, CUDA graphs)"
  run env OUT="$RES/soak-gsq" "$G/bench/soak.sh" gsq --hours 24 --conc 2 || true
fi

step "pack + upload results"
TAR=$W/gsq-phaseb-$(hostname)-$(date -u +%Y%m%dT%H%M%SZ).tar.gz
if [ $DRY = 0 ]; then
  { nvidia-smi; git -C "$G" rev-parse HEAD; uv pip freeze --python "$G/.venv/bin/python"; } > "$RES/box.txt" 2>&1
  tar -C "$W" --exclude='*/llama' --exclude='*/vllm' --exclude='*.f32' -czf "$TAR" results
fi
if [ -n "$UPLOAD" ]; then run rsync -a "$TAR" "$UPLOAD/"; else echo "results: $TAR (no --upload given)"; fi
