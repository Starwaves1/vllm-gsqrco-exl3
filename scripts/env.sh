# shellcheck shell=bash
# source me: shared settings and guards for the Phase B harnesses
# (serve-gsq.sh, serve-baseline.sh, tests/gpu, bench/*, cloud/bootstrap.sh).
#
# Everything is overridable from the environment. Guards:
#   - port: default 18090; 18080 (vision-proxy) and 18081 (production vLLM) are refused
#     outright, there is no override;
#   - GPU: gsq_require_gpu aborts unless GSQ_ALLOW_GPU=1;
#   - KV tier: the plugin's HF config dir must be this repo's hf-config/<name>, never a
#     production model dir, and the fs tier root must not be production's (namespace
#     collision: the tier folder is <root>/<model_config.model>_<config hash>).

GSQ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export GSQ_ROOT
export GSQ_VENV=${GSQ_VENV:-$GSQ_ROOT/.venv}
export GSQ_PROD_ARGV=${GSQ_PROD_ARGV:-$GSQ_ROOT/env/prod-serve-argv.txt}
export GSQ_PORT=${GSQ_PORT:-18090}
export GSQ_HOST=${GSQ_HOST:-127.0.0.1}
export GSQ_URL=${GSQ_URL:-http://127.0.0.1:$GSQ_PORT}
export GSQ_RUNS=${GSQ_RUNS:-$GSQ_ROOT/runs}          # gitignored

# Production facts (production-vllm-server-info.json, 2026-09-27). Read-only.
export GSQ_PROD_REPO=${GSQ_PROD_REPO:-$HOME/qwen38-27b-rtx3090}
GSQ_PROD_MODEL_DIR=$(sed -n 4p "$GSQ_PROD_ARGV")
export GSQ_PROD_MODEL_DIR
export GSQ_PROD_KV_TIER_ROOT=/mnt/kvcache/tier
export GSQ_PROD_CHAT_TEMPLATE_SHA256=d1f22a89eac3609dcfaa7b471b1f7d23bee2f084d275d26f4f8231d1d7908f4e

# The model under test and its HF config dir (Phase A item 4).
export GSQ_MODEL_NAME=Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp
export GSQ_GGUF_REPO=ukisai/Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF
export GSQ_GGUF_FILE=$GSQ_MODEL_NAME.gguf
export GSQ_GGUF_SHA256=9aecf1cd41b2cb2f32a74e0d889e33855ebef43b26f43b43feb5720239e677e5
if [ -z "${GSQ_GGUF:-}" ]; then
  for _g in "$GSQ_ROOT/models/$GSQ_GGUF_FILE" \
            "$GSQ_PROD_REPO/models/Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF/$GSQ_GGUF_FILE"; do
    [ -f "$_g" ] && GSQ_GGUF=$_g && break
  done
  GSQ_GGUF=${GSQ_GGUF:-$GSQ_ROOT/models/$GSQ_GGUF_FILE}
  unset _g
fi
export GSQ_GGUF
export GSQ_HF_CONFIG=${GSQ_HF_CONFIG:-$GSQ_ROOT/hf-config/$GSQ_MODEL_NAME}

# Baseline (W4A16) for A/B runs: production's model by default.
export GSQ_BASELINE_MODEL=${GSQ_BASELINE_MODEL:-$GSQ_PROD_MODEL_DIR}

# KV tiers. The fs tier gets its own root (never production's) and a smaller cap:
# on this host it shares the 400 GB drive with production's 300 GB tier.
# GSQ_KV_FS_TIER=0 drops the fs tier (what production's start script does when the
# drive is missing).
export GSQ_KV_TIER_ROOT=${GSQ_KV_TIER_ROOT:-/mnt/kvcache/gsq-tier}
export GSQ_KV_TIER_MAX_BYTES=${GSQ_KV_TIER_MAX_BYTES:-100000000000}
export GSQ_KV_FS_TIER=${GSQ_KV_FS_TIER:-1}

# Server environment, as production's start script sets it (STATUS.md, "Production's
# start script"). The API key is a local test key, never production's.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTHONHASHSEED=0
export GSQ_API_KEY=${GSQ_API_KEY:-gsq-local-test}
# vLLM general plugins in this venv: the two LoRA resolvers (production has them too)
# and gguf. The serve scripts pick the list; unset would load all three.
export GSQ_PLUGINS_OFF=lora_filesystem_resolver,lora_hf_hub_resolver
export GSQ_PLUGINS_ON=$GSQ_PLUGINS_OFF,gguf

gsq_die() { echo "gsq: $*" >&2; exit 2; }

gsq_check_port() {  # refuse production's ports, whatever the caller asks for
  case "$GSQ_PORT" in
    18080|18081) gsq_die "port $GSQ_PORT belongs to production (18080 vision-proxy, 18081 vLLM); refused" ;;
  esac
  case "$GSQ_URL" in
    *:18080|*:18080/*|*:18081|*:18081/*) gsq_die "GSQ_URL=$GSQ_URL points at production; refused" ;;
  esac
}

gsq_require_gpu() {
  [ "${GSQ_ALLOW_GPU:-0}" = 1 ] || gsq_die "GPU use is opt-in: set GSQ_ALLOW_GPU=1 (never while DeepSWE run 1 or production uses this GPU)"
}

# Production is "live" if anything listens on 18080/18081 or a production-venv vLLM runs.
# No GPU queries here (safe to call before the GPU gate).
gsq_prod_live() {
  if ss -Hltn 2>/dev/null | awk '{print $4}' | grep -qE ':(18080|18081)$'; then return 0; fi
  pgrep -f "$GSQ_PROD_REPO/venv/bin/vllm serve" >/dev/null 2>&1
}

gsq_refuse_if_prod_live() {  # $1 = name of the override variable
  local ov=$1
  if gsq_prod_live; then
    [ "${!ov:-0}" = 1 ] && { echo "gsq: WARNING production looks live on this host; $ov=1 given, continuing" >&2; return 0; }
    gsq_die "production is live on this host (port 18080/18081 or $GSQ_PROD_REPO/venv vllm); refusing to share its GPU. Override: $ov=1"
  fi
}

_gsq_real() { realpath -m -- "$1"; }

# Namespace rule: the fs KV tier folder is named after model_config.model, which the
# plugin sets to the HF config dir. It must be this repo's own dir.
gsq_assert_hf_config() {
  local d; d=$(_gsq_real "$GSQ_HF_CONFIG")
  [ "$d" = "$(_gsq_real "$GSQ_ROOT/hf-config/$GSQ_MODEL_NAME")" ] \
    || gsq_die "GSQ_HF_CONFIG=$GSQ_HF_CONFIG is not $GSQ_ROOT/hf-config/$GSQ_MODEL_NAME (KV-tier namespace rule)"
  case "$d/" in
    "$(_gsq_real "$GSQ_PROD_MODEL_DIR")"/*|"$(_gsq_real "$GSQ_PROD_REPO")"/*)
      gsq_die "HF config dir $d is inside production's tree; refused (KV-tier namespace collision)" ;;
  esac
  [ -f "$d/config.json" ] && [ -f "$d/PROVENANCE.json" ] || gsq_die "$d lacks config.json/PROVENANCE.json (run tools/make_hf_config.py build)"
  local s; s=$(sha256sum "$d/chat_template.jinja" | cut -d' ' -f1)
  [ "$s" = "$GSQ_PROD_CHAT_TEMPLATE_SHA256" ] || gsq_die "$d/chat_template.jinja sha256 $s != production's template"
}

gsq_assert_tier_root() {
  [ "$GSQ_KV_FS_TIER" = 1 ] || return 0
  [ "$(_gsq_real "$GSQ_KV_TIER_ROOT")" != "$GSQ_PROD_KV_TIER_ROOT" ] \
    || gsq_die "GSQ_KV_TIER_ROOT is production's tier root; refused"
  case "$(_gsq_real "$GSQ_KV_TIER_ROOT")/" in
    "$GSQ_PROD_KV_TIER_ROOT"/*) gsq_die "GSQ_KV_TIER_ROOT is inside production's tier root; refused" ;;
  esac
  local p; p=$(dirname "$(_gsq_real "$GSQ_KV_TIER_ROOT")")
  [ -d "$GSQ_KV_TIER_ROOT" ] || [ -w "$p" ] \
    || gsq_die "fs KV tier root $GSQ_KV_TIER_ROOT cannot be created ($p not writable); set GSQ_KV_TIER_ROOT=<dir> or GSQ_KV_FS_TIER=0"
}

# Production argv (one arg per line) into the array GSQ_ARGV.
gsq_load_prod_argv() {
  mapfile -t GSQ_ARGV < "$GSQ_PROD_ARGV"
  [ "${GSQ_ARGV[2]}" = serve ] || gsq_die "$GSQ_PROD_ARGV: unexpected layout"
}

# Rewrite GSQ_ARGV for a harness server. $1 = model positional; rest = extra args
# appended at the end. Changes: interpreter/venv, model, --port, --host, --chat-template
# (same bytes, repo copy), fs tier root/cap in --kv-transfer-config, optional CPU tier
# size (GSQ_CPU_TIER_BYTES) and --max-model-len (GSQ_MAX_MODEL_LEN). Everything else is production's.
gsq_rewrite_argv() {
  local model=$1; shift
  local out=("$GSQ_VENV/bin/python" "$GSQ_VENV/bin/vllm" serve "$model")
  local i=4 a v
  while [ $i -lt ${#GSQ_ARGV[@]} ]; do
    a=${GSQ_ARGV[$i]}; v=${GSQ_ARGV[$((i+1))]:-}
    case "$a" in
      --port) out+=(--port "$GSQ_PORT"); i=$((i+2)); continue ;;
      --host) out+=(--host "$GSQ_HOST"); i=$((i+2)); continue ;;
      --chat-template) out+=(--chat-template "$GSQ_HF_CONFIG/chat_template.jinja"); i=$((i+2)); continue ;;
      --max-model-len) out+=(--max-model-len "${GSQ_MAX_MODEL_LEN:-$v}"); i=$((i+2)); continue ;;
      --kv-transfer-config)
        local j=$v
        # GSQ_CPU_TIER_BYTES: smaller CPU tier for a box whose /dev/shm can't hold 24 GiB
        [ -n "${GSQ_CPU_TIER_BYTES:-}" ] && j=${j/\"cpu_bytes_to_use\":25769803776/\"cpu_bytes_to_use\":$GSQ_CPU_TIER_BYTES}
        if [ "$GSQ_KV_FS_TIER" = 1 ]; then
          j=${j/\"root_dir\":\"$GSQ_PROD_KV_TIER_ROOT\"/\"root_dir\":\"$GSQ_KV_TIER_ROOT\"}
          j=${j/\"max_bytes\":300000000000/\"max_bytes\":$GSQ_KV_TIER_MAX_BYTES}
          [[ $j == *"\"root_dir\":\"$GSQ_KV_TIER_ROOT\""* ]] || gsq_die "could not rewrite fs tier root_dir"
        else
          j=$(printf '%s' "$j" | sed 's/,"secondary_tiers":\[.*\]//')
          [[ $j != *secondary_tiers* ]] || gsq_die "could not drop the fs tier"
        fi
        out+=(--kv-transfer-config "$j"); i=$((i+2)); continue ;;
    esac
    out+=("$a"); i=$((i+1))
  done
  out+=("$@")
  GSQ_ARGV=("${out[@]}")
}

gsq_print_argv_diff() {  # $1 = label, $2 = VLLM_PLUGINS value
  echo "=== $1: argv vs production ($GSQ_PROD_ARGV); '<' production, '>' this launch"
  diff <(cat "$GSQ_PROD_ARGV") <(printf '%s\n' "${GSQ_ARGV[@]}") || true
  echo "=== env: VLLM_PLUGINS=$2 (production: unset = all installed; production's venv has no gguf)"
  echo "=== env: VLLM_API_KEY=<local test key>, PYTHONHASHSEED=$PYTHONHASHSEED, VLLM_USE_FLASHINFER_SAMPLER=$VLLM_USE_FLASHINFER_SAMPLER, PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
}

# Launch GSQ_ARGV in the foreground (exec keeps the pid in $GSQ_RUNS/serve-$1.pid).
# GSQ_LOG=<file>: also append all output there.
gsq_exec_server() {  # $1 = kind (gsq|baseline), $2 = VLLM_PLUGINS
  mkdir -p "$GSQ_RUNS"
  [ "$GSQ_KV_FS_TIER" = 1 ] && mkdir -p "$GSQ_KV_TIER_ROOT"
  echo $$ > "$GSQ_RUNS/serve-$1.pid"
  export VLLM_PLUGINS=$2 VLLM_API_KEY=$GSQ_API_KEY
  if [ -n "${GSQ_LOG:-}" ]; then
    mkdir -p "$(dirname "$GSQ_LOG")"
    exec > >(tee -a "$GSQ_LOG") 2>&1
  fi
  exec "${GSQ_ARGV[@]}"
}

# Python entry points: gate the GPU the same way (they import tools/no_gpu.py unless
# GSQ_ALLOW_GPU=1). Wait for a server's /health.
gsq_wait_health() {  # $1 = timeout seconds (default 1800), $2 = pid to watch (optional)
  local t=${1:-1800} pid=${2:-} n=0
  until curl -sf -o /dev/null "$GSQ_URL/health"; do
    [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null && { echo "gsq: server pid $pid exited" >&2; return 1; }
    n=$((n+5)); [ $n -ge "$t" ] && { echo "gsq: no /health after ${t}s" >&2; return 1; }
    sleep 5
  done
}
