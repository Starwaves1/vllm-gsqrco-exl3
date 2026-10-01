# shellcheck shell=bash
# Round-3 box helpers: source me from the NN-*.sh scripts (one gpuq job each).
#
# Layout on the box (override any of these from the environment):
#   R3_WT          this repo checkout (scripts, harness, hf-config, env/prod-main-serve-argv.txt);
#                  default: derived from this file's location
#   R3_PLUGIN_WT   checkout whose plugin/ holds the built Route L _C_gguf .so (default
#                  /workspace/wt-gsq-32ae6ec, main 32ae6ec built under /workspace/venv-main)
#   GSQ_VENV       vLLM main venv (default /workspace/venv-main, = production's venv-main)
#   R3_LOGS        logs and results root (default /workspace/logs/r3); each script writes
#                  $R3_LOGS/<script>/ (run.log, summary.txt, per-server dirs) and replaces it on rerun
# The server runs production's main argv (env/prod-main-serve-argv.txt) through scripts/env.sh's
# rewrite: only interpreter, model (.gguf) + --hf-config-path/--tokenizer, host/port, chat template
# (same bytes), KV tier root/cap and CPU tier size differ, plus what each script changes on
# purpose (R3_MUT, printed in argv-diff.txt).
set -uo pipefail

R3_S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
R3_WT=${R3_WT:-$(cd "$R3_S/../../../.." && pwd)}
R3_PLUGIN_WT=${R3_PLUGIN_WT:-/workspace/wt-gsq-32ae6ec}
R3_LOGS=${R3_LOGS:-/workspace/logs/r3}
R3_EXPECT_VLLM=${R3_EXPECT_VLLM:-0.30.1rc1.dev285+gd28795f1a}
R3_PLAN=0
R3_STEP=init
R3_SP=
R3_MUT=()
R3_SERVE_PREFIX=()   # e.g. (taskset -c 0-5,32-37) for 26-idle-cpu-contention
export R3_PROMPT_CACHE=${R3_PROMPT_CACHE:-$R3_LOGS/_prompts}

r3_die() { echo "r3: FAIL [$R3_STEP]: $*" >&2; exit 2; }
r3_step() { R3_STEP=$1; echo; echo "=== [$(date -u +%T)] $*"; }

# r3_init NAME "$@": parse --plan, set $L, start logging, install the exit trap.
r3_init() {
  R3_NAME=$1${R3_TAG:+-$R3_TAG}; shift   # R3_TAG: a second run of a script (e.g. on a patched venv)
  for a in "$@"; do
    case $a in
      --plan) R3_PLAN=1 ;;
      *) echo "unknown arg: $a (only --plan)" >&2; exit 2 ;;
    esac
  done
  L=$R3_LOGS/$R3_NAME
  if [ $R3_PLAN = 1 ]; then
    echo "PLAN $R3_NAME (nothing runs): logs would go to $L"
    return 0
  fi
  [ -f /workspace/box-env.sh ] || r3_die "/workspace/box-env.sh missing (not the rental box?)"
  rm -rf "$L"; mkdir -p "$L"
  exec > >(tee -a "$L/run.log") 2>&1
  trap 'r3_on_exit $?' EXIT
  echo "r3 $R3_NAME start $(date -u +%FT%TZ) wt=$R3_WT ($(git -C "$R3_WT" rev-parse --short HEAD 2>/dev/null)) plugin=$R3_PLUGIN_WT ($(git -C "$R3_PLUGIN_WT" rev-parse --short HEAD 2>/dev/null))"
}

r3_on_exit() {
  local rc=$1
  r3_stop || true
  if [ "$rc" != 0 ]; then
    { echo "FAILED rc=$rc at step '$R3_STEP' ($(date -u +%FT%TZ)); see run.log"; echo; cat "$L/summary.txt" 2>/dev/null; } > "$L/summary.tmp"
    mv "$L/summary.tmp" "$L/summary.txt"
    echo "FAILED rc=$rc step=$R3_STEP" > "$L/FAILED"
    echo "r3 $R3_NAME FAILED rc=$rc at step '$R3_STEP'"
  else
    echo "r3 $R3_NAME done $(date -u +%FT%TZ)"
  fi
}

# Environment for every server: box paths, production's main env, Route L, our plugin build.
r3_env() {
  # shellcheck disable=SC1091
  source /workspace/box-env.sh
  export GSQ_VENV=${GSQ_VENV_OVERRIDE:-/workspace/venv-main}
  export GSQ_PROD_ARGV=$R3_WT/env/prod-main-serve-argv.txt
  export GSQ_KV_TIER_ROOT=/workspace/kvtier-r3 GSQ_KV_TIER_MAX_BYTES=${R3_FS_TIER_BYTES:-4000000000}
  export VLLM_GGUF_LCPP=1 VLLM_USE_V2_MODEL_RUNNER=0 GSQ_ALLOW_GPU=1
  export PYTHONPATH=$R3_PLUGIN_WT/plugin:$R3_PLUGIN_WT/tools${R3_EXTRA_PYTHONPATH:+:$R3_EXTRA_PYTHONPATH}
  # shellcheck disable=SC1091
  source "$R3_WT/scripts/env.sh"
  export VLLM_PLUGINS=$GSQ_PLUGINS_ON VLLM_API_KEY=$GSQ_API_KEY
  PY=$GSQ_VENV/bin/python
  LOAD=("$PY" "$R3_S/r3load.py")
}

r3_preflight() {
  r3_step preflight
  command -v nvidia-smi >/dev/null || r3_die "no nvidia-smi"
  local q; q=$(nvidia-smi --query-gpu=name,compute_cap,memory.total,power.limit --format=csv,noheader,nounits | head -1)
  echo "GPU: $q"
  case $q in *3090*) ;; *) r3_die "not an RTX 3090: $q" ;; esac
  local pl; pl=$(echo "$q" | awk -F', ' '{print int($4)}')
  [ "$pl" -ge 300 ] || r3_die "power limit $pl W < 300 W"
  local used; used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ "$used" -lt 2000 ] || r3_die "GPU already has $used MiB in use (another job?)"
  pgrep -f "VLLM::EngineCore|bin/vllm serve" >/dev/null && r3_die "a vLLM server is already running"
  [ "$($PY -c 'import vllm; print(vllm.__version__)' 2>/dev/null | tail -1)" = "$R3_EXPECT_VLLM" ] \
    || r3_die "$GSQ_VENV is not vLLM $R3_EXPECT_VLLM"
  local so=$R3_PLUGIN_WT/plugin/vllm_gguf_plugin/_C_gguf.abi3.so
  [ -f "$so" ] || r3_die "no plugin build at $so"
  grep -qa lcpp_ "$so" || r3_die "$so has no lcpp_* ops (Route L not built)"
  local got; got=$($PY -c 'import vllm_gguf_plugin as p; print(p.__file__)' 2>/dev/null | tail -1)
  [[ $got == "$R3_PLUGIN_WT"/plugin/* ]] || r3_die "plugin resolves to $got, not $R3_PLUGIN_WT/plugin"
  [ -f "$GSQ_GGUF" ] || r3_die "GGUF missing: $GSQ_GGUF"
  local free_gb; free_gb=$(df -BG --output=avail /workspace | tail -1 | tr -dc 0-9)
  [ "$free_gb" -ge "${R3_MIN_DISK_GB:-6}" ] || r3_die "only ${free_gb} GB free on /workspace (fs tier cap $GSQ_KV_TIER_MAX_BYTES + traces)"
  local shm; shm=$(df -B1 --output=avail /dev/shm | tail -1 | tr -dc 0-9)
  [ "$shm" -ge $((GSQ_CPU_TIER_BYTES + 1073741824)) ] || r3_die "/dev/shm has $shm bytes free, CPU tier needs $GSQ_CPU_TIER_BYTES"
  { echo "date $(date -u +%FT%TZ)"; echo "gpu $q"
    nvidia-smi --query-gpu=driver_version,clocks.max.sm,clocks.max.mem,pcie.link.gen.max,pcie.link.width.max --format=csv,noheader
    echo "cpu $(lscpu | sed -n 's/^Model name: *//p') x $(nproc)"; free -g | sed -n 2p
    echo "vllm $R3_EXPECT_VLLM venv $GSQ_VENV"; echo "plugin $got"
    echo "r3 wt $(git -C "$R3_WT" rev-parse HEAD 2>/dev/null)"; echo "plugin wt $(git -C "$R3_PLUGIN_WT" rev-parse HEAD 2>/dev/null)"
  } > "$L/box.txt"
  cat "$L/box.txt"
}

# --- argv mutations (applied to GSQ_ARGV after the standard rewrite) -------------------------
# R3_MUT entries: "drop|--flag" (flag + its value), "set|--flag|value" (replace value or append),
# "swap|--old|--new" (valueless flag), "add|--flag|value", "flag|--flag" (append a valueless flag).
r3_apply_mut() {
  local m op f v i out
  for m in "${R3_MUT[@]}"; do
    IFS='|' read -r op f v <<< "$m"
    out=(); i=0
    case $op in
      drop)
        local hit=0
        while [ $i -lt ${#GSQ_ARGV[@]} ]; do
          if [ "${GSQ_ARGV[$i]}" = "$f" ]; then i=$((i + 2)); hit=1; continue; fi
          out+=("${GSQ_ARGV[$i]}"); i=$((i + 1))
        done
        [ $hit = 1 ] || r3_die "mutation $m: $f not in argv" ;;
      set)
        local hit=0
        while [ $i -lt ${#GSQ_ARGV[@]} ]; do
          if [ "${GSQ_ARGV[$i]}" = "$f" ]; then out+=("$f" "$v"); i=$((i + 2)); hit=1; continue; fi
          out+=("${GSQ_ARGV[$i]}"); i=$((i + 1))
        done
        [ $hit = 1 ] || out+=("$f" "$v") ;;
      swap)
        local hit=0
        for a in "${GSQ_ARGV[@]}"; do
          if [ "$a" = "$f" ]; then out+=("$v"); hit=1; else out+=("$a"); fi
        done
        [ $hit = 1 ] || r3_die "mutation $m: $f not in argv" ;;
      add) out=("${GSQ_ARGV[@]}" "$f" "$v") ;;
      flag) out=("${GSQ_ARGV[@]}" "$f") ;;
      *) r3_die "bad mutation $m" ;;
    esac
    GSQ_ARGV=("${out[@]}")
  done
}

r3_build_argv() {
  gsq_load_prod_argv
  if [ "${R3_MODEL_KIND:-gsq}" = baseline ]; then  # production's previous W4A16 (-fast), stock vLLM path
    gsq_rewrite_argv "$GSQ_BASELINE_MODEL"
    export VLLM_PLUGINS=$GSQ_PLUGINS_OFF
  else
    gsq_rewrite_argv "$GSQ_GGUF" --hf-config-path "$GSQ_HF_CONFIG" --tokenizer "$GSQ_HF_CONFIG"
    export VLLM_PLUGINS=$GSQ_PLUGINS_ON
  fi
  r3_apply_mut
}

# r3_serve TAG: start the server into $L/TAG (R=$L/TAG), wait for /health, record its facts.
r3_serve() {
  local tag=$1
  R=$L/$tag; mkdir -p "$R"
  r3_step "serve $tag"
  gsq_check_port; gsq_assert_hf_config; gsq_assert_tier_root
  r3_build_argv
  printf '%s\n' "${GSQ_ARGV[@]}" > "$R/argv.txt"
  diff "$GSQ_PROD_ARGV" "$R/argv.txt" > "$R/argv-diff.txt"
  echo "argv vs production ('<' production, '>' this server):"; cat "$R/argv-diff.txt"
  mkdir -p "$GSQ_KV_TIER_ROOT"
  setsid "${R3_SERVE_PREFIX[@]}" "${GSQ_ARGV[@]}" > "$R/server.log" 2>&1 &
  R3_SP=$!
  echo "$R3_SP" > "$R/server.pid"
  local t0=$SECONDS
  gsq_wait_health 2400 "$R3_SP" || { tail -40 "$R/server.log"; r3_die "server $tag did not come up"; }
  echo "server $tag healthy after $((SECONDS - t0)) s"
  ENGINE_PID=$(pgrep -f "VLLM::EngineCore" | head -1)
  API_PID=$R3_SP
  echo "$ENGINE_PID" > "$R/engine.pid"
  grep -E "GPU KV cache size|Maximum concurrency|draft head|cudagraph_mode|CUDAGraphMode|Dynamic speculative|async.sched|Using .* backend|OffloadingConnector|Loading weights took|Model loading took" \
    "$R/server.log" | cut -c1-260 | sort -u > "$R/server-facts.txt"
  cat "$R/server-facts.txt"
  grep -q "61440-token draft head" "$R/server.log" || echo "r3: WARNING: no '61440-token draft head' line in the server log"
}

r3_stop() {
  [ -n "$R3_SP" ] || { pgrep -f "VLLM::EngineCore|bin/vllm serve" >/dev/null || return 0; }
  if [ -n "$R3_SP" ] && kill -0 "$R3_SP" 2>/dev/null; then
    kill -INT -- "-$R3_SP" 2>/dev/null
    for _ in $(seq 60); do kill -0 "$R3_SP" 2>/dev/null || break; sleep 2; done
    kill -9 -- "-$R3_SP" 2>/dev/null
  fi
  pkill -9 -f "VLLM::EngineCore" 2>/dev/null; pkill -9 -f "bin/vllm serve" 2>/dev/null
  sleep 3
  R3_SP=
  box_clean_shm 2>/dev/null || true
  rm -rf /workspace/kvtier-r3
}

# r3_summary LINE...: append to summary.txt
r3_summary() { printf '%s\n' "$@" >> "$L/summary.txt"; }

# Production's reference numbers (prod-profile-20261001.md section 1 + 6), for the summaries.
R3_PROD_REF="production 2026-10-01 (vLLM main, same argv, 9600X host): clean pooled ms/step n=1 58.8, n=2 63.6 (ctx 204k, idle 13.2), n=4 71.4, n=8 72.8 (ctx 195k, idle 15.4), n=9 80.5 (k=2); idle flat 12-16 ms/step at n=1-9; box 0.27.1 k=3 ladder (short ctx) 27.9/31.6/35.7/44.2 ms/step c=1/2/4/8 with ~4.7 ms host idle at c=1"

# r3_variant NAME FUNC: run FUNC in a subshell (its r3_die ends only that variant), then stop the
# server. Failures are collected in R3_FAILED and reported by r3_finish.
R3_FAILED=()
r3_variant() {
  local name=$1 fn=$2 rc
  ( trap - EXIT; "$fn" ); rc=$?
  r3_stop
  if [ $rc != 0 ]; then
    R3_FAILED+=("$name")
    r3_summary "variant $name: FAILED rc=$rc (see run.log and $L/$name/server.log)"
  fi
  return 0
}

# r3_finish: exit non-zero (loudly) if any variant failed.
r3_finish() {
  if [ ${#R3_FAILED[@]} -gt 0 ]; then
    R3_STEP="variants failed: ${R3_FAILED[*]}"
    exit 3
  fi
}
