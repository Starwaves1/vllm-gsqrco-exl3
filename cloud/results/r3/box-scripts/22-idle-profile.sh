#!/bin/bash
# shellcheck disable=SC2011,SC2012  # file names here are ours (no spaces)
# R3-22 where the per-step host time goes, on production's main argv at c=2 x 96k (as 20's c2).
# One server with three additions that do not change the engine's code path:
#   - pyhook/sitecustomize.py on PYTHONPATH: R3_INSTR_DIR = wall/CPU timers around the engine loop,
#     scheduler, model runner, KV connector + tiering manager hooks and CUDA syncs (EngineCore main
#     thread, one line per cycle); R3_PTRACE_ANY = lets py-spy attach (yama scope 1, no CAP_SYS_PTRACE)
#   - VLLM_CUSTOM_SCOPES_FOR_PROFILING=1 (record_function scopes for the trace; ~0.1 ms/step when idle)
#   - --profiler-config torch, 25 iterations per start, CPU+CUDA activities (CUPTI), no stacks
# Sequence: warm, fill the CPU tier past the write-back watermark, c=2 decode; 60 s window with
# timers + /metrics; then (streams still decoding) py-spy dumps, py-spy record 30 s of EngineCore and
# API server, one torch-profiler capture (~25 steps).
# nsys is used instead of nothing only if present (not in this container): CUPTI via torch.
# Output: /workspace/logs/r3/22-idle-profile/{summary.txt, prof/{instr.txt, trace.txt, pyspy-*.txt, ...}}
# GPU time: ~17 min.
#   bash 22-idle-profile.sh [--plan]
source "$(dirname "$0")/lib.sh"
r3_init 22-idle-profile "$@"
PYSPY=${R3_PYSPY:-/workspace/r3-tools/bin/py-spy}
if [ $R3_PLAN = 1 ]; then
  sed -n '2,16p' "$0"; echo "py-spy: $PYSPY; hook: $R3_S/22-hook.sh"; exit 0
fi
r3_env
r3_preflight
[ -x "$PYSPY" ] || r3_die "py-spy missing at $PYSPY (uv venv /workspace/r3-tools && uv pip install --python /workspace/r3-tools/bin/python py-spy)"
command -v nsys >/dev/null && echo "nsys present: $(nsys --version) (not used: torch/CUPTI keeps the scopes)" || echo "nsys: not available in this container; torch profiler with CUPTI"

export R3_EXTRA_PYTHONPATH=$R3_S/pyhook R3_PTRACE_ANY=1 VLLM_CUSTOM_SCOPES_FOR_PROFILING=1
r3_env   # again, for the PYTHONPATH with the hook
export R3_INSTR_DIR=$L/prof/instr
PROF="{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$L/prof/trace\",\"torch_profiler_with_stack\":false,\"torch_profiler_use_gzip\":true,\"torch_profiler_dump_cuda_time_total\":false,\"ignore_frontend\":true,\"delay_iterations\":0,\"max_iterations\":25}"
R3_MUT=("add|--profiler-config|$PROF")
r3_serve prof
"${LOAD[@]}" warm || r3_die warm
r3_step fill; "${LOAD[@]}" fill --n 2 --tokens 90000 --conc 2 || r3_die fill
export R ENGINE_PID API_PID PYSPY
r3_step "c2 window + hook"
"${LOAD[@]}" steady --conc 2 --tokens 96000 --max-tokens 16000 --window 60 --k 5 --tag c2 --out "$R" \
  --hook "bash $R3_S/22-hook.sh" || r3_die "c2 window/hook"
r3_stop

r3_step analyze
T0=$("$PY" -c "import json;print(json.load(open('$R/c2-steady.json'))['window_t0'])")
T1=$("$PY" -c "import json;print(json.load(open('$R/c2-steady.json'))['window_t1'])")
IN=$(ls "$R3_INSTR_DIR"/instr-*.jsonl 2>/dev/null | xargs -r wc -l | sort -n | grep -v total | tail -1 | awk '{print $2}')
[ -n "$IN" ] || r3_die "no instrumentation output in $R3_INSTR_DIR"
"$PY" "$R3_S/r3analyze.py" instr "$IN" --t0 "$T0" --t1 "$T1" --json "$R/instr.json" > "$R/instr.txt" || r3_die "instr analysis"
TR=$(ls -t "$L"/prof/trace/*.pt.trace.json* 2>/dev/null | head -1)
[ -n "$TR" ] || r3_die "no torch trace in $L/prof/trace"
"$PY" "$R3_S/r3analyze.py" trace "$TR" --json "$R/trace.json" > "$R/trace.txt" || r3_die "trace analysis"
for f in "$R"/pyspy-*.raw; do
  [ -s "$f" ] || continue
  "$PY" "$R3_S/r3analyze.py" pyspy "$f" --json "${f%.raw}.json" > "${f%.raw}.txt" 2>&1 || echo "pyspy analysis failed for $f"
done
r3_summary "R3-22 idle profile (box, $(date -u +%F)), c=2 x 96k, k=5, production's main argv + timers/scopes/profiler" \
  "$(cat "$R/lines.txt")" "" "--- engine-cycle timers (unprofiled window) ---" "$(cat "$R/instr.txt")" "" \
  "--- torch profiler (CUPTI), ~25 steps ---" "$(cat "$R/trace.txt")" ""
for f in "$R"/pyspy-*.txt; do [ -f "$f" ] && r3_summary "--- $(basename "$f") ---" "$(head -45 "$f")" ""; done
grep -h "py-spy" "$R/hook.log" 2>/dev/null | grep -i -E "fail|error|unavailable" | sed 's/^/DEGRADED: /' >> "$L/summary.txt"
r3_summary "reference: $R3_PROD_REF"
cat "$L/summary.txt"
