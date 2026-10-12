#!/bin/bash
# R3-58 does a neighbour request corrupt the others? 28's echo+prompt_logprobs side client at c=4
# (<= 8 running) corrupted 26/30 main answers. Production's original main argv; main load = 30
# non-streamed chat answers at c=4, T=0, reasoning on, 600 tokens, beside one client sending short
# completions back to back, by kind: none | plain (4 tokens) | plain1 (1 token: prefill only) |
# lp (logprobs) | plp (prompt_logprobs) | echo (echo + logprobs). Servers: GSQ (this checkout's
# plugin), GSQ + both hotfix vLLM patches (prompt logprobs chunked + hidden states copied before the
# drafter), stock W4A16. Output: /workspace/logs/r3/58-side-client/. GPU time ~45 min.
#   bash 58-side-client.sh [--plan]
source "$(dirname "$0")/lib.sh"
r3_init 58-side-client "$@"
if [ "$R3_PLAN" = 1 ]; then sed -n '2,10p' "$0"; exit 0; fi
base=$(git -C "$R3_PLUGIN_WT" rev-parse HEAD)
git -C "$R3_WT" diff --quiet "$base" HEAD -- plugin/vllm_gguf_plugin/csrc plugin/setup.py || r3_die "csrc differs"
cp "$R3_PLUGIN_WT/plugin/vllm_gguf_plugin/_C_gguf.abi3.so" "$R3_WT/plugin/vllm_gguf_plugin/"
export R3_PLUGIN_WT=$R3_WT
r3_env
r3_preflight
P=$R3_S/../patches
bash "$R3_S/overlay-venv.sh" /workspace/venv-r3-plp2 "$P/prompt-logprobs-chunked.patch" "$P/prompt-logprobs-after-drafter.patch" \
  > "$L/overlay.log" 2>&1 || r3_die "overlay venv"
SIDES=none,plain,plain1,lp,plp,echo
run() {  # tag
  r3_serve "$1"
  "${LOAD[@]}" warm || r3_die warm
  "$PY" "$R3_S/r3tok.py" run --nonstream --max-tokens 600 --conc 4 --temps "" --side "$SIDES" --side-conc 4 \
    --out "$L/runs" --tag "$1" || r3_die "r3tok $1"
}
v_gsq() { run gsq; }
v_gsqpatched() { export GSQ_VENV_OVERRIDE=/workspace/venv-r3-plp2; r3_env; run gsq-patched; }
v_w4a16() { export R3_MODEL_KIND=baseline; run w4a16; }
report() {
  "$PY" "$R3_S/r3tok.py" report "$L/runs" --ref none --details 30 > "$L/report.txt" 2>&1
  { cat "$L/report.txt"; for f in "${R3_FAILED[@]}"; do echo "variant $f: FAILED"; done; } > "$L/summary.txt"
}
for v in ${R3_58_ONLY:-gsq w4a16 gsqpatched}; do
  r3_step "$v"; r3_variant "$v" "v_$v"; report
done
cat "$L/summary.txt"
r3_finish
