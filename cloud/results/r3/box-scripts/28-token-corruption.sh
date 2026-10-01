#!/bin/bash
# R3-28 wrong tokens at >= 9 running (prod-garbled-tokens-20261001.md: GSQ only, 26 % of answers at
# 9+, 0.1 % at <= 8, W4A16 clean). Production's ORIGINAL main argv (max-num-seqs 16, schedule
# [[1,4,5],[5,8,3],[9,16,2]]), plugin = this checkout (32ae6ec kernels + VLLM_GGUF_MMA_K switch).
#   1. repro  : prod argv; non-streamed chat, reasoning on, max 600 tokens, T=1.0 (top_k 20,
#               top_p 0.95) and T=0, at c=8 (control), 9, 12, 16 (4 x c requests per batch)
#   2. cells at c=9 and 12, T=1.0, plus a pass beside an echo/prompt_logprobs client:
#        {graphs | --enforce-eager} x {our 9..32-row kernels | VLLM_GGUF_MMA_K=0 (vendored MMQ for the
#        draft head and every Q4_K/IQ4_XS/IQ2_S product at 9..32 rows)} x {k=2 tier | capped at 8
#        (max-num-seqs 8, [[1,4,5],[5,8,3]]: production's mitigation)}, plus k=3 at 9..16.
# Corruption flags per answer (r3tok.py): EOS inside reasoning, characters outside Latin/Greek/
# punctuation/math/emoji, a fragment repeated 4+ times, text != decode(ids), HTTP/UTF-8 errors.
# R3_28_ONLY="..." picks cells. Output: /workspace/logs/r3/28-token-corruption/ (summary = report so far).
# GPU time: ~2 h for everything; the repro alone ~20 min.
#   bash 28-token-corruption.sh [--plan]
source "$(dirname "$0")/lib.sh"
r3_init 28-token-corruption "$@"
if [ "$R3_PLAN" = 1 ]; then sed -n '2,16p' "$0"; exit 0; fi
base=$(git -C "$R3_PLUGIN_WT" rev-parse HEAD)
git -C "$R3_WT" diff --quiet "$base" HEAD -- plugin/vllm_gguf_plugin/csrc plugin/setup.py || r3_die "csrc differs from $R3_PLUGIN_WT"
cp "$R3_PLUGIN_WT/plugin/vllm_gguf_plugin/_C_gguf.abi3.so" "$R3_WT/plugin/vllm_gguf_plugin/"
export R3_PLUGIN_WT=$R3_WT
r3_env
r3_preflight
grep -qF '[9,16,2]' "$GSQ_PROD_ARGV" || r3_die "argv file is not the original 9-16 tier"
SPEC='{"method":"mtp","num_speculative_tokens":5,"draft_sample_method":"probabilistic","num_speculative_tokens_per_batch_size":'
CAP8=("set|--max-num-seqs|8" "set|--speculative-config|${SPEC}[[1,4,5],[5,8,3]]}")
K3AT9=("set|--speculative-config|${SPEC}[[1,4,5],[5,16,3]]}")
tok() { "$PY" "$R3_S/r3tok.py" run --nonstream --max-tokens 600 --per-conc 4 --out "$L/runs" "$@" || r3_die "r3tok $*"; }
cell() {  # tag graphs|eager ours|mmq k2|cap8|k3at9
  local tag=$1
  R3_MUT=()
  [ "$2" = eager ] && R3_MUT+=("flag|--enforce-eager")
  [ "$3" = mmq ] && export VLLM_GGUF_MMA_K=0
  [ "$4" = cap8 ] && R3_MUT+=("${CAP8[@]}")
  [ "$4" = k3at9 ] && R3_MUT+=("${K3AT9[@]}")
  r3_serve "$tag"
  "${LOAD[@]}" warm || r3_die warm
  tok --tag "$tag" --conc 9,12 --temps 1.0 --plp-pass
}
v_repro() { r3_serve prod; "${LOAD[@]}" warm || r3_die warm; tok --tag prod --conc 8,9,12,16 --temps 1.0,0; }
v_g_mmq_k2()    { cell g-mmq-k2 graphs mmq k2; }
v_e_ours_k2()   { cell e-ours-k2 eager ours k2; }
v_g_ours_k3at9(){ cell g-ours-k3at9 graphs ours k3at9; }
v_g_ours_cap8() { cell g-ours-cap8 graphs ours cap8; }
v_e_mmq_k2()    { cell e-mmq-k2 eager mmq k2; }
v_g_mmq_cap8()  { cell g-mmq-cap8 graphs mmq cap8; }
v_e_ours_cap8() { cell e-ours-cap8 eager ours cap8; }
v_e_mmq_cap8()  { cell e-mmq-cap8 eager mmq cap8; }
report() {
  "$PY" "$R3_S/r3tok.py" report "$L/runs" --ref none --details 40 > "$L/report.txt" 2>&1
  { cat "$L/report.txt"; for f in "${R3_FAILED[@]}"; do echo "variant $f: FAILED (see run.log)"; done; } > "$L/summary.txt"
}
for v in ${R3_28_ONLY:-repro g_mmq_k2 e_ours_k2 g_ours_k3at9 g_ours_cap8 e_mmq_k2 g_mmq_cap8 e_ours_cap8 e_mmq_cap8}; do
  r3_step "$v"; r3_variant "$v" "v_$v"
  report
done
cat "$L/summary.txt"
r3_finish
