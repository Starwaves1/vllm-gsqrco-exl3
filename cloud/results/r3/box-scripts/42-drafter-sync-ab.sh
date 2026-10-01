#!/bin/bash
# R3-42 A/B of patches/drafter-host-seq-lens.patch (vLLM side; a proposal, production untouched):
# the MTP drafter follows seq_lens on the host across draft passes and FlashInfer's build uses the
# exact host lengths, so a decode step no longer blocks on seq_lens.cpu() once per draft pass
# (k=5: ~6 syncs -> 1 early event wait). Same job, same box, two servers:
#   stock    /workspace/venv-main
#   patched  /workspace/venv-r3-drafter (overlay-venv.sh: hardlinked copy + the patch)
# Each: warm, CPU tier fill, c2 x 96k window (k=5), c8 x 20k window (k=3), greedy probe (4 x 2k
# prompts together, 256 tokens, logprobs) for numerics. Pass: probe texts identical, acceptance
# within noise, and the ms/step change.
# Output: /workspace/logs/r3/42-drafter-sync-ab/{summary.txt, stock/, patched/}
# GPU time: ~40 min.
#   bash 42-drafter-sync-ab.sh [--plan]
source "$(dirname "$0")/lib.sh"
r3_init 42-drafter-sync-ab "$@"
PATCH=$R3_S/../patches/drafter-host-seq-lens.patch
VR=/workspace/venv-r3-drafter
if [ $R3_PLAN = 1 ]; then sed -n '2,14p' "$0"; echo "patch: $PATCH"; exit 0; fi
r3_step overlay
bash "$R3_S/overlay-venv.sh" "$VR" "$PATCH" || r3_die "overlay venv"
r3_env
r3_preflight
run() {  # tag venv
  export GSQ_VENV_OVERRIDE=$2
  r3_env
  r3_serve "$1"
  "${LOAD[@]}" warm || r3_die warm
  "${LOAD[@]}" probe --n 4 --tokens 2000 --max-tokens 256 --tag probe --out "$R" || r3_die probe
  "${LOAD[@]}" fill --n 2 --tokens 90000 --conc 2 || r3_die fill
  "${LOAD[@]}" steady --conc 2 --tokens 96000 --max-tokens 12000 --window 90 --k 5 --tag c2 --out "$R" || r3_die c2
  "${LOAD[@]}" steady --conc 8 --tokens 18000 --max-tokens 9000 --window 90 --k 3 --tag c8 --out "$R" || r3_die c8
}
v_stock() { run stock /workspace/venv-main; }
v_patched() { run patched "$VR"; }
r3_summary "R3-42 drafter host seq_lens A/B (box, $(date -u +%F)); patch $(sha256sum "$PATCH" | cut -c1-12)"
for v in stock patched; do
  r3_step "$v"; r3_variant "$v" "v_$v"
  [ -f "$L/$v/lines.txt" ] && r3_summary "$(sed "s/^/$v /" "$L/$v/lines.txt")"
done
"$PY" - "$L" >> "$L/summary.txt" <<'PYEOF' || echo "probe compare failed" >> "$L/summary.txt"
import json, sys, os
L = sys.argv[1]
a = json.load(open(os.path.join(L, "stock", "probe-probe.json")))
b = json.load(open(os.path.join(L, "patched", "probe-probe.json")))
for i, (x, y) in enumerate(zip(a, b)):
    tx, ty = x["tokens"] or [], y["tokens"] or []
    d = next((j for j, (p, q) in enumerate(zip(tx, ty)) if p != q), None)
    print(f"probe {i}: {'identical' if x['text'] == y['text'] else f'DIFFERS at token {d}'} ({len(tx)} tokens)")
PYEOF
grep -q DIFFERS "$L/summary.txt" && r3_summary "numerics: greedy outputs differ -> the patch changes attention inputs: NOT acceptable until explained"
cat "$L/summary.txt"
r3_finish
