#!/bin/bash
# EXL3 phase 1, job 02 (~5 min): the pruned MTP draft head for the real checkpoint
# (tools/exl3_draft_head.py): mtp_draft_head.safetensors (mtp.draft_lm_head.weight, bf16
# [40960, 5120]) + mtp_draft_vocab_ids.pt + the index entry, from production's 40,960 ids. Then a
# check that the written rows reproduce the EXL3 lm_head: x @ rows^T vs exl3_linear(x, lm_head)[ids].
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
job_log 02-draft-head
require_idle_gpu
O=$R/02-draft-head; rm -rf "$O"; mkdir -p "$O"
[ "$(sha256sum "$DRAFT_IDS" | cut -d' ' -f1)" = "$DRAFT_IDS_SHA256" ] || die "$DRAFT_IDS is not production's id list"
"$GSQ_VENV/bin/python" tools/exl3_draft_head.py "$EXL3_MODEL" --ids "$DRAFT_IDS" 2>&1 | tee "$O/draft-head.log"
"$GSQ_VENV/bin/python" - "$EXL3_MODEL" <<'PY' 2>&1 | tee "$O/check.txt"
import json, sys, torch
from safetensors import safe_open
sys.path.insert(0, "tests/gpu")
import exl3_cases as C
from vllm_exl3_plugin import ops
d = sys.argv[1]
idx = json.load(open(f"{d}/model.safetensors.index.json"))["weight_map"]
assert idx["mtp.draft_lm_head.weight"] == "mtp_draft_head.safetensors"
with safe_open(f"{d}/mtp_draft_head.safetensors", framework="pt") as f:
    rows = f.get_tensor("mtp.draft_lm_head.weight")
ids = torch.load(f"{d}/mtp_draft_vocab_ids.pt")
print("draft head", tuple(rows.shape), rows.dtype, "ids", ids.numel(), "sorted", bool((ids[1:] > ids[:-1]).all()))
assert rows.shape == (40960, 5120) and rows.dtype == torch.bfloat16 and ids.numel() == 40960
w = C.load(torch, C.HEAD)
x = torch.randn(8, 5120, generator=torch.Generator().manual_seed(1)).half().cuda()
full = ops.exl3_linear(x, w["trellis"], w["suh"], w["svh"], w["mcg"], w["mul1"], True)[:, ids.cuda()]
draft = x.float() @ rows.cuda().float().t()
s = C.err_stats(torch, draft, full.double())
agree = (draft.argmax(1) == full.argmax(1)).float().mean().item()
print("draft rows vs EXL3 lm_head on the ids:", s, "argmax agreement", agree)
assert s["rel_rms"] < 1e-2 and agree == 1.0, "draft head rows do not reproduce the lm_head"
print("DRAFT_HEAD_OK")
PY
ls -la "$EXL3_MODEL"/mtp_draft_* "$EXL3_MODEL"/model.safetensors.index.json* | tee -a "$O/check.txt"
grep -q DRAFT_HEAD_OK "$O/check.txt" || die "draft head check failed"
keep "$O" 02-draft-head "$O/draft-head.log" "$O/check.txt"
