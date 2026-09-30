"""Write the pruned MTP draft head for an EXL3 checkpoint, once, as bf16.

Production's vLLM overlay (qwen3_5_mtp.py, "syv patch") builds `mtp.draft_lm_head`, a
ParallelLMHead over the token ids in `mtp_draft_vocab_ids.pt` (40,960 in production), whenever
that file sits in the model directory, and loads its weight from the checkpoint as
`mtp.draft_lm_head.weight`. For W4A16 and GGUF the rows are a lossless slice of the quantized
lm_head. An EXL3 lm_head cannot be sliced that way: its output Hadamard mixes each block of
128 output rows (svh, had_r_128), so single rows do not exist in quantized form.

Design (the simpler of the two options): dequantize the lm_head once to its original basis
(exl3_dequant with had=True: reconstruct_had_slice, both Hadamards and suh/svh folded in), in
32768-column slices, keep the id rows, and write them as a plain bf16 tensor. At serve time the
draft head is then an ordinary unquantized ParallelLMHead (EXL3Config returns no quant method
for `draft_lm_head`): no runtime dequant, no dependency on the target's lm_head, 0.39 GiB.
The alternative, dequantizing inside vLLM at load, needs the target's lm_head loaded first and
a custom loader path; it buys nothing, since the rows are fixed per checkpoint and id list.
The rows are the fp16 dequant rounded to bf16 (the model's dtype, as vLLM would cast them).
They only change what the drafter proposes, never what is accepted.

Writes into MODEL_DIR (a plain local copy, not an HF cache snapshot: files there are symlinks
into the blob store; this tool replaces the index file rather than writing through a link):
  mtp_draft_head.safetensors   mtp.draft_lm_head.weight, bf16 [len(ids), hidden]
  model.safetensors.index.json + weight_map entry for it (vLLM's default loader only reads
                               files the index lists); the original is kept once as
                               model.safetensors.index.json.orig
  mtp_draft_vocab_ids.pt       the ids, int64, sorted (the overlay's switch; MTP_DRAFT_VOCAB=0
                               turns it off at serve time)

Needs a CUDA device and the built _C_exl3 (GPU phase). Unit test: tests/cpu/test_exl3_draft_head.py
(synthetic checkpoint, CPU dequant stand-in).

usage: python tools/exl3_draft_head.py MODEL_DIR --ids IDS.json|IDS.pt
       (production's 40,960 ids: ~/qwen38-27b-rtx3090/prepare/draft_vocab_ids.json)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRAFT_TENSOR = "mtp.draft_lm_head.weight"
DRAFT_FILE = "mtp_draft_head.safetensors"
IDS_FILE = "mtp_draft_vocab_ids.pt"
INDEX = "model.safetensors.index.json"
SLICE_N = 32768  # exllamav3 MAX_RECONSTRUCT_SLICE_N: 5120 x 32768 fp16 = 320 MiB per slice


def load_ids(path: str) -> torch.Tensor:
    if path.endswith(".json"):
        with open(path) as f:
            ids = torch.tensor(json.load(f), dtype=torch.int64)
    else:
        ids = torch.load(path, map_location="cpu", weights_only=True).to(torch.int64)
    return ids.reshape(-1)


def _check_ids(ids: torch.Tensor, vocab: int) -> torch.Tensor:
    ids = ids.reshape(-1).to(torch.int64)
    if ids.numel() == 0:
        raise ValueError("no draft ids")
    if int(ids.min()) < 0 or int(ids.max()) >= vocab:
        raise ValueError(f"draft ids must lie in [0, {vocab})")
    if torch.unique(ids).numel() != ids.numel():
        raise ValueError("draft ids must be unique")
    return torch.sort(ids).values


def _gpu_dequant(trellis, suh, svh, mcg, mul1, n_start, n_count):
    sys.path.insert(0, os.path.join(ROOT, "plugin-exl3"))
    from vllm_exl3_plugin import _C_exl3  # noqa: F401  (registers torch.ops._C_exl3)

    return torch.ops._C_exl3.exl3_dequant(trellis, suh, svh, mcg, mul1, n_start, n_count, True)


def build_draft_head(model_dir: str, ids: torch.Tensor, dequant=None, device: str = "cuda",
                     slice_n: int = SLICE_N) -> dict:
    """Write the draft head files into model_dir (module docstring). dequant(trellis, suh, svh,
    mcg, mul1, n_start, n_count) -> fp16 [k, n_count] of the original-basis weight W[in, out];
    default: torch.ops._C_exl3.exl3_dequant(..., had=True) on `device`."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    dequant = dequant or _gpu_dequant
    index_path = os.path.join(model_dir, INDEX)
    with open(index_path) as f:
        index = json.load(f)
    wm = index["weight_map"]

    def tensor(name):
        with safe_open(os.path.join(model_dir, wm[name]), framework="pt") as f:
            return f.get_tensor(name)

    mcg, mul1 = "lm_head.mcg" in wm, "lm_head.mul1" in wm
    trellis, suh, svh = (tensor(f"lm_head.{t}").to(device) for t in ("trellis", "suh", "svh"))
    k, n = trellis.shape[0] * 16, trellis.shape[1] * 16
    ids = _check_ids(ids, n)

    rows = torch.empty(ids.numel(), k, dtype=torch.bfloat16)
    for s in range(0, n, slice_n):
        c = min(slice_n, n - s)
        sel = ((ids >= s) & (ids < s + c)).nonzero().reshape(-1)
        if sel.numel() == 0:
            continue
        w = dequant(trellis, suh, svh, mcg, mul1, s, c)  # fp16 [k, c]
        if tuple(w.shape) != (k, c) or w.dtype != torch.half:
            raise ValueError(f"dequant returned {tuple(w.shape)} {w.dtype}, expected fp16 [{k}, {c}]")
        cols = (ids[sel] - s).to(w.device)
        rows[sel] = w.index_select(1, cols).t().to(torch.bfloat16).cpu()

    save_file({DRAFT_TENSOR: rows.contiguous()}, os.path.join(model_dir, DRAFT_FILE),
              metadata={"format": "pt", "source": "tools/exl3_draft_head.py: lm_head dequant rows, bf16"})
    orig = index_path + ".orig"
    if not os.path.exists(orig):
        shutil.copyfile(index_path, orig)  # follows a symlink: copies the content
    wm[DRAFT_TENSOR] = DRAFT_FILE
    tmp = index_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(index, f, indent=2)
    os.replace(tmp, index_path)  # replaces a symlinked index, never writes through it
    torch.save(ids, os.path.join(model_dir, IDS_FILE))
    return {"rows": ids.numel(), "hidden": k, "vocab": n, "bytes": rows.numel() * 2}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model_dir")
    ap.add_argument("--ids", required=True, help="draft token ids: JSON list or torch .pt")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    info = build_draft_head(a.model_dir, load_ids(a.ids), device=a.device)
    print(f"wrote {DRAFT_FILE}: {info['rows']} x {info['hidden']} bf16 "
          f"({info['bytes'] / 2**30:.2f} GiB) of {info['vocab']} rows; {IDS_FILE}; index updated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
