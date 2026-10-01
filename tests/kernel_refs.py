# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU reference models and inputs for the quantized matmul kernel tests.

The CUDA kernels quantize activations to q8_1 before the dot product
(csrc/gguf/gguf_kernel.cu, quantize_q8_1): per 32-value block, float
d = amax / 127, q = round(x / d) (half away from zero), and they store half(d)
and half(sum x). References (float64 on the CPU); a kernel result must match at
least one of them tightly and full precision loosely:

  q81     y = dequant(q8_1(x)) @ W.T                           MMVQ, most MMQ
  xsum    as q81, but the K-quant min term uses half(sum x)     MMQ Q4_K / Q5_K
  wround  y = x @ round_to_act_dtype(W).T                       dequant + matmul
  d2s6    llama.cpp b11211 MMQ Q2_K (the lcpp ops): q8 scale per 64 values,
          min term from half(sum x) per 16 for the first 96 of every 128
          values and from the quantized sum for the last 32 (quantize.cu
          MMQ_Q8_1_DS_LAYOUT_D2S6, vecdotq.cuh vec_dot_q2_K_q8_1_impl_mmq)
  full    y = x @ W.T

Not modelled, all far below the tolerances: b11211's MMQ quantizes with
x * (127 / amax) and keeps d in float for most types, and its tile loaders
round K-quant sub-block scales to half (mmq-load-tiles.cuh).

With --use_fast_math a rare q can be one off, which the tolerances absorb. An x
on a rounding tie (x / d = j + 0.5) may round either way on the GPU, so the
tests feed tie-free x (make_x).
"""

import numpy as np
import torch

QK8_1 = 32
# K-quant blocks: (first, end) byte of the quant bits and the block size.
# Zeroing the quant bits makes gguf-py dequantize to -(dmin * m) per sub-block,
# which isolates the min term.
KQUANT_MIN_SPLIT = {
    "Q4_K": (16, 144, 144),
    "Q5_K": (16, 176, 176),
    "Q2_K": (16, 80, 84),
}

# Relative errors (max over rows of ||y - ref|| / ||ref||), calibrated on an
# RTX 3090: worst reference-model error 2.5e-3 (bf16) / 1.24e-3 (fp16), worst
# error vs full precision 1.5e-2, except Q4_K through MMQ (7e-2 direct, 9e-2
# through the routing function) whose min term uses half(sum x) as ggml's MMQ
# does: it matches the xsum model to 2.5e-3.
TIGHT = {torch.bfloat16: 5e-3, torch.float16: 2.5e-3}
LOOSE = 3e-2
LOOSE_XSUM = 1.5e-1


def dequant(raw: np.ndarray, type_name: str) -> torch.Tensor:
    """gguf-py's dequantization of [rows, row_bytes] blocks, float64."""
    import gguf

    qt = gguf.GGMLQuantizationType[type_name]
    w = gguf.quants.dequantize(np.asarray(raw), qt)
    return torch.from_numpy(np.ascontiguousarray(w, dtype=np.float64))


def q8_1(x: torch.Tensor, qk: int = QK8_1) -> tuple[torch.Tensor, torch.Tensor]:
    """-> (x' = q * float(half(d)), x_avg = float(half(sum x)) / qk broadcast
    per block). qk: values per scale."""
    xf = x.to(torch.float32)
    n, k = xf.shape
    b = xf.view(n, k // qk, qk)
    d = b.abs().amax(-1, keepdim=True) / 127.0
    safe = torch.where(d == 0, torch.ones_like(d), d)
    t = b / safe
    q = torch.where(
        d == 0, torch.zeros_like(t), torch.sign(t) * torch.floor(t.abs() + 0.5)
    )
    dh = d.to(torch.float16).to(torch.float64)
    xq = (q.to(torch.float64) * dh).view(n, k)
    s = b.to(torch.float64).sum(-1, keepdim=True).to(torch.float16).to(torch.float64)
    xavg = (s / qk).expand(-1, -1, qk).reshape(n, k)
    return xq, xavg


def _min_part(raw: np.ndarray, type_name: str) -> torch.Tensor:
    """dmin * m per sub-block (W = scale part - this), by zeroing the quant bits."""
    lo, hi, bsz = KQUANT_MIN_SPLIT[type_name]
    z = np.array(raw, copy=True)
    z.reshape(z.shape[0], -1, bsz)[:, :, lo:hi] = 0
    return -dequant(z, type_name)


def _d2s6_sums(x: torch.Tensor, xq: torch.Tensor) -> torch.Tensor:
    """Per-value stand-in for D2S6's partial sums: half(sum x) / 16 per 16
    values for positions 0-95 of every 128, the quantized mean for 96-127."""
    n, k = x.shape
    s = x.to(torch.float32).view(n, k // 16, 16).to(torch.float64).sum(-1, keepdim=True)
    s = (s.to(torch.float16).to(torch.float64) / 16).expand(-1, -1, 16)
    s = s.reshape(n, k // 128, 128)
    q = xq.view(n, k // 128, 128).clone()
    q[:, :, :96] = s[:, :, :96]
    qm = q[:, :, 96:].reshape(n, k // 128, 2, 16).mean(-1, keepdim=True)
    q[:, :, 96:] = qm.expand(-1, -1, -1, 16).reshape(n, k // 128, 32)
    return q.view(n, k)


def refs(
    raw: np.ndarray, type_name: str, x: torch.Tensor, mmq: bool, lcpp: bool = False
) -> dict:
    W = dequant(raw, type_name)
    x64 = x.to(torch.float64)
    xq, xavg = q8_1(x)
    out = {
        "full": x64 @ W.T,
        "q81": xq @ W.T,
        "wround": x64 @ W.to(x.dtype).to(torch.float64).T,
    }
    if mmq and type_name in ("Q4_K", "Q5_K"):
        Wmin = _min_part(raw, type_name)  # constant within each 32-value block
        out["xsum"] = xq @ (W + Wmin).T - xavg @ Wmin.T
    if mmq and lcpp and type_name == "Q2_K":  # mmq.cuh: D2S6 layout
        Wmin = _min_part(raw, type_name)  # constant within each 16-value block
        xq64, _ = q8_1(x, 64)
        out["d2s6"] = xq64 @ (W + Wmin).T - _d2s6_sums(x, xq64) @ Wmin.T
    return out


def rel_err(y: torch.Tensor, ref: torch.Tensor) -> float:
    """max over rows of ||y - ref|| / ||ref||."""
    y = y.to(torch.float64).cpu()
    num = (y - ref).norm(dim=-1)
    den = ref.norm(dim=-1).clamp_min(1e-30)
    return float((num / den).max())


def check(
    y: torch.Tensor,
    raw: np.ndarray,
    type_name: str,
    x: torch.Tensor,
    mmq: bool,
    lcpp: bool = False,
):
    """y within TIGHT of some reference model and LOOSE of full precision."""
    r = refs(raw, type_name, x.cpu(), mmq, lcpp)
    errs = {k: rel_err(y, v) for k, v in r.items()}
    tight = TIGHT[x.dtype]
    best = min(v for k, v in errs.items() if k != "full")
    assert best <= tight, f"no reference model within {tight}: {errs}"
    loose = LOOSE_XSUM if "xsum" in errs or "d2s6" in errs else LOOSE
    assert errs["full"] <= loose, f"too far from full precision: {errs}"


def make_x(n: int, k: int, dtype: torch.dtype, seed: int = 0) -> torch.Tensor:
    """Hidden-state-like activations (N(0, 1) with a few 20x outlier channels),
    on the CPU, with no q8_1 rounding ties (x / d = j + 0.5 for the block's
    d = amax / 127, per 32 and per 64 values): 16-bit x hits them often and the
    GPU's fast-math division may round them either way. One ulp toward zero
    moves such an x / d by ~0.25; a nudged amax can create new ties, hence the
    loop (it converges in 2-4 passes)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, k, generator=g)
    x[:, torch.randperm(k, generator=g)[: max(1, k // 256)]] *= 20
    x = x.to(dtype)
    if x.dtype == torch.float32:
        return x
    for _ in range(8):
        tie = torch.zeros(n, k, dtype=torch.bool)
        for qk in (32, 64):
            b = x.float().abs().view(n, k // qk, qk)
            t = b * (127 / b.amax(-1, keepdim=True))
            tie |= ((t - t.floor() - 0.5).abs() < 1e-4).view(n, k)
        if not tie.any():
            return x
        x = torch.where(tie, (x.view(torch.int16) - 1).view(x.dtype), x)
    raise AssertionError("could not remove q8_1 rounding ties from x")


def sample_weight(type_name: str, rows: int, blocks: int, seed: int = 0) -> np.ndarray:
    """[rows, blocks * block bytes] uint8: blocks of the type's sample GGUF
    (tests/utils.py, hidden size 1024), a row being any `blocks` of them. Blocks
    are independent, so any row count and K (a multiple of the block size) can
    be built from real quantizer output."""
    import gguf

    from .utils import get_gguf_sample_tensors

    qt = gguf.GGMLQuantizationType[type_name]
    bsz = gguf.GGML_QUANT_SIZES[qt][1]
    pool = np.concatenate(
        [t.data.reshape(-1, bsz) for t in get_gguf_sample_tensors(1024, qt)]
    )
    idx = np.random.default_rng(seed).integers(0, len(pool), rows * blocks)
    return np.ascontiguousarray(pool[idx].reshape(rows, blocks * bsz))


def poison_cuda_allocator() -> None:
    """Fill the caching allocator's free blocks with 0xFF (NaN in every float
    dtype, -1 as int8), so uninitialised scratch or output a kernel reads or
    leaves unwritten holds garbage, not stale zeros."""
    keep = [
        torch.full((1 << p,), 0xFF, dtype=torch.uint8, device="cuda")
        for p in range(9, 28)
        for _ in range(2)
    ]
    del keep
