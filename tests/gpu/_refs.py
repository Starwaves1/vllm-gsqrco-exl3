"""CPU reference models for the plugin's quantized matmuls (float64 on the CPU).

The CUDA kernels quantize activations to q8_1 before the dot product
(csrc/gguf/gguf_kernel.cu:32-67): per 32-element block, float d = amax/127,
q = roundf(x/d) (half away from zero), and they store half(d) and half(sum x). With
--use_fast_math a rare q can be off by one, which the tolerances absorb.

Reference models (a kernel result must match at least one tightly):
  q81     y = dequant(q8_1(x)) @ W.T                         MMVQ, and MMQ except below
  xsum    as q81, but the K-quant min term uses half(sum x)   MMQ Q4_K/Q5_K (vecdotq.cuh:353-378)
  wround  y = x @ round_to_act_dtype(W).T                     Triton path / dequant+matmul path
and loosely match the full-precision product  full = x @ W.T.
"""

import numpy as np
import torch

QK8_1 = 32
# K-quant block layouts: bytes of the header (d, dmin, scales) before the quant bits,
# and the block size in bytes (ggml-common.h). Zeroing the quant bits makes gguf-py
# dequantize to -(dmin * m) per sub-block, which isolates the min term.
KQUANT_MIN_SPLIT = {"Q4_K": (16, 144), "Q5_K": (16, 176)}


def dequant(raw: np.ndarray, type_name: str) -> torch.Tensor:
    import gguf

    qt = gguf.GGMLQuantizationType[type_name]
    return torch.from_numpy(np.ascontiguousarray(gguf.quants.dequantize(np.asarray(raw), qt), dtype=np.float64))


def q8_1(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """-> (x' = q * float(half(d)), x_avg = float(half(sum x)) / 32 broadcast per block)."""
    xf = x.to(torch.float32)
    n, k = xf.shape
    b = xf.view(n, k // QK8_1, QK8_1)
    d = b.abs().amax(-1, keepdim=True) / 127.0
    safe = torch.where(d == 0, torch.ones_like(d), d)
    t = b / safe
    q = torch.where(d == 0, torch.zeros_like(t), torch.sign(t) * torch.floor(t.abs() + 0.5))
    dh = d.to(torch.float16).to(torch.float64)
    xq = (q.to(torch.float64) * dh).view(n, k)
    s = b.to(torch.float64).sum(-1, keepdim=True).to(torch.float16).to(torch.float64)
    xavg = (s / QK8_1).expand(-1, -1, QK8_1).reshape(n, k)
    return xq, xavg


def refs(raw: np.ndarray, type_name: str, x: torch.Tensor, mmq: bool) -> dict[str, torch.Tensor]:
    W = dequant(raw, type_name)
    x64 = x.to(torch.float64)
    xq, xavg = q8_1(x)
    out = {"full": x64 @ W.T, "q81": xq @ W.T, "wround": x64 @ W.to(x.dtype).to(torch.float64).T}
    if mmq and type_name in KQUANT_MIN_SPLIT:
        head, bsz = KQUANT_MIN_SPLIT[type_name]
        z = np.array(raw, copy=True)
        blocks = z.reshape(z.shape[0], -1, bsz)
        blocks[:, :, head:] = 0
        Wmin = -dequant(z, type_name)          # dmin * m, constant within each 32-block
        out["xsum"] = xq @ (W + Wmin).T - xavg @ Wmin.T
    return out


def rel_err(y: torch.Tensor, ref: torch.Tensor) -> float:
    """max over token rows of ||y - ref|| / ||ref||."""
    y = y.to(torch.float64).cpu()
    num = (y - ref).norm(dim=-1)
    den = ref.norm(dim=-1).clamp_min(1e-30)
    return float((num / den).max())
