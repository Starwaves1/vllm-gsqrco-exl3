"""CPU reference models for the plugin's quantized matmuls (float64 on the CPU).

The CUDA kernels quantize activations to q8_1 before the dot product
(csrc/gguf/gguf_kernel.cu:32-67): per 32-element block, float d = amax/127,
q = roundf(x/d) (half away from zero), and they store half(d) and half(sum x). With
--use_fast_math a rare q can be off by one, which the tolerances absorb.

Reference models (a kernel result must match at least one tightly):
  q81     y = dequant(q8_1(x)) @ W.T                         MMVQ, and MMQ except below
  xsum    as q81, but the K-quant min term uses half(sum x)   MMQ Q4_K/Q5_K (vecdotq.cuh:353-378)
  wround  y = x @ round_to_act_dtype(W).T                     Triton path / dequant+matmul path
  d2s6    lcpp (b11211) MMQ Q2_K: q8 scale per 64 values, min term from half(sum x) per 16
          for the first 96 of every 128 values and from the q sum for the last 32
          (quantize.cu MMQ_Q8_1_DS_LAYOUT_D2S6, vecdotq.cuh vec_dot_q2_K_q8_1_impl_mmq)
and loosely match the full-precision product  full = x @ W.T.
b11211 MMVQ and its other MMQ layouts (D4: float d per 32; DS4: half d + half sum per 32)
are q81 / xsum up to the rounding of d, which is far below the tolerances.
"""

import numpy as np
import torch

QK8_1 = 32
# K-quant block layouts: bytes of the header (d, dmin, scales) before the quant bits,
# and the block size in bytes (ggml-common.h). Zeroing the quant bits makes gguf-py
# dequantize to -(dmin * m) per sub-block, which isolates the min term.
# (first, end) byte of the quant bits in the block, and the block size.
KQUANT_MIN_SPLIT = {"Q4_K": (16, 144, 144), "Q5_K": (16, 176, 176), "Q2_K": (16, 80, 84)}


def dequant(raw: np.ndarray, type_name: str) -> torch.Tensor:
    import gguf

    qt = gguf.GGMLQuantizationType[type_name]
    return torch.from_numpy(np.ascontiguousarray(gguf.quants.dequantize(np.asarray(raw), qt), dtype=np.float64))


TIE = 1e-4  # |x/d| this close to k + 0.5 may round either way on the GPU


def q8_1(x: torch.Tensor, qk: int = QK8_1, lcpp_mmq: str | None = None):
    """-> (x' = q * float(half(d)), x_avg = float(half(sum x)) / 32 broadcast per block,
    tie = d where x/d is within TIE of a rounding tie, else 0).
    qk: values per scale (64 for the D2S6 layout; x_avg is then per 64, unused).
    lcpp_mmq: b11211's quantize_mmq_q8_1 instead, q = roundf(x * (127/amax)) and
    d = 1/(127/amax), kept in float ("D4") or rounded to half ("DS4", "D2S6").
    Near ties the GPU's q is not predictable: --use_fast_math turns 127/amax (and the
    stock kernels' x/d) into approximate division, and 16-bit x hits exact ties (x/d = 2.5)
    often enough. One flipped q in a 20x outlier channel moves a whole output row by
    ~3e-3 relative, so the references carry that slack (see refs)."""
    xf = x.to(torch.float32)
    n, k = xf.shape
    b = xf.view(n, k // qk, qk)
    amax = b.abs().amax(-1, keepdim=True)
    if lcpp_mmq:
        inv = 127.0 / torch.where(amax == 0, torch.ones_like(amax), amax)
        t = b * inv
        d = torch.where(amax == 0, torch.zeros_like(amax), 1.0 / inv)
    else:
        d = amax / 127.0
        t = b / torch.where(d == 0, torch.ones_like(d), d)
    q = torch.where(d == 0, torch.zeros_like(t), torch.sign(t) * torch.floor(t.abs() + 0.5))
    dh = d.to(torch.float64) if lcpp_mmq == "D4" else d.to(torch.float16).to(torch.float64)
    xq = (q.to(torch.float64) * dh).view(n, k)
    s = b.to(torch.float64).sum(-1, keepdim=True).to(torch.float16).to(torch.float64)
    xavg = (s / qk).expand(-1, -1, qk).reshape(n, k)
    near = ((t.abs() - t.abs().floor() - 0.5).abs() < TIE) & (d != 0)
    tie = torch.where(near, dh.expand_as(t), torch.zeros_like(t, dtype=torch.float64)).view(n, k)
    return xq, xavg, tie


def _min_part(raw: np.ndarray, type_name: str) -> torch.Tensor:
    """dmin * m per sub-block (>= 0; W = W_scale_part - this), by zeroing the quant bits."""
    lo, hi, bsz = KQUANT_MIN_SPLIT[type_name]
    z = np.array(raw, copy=True)
    z.reshape(z.shape[0], -1, bsz)[:, :, lo:hi] = 0
    return -dequant(z, type_name)


def _d2s6_sums(x: torch.Tensor, xq: torch.Tensor) -> torch.Tensor:
    """Per-value stand-in for D2S6's partial sums: half(sum x)/16 per 16 values for
    positions 0-95 of every 128, the quantized mean for positions 96-127."""
    n, k = x.shape
    s = x.to(torch.float32).view(n, k // 16, 16).to(torch.float64).sum(-1, keepdim=True)
    s = (s.to(torch.float16).to(torch.float64) / 16).expand(-1, -1, 16).reshape(n, k // 128, 128)
    q = xq.view(n, k // 128, 128).clone()
    q[:, :, :96] = s[:, :, :96]
    qm = q[:, :, 96:].reshape(n, k // 128, 2, 16).mean(-1, keepdim=True).expand(-1, -1, -1, 16)
    q[:, :, 96:] = qm.reshape(n, k // 128, 32)
    return q.view(n, k)


def refs(raw: np.ndarray, type_name: str, x: torch.Tensor, mmq: bool,
         lcpp: bool = False) -> dict[str, tuple[torch.Tensor, torch.Tensor | None]]:
    """model -> (reference output, per-element slack for near-tie q flips, or None)."""
    W = dequant(raw, type_name)
    x64 = x.to(torch.float64)
    layout = None
    if mmq and lcpp:  # mmq.cuh mmq_get_q8_1_ds_layout
        layout = {"Q2_K": "D2S6", "Q4_K": "DS4", "Q5_K": "DS4"}.get(type_name, "D4")
    xq, xavg, tie = q8_1(x, lcpp_mmq=layout)
    out = {"full": (x64 @ W.T, None), "q81": (xq @ W.T, tie @ W.abs().T),
           "wround": (x64 @ W.to(x.dtype).to(torch.float64).T, None)}
    if mmq and type_name in ("Q4_K", "Q5_K"):
        Wmin = _min_part(raw, type_name)       # dmin * m, constant within each 32-block
        out["xsum"] = (xq @ (W + Wmin).T - xavg @ Wmin.T, tie @ (W + Wmin).abs().T)
    if layout == "D2S6":
        Wmin = _min_part(raw, type_name)       # constant within each 16-block
        xq64, _, tie64 = q8_1(x, 64, layout)
        out["d2s6"] = (xq64 @ (W + Wmin).T - _d2s6_sums(x, xq64) @ Wmin.T,
                       tie64 @ ((W + Wmin).abs() + Wmin.abs()).T)
    return out


def rel_err(y: torch.Tensor, ref: torch.Tensor, slack: torch.Tensor | None = None) -> float:
    """max over token rows of ||y - ref|| / ||ref||, each |y - ref| first reduced by its slack."""
    y = y.to(torch.float64).cpu()
    diff = (y - ref).abs() if slack is None else ((y - ref).abs() - slack).clamp_min(0)
    num = diff.norm(dim=-1)
    den = ref.norm(dim=-1).clamp_min(1e-30)
    return float((num / den).max())
