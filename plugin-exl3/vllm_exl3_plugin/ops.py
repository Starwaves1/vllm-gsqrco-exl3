# SPDX-License-Identifier: Apache-2.0
"""torch.ops._C_exl3 (csrc/exl3_shim.cu) behind one routed custom op.

`exl3_linear(x, trellis, suh, svh, mcg, mul1, out_fp32)` is x @ W for one EXL3 tensor W,
x [n, k] fp16 (the kernels' activation dtype; the linear method casts once per layer), result
fp32 or fp16. It is registered as the vLLM custom op
`torch.ops.vllm._exl3_linear` (with a fake impl), so torch.compile sees one opaque node per
EXL3 tensor and the row-count choice below is made per call, not when the graph is traced.

Routing (`_exl3_op`, pinned by tests/cpu/test_exl3_routing.py), n = activation rows:

| n            | op                   | what                                                    |
|--------------|----------------------|---------------------------------------------------------|
| 1..144       | exl3_gemm            | vendored kernel (GEMV where its heuristic picks it)     |
| 17..144      | MULTI_ROW_OP if set  | multi-row kernel for the tensors it takes; else exl3_gemm |
| 145..1023    | recon_hgemm          | Hadamard x, rotated dequant, hgemm, Hadamard y          |
| >= 1024      | recon_had_hgemm      | original-basis dequant (Hadamards folded in), hgemm     |

The multi-row kernel is trellis-serve's Marlin-EXL3 (csrc/trellis_serve, op exl3_gemm_mr,
_C_exl3_mr), switched by EXL3_MR (read once at import; default 0 = the phase-1 table):

| EXL3_MR | tensors on exl3_gemm_mr | rows | layout |
|---------|--------------------------|------|--------|
| 0       | none                     |      |        |
| 1       | K3, K5 (mul1)            | 17..144 | the stored int16 trellis, read as int32 (no copy) |
| 2       | 1, plus K4 (mul1)        | K4: 1..144 | exl3_mr_repack at load: lossless word permutation, the only resident copy; unpacked per call above 144 rows |

K2, K6 and other codebooks stay on exl3_gemm. A repacked (int32) trellis is routed by its
dtype, since exl3_gemm and the dequant cannot read it.

This is exllamav3's own dispatch (LinearEXL3.forward / reconstruct_hgemm, d3739fd):
AUTO_RECONSTRUCT_THRESHOLD 144, fused reconstruct from 1024 rows, dequant in 32768-column
slices. GEMM_MAX_ROWS must stay above vLLM's largest CUDA-graph capture size (48 in
production), so captured graphs only hold exl3_gemm (warmed, no dequant buffers).
"""

from __future__ import annotations

import os

import torch

try:
    from . import _C_exl3  # noqa: F401  (registers torch.ops._C_exl3)

    OPS_AVAILABLE = True
except ImportError:
    OPS_AVAILABLE = False

try:
    from . import _C_exl3_mr  # noqa: F401  (registers torch.ops._C_exl3.exl3_gemm_mr, exl3_mr_*)

    MR_AVAILABLE = True
except ImportError:
    MR_AVAILABLE = False

GEMM_MAX_ROWS = 144  # exllamav3 AUTO_RECONSTRUCT_THRESHOLD
FUSED_RECON_MIN_ROWS = 1024  # LinearEXL3.reconstruct_hgemm: fused reconstruct from here
RECON_SLICE_N = 32768  # MAX_RECONSTRUCT_SLICE_N: dequant at most this many columns at once

# Multi-row decode kernel (the MTP verify regime: 12..48 rows at c=2..16 on production's k=5/3/2): the name of
# a torch.ops._C_exl3 op with exl3_gemm's signature, routed at MULTI_ROW_MIN..MULTI_ROW_MAX
# rows for the tensors it takes. None: those rows stay on the vendored exl3_gemm, which
# re-streams the weight once per 16 rows. EXL3_MR (table above) sets it to exl3_gemm_mr,
# which is accepted only after 10-mr-parity (cloud/results/exl3-opt/box-scripts).
MR_MODE = int(os.environ.get("EXL3_MR", "0"))
if MR_MODE not in (0, 1, 2):
    raise ValueError(f"EXL3_MR={MR_MODE}: 0 (off), 1 (K3/K5) or 2 (K3/K5 and repacked K4)")
MR_OP = "exl3_gemm_mr"
MULTI_ROW_OP: str | None = MR_OP if MR_MODE else None
MULTI_ROW_MIN, MULTI_ROW_MAX = 17, GEMM_MAX_ROWS
MR_TILE_WIDTHS = (48, 80)  # K3, K5: exl3_gemm_mr reads the stored trellis
MR_REPACK_TILE_WIDTH = 64  # K4: exl3_mr_repack first (EXL3_MR=2)

EXL3_GEMM = "exl3_gemm"
RECON_HGEMM = "recon_hgemm"
RECON_HAD_HGEMM = "recon_had_hgemm"


def _exl3_op(n: int, mr_ok: bool = True, repacked: bool = False) -> str:
    """The route for n activation rows (tables in the module docstring). mr_ok: the multi-row
    kernel takes this tensor as stored; repacked: the trellis is in its layout (int32)."""
    if n <= GEMM_MAX_ROWS:
        if repacked:
            return MR_OP
        if MULTI_ROW_OP is not None and mr_ok and MULTI_ROW_MIN <= n <= MULTI_ROW_MAX:
            return MULTI_ROW_OP
        return EXL3_GEMM
    return RECON_HAD_HGEMM if n >= FUSED_RECON_MIN_ROWS else RECON_HGEMM


def mr_takes(tile_width: int, mul1: bool) -> bool:
    """exl3_gemm_mr reads this tensor as stored (K3/K5, mul1)."""
    return mul1 and tile_width in MR_TILE_WIDTHS


def mr_repacks(tile_width: int, mul1: bool) -> bool:
    """EXL3_MR=2 replaces this tensor's trellis by exl3_mr_repack's layout at load (K4, mul1)."""
    return MR_MODE == 2 and mul1 and tile_width == MR_REPACK_TILE_WIDTH


def out_features(trellis: torch.Tensor) -> int:
    """n of a stored (int16 [k/16, n/16, 16K]) or repacked (int32 [k/16, n/64, 32, 4]) trellis."""
    return trellis.shape[1] * (64 if trellis.dtype == torch.int32 else 16)


def _recon_hgemm(x, trellis, suh, svh, mcg: bool, mul1: bool, fused: bool) -> torch.Tensor:
    """exllamav3's reconstruct_hgemm: dequantize W in column slices, fp16 GEMM, fp16 out."""
    ops = torch.ops._C_exl3
    xh = x
    if not fused:  # rotated basis: the input Hadamard (with suh) on x, the output one on y
        xh = ops.exl3_had_r_128(xh, suh, None, 1.0)
    n = trellis.shape[1] * 16
    ys = [
        ops.exl3_hgemm(xh, ops.exl3_dequant(trellis, suh, svh, mcg, mul1, s,
                                            min(RECON_SLICE_N, n - s), fused))
        for s in range(0, n, RECON_SLICE_N)
    ]
    y = ys[0] if len(ys) == 1 else torch.cat(ys, dim=1)
    if not fused:
        y = ops.exl3_had_r_128(y, None, svh, 1.0)
    return y


def exl3_linear(
    x: torch.Tensor,
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    mcg: bool,
    mul1: bool,
    out_fp32: bool,
) -> torch.Tensor:
    repacked = trellis.dtype == torch.int32
    name = _exl3_op(x.shape[0], not repacked and mr_takes(trellis.shape[2], mul1), repacked)
    if name == RECON_HGEMM or name == RECON_HAD_HGEMM:
        if repacked:  # the dequant reads exllamav3's layout: one transient copy per call
            trellis = torch.ops._C_exl3.exl3_mr_unpack(trellis)
        y = _recon_hgemm(x, trellis, suh, svh, mcg, mul1, name == RECON_HAD_HGEMM)
        return y.float() if out_fp32 else y
    return getattr(torch.ops._C_exl3, name)(x.contiguous(), trellis, suh, svh, mcg, mul1, out_fp32)


def exl3_linear_fake(
    x: torch.Tensor,
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    mcg: bool,
    mul1: bool,
    out_fp32: bool,
) -> torch.Tensor:
    return x.new_empty(x.shape[0], out_features(trellis), dtype=torch.float if out_fp32 else torch.half)


def _register() -> None:
    from vllm.utils.torch_utils import direct_register_custom_op

    direct_register_custom_op(
        op_name="_exl3_linear", op_func=exl3_linear, fake_impl=exl3_linear_fake
    )


_register()
