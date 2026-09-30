# SPDX-License-Identifier: Apache-2.0
"""torch.ops._C_exl3 (csrc/exl3_shim.cu) behind one routed custom op.

`exl3_linear(x, trellis, suh, svh, mcg, mul1)` is x @ W for one EXL3 tensor W, x [n, k]
fp16/bf16, result in x's dtype. It is registered as the vLLM custom op
`torch.ops.vllm._exl3_linear` (with a fake impl), so torch.compile sees one opaque node per
EXL3 tensor and the row-count choice below is made per call, not when the graph is traced.

Routing (`_exl3_op`, pinned by tests/cpu/test_exl3_routing.py), n = activation rows:

| n            | op                   | what                                                    |
|--------------|----------------------|---------------------------------------------------------|
| 1..144       | exl3_gemm            | vendored kernel (GEMV where its heuristic picks it)     |
| 17..144      | MULTI_ROW_OP if set  | hook for a multi-row kernel; None (default): exl3_gemm   |
| 145..1023    | recon_hgemm          | Hadamard x, rotated dequant, hgemm, Hadamard y          |
| >= 1024      | recon_had_hgemm      | original-basis dequant (Hadamards folded in), hgemm     |

This is exllamav3's own dispatch (LinearEXL3.forward / reconstruct_hgemm, d3739fd):
AUTO_RECONSTRUCT_THRESHOLD 144, fused reconstruct from 1024 rows, dequant in 32768-column
slices. GEMM_MAX_ROWS must stay above vLLM's largest CUDA-graph capture size (48 in
production), so captured graphs only hold exl3_gemm (warmed, no dequant buffers).
"""

from __future__ import annotations

import torch

try:
    from . import _C_exl3  # noqa: F401  (registers torch.ops._C_exl3)

    OPS_AVAILABLE = True
except ImportError:
    OPS_AVAILABLE = False

GEMM_MAX_ROWS = 144  # exllamav3 AUTO_RECONSTRUCT_THRESHOLD
FUSED_RECON_MIN_ROWS = 1024  # LinearEXL3.reconstruct_hgemm: fused reconstruct from here
RECON_SLICE_N = 32768  # MAX_RECONSTRUCT_SLICE_N: dequant at most this many columns at once

# Hook for a multi-row decode kernel (the MTP verify regime, 4..64 rows at c >= 2 with k=3):
# the name of a torch.ops._C_exl3 op with exl3_gemm's signature, routed at
# MULTI_ROW_MIN..MULTI_ROW_MAX rows. None: those rows stay on the vendored exl3_gemm, which
# re-streams the weight once per 16 rows. First candidate: trellis-serve's Marlin-EXL3 (MIT),
# accepted only after a bit-exact decode check against exl3_dequant (EXL3.md, GPU phase 4).
MULTI_ROW_OP: str | None = None
MULTI_ROW_MIN, MULTI_ROW_MAX = 17, GEMM_MAX_ROWS

EXL3_GEMM = "exl3_gemm"
RECON_HGEMM = "recon_hgemm"
RECON_HAD_HGEMM = "recon_had_hgemm"


def _exl3_op(n: int) -> str:
    """The route for n activation rows (table in the module docstring)."""
    if n <= GEMM_MAX_ROWS:
        if MULTI_ROW_OP is not None and MULTI_ROW_MIN <= n <= MULTI_ROW_MAX:
            return MULTI_ROW_OP
        return EXL3_GEMM
    return RECON_HAD_HGEMM if n >= FUSED_RECON_MIN_ROWS else RECON_HGEMM


def _recon_hgemm(x, trellis, suh, svh, mcg: bool, mul1: bool, fused: bool) -> torch.Tensor:
    """exllamav3's reconstruct_hgemm: dequantize W in column slices, fp16 GEMM."""
    ops = torch.ops._C_exl3
    xh = x.to(torch.half)
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
    return y.to(x.dtype)


def exl3_linear(
    x: torch.Tensor,
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    mcg: bool,
    mul1: bool,
) -> torch.Tensor:
    name = _exl3_op(x.shape[0])
    if name == RECON_HGEMM or name == RECON_HAD_HGEMM:
        return _recon_hgemm(x, trellis, suh, svh, mcg, mul1, name == RECON_HAD_HGEMM)
    return getattr(torch.ops._C_exl3, name)(x.contiguous(), trellis, suh, svh, mcg, mul1)


def exl3_linear_fake(
    x: torch.Tensor,
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    mcg: bool,
    mul1: bool,
) -> torch.Tensor:
    return x.new_empty(x.shape[0], trellis.shape[1] * 16)


def _register() -> None:
    from vllm.utils.torch_utils import direct_register_custom_op

    direct_register_custom_op(
        op_name="_exl3_linear", op_func=exl3_linear, fake_impl=exl3_linear_fake
    )


_register()
