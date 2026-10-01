# SPDX-License-Identifier: Apache-2.0
"""torch.ops._C_exl3 (csrc/exl3_shim.cu) behind one routed custom op.

`exl3_linear(x, trellis, suh, svh, mcg, mul1, out_fp32)` is x @ W for one EXL3 tensor W,
x [n, k] fp16 (the kernels' activation dtype; the linear method casts once per layer), result
fp32 or fp16; or x bf16 and the result bf16 (ops.MR_GLUE: exl3_gemm_mr converts inside its
launches, the other routes cast). It is registered as the vLLM custom op
`torch.ops.vllm._exl3_linear` (with a fake impl), so torch.compile sees one opaque node per
EXL3 tensor and the row-count choice below is made per call, not when the graph is traced.

Routing (`_exl3_op`, pinned by tests/cpu/test_exl3_routing.py), n = activation rows:

| n            | op                   | what                                                    |
|--------------|----------------------|---------------------------------------------------------|
| 1..144       | exl3_gemm            | vendored kernel (GEMV where its heuristic picks it)     |
| 1..384       | MULTI_ROW_OP if set  | multi-row kernel for the tensors it takes (EXL3_MR, default on): before the rows above and below |
| 145..1023    | recon_hgemm          | Hadamard x, rotated dequant, hgemm, Hadamard y          |
| >= 1024      | recon_had_hgemm      | original-basis dequant (Hadamards folded in), hgemm     |

The multi-row kernel is trellis-serve's Marlin-EXL3 (csrc/trellis_serve, op exl3_gemm_mr,
_C_exl3_mr), switched by EXL3_MR (read once at import; default 2; 0 = the phase-1 table):

| EXL3_MR | tensors on exl3_gemm_mr | rows | layout |
|---------|--------------------------|------|--------|
| 0       | none                     |      |        |
| 1       | K3, K5 (mul1)            | 1..384 | the stored int16 trellis, read as int32 (no copy) |
| 2       | 1, plus K4 (mul1)        | K4: 1..384 | exl3_mr_repack at load: lossless word permutation, the only resident copy; unpacked per call above 384 rows |

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
# default 2 (job 12, pass 2 T=0, c=1/2/4/8: 33.7/38.9/52.0/52.4 ms/step vs 39.9/43.0/69.9/72.1 at 0)
MR_MODE = int(os.environ.get("EXL3_MR", "2"))
if MR_MODE not in (0, 1, 2):
    raise ValueError(f"EXL3_MR={MR_MODE}: 0 (off), 1 (K3/K5) or 2 (K3/K5 and repacked K4)")
MR_OP = "exl3_gemm_mr"
# glue (with EXL3_MR=2): a bf16 model passes bf16 activations and gets bf16 back (exl3_linear's bf16
# contract), so exl3_gemm_mr skips the cast in and the fp32 round trip out. Same bits, fewer launches.
MR_GLUE = MR_MODE == 2
MULTI_ROW_OP: str | None = MR_OP if MR_MODE else None
# MULTI_ROW_MIN: K3/K5 take exl3_gemm_mr from 1 row (job 12: from 17 rows instead costs +3.3
# ms/step at c=1, +1.5 at c=2; job 11: mr beats exl3_gemm at 1..16 rows too)
# MULTI_ROW_MAX: past exllamav3's 144 the multi-row kernel still beats the dequant route; job 11
# (3090, erlidev, model sum over every eligible tensor): 192 rows 170 vs 280 ms, 256: 221 vs 299,
# 384: 326 vs 354, 512: 435 vs 424. So to 384 (K2, the one class it cannot take, keeps 144).
MULTI_ROW_MIN, MULTI_ROW_MAX = 1, 384
MR_TILE_WIDTHS = (48, 80)  # K3, K5: exl3_gemm_mr reads the stored trellis
MR_REPACK_TILE_WIDTH = 64  # K4: exl3_mr_repack first (EXL3_MR=2)

EXL3_GEMM = "exl3_gemm"
RECON_HGEMM = "recon_hgemm"
RECON_HAD_HGEMM = "recon_had_hgemm"


def _exl3_op(n: int, mr_ok: bool = True, repacked: bool = False) -> str:
    """The route for n activation rows (tables in the module docstring). mr_ok: the multi-row
    kernel takes this tensor as stored; repacked: the trellis is in its layout (int32)."""
    if repacked and n <= MULTI_ROW_MAX:
        return MR_OP
    if MULTI_ROW_OP is not None and mr_ok and MULTI_ROW_MIN <= n <= MULTI_ROW_MAX:
        return MULTI_ROW_OP
    if n <= GEMM_MAX_ROWS:
        return EXL3_GEMM
    return RECON_HAD_HGEMM if n >= FUSED_RECON_MIN_ROWS else RECON_HGEMM


def mr_takes(tile_width: int, mul1: bool) -> bool:
    """exl3_gemm_mr reads this tensor as stored (K3/K5, mul1)."""
    return mul1 and tile_width in MR_TILE_WIDTHS


def mr_repacks(tile_width: int, mul1: bool) -> bool:
    """EXL3_MR=2 replaces this tensor's trellis by exl3_mr_repack's layout at load (K4, mul1)."""
    return MR_MODE == 2 and mul1 and tile_width == MR_REPACK_TILE_WIDTH


def repack_k4_(trellis: torch.Tensor, chunk_bytes: int = 64 << 20) -> torch.Tensor:
    """exl3_mr_repack in place: the same word permutation (32-bit word l of tile (i, 4g + j) to
    [i, g, l, j]) on the tensor's own storage, a few k/16 rows at a time; returns the int32
    [k/16, n/64, 32, 4] view. No second copy of the weight ever exists, whoever still references
    the int16 tensor (job 12 run 1: an allocating repack left 23.05 GiB live at load and OOMed)."""
    kt, nt, width = trellis.shape
    assert trellis.dtype == torch.int16 and width == MR_REPACK_TILE_WIDTH and nt % 4 == 0 and trellis.is_contiguous()
    w = trellis.view(torch.int32).view(kt, nt // 4, 4, 32)
    step = max(1, chunk_bytes // (nt * 128))
    for i in range(0, kt, step):
        blk = w[i:i + step]
        blk.view(-1).copy_(blk.permute(0, 1, 3, 2).contiguous().view(-1))
    return w.view(kt, nt // 4, 32, 4)


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
    if repacked and out_features(trellis) > RECON_SLICE_N:
        name = MR_OP  # the lm_head above 384 rows (prompt_logprobs only): no 0.6 GiB unpack copy outside
        # vLLM's profiled budget; speed unmeasured there (job 11: mr ~ dequant at 512 rows)
    bf16 = x.dtype == torch.bfloat16  # bf16 in, bf16 out; only exl3_gemm_mr takes it directly
    if name == RECON_HGEMM or name == RECON_HAD_HGEMM:
        if repacked:  # the dequant reads exllamav3's layout: one transient copy per call
            trellis = torch.ops._C_exl3.exl3_mr_unpack(trellis)
        y = _recon_hgemm(x.half() if bf16 else x, trellis, suh, svh, mcg, mul1, name == RECON_HAD_HGEMM)
        return y.to(torch.bfloat16) if bf16 else y.float() if out_fp32 else y
    if bf16 and name != MR_OP:
        return getattr(torch.ops._C_exl3, name)(x.half(), trellis, suh, svh, mcg, mul1, True).to(torch.bfloat16)
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
    dtype = torch.bfloat16 if x.dtype == torch.bfloat16 else torch.float if out_fp32 else torch.half
    return x.new_empty(x.shape[0], out_features(trellis), dtype=dtype)


def exl3_linear_parts(
    x: torch.Tensor,
    trellis: list[torch.Tensor],
    suh: list[torch.Tensor],
    svh: list[torch.Tensor],
    mcg: bool,
    mul1: bool,
    out_fp32: bool,
) -> torch.Tensor:
    """One layer: exl3_linear per part, concatenated along the outputs. One opaque op per layer, so
    torch.compile never sees the cat: inductor's split-of-cat pass would otherwise hand the model's
    split of it (GDN's z) back as the part's own buffer, with a stride the next compiled piece was
    not traced with (job 12 run 2: z of qkvz expected stride 16384, got 6144)."""
    ys = [exl3_linear(x, t, s, v, mcg, mul1, out_fp32) for t, s, v in zip(trellis, suh, svh)]
    return ys[0] if len(ys) == 1 else torch.cat(ys, dim=1)


def exl3_linear_parts_fake(
    x: torch.Tensor,
    trellis: list[torch.Tensor],
    suh: list[torch.Tensor],
    svh: list[torch.Tensor],
    mcg: bool,
    mul1: bool,
    out_fp32: bool,
) -> torch.Tensor:
    dtype = torch.bfloat16 if x.dtype == torch.bfloat16 else torch.float if out_fp32 else torch.half
    return x.new_empty(x.shape[0], sum(out_features(t) for t in trellis), dtype=dtype)


# EXL3_EMBED_HOST (default 1): the bf16 token embedding lives in page-locked host memory
# (quantization/embedding.py); rows are gathered to the GPU per step by exl3_embed_host. Job 12:
# no ms/step change beyond run-to-run spread; frees 2.37 GiB (KV 198,162 -> 264,993 tokens at
# 196,608; 200,000 fits, job 14).
EMBED_HOST = os.environ.get("EXL3_EMBED_HOST", "1") == "1"
# EXL3_DRAFT_FP8 (A/B while measured): the MTP draft head as fp8 weights (quantization/draft_head.py)
DRAFT_FP8 = os.environ.get("EXL3_DRAFT_FP8", "0") == "1"


def exl3_embed_host(ids: torch.Tensor, table_id: int, cols: int) -> torch.Tensor:
    flat = ids.reshape(-1).contiguous()
    return torch.ops._C_exl3.exl3_embed_host(flat, table_id, cols).view(*ids.shape, cols)


def exl3_embed_host_fake(ids: torch.Tensor, table_id: int, cols: int) -> torch.Tensor:
    return ids.new_empty(*ids.shape, cols, dtype=torch.bfloat16)


def _register() -> None:
    from vllm.utils.torch_utils import direct_register_custom_op

    direct_register_custom_op(
        op_name="_exl3_linear", op_func=exl3_linear, fake_impl=exl3_linear_fake
    )
    direct_register_custom_op(
        op_name="_exl3_linear_parts", op_func=exl3_linear_parts, fake_impl=exl3_linear_parts_fake
    )
    direct_register_custom_op(
        op_name="_exl3_embed_host", op_func=exl3_embed_host, fake_impl=exl3_embed_host_fake
    )


_register()
