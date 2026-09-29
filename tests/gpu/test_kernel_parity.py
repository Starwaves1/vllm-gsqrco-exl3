"""Plugin CUDA kernels vs CPU references, per quant type in the GGUF, on real rows.

Covers: ggml_dequantize (vs gguf-py b11211), MMVQ ggml_mul_mat_vec_a8 at 1..16 tokens,
MMQ ggml_mul_mat_a8 at 16..512 tokens (K-quants only; the plugin has no IQ MMQ), and the
production routing function _fused_mul_mat_gguf on whole tensors (MMVQ below the
mmvq_safe threshold, else MMQ or dequantize + x @ W.T).

Tolerances calibrated on an RTX 3090 at e2b8ad5 (2026-09-28, cloud/results/phase1): worst
reference-model error 2.5e-3 (bf16) / 1.24e-3 (fp16); worst error vs full precision 1.5e-2,
except Q4_K through MMQ (7.0e-2 direct, 9.0e-2 via routing), which matches the xsum model
to 2.5e-3: its min term uses half(sum x), as in ggml's MMQ. ggml_dequantize accepts float32.
Inputs are tie-free (_x): an x on a q8_1 rounding tie may round either way on the GPU.

Route L (llama.cpp b11211 MMVQ/MMQ behind csrc/lcpp_shim.cu; needs the VLLM_GGUF_BUILD_LCPP=1
build, skipped otherwise): lcpp_mul_mat_vec_q at 1..8 tokens and lcpp_mul_mat_q at 1..2048
against the same references (plus the D2S6 model for Q2_K MMQ, see _refs.py), with the
allocator's free blocks poisoned (0xFF) first so an unzeroed scratch read shows up; CUDA-graph
capture + replay must be bit-exact with an eager call. The shim's own kernels for 1..8 rows,
lcpp_mul_mat_vec_iq3 and lcpp_mul_mat_vec_iq3_mma (IQ3_S/IQ3_XXS, dp4a and int8 tensor cores)
and lcpp_mul_mat_vec_own (Q4_K/IQ2_S, lcpp_owned_k4.cu), are checked the same way and against
vendored MMVQ on fp32 X.
Run the file with VLLM_GGUF_LCPP=1 and
the routing tests go through Route L too (then the mixed-shard layer test also runs).
"""

import pytest

from gsq_gpu import QUANT_TYPES

ROWS = 512                          # rows per kernel test (real rows from the GGUF)
MMVQ_TOKENS = [1, 2, 3, 4, 8, 16]    # 4 = MTP k=3 verify
MMQ_TOKENS = [16, 64, 512]
ROUTE_TOKENS = [1, 4, 8, 9, 16, 32, 128, 512]
TIGHT = {"bfloat16": 5e-3, "float16": 2.5e-3}
# vs full precision. The MMQ Q4_K/Q5_K x-sum model is further from full than q81: its scale
# and min terms no longer share the q8_1 error, so they stop cancelling (the 20x outlier
# channels in _x make it large).
LOOSE = 3e-2
LOOSE_XSUM = 1.5e-1                  # Q4_K via MMQ (and Q2_K via lcpp MMQ), see the module docstring
DQ_DTYPES = ["float32", "float16", "bfloat16"]
MIN_TERM = ("Q4_K", "Q5_K")                       # MMQ types whose x-sum min term is far from full

LCPP_TYPES = [q for q in QUANT_TYPES if q != "IQ1_M"]  # IQ1_M keeps the e2b8ad5 path
LCPP_MMVQ_TOKENS = [1, 2, 3, 4, 5, 6, 7, 8]
# 1..8: MMQ below upstream's J_max tail (only the shim's zeroed 128-block tail protects the
# reads); 128 = production's prefill chunk (--long-prefill-token-threshold 128).
LCPP_MMQ_TOKENS = [1, 2, 3, 5, 7, 8, 9, 16, 64, 128, 512, 2048]
IQ3_TYPES = ["IQ3_S", "IQ3_XXS"]     # the owned lcpp_mul_mat_vec_iq3[_mma] kernels
IQ3_OPS = ["lcpp_mul_mat_vec_iq3", "lcpp_mul_mat_vec_iq3_mma"]  # dp4a, int8 tensor cores
OWN_TYPES = ["Q4_K", "IQ2_S"]  # the owned lcpp_mul_mat_vec_own kernel
OWNED = [(t, op) for op in IQ3_OPS for t in IQ3_TYPES] + [(t, "lcpp_mul_mat_vec_own") for t in OWN_TYPES]


def _sample(tensors_by_type, name, rows=ROWS, big=None):
    ts = tensors_by_type.get(name)
    if not ts:
        pytest.skip(f"{name} not in this GGUF")
    if big is not None:  # routing depends on the tensor's row count (> 5120 or not)
        ts = [t for t in ts if (int(t.shape[1]) > 5120) == big] or pytest.skip(f"no {name} tensor with rows>5120 == {big}")
    t = ts[0]
    data = t.data if rows is None else t.data[:rows]
    return t, data


def _x(n, k, dtype, seed=0):
    import torch

    g = torch.Generator().manual_seed(seed)
    # hidden-state-like: mostly N(0,1) with a few large outlier channels
    x = torch.randn(n, k, generator=g)
    x[:, torch.randperm(k, generator=g)[: max(1, k // 256)]] *= 20
    x = x.to(getattr(torch, dtype))
    if x.dtype == torch.float32:
        return x
    # No q8_1 rounding ties (x/d = j + 0.5 for the block's d = amax/127, per 32 and per 64 values):
    # 16-bit x hits them often (x = amax/2), and the GPU's fast-math division may round them either
    # way, so no reference could be exact. One ulp toward zero moves such an x/d by ~0.25 (a nudged
    # 32-block amax can create new ties, hence the loop; it converges in 2-4 passes).
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


def _check(y, raw, name, x, mmq, lcpp=False):
    import _refs

    r = _refs.refs(raw, name, x.cpu(), mmq, lcpp)
    errs = {k: _refs.rel_err(y, v) for k, v in r.items()}
    tight = TIGHT[str(x.dtype).split(".")[-1]]
    print(f"\n{name} n={x.shape[0]} {x.dtype} mmq={mmq} lcpp={lcpp}: " + " ".join(f"{k}={v:.2e}" for k, v in errs.items()))
    best = min(v for k, v in errs.items() if k != "full")
    assert best <= tight, f"no reference model within {tight}: {errs}"
    loose = LOOSE_XSUM if "xsum" in errs or "d2s6" in errs else LOOSE
    assert errs["full"] <= loose, f"too far from full precision: {errs}"


@pytest.mark.parametrize("dtype", DQ_DTYPES)
@pytest.mark.parametrize("name", QUANT_TYPES)
def test_dequantize(tensors_by_type, name, dtype):
    import gguf
    import numpy as np
    import torch

    import _refs
    from vllm_gguf_plugin import ops

    t, raw = _sample(tensors_by_type, name)
    qt = int(gguf.GGMLQuantizationType[name])
    ref = _refs.dequant(raw, name)
    w = torch.from_numpy(np.ascontiguousarray(raw)).cuda()
    _poison_allocator()  # the output is uninitialised: an element left unwritten shows up
    out = ops.ggml_dequantize(w, qt, ref.shape[0], ref.shape[1], getattr(torch, dtype)).cpu()
    torch.cuda.synchronize()
    exp = ref.to(getattr(torch, dtype))
    if name in ("Q2_K", "Q4_K", "Q6_K"):
        # dequantize.cuh does K-quants in fp16, ggml in fp32: not bit-exact. Same bound as the
        # strict xfail + test_cuda_kquant_dequant_error_is_fp16_sized in tests/cpu/test_dequant_fixtures.py
        # (2**-9 of the row's absmax), plus half an output ulp for the final rounding.
        tol = 2.0**-9 * ref.abs().amax(dim=1, keepdim=True) + torch.finfo(out.dtype).eps / 2 * ref.abs()
        assert ((out.float() - ref).abs() <= tol).all(), ((out.float() - ref).abs() / tol).max()
    elif dtype == "float32":
        # same float math on both sides; allow only 1-ulp ordering differences
        torch.testing.assert_close(out, exp, rtol=2e-7, atol=0)
    else:
        torch.testing.assert_close(out, exp, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
@pytest.mark.parametrize("n", MMVQ_TOKENS)
@pytest.mark.parametrize("name", QUANT_TYPES)
def test_mmvq(tensors_by_type, name, n, dtype):
    import gguf
    import numpy as np
    import torch

    from vllm_gguf_plugin import ops

    t, raw = _sample(tensors_by_type, name)
    k = int(t.shape[0])
    x = _x(n, k, dtype, seed=n)
    w = torch.from_numpy(np.ascontiguousarray(raw)).cuda()
    y = ops.ggml_mul_mat_vec_a8(w, x.cuda(), int(gguf.GGMLQuantizationType[name]), w.shape[0])
    torch.cuda.synchronize()
    assert y.shape == (n, raw.shape[0]) and y.dtype == x.dtype
    _check(y, raw, name, x, mmq=False)


@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
@pytest.mark.parametrize("n", MMQ_TOKENS)
@pytest.mark.parametrize("name", [q for q in QUANT_TYPES if not q.startswith("IQ")])
def test_mmq(tensors_by_type, name, n, dtype):
    import gguf
    import numpy as np
    import torch

    from vllm_gguf_plugin import ops

    t, raw = _sample(tensors_by_type, name)
    x = _x(n, int(t.shape[0]), dtype, seed=100 + n)
    w = torch.from_numpy(np.ascontiguousarray(raw)).cuda()
    y = ops.ggml_mul_mat_a8(w, x.cuda(), int(gguf.GGMLQuantizationType[name]), w.shape[0])
    torch.cuda.synchronize()
    _check(y, raw, name, x, mmq=True)


@pytest.mark.parametrize("big", [True, False])
@pytest.mark.parametrize("n", ROUTE_TOKENS)
@pytest.mark.parametrize("name", QUANT_TYPES)
def test_routing_whole_tensor(tensors_by_type, name, n, big):
    """The function production calls, on a whole tensor (routing depends on its rows).
    Reference: the kernel's own dequantize (checked above) in float64, since a CPU
    dequant of a 17408x5120 tensor per case would dominate the run time."""
    import gguf
    import numpy as np
    import torch

    import _refs
    from vllm_gguf_plugin import ops
    from vllm_gguf_plugin.quantization.linear import _fused_mul_mat_gguf

    t, raw = _sample(tensors_by_type, name, rows=None, big=big)
    qt = int(gguf.GGMLQuantizationType[name])
    w = torch.from_numpy(np.ascontiguousarray(raw)).cuda()
    x = _x(n, int(t.shape[0]), "bfloat16", seed=200 + n)
    y = _fused_mul_mat_gguf(x.cuda(), w, qt)
    W = ops.ggml_dequantize(w, qt, int(t.shape[1]), int(t.shape[0]), torch.float32).double()
    ref = (x.cuda().double() @ W.T).cpu()
    torch.cuda.synchronize()
    e = _refs.rel_err(y, ref)
    print(f"\n{name} rows={t.shape[1]} n={n}: rel err vs full {e:.2e}")
    min_term = name in MIN_TERM or (name == "Q2_K" and ops.LCPP_ENABLED)  # lcpp MMQ: D2S6
    assert e <= (LOOSE_XSUM if min_term else LOOSE)


# ---------------------------------------------------------------------------- Route L


def _lcpp():
    import torch

    from vllm_gguf_plugin import ops  # noqa: F401  (loads _C_gguf)

    if not hasattr(torch.ops._C_gguf, "lcpp_mul_mat_q"):
        pytest.skip("_C_gguf built without VLLM_GGUF_BUILD_LCPP=1")
    return torch.ops._C_gguf


def _poison_allocator():
    """Fill the caching allocator's free blocks with 0xFF (NaN as fp32, -1 as int8), so the
    op's uninitialised scratch holds garbage, not stale zeros."""
    import torch

    keep = [torch.full((1 << p,), 0xFF, dtype=torch.uint8, device="cuda") for p in range(9, 28) for _ in range(2)]
    del keep


def _lcpp_case(tensors_by_type, name, n, dtype, seed, rows=ROWS):
    import gguf
    import numpy as np
    import torch

    t, raw = _sample(tensors_by_type, name, rows=rows)
    x = _x(n, int(t.shape[0]), dtype, seed=seed)
    w = torch.from_numpy(np.ascontiguousarray(raw)).cuda()
    return raw, x, w, int(gguf.GGMLQuantizationType[name])


@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
@pytest.mark.parametrize("n", LCPP_MMVQ_TOKENS)
@pytest.mark.parametrize("name", LCPP_TYPES)
def test_lcpp_mmvq(tensors_by_type, name, n, dtype):
    import torch

    C = _lcpp()
    raw, x, w, qt = _lcpp_case(tensors_by_type, name, n, dtype, seed=300 + n)
    _poison_allocator()
    y = C.lcpp_mul_mat_vec_q(w, x.cuda(), qt, w.shape[0])
    torch.cuda.synchronize()
    assert y.shape == (n, raw.shape[0]) and y.dtype == x.dtype
    _check(y, raw, name, x, mmq=False, lcpp=True)


@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
@pytest.mark.parametrize("n", LCPP_MMQ_TOKENS)
@pytest.mark.parametrize("name", LCPP_TYPES)
def test_lcpp_mmq(tensors_by_type, name, n, dtype):
    import torch

    C = _lcpp()
    raw, x, w, qt = _lcpp_case(tensors_by_type, name, n, dtype, seed=400 + n)
    _poison_allocator()
    y = C.lcpp_mul_mat_q(w, x.cuda(), qt, w.shape[0])
    torch.cuda.synchronize()
    assert y.shape == (n, raw.shape[0]) and y.dtype == x.dtype
    _check(y, raw, name, x, mmq=True, lcpp=True)


@pytest.mark.parametrize("n", [5, 128])
@pytest.mark.parametrize("name", LCPP_TYPES)
def test_lcpp_mmq_odd_rows(tensors_by_type, name, n):
    """W rows not a multiple of 128: MMQ's fallback tiles."""
    import torch

    C = _lcpp()
    raw, x, w, qt = _lcpp_case(tensors_by_type, name, n, "bfloat16", seed=500 + n, rows=200)
    _poison_allocator()
    y = C.lcpp_mul_mat_q(w, x.cuda(), qt, w.shape[0])
    torch.cuda.synchronize()
    _check(y, raw, name, x, mmq=True, lcpp=True)


@pytest.mark.parametrize("shape", ["real", "row_tail", "k_tail", "odd_rows", "k_min", "few_rows", "many_tiles"])
@pytest.mark.parametrize("dtype", ["bfloat16", "float16", "float32"])
@pytest.mark.parametrize("n", LCPP_MMVQ_TOKENS)
@pytest.mark.parametrize("name,op", OWNED)
def test_lcpp_iq3(tensors_by_type, op, name, n, dtype, shape):
    """The owned kernels (lcpp_mul_mat_vec_iq3, dp4a; lcpp_mul_mat_vec_iq3_mma, int8 tensor
    cores; lcpp_mul_mat_vec_own, Q4_K/IQ2_S dp4a) take MMVQ's q8_1 input and compute each 32-value
    slice's scaled integer sum exactly as the vendored vec_dot; only the fp32 order of a row's
    slice terms differs (the mma kernel also applies d_w once per weight block; Q4_K: also the
    fp32 rounding of its scale and min terms). 16-bit X: the CPU reference models, as for MMVQ. fp32 X
    (fp32 output, no final rounding): within 1e-5 of vendored MMVQ itself. row_tail: 202 rows (the last CTA's 16
    rows are part-filled; even, because vendored MMVQ, the reference, reads one weight row past
    the end at an odd row count: compute-sanitizer memcheck); k_tail: K = 4608 (the last staged
    chunk is 16 q8_1 blocks, not 32; this model's K are all multiples of 1024). odd_rows: 201
    rows of K = 4608, so W's size is 4 mod 8 (fp32 reference for the mma kernel: the dp4a
    kernel, which stays in bounds); k_min: K = 512 (2 weight blocks: the mma kernel's warps 2
    and 3 have none); few_rows: 20 rows (fewer 16-row tiles than CTAs); many_tiles: the 512
    rows 16 times over, fp32 only (more tiles than resident CTAs: each CTA takes several)."""
    import _refs
    import gguf
    import numpy as np
    import torch

    if shape == "many_tiles" and dtype != "float32":
        pytest.skip("fp32 only: the reference is MMVQ on the GPU")
    if shape == "odd_rows" and op == "lcpp_mul_mat_vec_own" and dtype == "float32":
        pytest.skip("fp32 reference is MMVQ, which reads past this W (no in-bounds Q4_K/IQ2_S one)")
    C = _lcpp()
    qt = gguf.GGMLQuantizationType[name]
    rows = {"row_tail": 202, "odd_rows": 201, "few_rows": 20}.get(shape, ROWS)
    _, raw = _sample(tensors_by_type, name, rows=rows)
    if shape == "many_tiles":
        raw = np.tile(raw, (16, 1))
    bsz = gguf.GGML_QUANT_SIZES[qt][1]
    blocks = {"k_tail": 18, "odd_rows": 18, "k_min": 2}.get(shape)
    raw = np.ascontiguousarray(raw[:, : blocks * bsz] if blocks else raw)
    x = _x(n, raw.shape[1] // bsz * 256, dtype, seed=1000 + n)
    w = torch.from_numpy(raw).cuda()
    _poison_allocator()
    y = getattr(C, op)(w, x.cuda(), int(qt), w.shape[0])
    mma_odd = shape == "odd_rows" and op.endswith("_mma")  # under memcheck MMVQ would read past W
    ref_op = C.lcpp_mul_mat_vec_iq3 if mma_odd else C.lcpp_mul_mat_vec_q
    ref = ref_op(w, x.cuda(), int(qt), w.shape[0]) if dtype == "float32" else None
    torch.cuda.synchronize()
    assert y.shape == (n, raw.shape[0]) and y.dtype == x.dtype
    if dtype == "float32":
        err = _refs.rel_err(y, ref.double().cpu())
        print(f"\n{name} n={n} {shape}: vs MMVQ rel {err:.1e}, bit-equal {(y == ref).float().mean().item():.3f}")
        assert err <= 1e-5
    else:
        _check(y, raw, name, x, mmq=False, lcpp=True)
        # 16-bit output (written by the dp4a IQ3 kernel, cast from fp32 by the others): equal
        # to the op's fp32 output cast by torch (same q8_1 bytes: 16-bit to float is exact)
        y32 = getattr(C, op)(w, x.cuda().float(), int(qt), w.shape[0])
        assert torch.equal(y, y32.to(y.dtype))


@pytest.mark.parametrize("n", [1, 4, 8])
@pytest.mark.parametrize("name", LCPP_TYPES)
def test_lcpp_x_q8(tensors_by_type, name, n):
    """The 1..8-row ops on X quantized beforehand (x_q8, as apply() shares one quantization
    among a layer's shard runs) return exactly what they return quantizing X themselves."""
    import torch

    C = _lcpp()
    _, x, w, qt = _lcpp_case(tensors_by_type, name, n, "bfloat16", seed=950 + n)
    x = x.cuda()
    q8 = C.lcpp_quantize_q8_1(x, qt, False, False)
    ops_ = [C.lcpp_mul_mat_vec_q] + [getattr(C, op) for t, op in OWNED if t == name]
    for op in ops_:
        assert torch.equal(op(w, x, qt, w.shape[0], q8), op(w, x, qt, w.shape[0]))


def test_quantize_x_q8_1_mixed_route():
    """A layer's runs share one q8_1 quantization of X even when the first run is not a Route L
    type (blk.13 gate_up: IQ1_M, stock path, then IQ2_*): the bytes are those of the Route L run."""
    import gguf
    import torch

    from vllm_gguf_plugin import ops
    from vllm_gguf_plugin.quantization.linear import _quantize_x_q8_1

    C = _lcpp()
    if not ops.LCPP_ENABLED:
        pytest.skip("needs VLLM_GGUF_LCPP=1")
    T = gguf.GGMLQuantizationType
    x = _x(4, 5120, "bfloat16", seed=77).cuda()
    q8 = _quantize_x_q8_1(x, [int(T.IQ1_M), int(T.IQ2_XS)], [5120, 5120])
    assert torch.equal(q8, C.lcpp_quantize_q8_1(x, int(T.IQ2_XS), False, False))
    # 8 rows: MMVQ is not routed there, the Q4_K kernel is (above 2048 weight rows)
    x = _x(8, 5120, "bfloat16", seed=78).cuda()
    q8 = _quantize_x_q8_1(x, [int(T.Q4_K)], [4096])
    assert torch.equal(q8, C.lcpp_quantize_q8_1(x, int(T.Q4_K), False, False))


@pytest.mark.parametrize("n", [1, 2, 4, 8, 9])
def test_unquantized_small_n(n):
    """A BF16 GGUF weight shaped like GDN in_proj_ba (96 x 5120): at <= 8 rows the product is a
    batched gemv, above that F.linear; both accumulate in fp32, so both sit within bf16 output
    rounding of the fp64 product, and the result is a contiguous [n, 96] bf16 tensor. A weight
    with more than 128 rows stays on F.linear (the gemv reads it once per row)."""
    import _refs
    import torch

    from vllm_gguf_plugin.quantization.linear import _unquantized_gemm

    g = torch.Generator().manual_seed(n)
    w = (torch.randn(96, 5120, generator=g) * 0.02).bfloat16().cuda()
    x = _x(n, 5120, "bfloat16", seed=n).cuda()
    y = _unquantized_gemm(x, w)
    ref = (x.double() @ w.double().T).cpu()
    assert y.shape == (n, 96) and y.dtype == torch.bfloat16 and y.is_contiguous()
    assert _refs.rel_err(y, ref) <= 4e-3
    big = torch.cat([w, w])  # 192 rows
    assert torch.equal(_unquantized_gemm(x, big), torch.nn.functional.linear(x, big))


@pytest.mark.parametrize("x_kind", ["bfloat16", "float16", "float32", "rowstride"])
@pytest.mark.parametrize("n", [1, 4, 9])
@pytest.mark.parametrize("mmq", [False, True], ids=["q8_1", "mmq"])
@pytest.mark.parametrize("name", LCPP_TYPES)
def test_lcpp_quantize_vs_vendored(name, mmq, n, x_kind):
    """The shim's own q8_1 quantizer (reads fp32/fp16/bf16 X) writes the same bytes as the
    vendored fp32 quantizers (quantize.cu) on X.float(): every quant, scale and partial sum,
    for MMVQ's block_q8_1 and in each type's MMQ ds layout (D4 / DS4 / D2S6)."""
    import gguf
    import torch

    C = _lcpp()
    k, qt = 5120, int(gguf.GGMLQuantizationType[name])
    x = _x(n, k, "bfloat16" if x_kind == "rowstride" else x_kind, seed=900 + n).cuda()
    x[0, 128:256] = 0  # all-zero blocks: the amax == 0 branches
    if x_kind == "rowstride":
        x = torch.cat([x, x[:, :512]], 1)[:, :k]  # row stride k + 512
    ours = C.lcpp_quantize_q8_1(x, qt, mmq, False)
    ref = C.lcpp_quantize_q8_1(x.float().contiguous(), qt, mmq, True)
    torch.cuda.synchronize()
    assert torch.equal(ours, ref)


@pytest.mark.parametrize("op_n", [("mmvq", n) for n in (1, 4, 8)] + [("mmq", n) for n in (1, 5, 8, 9, 128, 512)],
                         ids=lambda p: f"{p[0]}-{p[1]}")
@pytest.mark.parametrize("name", LCPP_TYPES)
def test_lcpp_graph_replay(tensors_by_type, name, op_n):
    """Captured on torch's (non-default) capture stream, replayed with new X: bit-exact with an
    eager call on the default stream, twice. A launch on any other stream would either fail the
    capture or leave static_y stale."""
    C = _lcpp()
    op, n = op_n
    _graph_replay(tensors_by_type, name, n, C.lcpp_mul_mat_vec_q if op == "mmvq" else C.lcpp_mul_mat_q)


@pytest.mark.parametrize("n", [1, 4, 6, 8])
@pytest.mark.parametrize("name,op", OWNED)
def test_lcpp_iq3_graph_replay(tensors_by_type, op, name, n):
    _graph_replay(tensors_by_type, name, n, getattr(_lcpp(), op))


def _graph_replay(tensors_by_type, name, n, fn):
    import torch

    _, x1, w, qt = _lcpp_case(tensors_by_type, name, n, "bfloat16", seed=600 + n)
    x2 = _x(n, x1.shape[1], "bfloat16", seed=700 + n).cuda()
    static_x = x1.cuda()
    fn(w, static_x, qt, w.shape[0])  # warm-up: cudaFuncSetAttribute runs on first use
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_y = fn(w, static_x, qt, w.shape[0])
    ref = fn(w, x2, qt, w.shape[0])
    for _ in range(2):
        static_x.copy_(x2)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(static_y, ref)
        static_x.copy_(x1.cuda())  # the next replay must see the copy again
    assert not torch.equal(fn(w, x1.cuda(), qt, w.shape[0]), ref)


def _routed(w, x, qt):
    """One product through the production routing (the kernel apply() picks for this type and n)."""
    from vllm_gguf_plugin.quantization.linear import _fused_mul_mat_gguf

    return _fused_mul_mat_gguf(x, w, qt)


def _padded_layer(shards, qts, k, monkeypatch):
    """A fused layer from GGUF shards through GGUFLinearMethod's padded-weight build."""
    import torch
    import vllm.model_executor.parameter as vparam  # GGUFWeightParameter asks for the TP rank

    from vllm_gguf_plugin.quantization.linear import GGUFLinearMethod

    monkeypatch.setattr(vparam, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(vparam, "get_tensor_model_parallel_world_size", lambda: 1)
    layer = torch.nn.Module()
    w = torch.nn.Parameter(torch.empty(0, dtype=torch.uint8, device="cuda"), requires_grad=False)
    ids = list(range(len(shards)))
    w.data_container, w.shard_id, w.shard_id_map = list(shards), ids, {i: i for i in ids}
    w.weight_loader, w.input_dim, w.output_dim = None, 1, 0
    w.tensor_shape = (sum(s.shape[0] for s in shards), k)
    layer.register_parameter("weight", w)
    layer.weight_type = type("WT", (), {"weight_type": qts[0], "shard_weight_type": dict(zip(ids, qts))})()
    method = GGUFLinearMethod(None)
    method._create_padded_weight_param(layer)
    return layer, method


def _mixed_block(gguf_reader, a, b, a_narrower=False):
    """First block whose tensors a and b have different Route L types (with a_narrower: a's
    rows hold fewer bytes than b's, so a's run is packed tighter than the padded row)."""
    by_name = {t.name: t for t in gguf_reader.tensors}
    for i in range(64):
        ta, tb = by_name.get(f"blk.{i}.{a}.weight"), by_name.get(f"blk.{i}.{b}.weight")
        if ta is not None and tb is not None and ta.tensor_type != tb.tensor_type \
                and {ta.tensor_type.name, tb.tensor_type.name} <= set(LCPP_TYPES) \
                and (not a_narrower or ta.data.shape[1] < tb.data.shape[1]):
            return i, ta, tb
    pytest.skip(f"no block with a mixed-type {a}/{b} pair on Route L types")


@pytest.mark.parametrize("n", [1, 4, 8, 9, 128])
def test_lcpp_mixed_shard_layer(tensors_by_type, gguf_reader, n, monkeypatch):
    """A fused gate/up layer whose shards have different quant types, through
    GGUFLinearMethod's padded-weight build and apply(): with VLLM_GGUF_LCPP=1 each shard is
    stored contiguously and passed as a view (no copy), and the result is bit-exact with the
    op on each shard alone."""
    import numpy as np
    import torch

    from vllm_gguf_plugin import ops

    _lcpp()  # skips without the lcpp build
    if not ops.LCPP_ENABLED:
        pytest.skip("needs VLLM_GGUF_LCPP=1")
    blk, *ts = _mixed_block(gguf_reader, "ffn_gate", "ffn_up")
    shards = [torch.from_numpy(np.ascontiguousarray(t.data)).cuda() for t in ts]
    qts = [int(t.tensor_type) for t in ts]
    assert shards[0].shape[1] != shards[1].shape[1]  # different row bytes: the padded case
    layer, method = _padded_layer(shards, qts, int(ts[0].shape[0]), monkeypatch)

    from vllm_gguf_plugin.quantization.linear import _shard_weight

    padded = layer.weight
    lo, hi = padded.data_ptr(), padded.data_ptr() + padded.numel()
    for i, s in enumerate(shards):
        v = _shard_weight(padded, *padded.shard_offset_map[i])
        assert lo <= v.data_ptr() < hi and torch.equal(v, s)  # a view with the shard's bytes

    x = _x(n, int(ts[0].shape[0]), "bfloat16", seed=800 + n).cuda()
    y = method.apply(layer, x)
    ref = torch.cat([_routed(s, x, q) for s, q in zip(shards, qts)], dim=1)
    torch.cuda.synchronize()
    print(f"\nblk.{blk} gate {ts[0].tensor_type.name} + up {ts[1].tensor_type.name}, n={n}")
    assert torch.equal(y, ref)


@pytest.mark.parametrize("a_narrower", [False, True], ids=["qkv_widest", "qkv_narrower"])
@pytest.mark.parametrize("n", [1, 4, 8, 9, 128])
def test_lcpp_same_type_run(tensors_by_type, gguf_reader, n, a_narrower, monkeypatch):
    """GDN in_proj_qkvz: shards q, k, v are row slices of one attn_qkv tensor (one type) and z is
    attn_gate (another type). apply() runs one product for the q/k/v run and one for z. Against
    the routed op on each of the four shards alone it is bit-exact through MMVQ and the IQ3
    kernel (rows are independent); through MMQ, stream-k splits K differently for a 10240-row
    than a 2048-row product, so the fp32 partial sums add in another order and ~1 bf16 ulp can
    flip (measured max 0.03)."""
    import numpy as np
    import torch

    from vllm_gguf_plugin import ops

    _lcpp()  # skips without the lcpp build
    if not ops.LCPP_ENABLED:
        pytest.skip("needs VLLM_GGUF_LCPP=1")
    import _refs

    blk, tqkv, tz = _mixed_block(gguf_reader, "attn_qkv", "attn_gate", a_narrower)
    qkv = torch.from_numpy(np.ascontiguousarray(tqkv.data)).cuda()
    z = torch.from_numpy(np.ascontiguousarray(tz.data)).cuda()
    shards = [qkv[:2048], qkv[2048:4096], qkv[4096:], z]
    qts = [int(tqkv.tensor_type)] * 3 + [int(tz.tensor_type)]
    layer, method = _padded_layer(shards, qts, int(tqkv.shape[0]), monkeypatch)

    x = _x(n, int(tqkv.shape[0]), "bfloat16", seed=850 + n).cuda()
    y = method.apply(layer, x)
    per_shard = torch.cat([_routed(s, x, q) for s, q in zip(shards, qts)], dim=1)
    whole = torch.cat([_routed(qkv, x, qts[0]), _routed(z, x, qts[3])], dim=1)
    torch.cuda.synchronize()
    print(f"\nblk.{blk} qkv {tqkv.tensor_type.name} + z {tz.tensor_type.name}, n={n}: "
          f"max |run - per shard| {(y.float() - per_shard.float()).abs().max().item():.3g}")
    assert torch.equal(y, whole)  # one product per run
    if n <= 8:  # MMVQ / the IQ3 kernel (this layer is IQ3_S + IQ3_XXS)
        assert torch.equal(y, per_shard)
    else:
        assert _refs.rel_err(y, per_shard.double().cpu()) <= 1e-3
