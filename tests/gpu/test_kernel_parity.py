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
capture + replay must be bit-exact with an eager call. Run the file with VLLM_GGUF_LCPP=1 and
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
    import torch

    C = _lcpp()
    op, n = op_n
    fn = C.lcpp_mul_mat_vec_q if op == "mmvq" else C.lcpp_mul_mat_q
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


def _mixed_block(gguf_reader, a, b):
    """First block whose tensors a and b have different Route L types."""
    by_name = {t.name: t for t in gguf_reader.tensors}
    for i in range(64):
        ta, tb = by_name.get(f"blk.{i}.{a}.weight"), by_name.get(f"blk.{i}.{b}.weight")
        if ta is not None and tb is not None and ta.tensor_type != tb.tensor_type \
                and {ta.tensor_type.name, tb.tensor_type.name} <= set(LCPP_TYPES):
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

    C = _lcpp()
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
    fn = C.lcpp_mul_mat_vec_q if n < 8 else C.lcpp_mul_mat_q
    ref = torch.cat([fn(s, x, q, s.shape[0]) for s, q in zip(shards, qts)], dim=1)
    torch.cuda.synchronize()
    print(f"\nblk.{blk} gate {ts[0].tensor_type.name} + up {ts[1].tensor_type.name}, n={n}")
    assert torch.equal(y, ref)


@pytest.mark.parametrize("n", [1, 4, 8, 9, 128])
def test_lcpp_same_type_run(tensors_by_type, gguf_reader, n, monkeypatch):
    """GDN in_proj_qkvz: shards q, k, v are row slices of one attn_qkv tensor (one type) and z is
    attn_gate (another type). apply() runs one product for the q/k/v run and one for z. Against
    the op on each of the four shards alone it is bit-exact through MMVQ (rows are independent);
    through MMQ, stream-k splits K differently for a 10240-row than a 2048-row product, so the
    fp32 partial sums add in another order and ~1 bf16 ulp can flip (measured max 0.03)."""
    import numpy as np
    import torch

    from vllm_gguf_plugin import ops

    C = _lcpp()
    if not ops.LCPP_ENABLED:
        pytest.skip("needs VLLM_GGUF_LCPP=1")
    import _refs

    blk, tqkv, tz = _mixed_block(gguf_reader, "attn_qkv", "attn_gate")
    qkv = torch.from_numpy(np.ascontiguousarray(tqkv.data)).cuda()
    z = torch.from_numpy(np.ascontiguousarray(tz.data)).cuda()
    shards = [qkv[:2048], qkv[2048:4096], qkv[4096:], z]
    qts = [int(tqkv.tensor_type)] * 3 + [int(tz.tensor_type)]
    layer, method = _padded_layer(shards, qts, int(tqkv.shape[0]), monkeypatch)

    x = _x(n, int(tqkv.shape[0]), "bfloat16", seed=850 + n).cuda()
    fn = C.lcpp_mul_mat_vec_q if n < 8 else C.lcpp_mul_mat_q
    y = method.apply(layer, x)
    per_shard = torch.cat([fn(s, x, q, s.shape[0]) for s, q in zip(shards, qts)], dim=1)
    whole = torch.cat([fn(qkv, x, qts[0], qkv.shape[0]), fn(z, x, qts[3], z.shape[0])], dim=1)
    torch.cuda.synchronize()
    print(f"\nblk.{blk} qkv {tqkv.tensor_type.name} + z {tz.tensor_type.name}, n={n}: "
          f"max |run - per shard| {(y.float() - per_shard.float()).abs().max().item():.3g}")
    assert torch.equal(y, whole)  # one product per run
    if n < 8:
        assert torch.equal(y, per_shard)
    else:
        assert _refs.rel_err(y, per_shard.double().cpu()) <= 1e-3
