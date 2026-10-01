# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The lcpp ops (csrc/lcpp_shim.cu; built with VLLM_GGUF_BUILD_LCPP=1, skipped
otherwise) against CPU reference models (tests/kernel_refs.py) and against each
other, with the caching allocator's free blocks poisoned first so that an
unzeroed scratch read shows up. CUDA-graph capture and replay must be bit-exact
with an eager call. The routing tests (apply(), mixed-type layers) need
VLLM_GGUF_LCPP=1.

Weights are blocks of the sample GGUFs (kernel_refs.sample_weight) in the
shapes each kernel needs; K = 5120 unless noted.
"""

import gguf
import pytest
import torch

from .kernel_refs import (
    LOOSE_XSUM,
    check,
    make_x,
    poison_cuda_allocator,
    rel_err,
    sample_weight,
)

if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)

LCPP_TYPES = [
    "IQ2_S",
    "IQ2_XS",
    "IQ2_XXS",
    "IQ3_S",
    "IQ3_XXS",
    "IQ4_XS",
    "Q2_K",
    "Q4_K",
    "Q6_K",
]
ROWS, BLOCKS = 512, 20
LCPP_MMVQ_TOKENS = [1, 2, 3, 4, 5, 6, 7, 8]
# 1..7: MMQ below upstream's J_max read tail (only the shim's zeroed tail keeps
# the reads defined); 128 and 2048: prefill chunks
LCPP_MMQ_TOKENS = [1, 2, 3, 5, 7, 8, 9, 16, 64, 128, 512, 2048]
IQ3_TYPES = ["IQ3_S", "IQ3_XXS"]
IQ3_OPS = ["lcpp_mul_mat_vec_iq3", "lcpp_mul_mat_vec_iq3_mma"]
OWN_TYPES = ["Q4_K", "IQ2_S"]
OWNED = [(t, op) for op in IQ3_OPS for t in IQ3_TYPES] + [
    (t, "lcpp_mul_mat_vec_own") for t in OWN_TYPES
]


def _qt(name: str) -> int:
    return int(gguf.GGMLQuantizationType[name])


def _lcpp():
    from vllm_gguf_plugin import ops  # noqa: F401  (loads _C_gguf)

    if not hasattr(torch.ops._C_gguf, "lcpp_mul_mat_q"):
        pytest.skip("_C_gguf built without VLLM_GGUF_BUILD_LCPP=1")
    return torch.ops._C_gguf


def _routing():
    from vllm_gguf_plugin import ops

    _lcpp()
    if not ops.LCPP_ENABLED:
        pytest.skip("needs VLLM_GGUF_LCPP=1")


def _case(name, n, dtype, seed, rows=ROWS, blocks=BLOCKS):
    raw = sample_weight(name, rows, blocks)
    x = make_x(n, blocks * 256, dtype, seed=seed)
    return raw, x, torch.from_numpy(raw).cuda(), _qt(name)


def _graph_replay(name, n, fn, rows=ROWS, prep=None):
    """Captured on torch's capture stream, replayed with new X: bit-exact with
    an eager call on the default stream, twice. A launch on another stream
    would fail the capture or leave static_y stale."""
    _, x1, w, qt = _case(name, n, torch.bfloat16, 600 + n, rows=rows)
    if prep is not None:  # e.g. pack W
        w = prep(w, qt)
    x2 = make_x(n, x1.shape[1], torch.bfloat16, seed=700 + n).cuda()
    static_x = x1.cuda()
    fn(w, static_x, qt, w.shape[0])  # warm-up: first-use attribute setup
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
        static_x.copy_(x1.cuda())
    assert not torch.equal(fn(w, x1.cuda(), qt, w.shape[0]), ref)


# ------------------------------------------------ vendored MMVQ / MMQ


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=str)
@pytest.mark.parametrize("n", LCPP_MMVQ_TOKENS)
@pytest.mark.parametrize("name", LCPP_TYPES)
def test_lcpp_mmvq(name, n, dtype):
    C = _lcpp()
    raw, x, w, qt = _case(name, n, dtype, 300 + n)
    poison_cuda_allocator()
    y = C.lcpp_mul_mat_vec_q(w, x.cuda(), qt, w.shape[0])
    assert y.shape == (n, ROWS) and y.dtype == dtype
    check(y, raw, name, x, mmq=False, lcpp=True)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=str)
@pytest.mark.parametrize("n", LCPP_MMQ_TOKENS)
@pytest.mark.parametrize("name", LCPP_TYPES)
def test_lcpp_mmq(name, n, dtype):
    C = _lcpp()
    raw, x, w, qt = _case(name, n, dtype, 400 + n)
    poison_cuda_allocator()
    y = C.lcpp_mul_mat_q(w, x.cuda(), qt, w.shape[0])
    assert y.shape == (n, ROWS) and y.dtype == dtype
    check(y, raw, name, x, mmq=True, lcpp=True)


@pytest.mark.parametrize("n", [5, 128])
@pytest.mark.parametrize("name", LCPP_TYPES)
def test_lcpp_mmq_odd_rows(name, n):
    """W rows not a multiple of 128: MMQ's fallback tiles."""
    C = _lcpp()
    raw, x, w, qt = _case(name, n, torch.bfloat16, 500 + n, rows=200)
    poison_cuda_allocator()
    y = C.lcpp_mul_mat_q(w, x.cuda(), qt, w.shape[0])
    check(y, raw, name, x, mmq=True, lcpp=True)


@pytest.mark.parametrize(
    "op_n",
    [("mmvq", n) for n in (1, 4, 8)] + [("mmq", n) for n in (1, 5, 8, 9, 128, 512)],
    ids=lambda p: f"{p[0]}-{p[1]}",
)
@pytest.mark.parametrize("name", LCPP_TYPES)
def test_lcpp_graph_replay(name, op_n):
    C = _lcpp()
    op, n = op_n
    fn = C.lcpp_mul_mat_vec_q if op == "mmvq" else C.lcpp_mul_mat_q
    _graph_replay(name, n, fn)


def _routed(w, x, qt):
    """One product through linear.py's routing."""
    from vllm_gguf_plugin.quantization.linear import _fused_mul_mat_gguf

    return _fused_mul_mat_gguf(x, w, qt)


def _padded_layer(shards, qts, k, monkeypatch):
    """A fused layer from GGUF shards through GGUFLinearMethod's padded-weight
    build."""
    import vllm.model_executor.parameter as vparam

    from vllm_gguf_plugin.quantization.linear import GGUFLinearMethod

    monkeypatch.setattr(vparam, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(vparam, "get_tensor_model_parallel_world_size", lambda: 1)
    layer = torch.nn.Module()
    w = torch.nn.Parameter(
        torch.empty(0, dtype=torch.uint8, device="cuda"), requires_grad=False
    )
    ids = list(range(len(shards)))
    w.data_container, w.shard_id, w.shard_id_map = (
        list(shards),
        ids,
        {i: i for i in ids},
    )
    w.weight_loader, w.input_dim, w.output_dim = None, 1, 0
    w.tensor_shape = (sum(s.shape[0] for s in shards), k)
    layer.register_parameter("weight", w)
    wt = {"weight_type": qts[0], "shard_weight_type": dict(zip(ids, qts))}
    layer.weight_type = type("WT", (), wt)()
    method = GGUFLinearMethod(None)
    method._create_padded_weight_param(layer)
    return layer, method


@pytest.mark.parametrize(
    "types",
    [("IQ4_XS", "Q4_K"), ("IQ3_XXS", "IQ2_S"), ("Q6_K", "IQ3_S")],
    ids="+".join,
)
@pytest.mark.parametrize("n", [1, 4, 6, 7, 8, 9, 32, 128])
def test_lcpp_mixed_shard_layer(n, types, monkeypatch):
    """A fused gate/up layer whose two shards have different quant types (and
    row bytes), through the padded-weight build and apply(): each shard is
    stored contiguously and passed as a view (no copy), and the result is
    bit-exact with the routed op on each shard alone."""
    from vllm_gguf_plugin.quantization.linear import _shard_weight

    _routing()
    shards = [torch.from_numpy(sample_weight(t, 17408, BLOCKS)).cuda() for t in types]
    qts = [_qt(t) for t in types]
    layer, method = _padded_layer(shards, qts, BLOCKS * 256, monkeypatch)
    padded = layer.weight
    lo, hi = padded.data_ptr(), padded.data_ptr() + padded.numel()
    for i, s in enumerate(shards):
        v = _shard_weight(padded, *padded.shard_offset_map[i])
        assert lo <= v.data_ptr() < hi and torch.equal(v, s)
    x = make_x(n, BLOCKS * 256, torch.bfloat16, seed=800 + n).cuda()
    y = method.apply(layer, x)
    ref = torch.cat([_routed(s, x, q) for s, q in zip(shards, qts)], dim=1)
    assert torch.equal(y, ref)


def _qkvz(types):
    """GDN in_proj_qkvz-like shards: q, k, v row slices of one tensor of
    types[0] (2048 + 2048 + 6144 rows) and z of types[1] (6144 rows)."""
    qkv = torch.from_numpy(sample_weight(types[0], 10240, BLOCKS)).cuda()
    z = torch.from_numpy(sample_weight(types[1], 6144, BLOCKS, seed=1)).cuda()
    return qkv, z, [qkv[:2048], qkv[2048:4096], qkv[4096:], z]


@pytest.mark.parametrize(
    "types",
    [("IQ3_XXS", "Q4_K"), ("Q4_K", "IQ3_S"), ("IQ2_S", "IQ4_XS")],
    ids="+".join,
)
@pytest.mark.parametrize("n", [1, 4, 6, 7, 8, 9, 32, 128])
def test_lcpp_same_type_run(n, types, monkeypatch):
    """apply() runs one product for the q/k/v run and one for z, bit-exact with
    the routed op on the run's whole bytes. Against the four shards alone it is
    bit-exact where each shard takes its run's kernel at <= 8 rows (rows are
    independent); where MMQ splits K differently for the run than for a shard
    (stream-k), fp32 partial sums add in another order: within 1e-3; where a
    shard takes MMQ with its Q4_K min term and the run another kernel, within
    LOOSE_XSUM."""
    from vllm_gguf_plugin.quantization.linear import _lcpp_op

    _routing()
    qkv, z, shards = _qkvz(types)
    qts = [_qt(types[0])] * 3 + [_qt(types[1])]
    layer, method = _padded_layer(shards, qts, BLOCKS * 256, monkeypatch)
    x = make_x(n, BLOCKS * 256, torch.bfloat16, seed=850 + n).cuda()
    y = method.apply(layer, x)
    whole = torch.cat([_routed(qkv, x, qts[0]), _routed(z, x, qts[3])], dim=1)
    assert torch.equal(y, whole)
    per_shard = torch.cat([_routed(s, x, q) for s, q in zip(shards, qts)], dim=1)
    run_rows = [qkv.shape[0]] * 3 + [z.shape[0]]
    pairs = [
        (_lcpp_op(n, q, s.shape[0]), _lcpp_op(n, q, r))
        for s, q, r in zip(shards, qts, run_rows)
    ]
    if n <= 8 and all(a == b for a, b in pairs):
        assert torch.equal(y, per_shard)
    elif any(a != b and "lcpp_mul_mat_q" in (a, b) for a, b in pairs):
        assert rel_err(y, per_shard.double().cpu()) <= LOOSE_XSUM
    else:
        assert rel_err(y, per_shard.double().cpu()) <= 1e-3


# ------------------------------------------------ owned q8_1 quantizer


@pytest.mark.parametrize("x_kind", ["bfloat16", "float16", "float32", "rowstride"])
@pytest.mark.parametrize("n", [1, 4, 9])
@pytest.mark.parametrize("mmq", [False, True], ids=["q8_1", "mmq"])
@pytest.mark.parametrize("name", LCPP_TYPES)
def test_lcpp_quantize_vs_vendored(name, mmq, n, x_kind):
    """The shim's q8_1 quantizer (reads fp32 / fp16 / bf16 X) writes the same
    bytes as the vendored fp32 quantizers (quantize.cu) on X.float(): every
    quant, scale and partial sum, for MMVQ's block_q8_1 and each type's MMQ ds
    layout (D4 / DS4 / D2S6)."""
    C = _lcpp()
    k = BLOCKS * 256
    dtype = getattr(torch, "bfloat16" if x_kind == "rowstride" else x_kind)
    x = make_x(n, k, dtype, seed=900 + n).cuda()
    x[0, 128:256] = 0  # all-zero blocks: the amax == 0 branches
    if x_kind == "rowstride":
        x = torch.cat([x, x[:, :512]], 1)[:, :k]  # row stride k + 512
    ours = C.lcpp_quantize_q8_1(x, _qt(name), mmq, False)
    ref = C.lcpp_quantize_q8_1(x.float().contiguous(), _qt(name), mmq, True)
    assert torch.equal(ours, ref)


# ------------------------------------------------ owned 1..8-row kernels


@pytest.mark.parametrize(
    "shape",
    ["real", "row_tail", "k_tail", "odd_rows", "k_min", "few_rows", "many_tiles"],
)
@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float16, torch.float32], ids=str
)
@pytest.mark.parametrize("n", LCPP_MMVQ_TOKENS)
@pytest.mark.parametrize("name,op", OWNED)
def test_lcpp_owned_vec(op, name, n, dtype, shape):
    """The owned 1..8-row kernels take MMVQ's q8_1 input and compute each
    32-value slice's scaled integer sum exactly as the vendored vec_dot; only
    the fp32 order of a row's slice terms differs. 16-bit X: the CPU reference
    models, and the 16-bit output equals the op's fp32 output cast by torch.
    fp32 X: within 1e-5 of vendored MMVQ. Shapes (512 rows, K 5120 unless
    noted): row_tail 202 rows (the last 16-row CTA part-filled; even, because
    the reference MMVQ reads one row past W at an odd count); k_tail K 4608
    (the last staged chunk 16 q8_1 blocks, not 32); odd_rows 201 rows of K
    4608, W's size 4 mod 8 (fp32 reference for the mma kernel: the dp4a one);
    k_min K 512 (the mma kernel's warps 2 and 3 get no block); few_rows 20
    rows (fewer tiles than CTAs); many_tiles 8192 rows (each CTA takes several;
    fp32, and bf16 for lcpp_mul_mat_vec_own, whose routing starts above 2048
    rows)."""
    C = _lcpp()
    if not hasattr(C, op):
        pytest.skip(f"{op} not built")
    own_bf16 = op == "lcpp_mul_mat_vec_own" and dtype == torch.bfloat16
    if shape == "many_tiles" and dtype != torch.float32 and not own_bf16:
        pytest.skip("fp32 only: the reference is MMVQ on the GPU")
    if shape == "odd_rows" and op == "lcpp_mul_mat_vec_own" and dtype == torch.float32:
        pytest.skip("the fp32 reference, MMVQ, reads past this W")
    rows = {"row_tail": 202, "odd_rows": 201, "few_rows": 20, "many_tiles": 8192}
    blocks = {"k_tail": 18, "odd_rows": 18, "k_min": 2}.get(shape, BLOCKS)
    raw, x, w, qt = _case(name, n, dtype, 1000 + n, rows.get(shape, ROWS), blocks)
    poison_cuda_allocator()
    y = getattr(C, op)(w, x.cuda(), qt, w.shape[0])
    assert y.shape == (n, raw.shape[0]) and y.dtype == dtype
    if dtype == torch.float32:
        mma_odd = shape == "odd_rows" and op.endswith("_mma")
        ref_op = C.lcpp_mul_mat_vec_iq3 if mma_odd else C.lcpp_mul_mat_vec_q
        ref = ref_op(w, x.cuda(), qt, w.shape[0])
        assert rel_err(y, ref.double().cpu()) <= 1e-5
    else:
        check(y, raw, name, x, mmq=False, lcpp=True)
        y32 = getattr(C, op)(w, x.cuda().float(), qt, w.shape[0])
        assert torch.equal(y, y32.to(y.dtype))


@pytest.mark.parametrize("n", [1, 4, 6, 8])
@pytest.mark.parametrize("name,op", OWNED)
def test_lcpp_owned_vec_graph_replay(op, name, n):
    C = _lcpp()
    if not hasattr(C, op):
        pytest.skip(f"{op} not built")
    _graph_replay(name, n, getattr(C, op))
