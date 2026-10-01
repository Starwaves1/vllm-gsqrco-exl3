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

import os
import subprocess
import sys

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
    "IQ1_M",
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
LCPP_MMQ_TYPES = [q for q in LCPP_TYPES if q != "IQ1_M"]  # no IQ1_M MMQ
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
MMA_K_TYPES = ["Q4_K", "IQ4_XS", "IQ2_S"]
# 2 / 4 / 8 column tiles, full and part-filled; 1: the op's lower bound
MMA_K_TOKENS = [1, 9, 16, 17, 32, 33, 64]


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
@pytest.mark.parametrize("name", LCPP_MMQ_TYPES)
def test_lcpp_mmq(name, n, dtype):
    C = _lcpp()
    raw, x, w, qt = _case(name, n, dtype, 400 + n)
    poison_cuda_allocator()
    y = C.lcpp_mul_mat_q(w, x.cuda(), qt, w.shape[0])
    assert y.shape == (n, ROWS) and y.dtype == dtype
    check(y, raw, name, x, mmq=True, lcpp=True)


@pytest.mark.parametrize("n", [5, 128])
@pytest.mark.parametrize("name", LCPP_MMQ_TYPES)
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
    if op == "mmq" and name not in LCPP_MMQ_TYPES:
        pytest.skip(f"no {name} MMQ")
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
    k = x.shape[1]
    run_rows = [qkv.shape[0]] * 3 + [z.shape[0]]
    pairs = [
        (_lcpp_op(n, q, s.shape[0], k), _lcpp_op(n, q, r, k))
        for s, q, r in zip(shards, qts, run_rows)
    ]
    if n <= 8 and all(a == b for a, b in pairs):
        assert torch.equal(y, per_shard)
    elif any(
        a != b and "lcpp_mul_mat_q" in (a, b) and "lcpp_mul_mat_mma_k" not in (a, b)
        for a, b in pairs
    ):
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
    if mmq and name not in LCPP_MMQ_TYPES:
        pytest.skip(f"no {name} MMQ")
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


# ------------------------------------------------ owned 9..64-row mma_k


def _within_1ulp(y, ref):
    """16-bit y vs vendored MMQ's output in the same dtype: both round an fp32
    sum that differs only in the order of its terms, so they may be 1 ulp
    apart, never more, except where the sum cancels to near zero relative to
    the output's rms."""
    a, b = y.float(), ref.float()
    ulp = torch.where(
        b == 0, torch.zeros_like(b), (b.abs().frexp().exponent - 1).float().exp2()
    )
    ulp = ulp * (2.0**-7 if y.dtype == torch.bfloat16 else 2.0**-10)
    cancel = b.abs() < 1e-3 * b.pow(2).mean().sqrt()
    assert not (((a - b).abs() > ulp) & ~cancel).any()


@pytest.mark.parametrize(
    "shape", ["real", "row_tail", "k_tail", "down", "no_pieces", "big_tail"]
)
@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float16, torch.float32], ids=str
)
@pytest.mark.parametrize("n", MMA_K_TOKENS)
@pytest.mark.parametrize("name", MMA_K_TYPES)
def test_lcpp_mma_k(name, n, dtype, shape):
    """lcpp_mul_mat_mma_k takes MMQ's q8_1 layout and computes each slice's
    term with the vendored MMQ vec_dot's expression; only the fp32 order of the
    K sum differs. 16-bit X: the CPU reference models, and within 1 ulp of MMQ.
    fp32 X: within 1e-5 of MMQ. Shapes (512 rows, K 5120 unless noted):
    row_tail 202 rows (the last tile part-filled); k_tail K 4608; down K 17408;
    no_pieces 10496 rows at 9..16 columns (on an 82-SM GPU every CTA covers
    whole tiles: no fixup); big_tail 17398 rows (a part-filled last tile the
    last CTA covers whole there)."""
    C = _lcpp()
    if shape == "no_pieces" and not 9 <= n <= 16:
        pytest.skip("the no-fixup layout is for 2 column tiles")
    rows = {"row_tail": 202, "no_pieces": 10496, "big_tail": 17398}.get(shape, ROWS)
    blocks = {"k_tail": 18, "down": 68}.get(shape, BLOCKS)
    raw, x, w, qt = _case(name, n, dtype, 1100 + n, rows, blocks)
    poison_cuda_allocator()
    y = C.lcpp_mul_mat_mma_k(w, x.cuda(), qt, w.shape[0])
    ref = C.lcpp_mul_mat_q(w, x.cuda(), qt, w.shape[0])
    assert y.shape == (n, rows) and y.dtype == dtype
    if dtype == torch.float32:
        assert rel_err(y, ref.double().cpu()) <= 1e-5
    else:
        if rows <= ROWS:
            check(y, raw, name, x, mmq=True, lcpp=True)
        _within_1ulp(y, ref)


@pytest.mark.parametrize("n", [16, 32, 64])
@pytest.mark.parametrize(
    "shape", [(17408, 20), (5120, 68)], ids=lambda p: f"{p[0]}x{p[1] * 256}"
)
@pytest.mark.parametrize("name", MMA_K_TYPES)
def test_lcpp_mma_k_whole_tensor(name, shape, n):
    """Whole 17408 x 5120 and 5120 x 17408 weights (272 / 80 tiles, 20 / 68 K
    steps, shared by the resident CTAs), fp32 X: within 2e-6 of MMQ."""
    C = _lcpp()
    rows, blocks = shape
    _, x, w, qt = _case(name, n, torch.float32, 1200 + n, rows, blocks)
    x = x.cuda()
    poison_cuda_allocator()
    y = C.lcpp_mul_mat_mma_k(w, x, qt, rows)
    ref = C.lcpp_mul_mat_q(w, x, qt, rows)
    assert rel_err(y, ref.double().cpu()) <= 2e-6


@pytest.mark.parametrize("rows", [ROWS, 10496, 17408])
@pytest.mark.parametrize("n", [16, 33, 64])
@pytest.mark.parametrize("name", MMA_K_TYPES)
def test_lcpp_mma_k_graph_replay(name, n, rows):
    """512 rows (every tile shared by CTAs), 10496 (at 16 columns every CTA
    covers whole tiles), 17408."""
    _graph_replay(name, n, _lcpp().lcpp_mul_mat_mma_k, rows=rows)


_FIRST_CALL_IN_CAPTURE = r"""
import torch
from tests.kernel_refs import sample_weight
from vllm_gguf_plugin import ops  # noqa: F401  (loads _C_gguf, no CUDA call)
C = torch.ops._C_gguf
torch.zeros(1, device="cuda")  # a CUDA context, but no _C_gguf call before capture
fails = []
for name, qt in (("Q4_K", 12), ("IQ4_XS", 23), ("IQ2_S", 22)):
    for rows, n in ((17408, 16), (512, 17), (17408, 64)):
        w = torch.from_numpy(sample_weight(name, rows, 20)).cuda()
        g = torch.Generator().manual_seed(n)
        x1 = torch.randn(n, 5120, generator=g).bfloat16().cuda()
        x2 = torch.randn(n, 5120, generator=g).bfloat16().cuda()
        static_x = x1.clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_y = C.lcpp_mul_mat_mma_k(w, static_x, qt, rows)
        static_x.copy_(x2)
        graph.replay()
        torch.cuda.synchronize()
        if not torch.equal(static_y, C.lcpp_mul_mat_mma_k(w, x2, qt, rows)):
            fails.append(f"{name} rows={rows} n={n}")
print("FAILS", fails)
"""


def test_lcpp_mma_k_first_call_in_capture():
    """Each kernel instance's first call (its one-time cudaFuncSetAttribute,
    the shim's first device query) inside a CUDA-graph capture, in a fresh
    process: the capture must succeed and its replay equal an eager call."""
    _lcpp()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(
        os.environ, PYTHONPATH=os.pathsep.join([root, os.environ.get("PYTHONPATH", "")])
    )
    out = subprocess.run(
        [sys.executable, "-c", _FIRST_CALL_IN_CAPTURE],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
        cwd=root,
    )
    assert out.returncode == 0, out.stderr[-3000:]
    assert "FAILS []" in out.stdout, out.stdout[-2000:]


# ------------------------------------------------ packed IQ3

PACKED_TOKENS = list(range(1, 33))  # every fill of the 1, 2 and 4 column groups
TILED_RTOL = 1e-5  # lcpp_mul_mat_iq3_packed vs MMQ: fp32 reordering only
# every tile width (16 / 32 / 48 / 64 columns) full and part-filled, prefill
# chunks and a mixed step (129)
TILED_TOKENS = [1, 8, 16, 17, 32, 33, 48, 64, 65, 96, 128, 129, 200, 512, 2048]


def _packed(w, qt):
    from vllm_gguf_plugin.quantization import iq3_pack

    return iq3_pack.pack(w, qt)


@pytest.mark.parametrize("name", IQ3_TYPES)
def test_iq3_pack_roundtrip_cuda(name):
    """pack on the GPU equals pack on the CPU and round-trips."""
    from vllm_gguf_plugin.quantization import iq3_pack

    qt = _qt(name)
    w = torch.from_numpy(sample_weight(name, 18944, 4)).cuda()
    p = iq3_pack.pack(w, qt)
    assert p.shape == w.shape and not torch.equal(p, w)
    assert torch.equal(iq3_pack.unpack(p, qt), w)
    assert torch.equal(iq3_pack.pack(w[:256].cpu(), qt), p[:256].cpu())


@pytest.mark.parametrize("rows", [16, 17408])
@pytest.mark.parametrize("name", IQ3_TYPES)
def test_iq3_pack_inplace_peak(name, rows):
    """pack_ (the load path) equals pack, and its GPU scratch is at most
    min(tensor bytes, CHUNK_BYTES)."""
    from vllm_gguf_plugin.quantization import iq3_pack

    qt = _qt(name)
    w = torch.from_numpy(sample_weight(name, rows, BLOCKS)).cuda()
    ref = iq3_pack.pack(w, qt)
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    iq3_pack.pack_(w, qt)
    torch.cuda.synchronize()
    scratch = torch.cuda.max_memory_allocated() - base
    assert scratch <= min(w.numel(), iq3_pack.CHUNK_BYTES), (scratch, w.numel())
    assert torch.equal(w, ref)


@pytest.mark.parametrize("shape", ["real", "k_tail", "k_min", "few_rows", "many_tiles"])
@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float16, torch.float32], ids=str
)
@pytest.mark.parametrize("n", PACKED_TOKENS)
@pytest.mark.parametrize("name", IQ3_TYPES)
def test_lcpp_iq3_packed_vec(name, n, dtype, shape):
    """lcpp_mul_mat_vec_iq3_mma_packed on packed W is bit-exact with
    lcpp_mul_mat_vec_iq3_mma on the GGUF bytes, 8 activation rows at a time.
    k_tail K 4608; k_min K 512; few_rows 32 rows; many_tiles 8192 rows."""
    C = _lcpp()
    rows = {"few_rows": 32, "many_tiles": 8192}.get(shape, ROWS)
    blocks = {"k_tail": 18, "k_min": 2}.get(shape, BLOCKS)
    raw, x, w, qt = _case(name, n, dtype, 1100 + n, rows, blocks)
    x = x.cuda()
    p = _packed(w, qt)
    poison_cuda_allocator()
    y = C.lcpp_mul_mat_vec_iq3_mma_packed(p, x, qt, p.shape[0])
    ref = torch.cat(
        [C.lcpp_mul_mat_vec_iq3_mma(w, x[i : i + 8], qt, rows) for i in range(0, n, 8)]
    )
    assert y.shape == (n, rows) and y.dtype == dtype
    assert torch.equal(y, ref)
    if dtype != torch.float32 and shape == "real":
        check(y, raw, name, x.cpu(), mmq=False, lcpp=True)


@pytest.mark.parametrize("n", [1, 4, 8, 16, 32])
@pytest.mark.parametrize("name", IQ3_TYPES)
def test_lcpp_iq3_packed_vec_graph_replay(name, n):
    fn = _lcpp().lcpp_mul_mat_vec_iq3_mma_packed
    _graph_replay(name, n, fn, prep=_packed)


def _close_to_fp32(y, ref32, rtol):
    """y is ref32 rounded to y's dtype after an fp32 reordering error of at
    most rtol * max|ref32|."""
    ref32 = ref32.float()
    err = (y.float() - ref32).abs()
    half_ulp = 0 if y.dtype == torch.float32 else torch.finfo(y.dtype).eps / 2
    tol = half_ulp * ref32.abs() + rtol * ref32.abs().max()
    assert (err <= tol).all(), f"max err / tol {(err / tol).max().item():.3g}"


@pytest.mark.parametrize("shape", ["real", "rows_208", "k_tail", "k_min", "many_tiles"])
@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float16, torch.float32], ids=str
)
@pytest.mark.parametrize("n", TILED_TOKENS)
@pytest.mark.parametrize("name", IQ3_TYPES)
def test_lcpp_iq3_packed_tiled(name, n, dtype, shape):
    """lcpp_mul_mat_iq3_packed (tiled, packed W, any rows) against vendored MMQ
    on the GGUF bytes: the same q8_1 input and per-slice fp32 term, added in K
    order, so the fp32 results differ only where one of the two splits a
    tile's K range (TILED_RTOL); 16-bit outputs are that rounded. rows_208: 13
    16-row tiles; k_tail K 4608; k_min K 512 (every tile split over CTAs);
    many_tiles 8192 rows (whole-tile waves, then a split tail)."""
    C = _lcpp()
    rows = {"rows_208": 208, "many_tiles": 8192}.get(shape, ROWS)
    blocks = {"k_tail": 18, "k_min": 2}.get(shape, BLOCKS)
    raw, x, w, qt = _case(name, n, dtype, 1400 + n, rows, blocks)
    x = x.cuda()
    p = _packed(w, qt)
    poison_cuda_allocator()
    y = C.lcpp_mul_mat_iq3_packed(p, x, qt, rows)
    ref32 = C.lcpp_mul_mat_q(w, x.float(), qt, rows)  # the same q8_1 bytes
    assert y.shape == (n, rows) and y.dtype == dtype
    _close_to_fp32(y, ref32, TILED_RTOL)
    if dtype != torch.float32 and shape == "real":
        check(y, raw, name, x.cpu(), mmq=True, lcpp=True)


@pytest.mark.parametrize("n", [16, 32, 129, 512])
@pytest.mark.parametrize("name", IQ3_TYPES)
def test_lcpp_iq3_packed_tiled_graph_replay(name, n):
    _graph_replay(name, n, _lcpp().lcpp_mul_mat_iq3_packed, prep=_packed)


@pytest.mark.parametrize("n", [1, 4, 8, 9, 16, 32, 33, 128, 2048])
@pytest.mark.parametrize("name", IQ3_TYPES)
def test_routing_packed_whole_tensor(name, n):
    """_fused_mul_mat_gguf with packed=True: up to PACKED_VEC_MAX_ROWS the packed
    decode kernel (bit-exact with the mma kernel 8 rows at a time), above it
    the tiled one (bit-exact with lcpp_mul_mat_iq3_packed, within TILED_RTOL
    of MMQ on the GGUF bytes)."""
    from vllm_gguf_plugin.quantization.linear import (
        PACKED_VEC_MAX_ROWS,
        _fused_mul_mat_gguf,
    )

    C = _lcpp()
    _routing()
    raw, x, w, qt = _case(name, n, torch.bfloat16, 1200 + n, rows=18944)
    x = x.cuda()
    p = _packed(w, qt)
    poison_cuda_allocator()
    y = _fused_mul_mat_gguf(x, p, qt, True)
    if n <= PACKED_VEC_MAX_ROWS:
        ref = torch.cat(
            [
                C.lcpp_mul_mat_vec_iq3_mma(w, x[i : i + 8], qt, w.shape[0])
                for i in range(0, n, 8)
            ]
        )
    else:
        ref = C.lcpp_mul_mat_iq3_packed(p, x, qt, p.shape[0])
        _close_to_fp32(y, C.lcpp_mul_mat_q(w, x.float(), qt, w.shape[0]), TILED_RTOL)
    assert torch.equal(y, ref)
    if n in (16, 32, 128):
        check(y[:, :512], raw[:512], name, x.cpu(), mmq=n > 8, lcpp=True)


def _routed_packed_ref(w, x, qt, iq3):
    """What apply() must give for GGUF bytes w whose run is packed (iq3) or not."""
    from vllm_gguf_plugin.quantization.linear import PACKED_VEC_MAX_ROWS

    C = torch.ops._C_gguf
    if not iq3:
        return _routed(w, x, qt)
    if x.shape[0] <= PACKED_VEC_MAX_ROWS:
        return torch.cat(
            [
                C.lcpp_mul_mat_vec_iq3_mma(w, x[i : i + 8], qt, w.shape[0])
                for i in range(0, x.shape[0], 8)
            ]
        )
    return C.lcpp_mul_mat_iq3_packed(_packed(w, qt), x, qt, w.shape[0])


@pytest.mark.parametrize(
    "types", [("IQ3_XXS", "IQ3_S"), ("IQ3_S", "Q4_K")], ids=["z_iq3", "z_other"]
)
@pytest.mark.parametrize("n", [1, 4, 8, 16, 32, 128])
def test_lcpp_packed_layer(n, types, monkeypatch):
    """A qkvz layer (q/k/v run of one IQ3 type, z of another type, IQ3 or not)
    and a single-tensor layer, packed by GGUFLinearMethod._pack_iq3: every IQ3
    run is packed in place (no reallocation; other runs keep their bytes),
    iq3_packed is set, and apply() is bit-exact with the packed kernels on each
    IQ3 run and the usual routing on the others. A layer with a 200-row IQ3 run
    is not packed."""
    from vllm_gguf_plugin.quantization.linear import (
        _IQ3_TYPES,
        GGUFLinearMethod,
        _shard_weight,
    )

    _routing()
    qkv, z, shards = _qkvz(types)
    qts = [_qt(types[0])] * 3 + [_qt(types[1])]
    layer, method = _padded_layer(shards, qts, BLOCKS * 256, monkeypatch)
    ptr = layer.weight.data_ptr()
    method._pack_iq3(layer)
    assert layer.weight.data_ptr() == ptr and layer.weight.iq3_packed
    z_iq3 = qts[3] in _IQ3_TYPES
    if not z_iq3:
        assert torch.equal(
            _shard_weight(layer.weight, *layer.weight.shard_offset_map[3]), z
        )
    x = make_x(n, BLOCKS * 256, torch.bfloat16, seed=1300 + n).cuda()
    y = method.apply(layer, x)
    ref = torch.cat(
        [
            _routed_packed_ref(qkv, x, qts[0], True),
            _routed_packed_ref(z, x, qts[3], z_iq3),
        ],
        1,
    )
    assert torch.equal(y, ref)

    single = torch.nn.Module()  # one tensor, no shards
    single.register_parameter(
        "weight", torch.nn.Parameter(z.clone(), requires_grad=False)
    )
    single.weight.shard_id = []
    wt = {"weight_type": qts[3], "shard_weight_type": {}}
    single.weight_type = type("WT", (), wt)()
    method = GGUFLinearMethod(None)
    method._pack_iq3(single)
    assert getattr(single.weight, "iq3_packed", False) == z_iq3
    assert torch.equal(method.apply(single, x), _routed_packed_ref(z, x, qts[3], z_iq3))

    odd, method = _padded_layer([qkv[:200], z], qts[2:], BLOCKS * 256, monkeypatch)
    method._pack_iq3(odd)  # all or nothing: the 200-row IQ3 run stays GGUF
    assert not getattr(odd.weight, "iq3_packed", False)
    ref = torch.cat([_routed(qkv[:200], x, qts[0]), _routed(z, x, qts[3])], 1)
    assert torch.equal(method.apply(odd, x), ref)


# ------------------------------------------------ shared q8_1 X (x_q8)


@pytest.mark.parametrize("n", [1, 4, 8])
@pytest.mark.parametrize("name", LCPP_TYPES)
def test_lcpp_x_q8(name, n):
    """The q8_1-reading ops on X quantized beforehand (x_q8, as apply() shares
    one quantization between a layer's runs) return exactly what they return
    quantizing X themselves."""
    C = _lcpp()
    _, x, w, qt = _case(name, n, torch.bfloat16, 950 + n)
    x = x.cuda()
    q8 = C.lcpp_quantize_q8_1(x, qt, False, False)
    ops_ = [C.lcpp_mul_mat_vec_q] + [getattr(C, op) for t, op in OWNED if t == name]
    for op in ops_:
        assert torch.equal(op(w, x, qt, w.shape[0], q8), op(w, x, qt, w.shape[0]))
    if name in IQ3_TYPES:  # the packed decode kernel, on W packed
        p, op = _packed(w, qt), C.lcpp_mul_mat_vec_iq3_mma_packed
        assert torch.equal(op(p, x, qt, p.shape[0], q8), op(p, x, qt, p.shape[0]))


def test_quantize_x_q8_1_mixed_route():
    """A layer's runs share one q8_1 quantization of X even when the first run
    is not an lcpp type (Q5_K, stock path): the bytes are the lcpp run's."""
    from vllm_gguf_plugin.quantization.linear import _quantize_x_q8_1

    C = _lcpp()
    _routing()
    T = gguf.GGMLQuantizationType
    x = make_x(4, 5120, torch.bfloat16, seed=77).cuda()
    q8 = _quantize_x_q8_1(x, [int(T.Q5_K), int(T.IQ2_XS)], [5120, 5120])
    assert torch.equal(q8, C.lcpp_quantize_q8_1(x, int(T.IQ2_XS), False, False))
    # 8 rows: not MMVQ, but the Q4_K kernel (above 2048 weight rows)
    x = make_x(8, 5120, torch.bfloat16, seed=78).cuda()
    q8 = _quantize_x_q8_1(x, [int(T.Q4_K)], [4096])
    assert torch.equal(q8, C.lcpp_quantize_q8_1(x, int(T.Q4_K), False, False))


@pytest.mark.parametrize("n", [1, 4, 6, 8])
@pytest.mark.parametrize("name,op", OWNED)
def test_lcpp_x_q8_graph_replay(op, name, n):
    """The decode path in one graph: apply()'s shared quantize (_quantize_x_q8_1,
    with weight rows above the owned kernels' routing floor, so it fills) and
    the owned op reading that x_q8."""
    from vllm_gguf_plugin.quantization.linear import _quantize_x_q8_1

    f = getattr(_lcpp(), op)

    def fn(w, x, qt, rows):
        return f(w, x, qt, rows, _quantize_x_q8_1(x, [qt], [17408]))

    _graph_replay(name, n, fn)


# ------------------------------------------------ IQ1_M on MMVQ above 8 rows


@pytest.mark.parametrize("n", [9, 20, 32])
def test_lcpp_iq1_m_chunks(n):
    """IQ1_M above 8 rows (no IQ1_M MMQ): MMVQ on 8-row chunks, reading
    apply()'s shared q8_1 X in row slices, equals the chunks' own products,
    with or without x_q8."""
    from vllm_gguf_plugin.quantization.linear import (
        _fused_mul_mat_gguf,
        _quantize_x_q8_1,
    )

    C = _lcpp()
    _routing()
    _, x, w, qt = _case("IQ1_M", n, torch.bfloat16, 980 + n)
    x = x.cuda()
    want = torch.cat(
        [C.lcpp_mul_mat_vec_q(w, x[i : i + 8], qt, w.shape[0]) for i in range(0, n, 8)]
    )
    q8 = _quantize_x_q8_1(x, [qt], [w.shape[0]])
    assert torch.equal(_fused_mul_mat_gguf(x, w, qt), want)
    assert torch.equal(_fused_mul_mat_gguf(x, w, qt, False, q8), want)
