"""EXL3 multi-row kernel parity on the GPU (EXL3-OPT.md; job 10-mr-parity): trellis-serve's
Marlin-EXL3 behind torch.ops._C_exl3.exl3_gemm_mr (plugin-exl3/vllm_exl3_plugin/csrc/
exl3_mr_shim.cu, _C_exl3_mr) on the checkpoint's real tensors (tests/gpu/exl3_cases.py: K2,
K3, K4, K5 and the K4 lm_head for erlidev's).

  decode      level 1: the kernel's decoded weight equals exl3_dequant(had=False) (itself
              bit-exact with exllamav3's reconstruct in job 01) for every row and column:
              identity blocks through the rotated-basis GEMM (upstream exl3_gemm_marlin; an
              fp32 sum of one product and zeros is exact). K3/K5 read in place, K4 repacked.
  gemm        exl3_gemm_mr at 17/24/32/48/64/96/144 rows (and 1/8/16 for repacked K4) vs fp64
              (x @ exl3_dequant(had=True)), compared in the dtype the model keeps (bf16 for a
              bf16 model, fp16 otherwise) with exl3_gemm, the route it replaces: inside
              exl3_gemm's error x1.5 (floor 0.2 % of RMS, 2 % max), as job 01 holds exl3_gemm
              to exllamav3. The dequant + fp16 cuBLAS GEMM (exllamav3's >144-row path) is
              printed as a third reference.
  routing     exl3_linear with EXL3_MR 1 / 2 (monkeypatched): K2 stays on exl3_gemm, bit for
              bit; a repacked K4 above 144 rows (unpack + dequant) equals the stored one.
  determinism two calls give identical bits.
  bf16 io     bf16 x in, bf16 out (the glue, ops.MR_GLUE) == fp16 x, fp32 out, .to(bf16), bit for bit.
  graphs      after exl3_mr_warmup: captured exl3_gemm_mr replays == eager on new inputs; in a
              fresh process, a capture before warmup is refused without a device fault.

Environment as tests/gpu/test_exl3_kernels.py. Skips without the checkpoint or _C_exl3_mr.
"""

import os
import subprocess
import sys

import pytest

from gsq_gpu import ROOT
import exl3_cases as C

sys.path.insert(0, str(ROOT / "plugin-exl3"))

SLACK, FLOOR_RMS = 1.5, 2e-3
# max_rel floor 3e-2 (test_exl3_kernels.py's route-agreement bound): compared in bf16, a value at
# 6-8x the RMS carries 2^-9 rounding alone; job 10 run 1: K3-up 32 rows bf16 max_rel 0.0246 at
# rel_rms 1.97e-3 (exl3_gemm: 0.0163 / 2.80e-3), mr's rms below exl3_gemm's in 121 of 122 cases
FLOOR_MAX = 3e-2
MR_ROWS = [17, 24, 32, 48, 64, 96, 144, 192, 384]  # 192/384: the mr range past 144 (ops.MULTI_ROW_MAX)
K4_SMALL_ROWS = [1, 8, 16]  # repacked K4 takes every row count
BITS = {tid: v[1] for tid, v in C.TENSORS.items()}
MR_TIDS = [t for t in C.TENSORS if BITS[t] in (3, 4, 5)]


@pytest.fixture(scope="session")
def ops():
    if not C.MODEL.exists():
        pytest.skip(f"EXL3 checkpoint not found: {C.MODEL} (EXL3_MODEL)")
    from vllm_exl3_plugin import ops as o

    if not (o.OPS_AVAILABLE and o.MR_AVAILABLE):
        pytest.skip("_C_exl3 / _C_exl3_mr not built (VLLM_EXL3_BUILD=1)")
    return o


_W = {}


def weights(tid):
    """trellis/suh/svh/flags plus `b`, the tensor exl3_gemm_mr takes (stored or K4 repack)."""
    import torch

    if tid not in _W:
        _W.clear()
        torch.cuda.empty_cache()
        w = C.load(torch, tid)
        w["b"] = torch.ops._C_exl3.exl3_mr_repack(w["trellis"]) if BITS[tid] == 4 else w["trellis"]
        _W[tid] = w
    return _W[tid]


def dequant_fn(w, had):
    import torch

    return lambda s, c: torch.ops._C_exl3.exl3_dequant(w["trellis"], w["suh"], w["svh"], w["mcg"], w["mul1"],
                                                       s, c, had)


@pytest.mark.parametrize("tid", MR_TIDS)
def test_decode_exact(ops, tid):
    import torch
    from vllm_exl3_plugin import _C_exl3_mr as M

    w = weights(tid)
    _, K, k, n = C.TENSORS[tid]
    b = w["b"] if K == 4 else w["b"].view(torch.int32).view(k // 16, n // 64, 4, 8 * K)
    if K == 4:
        assert torch.equal(torch.ops._C_exl3.exl3_mr_unpack(b), w["trellis"]), "repack does not invert"
    eye = torch.eye(64, dtype=torch.half, device="cuda")
    bad = 0
    for s, cnt in C.slices(n):
        wd = dequant_fn(w, False)(s, cnt)  # rotated basis, fp16 [k, cnt]
        for r in range(0, k, 64):
            a = torch.zeros(64, k, dtype=torch.half, device="cuda")
            a[:, r:r + 64] = eye
            c = torch.empty(64, n, dtype=torch.half, device="cuda")
            M.exl3_gemm_marlin(a, b, c, 2)
            bad += int((c[:, s:s + cnt] != wd[r:r + 64]).sum().item())
        del wd
    assert bad == 0, f"{tid}: {bad} decoded weights differ from exl3_dequant"


def _bf16_or_half(y, out_fp32):
    import torch

    return y.to(torch.bfloat16) if out_fp32 else y


@pytest.mark.parametrize("out_fp32", [True, False], ids=["bf16model", "fp16model"])
@pytest.mark.parametrize("m", MR_ROWS + K4_SMALL_ROWS)
@pytest.mark.parametrize("tid", MR_TIDS)
def test_gemm_vs_fp64(ops, tid, m, out_fp32):
    import torch

    K, n = BITS[tid], C.TENSORS[tid][3]
    if m in K4_SMALL_ROWS and K != 4:
        pytest.skip("K3/K5 take exl3_gemm_mr from 17 rows only")
    w = weights(tid)
    x = C.make_x(torch, tid, m)
    args = (w["suh"], w["svh"], w["mcg"], w["mul1"], out_fp32)
    y = torch.ops._C_exl3.exl3_gemm_mr(x, w["b"], *args)
    assert y.shape == (m, n) and y.dtype == (torch.float if out_fp32 else torch.half)
    ref = C.fp64_ref(torch, x, dequant_fn(w, True), n)
    s = C.err_stats(torch, _bf16_or_half(y, out_fp32), ref)
    g = C.err_stats(torch, _bf16_or_half(torch.ops._C_exl3.exl3_gemm(x, w["trellis"], *args), out_fp32), ref)
    h = C.err_stats(torch, torch.cat([x @ dequant_fn(w, True)(s0, c0) for s0, c0 in C.slices(n)], dim=1), ref)
    print(f"\n{tid} m={m} {'bf16' if out_fp32 else 'fp16'} mr {s} | exl3_gemm {g} | dequant+fp16 GEMM {h}")
    assert s["finite"], "non-finite output"
    assert s["rel_rms"] <= max(SLACK * g["rel_rms"], FLOOR_RMS), (s, g)
    assert s["max_rel"] <= max(SLACK * g["max_rel"], FLOOR_MAX), (s, g)


@pytest.mark.parametrize("m", [1, 8, 17, 48, 144])
@pytest.mark.parametrize("tid", MR_TIDS)
def test_bf16_io_same_bits(ops, tid, m):
    """Glue (ops.MR_GLUE): bf16 x straight into exl3_gemm_mr == x.half() + fp32 out + .to(bf16), bit for bit."""
    import torch

    if m < 17 and BITS[tid] != 4:
        pytest.skip("K3/K5 take exl3_gemm_mr from 17 rows only")
    w = weights(tid)
    xb = C.make_x(torch, tid, m).to(torch.bfloat16)
    args = (w["b"], w["suh"], w["svh"], w["mcg"], w["mul1"], True)
    y = torch.ops._C_exl3.exl3_gemm_mr(xb, *args)
    assert y.dtype == torch.bfloat16
    assert torch.equal(y, torch.ops._C_exl3.exl3_gemm_mr(xb.half(), *args).to(torch.bfloat16))


@pytest.mark.parametrize("m", [1, 6, 24, 48])
def test_linear_parts_same_bits(ops, m):
    """The per-layer op on bf16 x (glue) == each part's exl3_linear on fp16 x, cast to bf16, concatenated."""
    import torch

    ws = [C.load(torch, t) for t in ("K4-kproj", "K5-kproj")]  # both k=5120: a k/v-like fused pair
    for w in ws:
        if w["trellis"].shape[2] == 64:
            w["trellis"] = ops.repack_k4_(w["trellis"])
    x = C.make_x(torch, "K4-kproj", m)
    args = ([w["trellis"] for w in ws], [w["suh"] for w in ws], [w["svh"] for w in ws], False, True, True)
    y = ops.exl3_linear_parts(x.to(torch.bfloat16), *args)
    want = torch.cat([ops.exl3_linear(x.to(torch.bfloat16).half(), w["trellis"], w["suh"], w["svh"], False, True, True)
                      .to(torch.bfloat16) for w in ws], 1)
    assert y.dtype == torch.bfloat16 and torch.equal(y, want)


@pytest.mark.parametrize("m", [1, 6, 24, 48, 144])
def test_gemm_mr_multi(ops, m):
    """A concatenated fused group (two tensors of one K, k=5120) in one exl3_gemm_mr_multi call ==
    the parts one by one, up to accumulation order (rel. rms <= 1e-3)."""
    import torch

    ws = [C.load(torch, t) for t in ("K3-up", "K3-up")]
    ws[1] = {**ws[1], "suh": ws[1]["suh"].flip(0).contiguous(), "svh": ws[1]["svh"].flip(0).contiguous()}
    x = C.make_x(torch, "K3-up", m)
    parts = [torch.ops._C_exl3.exl3_gemm_mr(x, w["trellis"], w["suh"], w["svh"], False, True, True) for w in ws]
    t = torch.cat([w["trellis"] for w in ws], dim=1)
    n = ws[0]["svh"].numel()
    y = torch.ops._C_exl3.exl3_gemm_mr_multi(x, t, torch.cat([w["suh"] for w in ws]), torch.cat([w["svh"] for w in ws]),
                                             [n], False, True, True)
    d = C.err_stats(torch, y, torch.cat(parts, 1).double())
    assert d["finite"] and d["rel_rms"] <= 1e-3, d


@pytest.mark.parametrize("tid", MR_TIDS)
def test_deterministic(ops, tid):
    import torch

    w = weights(tid)
    x = C.make_x(torch, tid, 48)
    args = (w["b"], w["suh"], w["svh"], w["mcg"], w["mul1"], True)
    assert torch.equal(torch.ops._C_exl3.exl3_gemm_mr(x, *args), torch.ops._C_exl3.exl3_gemm_mr(x, *args))


@pytest.mark.parametrize("m", [16, 17, 48, 144])
@pytest.mark.parametrize("tid", list(C.TENSORS))
def test_routing_mr1(ops, monkeypatch, tid, m):
    """EXL3_MR=1: K3/K5 at 17..144 rows on exl3_gemm_mr, everything else bit-identical to the
    phase-1 route (K2 and K4 stay on exl3_gemm)."""
    import torch

    monkeypatch.setattr(ops, "MR_MODE", 1)
    monkeypatch.setattr(ops, "MULTI_ROW_OP", ops.MR_OP)
    monkeypatch.setattr(ops, "MULTI_ROW_MIN", 17)  # the 16/17 boundary (the default is 1)
    w = weights(tid)
    x = C.make_x(torch, tid, m)
    args = (w["trellis"], w["suh"], w["svh"], w["mcg"], w["mul1"], True)
    y = ops.exl3_linear(x, *args)
    if BITS[tid] in (3, 5) and m >= 17:
        assert torch.equal(y, torch.ops._C_exl3.exl3_gemm_mr(x, *args))
    else:
        assert torch.equal(y, torch.ops._C_exl3.exl3_gemm(x, *args))


@pytest.mark.parametrize("m", [1, 48, 145, 385, 1024])
@pytest.mark.parametrize("tid", [t for t in MR_TIDS if BITS[t] == 4])
def test_routing_repacked(ops, monkeypatch, tid, m):
    """EXL3_MR=2: a repacked K4 runs exl3_gemm_mr to MULTI_ROW_MAX rows (the lm_head, n > 32768, at
    any row count); above, unpack + dequant gives the stored tensor's route bit for bit."""
    import torch

    monkeypatch.setattr(ops, "MR_MODE", 2)
    monkeypatch.setattr(ops, "MULTI_ROW_OP", ops.MR_OP)
    w = weights(tid)
    x = C.make_x(torch, tid, m)
    y = ops.exl3_linear(x, w["b"], w["suh"], w["svh"], w["mcg"], w["mul1"], True)
    if m <= ops.MULTI_ROW_MAX or C.TENSORS[tid][3] > ops.RECON_SLICE_N:
        want = torch.ops._C_exl3.exl3_gemm_mr(x, w["b"], w["suh"], w["svh"], w["mcg"], w["mul1"], True)
    else:
        want = ops.exl3_linear(x, w["trellis"], w["suh"], w["svh"], w["mcg"], w["mul1"], True)
    if m > ops.WIDE_CHUNK_ROWS and C.TENSORS[tid][3] > ops.RECON_SLICE_N:
        # the lm_head above 256 rows runs in row chunks: another k-split, fp32 sums in another order
        d = C.err_stats(torch, y, want.double())
        assert d["finite"] and d["rel_rms"] <= 1e-3, d
    else:
        assert torch.equal(y, want)


@pytest.mark.parametrize("m", [1, 4, 17, 24, 48])
@pytest.mark.parametrize("tid", ["K3-down", "K4-kproj", "K5-kproj", C.HEAD])
def test_graph_replay(ops, tid, m):
    import torch

    if tid not in MR_TIDS or (BITS[tid] != 4 and m < 17):
        pytest.skip("not routed to exl3_gemm_mr")
    w = weights(tid)
    args = (w["b"], w["suh"], w["svh"], w["mcg"], w["mul1"], True)
    torch.ops._C_exl3.exl3_mr_warmup(*args[:5], [m], True)
    x_static = C.make_x(torch, tid, m)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        torch.ops._C_exl3.exl3_gemm_mr(x_static, *args)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y_static = torch.ops._C_exl3.exl3_gemm_mr(x_static, *args)
    for i in range(3):
        x_new = torch.randn(m, x_static.shape[1], generator=torch.Generator().manual_seed(i + 7)).half().cuda()
        x_static.copy_(x_new)
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(y_static, torch.ops._C_exl3.exl3_gemm_mr(x_new, *args)), f"replay {i} differs"
    del g


@pytest.mark.parametrize("rows", [1, 6, 48, 2048])
def test_embed_host_gather(ops, rows):
    """exl3_embed_host on a registered (page-locked) table == F.embedding on the GPU copy, bit for
    bit; out-of-range ids give zero rows; a captured gather replays with new ids."""
    import torch

    from vllm_exl3_plugin.quantization.embedding import EXL3HostEmbeddingMethod

    torch.manual_seed(rows)
    g = torch.Generator().manual_seed(rows)
    v = 16384  # registered tables live for the process: a small one per case
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(torch.randn(v, 5120, device="cuda", dtype=torch.bfloat16), requires_grad=False)
    ref_w = layer.weight.data.clone()
    m = EXL3HostEmbeddingMethod()
    alloc0 = torch.cuda.memory_allocated()  # freed blocks go to torch's cache, not the driver
    m.process_weights_after_loading(layer)
    assert m.table is not None and layer.weight.shape == (0, 5120)
    assert alloc0 - torch.cuda.memory_allocated() >= ref_w.nbytes, "GPU copy not freed"
    ids = torch.randint(0, v, (rows,), generator=g).cuda()
    assert torch.equal(m.embedding(layer, ids), torch.nn.functional.embedding(ids, ref_w))
    bad = torch.tensor([-1, v, 7], device="cuda")
    out = m.embedding(layer, bad)
    assert torch.equal(out[:2], torch.zeros_like(out[:2])) and torch.equal(out[2], ref_w[7])
    static = ids.clone()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        y = m.embedding(layer, static)
    new = torch.randint(0, v, (rows,), generator=g).cuda()
    static.copy_(new)
    gr.replay()
    torch.cuda.synchronize()
    assert torch.equal(y, ref_w[new])


@pytest.mark.parametrize("mr", [0, 2])
def test_lm_head_many_rows_bounded(ops, monkeypatch, mr):
    """prompt_logprobs regression (job 18: EXL3_MR=0 OOMed inside the lm_head on a 4k-token prompt): the
    lm_head on 2048 rows in bf16 returns bf16 with scratch bounded by one 256-row chunk (peak minus the
    output <= 1.2 GiB), rows equal to a 256-row call on the same rows."""
    import torch

    monkeypatch.setattr(ops, "MR_MODE", mr)
    monkeypatch.setattr(ops, "MULTI_ROW_OP", ops.MR_OP if mr else None)
    w = C.load(torch, C.HEAD)
    t = ops.repack_k4_(w["trellis"]) if mr == 2 and w["trellis"].shape[2] == 64 else w["trellis"]
    x = C.make_x(torch, C.HEAD, 2048).to(torch.bfloat16)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    y = ops.exl3_linear(x, t, w["suh"], w["svh"], w["mcg"], w["mul1"], True)
    torch.cuda.synchronize()
    extra = torch.cuda.max_memory_allocated() - base - y.nbytes
    print(f"\nlm_head 2048 rows mr={mr}: output {y.nbytes / 2**30:.2f} GiB, scratch peak {extra / 2**30:.2f} GiB")
    assert y.dtype == torch.bfloat16 and bool(torch.isfinite(y).all()) and extra <= 1.2 * 2**30
    ref = ops.exl3_linear(x[256:512], t, w["suh"], w["svh"], w["mcg"], w["mul1"], True)
    assert torch.equal(y[256:512], ref)
    # fp16 x with fp32 out (the contract without the bf16 path): the chunked output keeps the fake's dtype
    y32 = ops.exl3_linear(x[:300].half(), t, w["suh"], w["svh"], w["mcg"], w["mul1"], True)
    assert y32.dtype == ops.exl3_linear_fake(x[:300].half(), t, None, None, False, True, True).dtype == torch.float


def _garbage(kind, rows, k):
    import torch

    if kind == "nan":
        return torch.full((rows, k), float("nan"))
    if kind == "inf":
        g = torch.full((rows, k), float("inf"))
        g[:, ::2] = float("-inf")
        return g
    return torch.full((rows, k), 65504.0) * torch.sign(torch.randn(rows, k))  # fp16 max, both signs


@pytest.mark.parametrize("pad", ["nan", "inf", "big"])
@pytest.mark.parametrize("n", [9, 12, 16])
@pytest.mark.parametrize("op", ["exl3_gemm", "exl3_gemm_mr"])
@pytest.mark.parametrize("tid", ["K2-up", "K3-down", "K4-mtp-up", "K5-kproj"])
def test_padded_rows_in_capture(ops, tid, op, n, pad):
    """The drafter at 9-16 running requests (k=2 tier): a 16-row CUDA graph with n real rows and
    padding rows holding NaN / +-inf / +-65504. The real rows' outputs stay finite and equal the
    eager result on the real rows alone (rel. rms <= 1e-3): no row of a padded batch leaks into another."""
    import torch

    if op == "exl3_gemm_mr" and BITS[tid] not in (3, 4, 5):
        pytest.skip("not on exl3_gemm_mr")
    if tid not in C.TENSORS:
        pytest.skip(f"{tid} not in this checkpoint's table")
    w = C.load(torch, tid)
    t = ops.repack_k4_(w["trellis"].clone()) if op == "exl3_gemm_mr" and BITS[tid] == 4 else w["trellis"]
    args = (t, w["suh"], w["svh"], w["mcg"], w["mul1"], True)
    if op == "exl3_gemm":
        torch.ops._C_exl3.exl3_warmup(*args[:5], [1, 2, 4, 8, 16], True)
    else:
        torch.ops._C_exl3.exl3_mr_warmup(*args[:5], [16], True)
    fn = getattr(torch.ops._C_exl3, op)
    k = w["suh"].numel()
    x_real = C.make_x(torch, tid, n)
    x16 = torch.cat([x_real, _garbage(pad, 16 - n, k).half().cuda()]) if n < 16 else x_real.clone()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn(x16, *args)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y16 = fn(x16, *args)
    g.replay()
    torch.cuda.synchronize()
    eager = fn(x_real, *args)
    real = y16[:n]
    d = C.err_stats(torch, real, eager.double())
    print(f"\npadded {tid} {op} n={n} pad={pad}: finite={d['finite']} bit-identical={torch.equal(real, eager)} {d}")
    assert d["finite"] and d["rel_rms"] <= 1e-3, d
    del g


@pytest.mark.parametrize("rows", [1, 4, 8])
def test_draft_head_fp8(ops, rows):
    """The fp8 draft head (Marlin) vs the bf16 head on the same rows: logits within e4m3's error
    (rel. rms <= 4 %); argmax kept for >= 90 % of rows (random logits have near-ties: 0.94-0.95
    measured; the real gate is MTP acceptance on the ladder)."""
    import torch

    from vllm_exl3_plugin.quantization.draft_head import EXL3DraftHeadFp8Method

    torch.manual_seed(rows)
    w = (torch.randn(40960, 5120, device="cuda") * 0.02).to(torch.bfloat16)
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(w.clone(), requires_grad=False)
    m = EXL3DraftHeadFp8Method()
    m.process_weights_after_loading(layer)
    assert m.marlin is not None and layer.weight.shape == (0, 5120)
    x = torch.randn(rows * 64, 5120, device="cuda").to(torch.bfloat16)
    y, ref = m.apply(layer, x).float(), (x @ w.T).float()
    rel = ((y - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
    agree = (y.argmax(1) == ref.argmax(1)).float().mean().item()
    print(f"\nfp8 draft head rows={rows * 64}: rel rms {rel:.4f}, argmax agree {agree:.3f}")
    assert rel <= 0.04 and agree >= 0.90


_UNWARMED = r"""
import json, sys, torch
sys.path.insert(0, sys.argv[1])
import exl3_cases as C
from vllm_exl3_plugin import _C_exl3_mr  # noqa: F401
w = C.load(torch, "K3-down")
x = C.make_x(torch, "K3-down", 32)
out = {"status": "ok"}
try:
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        torch.ops._C_exl3.exl3_gemm_mr(x, w["trellis"], w["suh"], w["svh"], False, True, True)
except RuntimeError as e:
    out = {"status": "rejected", "error": str(e).splitlines()[0][:300]}
torch.cuda.synchronize()
torch.ops._C_exl3.exl3_mr_warmup(w["trellis"], w["suh"], w["svh"], False, True, [32], True)
y = torch.ops._C_exl3.exl3_gemm_mr(x, w["trellis"], w["suh"], w["svh"], False, True, True)
torch.cuda.synchronize()
out["healthy_after"] = bool(torch.isfinite(y).all().item())
print(json.dumps(out))
"""


def test_unwarmed_capture_refused(ops):
    import json

    env = dict(os.environ, CUDA_LAUNCH_BLOCKING="1",
               PYTHONPATH=os.pathsep.join([str(ROOT / "tools"), str(ROOT / "plugin-exl3"), os.environ.get("PYTHONPATH", "")]))
    p = subprocess.run([sys.executable, "-c", _UNWARMED, str(ROOT / "tests/gpu")], capture_output=True, text=True,
                       timeout=600, env=env)
    assert p.returncode == 0, (p.stdout + p.stderr)[-3000:]
    res = json.loads(p.stdout.strip().splitlines()[-1])
    assert res["status"] == "rejected" and "was not warmed up" in res["error"], res
    assert res["healthy_after"], res
