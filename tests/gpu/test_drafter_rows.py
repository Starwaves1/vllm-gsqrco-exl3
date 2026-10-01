"""The MTP drafter's products at 9..16 rows, the batch the k=2 tier ([9,16,2]) gives the drafter,
inside CUDA graphs whose padded rows hold garbage (production 2026-10-01: wrong tokens only at 9+
running, only on GSQ; cloud/results/prod-garbled-tokens-20261001.md).

Targets are the drafter's real weights, routed as in production (`linear._fused_mul_mat_gguf`):
  - the pruned draft head: output.weight rows listed in hf-config/<name>/mtp_draft_vocab_ids.pt
    (Q4_K, 61,440 rows; lcpp_mul_mat_mma_k at 9..32 rows),
  - the MTP layer (blk.64, all Q6_K: vendored MMQ from 8 rows): eh_proj, attn_q, attn_output,
    ffn_gate, ffn_down,
  - the target lm_head (Q4_K, 248,320 rows) at the verify rows the schedule makes (k=2 at 9..10
    requests: 27, 30; k=3 at 5..8: 20..32), mma_k too.
Per (target, n), drafter tensors and one target tensor per quant type at 9..48 rows: the n real rows (hidden-state-like bf16) are padded to the next capture size with
NaN / inf / bf16-max / stale-random rows. Checked: real rows finite, identical to the same rows with
zero padding (rows are independent: nothing may leak across activation columns), within LOOSE_XSUM
of the dequantized fp32 product, and the same under CUDA-graph capture + replay with new real rows.
Run with VLLM_GGUF_LCPP=1 GSQ_ALLOW_GPU=1."""

import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
HF = ROOT / "hf-config" / "Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp"
CAPTURE = [1, 2, 4, 8, 16, 24, 32, 40, 48]  # vLLM's sizes up to max_cudagraph_capture_size 48
LOOSE_XSUM = 1.5e-1
PADS = ["nan", "inf", "big", "stale"]


def _padded_size(n):
    return next(c for c in CAPTURE if c >= n)


def _x(n, k, seed):
    import torch

    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, k, generator=g)
    x[:, torch.randperm(k, generator=g)[: k // 256]] *= 20
    return x.bfloat16()


def _pad(x, total, kind, seed):
    import torch

    n, k = x.shape
    if total == n:
        return x.clone()
    p = {"nan": torch.full((total - n, k), float("nan")), "inf": torch.full((total - n, k), float("inf")),
         "big": torch.full((total - n, k), 3.38e38), "zero": torch.zeros(total - n, k),
         "stale": _x(total - n, k, seed + 99).float() * 50}[kind]
    if kind == "inf":
        p[:, ::2] = float("-inf")
    return torch.cat([x, p.bfloat16()])


_W = {}


def _weight(gguf_reader, target):
    import gguf
    import numpy as np
    import torch

    if target in _W:
        return _W[target]
    by = {t.name: t for t in gguf_reader.tensors}
    if target in ("draft_head", "lm_head"):
        t = by["output.weight"]
        raw = np.asarray(t.data)
        if target == "draft_head":
            ids_path = HF / "mtp_draft_vocab_ids.pt"
            if not ids_path.exists():
                pytest.skip("no mtp_draft_vocab_ids.pt")
            ids = torch.load(ids_path, map_location="cpu", weights_only=True).numpy()
            raw = raw[ids]
    else:
        t = by[f"{target}.weight" if target.startswith("blk.") else f"blk.64.{target}.weight"]
        raw = np.asarray(t.data)
    qt = int(gguf.GGMLQuantizationType[t.tensor_type.name])
    w = torch.from_numpy(np.ascontiguousarray(raw)).cuda()
    _W.clear()  # one weight on the GPU at a time
    _W[target] = (w, qt, int(t.shape[0]))
    return _W[target]


# 9..16 requests at k=2: target verify and padded first draft pass = 3 rows per request (27..48),
# draft loop pass = 1 row per request (9..16)
CASES = ([("draft_head", n) for n in list(range(9, 17)) + [27, 30, 33, 36, 39, 42, 45, 48]]
         + [(t, n) for t in ("nextn.eh_proj", "attn_q", "attn_output", "ffn_gate", "ffn_down") for n in (9, 12, 16, 27, 33, 48)]
         + [("lm_head", n) for n in (20, 24, 27, 30, 32, 33, 36, 39, 42, 45, 48)]
         # one target tensor per quant type (routes: packed IQ3 tiled, mma_k, MMQ, IQ1_M chunked MMVQ)
         + [(t, n) for t in ("blk.1.ffn_down", "blk.2.ffn_gate", "blk.0.attn_qkv", "blk.1.ssm_out", "blk.0.ffn_down",
                             "blk.0.ffn_gate", "blk.0.ffn_up", "blk.7.attn_q", "blk.13.ffn_gate")
            for n in (9, 16, 27, 33, 48)])


@pytest.mark.parametrize("target,n", CASES, ids=[f"{t}-{n}" for t, n in CASES])
def test_drafter_rows_padding(gguf_reader, target, n):
    import torch

    from vllm_gguf_plugin import ops
    from vllm_gguf_plugin.quantization import linear

    if not ops.LCPP_ENABLED:
        pytest.skip("needs VLLM_GGUF_LCPP=1")
    w, qt, k = _weight(gguf_reader, target)
    W0 = w
    rows = w.shape[0]
    total = _padded_size(n)
    packed = False
    if qt in (int(linear.WeightType.IQ3_S), int(linear.WeightType.IQ3_XXS)) and rows % 16 == 0:
        from vllm_gguf_plugin.quantization import iq3_pack  # the model's IQ3 layers run packed
        w, packed = iq3_pack.pack(w, qt), True
    route = linear._lcpp_op(total, qt, rows, k, packed)
    f = lambda x: linear._fused_mul_mat_gguf(x, w, qt, None, packed)  # noqa: E731

    x = _x(n, k, seed=1000 + n).cuda()
    clean = f(_pad(x.cpu(), total, "zero", n).cuda())[:n].clone()
    torch.cuda.synchronize()
    assert torch.isfinite(clean).all(), f"{target} n={n} ({route}): non-finite with zero padding"
    # vs the dequantized fp32 product (row blocks, the 248k lm_head would not fit at once in fp32)
    num = den = 0.0
    for r0 in range(0, rows, 32768):
        r1 = min(rows, r0 + 32768)
        wd = W0[r0:r1] if packed else w[r0:r1]
        ref = x.float() @ ops.ggml_dequantize(wd, qt, r1 - r0, k, torch.float32).T
        num += (clean[:, r0:r1].float() - ref).pow(2).sum(-1)
        den += ref.pow(2).sum(-1)
    err = float((num.sqrt() / den.sqrt()).max())
    assert err <= LOOSE_XSUM, f"{target} n={n}: rel err {err:.3g}"

    bad = []
    for kind in PADS:
        y = f(_pad(x.cpu(), total, kind, n).cuda())[:n]
        torch.cuda.synchronize()
        if not torch.isfinite(y).all() or not torch.equal(y, clean):
            bad.append(f"eager pad={kind}: finite={bool(torch.isfinite(y).all())} "
                       f"maxdiff={float((y.float() - clean.float()).abs().nan_to_num(1e30).max()):.3g}")

    # CUDA graph: capture at the padded size, replay with new real rows and garbage padding
    static_x = _pad(x.cpu(), total, "zero", n).cuda()
    f(static_x)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_y = f(static_x)
    for j, kind in enumerate(PADS):
        x2 = _x(n, k, seed=2000 + 10 * n + j).cuda()
        ref2 = f(_pad(x2.cpu(), total, "zero", n).cuda())[:n].clone()
        static_x.copy_(_pad(x2.cpu(), total, kind, n).cuda())
        graph.replay()
        torch.cuda.synchronize()
        y = static_y[:n]
        if not torch.isfinite(y).all() or not torch.equal(y, ref2):
            bad.append(f"graph pad={kind}: finite={bool(torch.isfinite(y).all())} "
                       f"maxdiff={float((y.float() - ref2.float()).abs().nan_to_num(1e30).max()):.3g}")
    del graph
    print(f"\n{target} {rows}x{k} n={n} padded {total} route {route}: rel err {err:.2e}; "
          + ("ok" if not bad else "; ".join(bad)))
    assert not bad, f"{target} n={n} padded {total} ({route}): " + "; ".join(bad)


def test_env():
    assert os.environ.get("VLLM_GGUF_LCPP") == "1", "run with VLLM_GGUF_LCPP=1"


# Exact row counts (no padding), the shapes a mixed or eager batch can hand the ops, for
# compute-sanitizer memcheck runs (PYTORCH_NO_CUDA_MEMORY_CACHING=1: every tensor its own
# allocation, so an out-of-bounds read or write of X, x_q8, scratch or the output is reported).
EXACT_N = list(range(9, 17)) + list(range(25, 34)) + [40, 41, 47, 48]
EXACT_TARGETS = ["draft_head", "lm_head", "nextn.eh_proj", "blk.1.ffn_down", "blk.2.ffn_gate", "blk.0.attn_qkv",
                 "blk.1.ssm_out", "blk.0.ffn_down", "blk.0.ffn_gate", "blk.0.ffn_up", "blk.7.attn_q", "blk.13.ffn_gate"]


@pytest.mark.parametrize("n", EXACT_N)
@pytest.mark.parametrize("target", EXACT_TARGETS)
def test_rows_exact(gguf_reader, target, n):
    import torch

    from vllm_gguf_plugin import ops
    from vllm_gguf_plugin.quantization import linear

    if not ops.LCPP_ENABLED:
        pytest.skip("needs VLLM_GGUF_LCPP=1")
    w, qt, k = _weight(gguf_reader, target)
    packed = False
    if qt in (int(linear.WeightType.IQ3_S), int(linear.WeightType.IQ3_XXS)) and w.shape[0] % 16 == 0:
        from vllm_gguf_plugin.quantization import iq3_pack
        w, packed = iq3_pack.pack(w, qt), True
    x = _x(n, k, seed=3000 + n).cuda()
    y = linear._fused_mul_mat_gguf(x, w, qt, None, packed)
    torch.cuda.synchronize()
    assert y.shape == (n, w.shape[0]) and torch.isfinite(y).all(), f"{target} n={n}"


@pytest.mark.parametrize("n", EXACT_N)
def test_rows_exact_mixed_qkv(gguf_reader, n):
    """A full-attention layer's fused qkv_proj with mixed shard types (blk.3: q IQ2_XXS, k / v IQ3_S,
    packed): apply()'s shared x_q8 quantize and the per-run ops reading it."""
    import gguf
    import numpy as np
    import torch

    from vllm_gguf_plugin import ops
    from vllm_gguf_plugin.quantization import iq3_pack, linear

    if not ops.LCPP_ENABLED:
        pytest.skip("needs VLLM_GGUF_LCPP=1")
    by = {t.name: t for t in gguf_reader.tensors}
    runs = []
    for s in ("attn_q", "attn_k", "attn_v"):
        t = by[f"blk.3.{s}.weight"]
        qt = int(gguf.GGMLQuantizationType[t.tensor_type.name])
        w = torch.from_numpy(np.ascontiguousarray(t.data)).cuda()
        runs.append((w, qt))
    packed = all(w.shape[0] % 16 == 0 for w, qt in runs if qt in (int(linear.WeightType.IQ3_S), int(linear.WeightType.IQ3_XXS)))
    if packed:
        runs = [(iq3_pack.pack(w, qt) if qt in (int(linear.WeightType.IQ3_S), int(linear.WeightType.IQ3_XXS)) else w, qt)
                for w, qt in runs]
    k = int(by["blk.3.attn_q.weight"].shape[0])
    x = _x(n, k, seed=4000 + n).cuda()
    x_q8 = linear._quantize_x_q8_1(x, [qt for _, qt in runs], [w.shape[0] for w, _ in runs], packed)
    ys = [linear._fused_mul_mat_gguf(x, w, qt, x_q8, packed) for w, qt in runs]
    alone = [linear._fused_mul_mat_gguf(x, w, qt, None, packed) for w, qt in runs]
    torch.cuda.synchronize()
    for (w, qt), y, a in zip(runs, ys, alone):
        assert torch.isfinite(y).all() and torch.equal(y, a), f"n={n} type {qt}: shared x_q8 != own quantize"
