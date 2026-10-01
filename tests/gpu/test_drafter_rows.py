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
Per (target, n): the n real rows (hidden-state-like bf16) are padded to the next capture size with
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
        t = by[f"blk.64.{target}.weight"]
        raw = np.asarray(t.data)
    qt = int(gguf.GGMLQuantizationType[t.tensor_type.name])
    w = torch.from_numpy(np.ascontiguousarray(raw)).cuda()
    _W.clear()  # one weight on the GPU at a time
    _W[target] = (w, qt, int(t.shape[0]))
    return _W[target]


CASES = ([("draft_head", n) for n in range(9, 17)]
         + [(t, n) for t in ("nextn.eh_proj", "attn_q", "attn_output", "ffn_gate", "ffn_down") for n in (9, 12, 16)]
         + [("lm_head", n) for n in (20, 24, 27, 30, 32)])


@pytest.mark.parametrize("target,n", CASES, ids=[f"{t}-{n}" for t, n in CASES])
def test_drafter_rows_padding(gguf_reader, target, n):
    import torch

    from vllm_gguf_plugin import ops
    from vllm_gguf_plugin.quantization import linear

    if not ops.LCPP_ENABLED:
        pytest.skip("needs VLLM_GGUF_LCPP=1")
    w, qt, k = _weight(gguf_reader, target)
    rows = w.shape[0]
    total = _padded_size(n)
    route = linear._lcpp_op(total, qt, rows, k)
    f = lambda x: linear._fused_mul_mat_gguf(x, w, qt)  # noqa: E731

    x = _x(n, k, seed=1000 + n).cuda()
    clean = f(_pad(x.cpu(), total, "zero", n).cuda())[:n].clone()
    torch.cuda.synchronize()
    assert torch.isfinite(clean).all(), f"{target} n={n} ({route}): non-finite with zero padding"
    # vs the dequantized fp32 product (row blocks, the 248k lm_head would not fit at once in fp32)
    num = den = 0.0
    for r0 in range(0, rows, 32768):
        r1 = min(rows, r0 + 32768)
        ref = x.float() @ ops.ggml_dequantize(w[r0:r1], qt, r1 - r0, k, torch.float32).T
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
