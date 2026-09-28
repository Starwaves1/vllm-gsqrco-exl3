"""Plugin CUDA kernels vs CPU references, per quant type in the GGUF, on real rows.

Covers: ggml_dequantize (vs gguf-py b11211), MMVQ ggml_mul_mat_vec_a8 at 1..16 tokens,
MMQ ggml_mul_mat_a8 at 16..512 tokens (K-quants only; the plugin has no IQ MMQ), and the
production routing function _fused_mul_mat_gguf on whole tensors (MMVQ below the
mmvq_safe threshold, else MMQ or dequantize + x @ W.T).

Tolerances calibrated on an RTX 3090 at e2b8ad5 (2026-09-28, cloud/results/phase1): worst
reference-model error 2.5e-3 (bf16) / 1.24e-3 (fp16); worst error vs full precision 1.5e-2,
except Q4_K through MMQ (7.0e-2 direct, 9.0e-2 via routing), which matches the xsum model
to 2.5e-3: its min term uses half(sum x), as in ggml's MMQ. ggml_dequantize accepts float32.
TODO(kernels): when multi-column MMVQ / IQ MMQ land, add their token counts and the new
routing thresholds here; the references do not change.
"""

import pytest

from gsq_gpu import QUANT_TYPES

ROWS = 512                          # rows per kernel test (real rows from the GGUF)
MMVQ_TOKENS = [1, 2, 3, 4, 8, 16]    # 4 = MTP k=3 verify
MMQ_TOKENS = [16, 64, 512]
ROUTE_TOKENS = [1, 4, 8, 16, 32, 512]
TIGHT = {"bfloat16": 5e-3, "float16": 2.5e-3}
# vs full precision. The MMQ Q4_K/Q5_K x-sum model is further from full than q81: its scale
# and min terms no longer share the q8_1 error, so they stop cancelling (the 20x outlier
# channels in _x make it large).
LOOSE = 3e-2
LOOSE_XSUM = 1.5e-1                               # Q4_K via MMQ, see the module docstring
DQ_DTYPES = ["float32", "float16", "bfloat16"]


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
    return x.to(getattr(torch, dtype))


def _check(y, raw, name, x, mmq):
    import _refs

    r = _refs.refs(raw, name, x.cpu(), mmq)
    errs = {k: _refs.rel_err(y, v) for k, v in r.items()}
    tight = TIGHT[str(x.dtype).split(".")[-1]]
    print(f"\n{name} n={x.shape[0]} {x.dtype} mmq={mmq}: " + " ".join(f"{k}={v:.2e}" for k, v in errs.items()))
    best = min(v for k, v in errs.items() if k != "full")
    assert best <= tight, f"no reference model within {tight}: {errs}"
    assert errs["full"] <= (LOOSE_XSUM if "xsum" in errs else LOOSE), f"too far from full precision: {errs}"


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
        torch.testing.assert_close(out, exp, rtol=0, atol=0)  # TODO(GPU): 1 ulp if rounding order differs


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
    assert e <= (LOOSE_XSUM if name in _refs.KQUANT_MIN_SPLIT else LOOSE)
