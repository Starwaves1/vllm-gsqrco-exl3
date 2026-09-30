"""EXL3 routing (vllm_exl3_plugin.ops._exl3_op) and the composite paths, CPU only.

The table in ops.py's docstring, at every boundary; the multi-row hook; and exl3_linear's
dequant + GEMM paths run against CPU stand-ins for the shim ops (torch.ops._C_exl3 patched):
the fused path (original-basis dequant, 32768-column slices) and the rotated one (Hadamard
with suh on x, rotated dequant, Hadamard with svh on y) must both equal x @ W.
"""

import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "plugin-exl3"))

GEMM, RECON, RECON_HAD = "exl3_gemm", "recon_hgemm", "recon_had_hgemm"


@pytest.mark.parametrize("n,want", [
    (1, GEMM), (2, GEMM), (8, GEMM), (16, GEMM), (17, GEMM), (48, GEMM), (64, GEMM), (144, GEMM),
    (145, RECON), (512, RECON), (1023, RECON),
    (1024, RECON_HAD), (2048, RECON_HAD), (65536, RECON_HAD),
])
def test_exl3_op(n, want):
    from vllm_exl3_plugin.ops import _exl3_op

    assert _exl3_op(n) == want


def test_capture_sizes_stay_on_gemm():
    """vLLM captures CUDA graphs up to 48 rows in production (max_cudagraph_capture_size):
    all of them must route to the warmed exl3_gemm, never to dequant + GEMM."""
    from vllm_exl3_plugin import ops

    assert ops.GEMM_MAX_ROWS >= 48
    assert all(ops._exl3_op(n) == GEMM for n in range(1, 49))


@pytest.mark.parametrize("n,want", [(1, GEMM), (16, GEMM), (17, "exl3_multi_row"), (64, "exl3_multi_row"),
                                    (144, "exl3_multi_row"), (145, RECON)])
def test_multi_row_hook(monkeypatch, n, want):
    from vllm_exl3_plugin import ops

    monkeypatch.setattr(ops, "MULTI_ROW_OP", "exl3_multi_row")
    assert ops._exl3_op(n) == want


class FakeShim:
    """CPU stand-ins for torch.ops._C_exl3 with the Hadamard taken as the identity, so the
    rotated weight is W / (suh x svh) and every path must reproduce x @ W."""

    def __init__(self, w, suh, svh):
        self.w, self.suh, self.svh = w, suh, svh
        self.calls = []

    def exl3_gemm(self, x, trellis, suh, svh, mcg, mul1, out_fp32):
        assert x.dtype == torch.half
        self.calls.append(("gemm", x.shape[0]))
        return (x.float() @ self.w).to(torch.float if out_fp32 else torch.half)

    def exl3_dequant(self, trellis, suh, svh, mcg, mul1, n_start, n_count, had):
        self.calls.append(("dequant", n_start, n_count, had))
        w = self.w if had else self.w / (self.suh[:, None].float() * self.svh[None, :].float())
        return w[:, n_start:n_start + n_count].half()

    def exl3_hgemm(self, a, b):
        self.calls.append(("hgemm", tuple(a.shape), tuple(b.shape)))
        assert a.dtype == b.dtype == torch.half
        return (a.float() @ b.float()).half()

    def exl3_had_r_128(self, x, pre, post, scale):
        assert (pre is None) != (post is None) and x.dtype == torch.half
        self.calls.append(("had", "pre" if pre is not None else "post"))
        return (x.float() * (pre if pre is not None else post).float() * scale).half()


@pytest.fixture
def fake(monkeypatch):
    k, n_out = 128, 32768 + 256  # two dequant slices
    g = torch.Generator().manual_seed(0)
    w = torch.randn(k, n_out, generator=g) * 0.05
    suh = (torch.rand(k, generator=g) + 0.5).half()
    svh = (torch.rand(n_out, generator=g) + 0.5).half()
    shim = FakeShim(w, suh, svh)
    for name in ("exl3_gemm", "exl3_dequant", "exl3_hgemm", "exl3_had_r_128"):
        monkeypatch.setattr(torch.ops._C_exl3, name, getattr(shim, name), raising=False)
    trellis = torch.empty(k // 16, n_out // 16, 64, dtype=torch.int16)
    return shim, trellis


@pytest.mark.parametrize("rows,path", [(4, "gemm"), (200, "rotated"), (1024, "fused")])
def test_exl3_linear_paths(fake, rows, path):
    from vllm_exl3_plugin.ops import exl3_linear

    shim, trellis = fake
    x = torch.randn(rows, shim.w.shape[0], generator=torch.Generator().manual_seed(1)).half()
    y = exl3_linear(x, trellis, shim.suh, shim.svh, False, True, True)
    ref = x.float() @ shim.w
    assert y.dtype == torch.float and y.shape == ref.shape
    assert torch.allclose(y.float(), ref, rtol=2e-2, atol=2e-2), (y.float() - ref).abs().max()
    kinds = [c[0] for c in shim.calls]
    n_out = shim.w.shape[1]
    if path == "gemm":
        assert shim.calls == [("gemm", rows)]
    elif path == "rotated":
        assert kinds == ["had", "dequant", "hgemm", "dequant", "hgemm", "had"]
        assert shim.calls[0] == ("had", "pre") and shim.calls[-1] == ("had", "post")
        assert shim.calls[1] == ("dequant", 0, 32768, False)
        assert shim.calls[3] == ("dequant", 32768, n_out - 32768, False)
    else:
        assert kinds == ["dequant", "hgemm", "dequant", "hgemm"]
        assert shim.calls[0] == ("dequant", 0, 32768, True)
        assert shim.calls[2] == ("dequant", 32768, n_out - 32768, True)
    assert exl3_linear(x, trellis, shim.suh, shim.svh, False, True, False).dtype == torch.half


def test_fake_impl_shape():
    from vllm_exl3_plugin.ops import exl3_linear_fake

    x = torch.empty(7, 5120, dtype=torch.half, device="meta")
    trellis = torch.empty(320, 64, 64, dtype=torch.int16, device="meta")
    y = exl3_linear_fake(x, trellis, None, None, False, True, True)
    assert y.shape == (7, 1024) and y.dtype == torch.float and y.device.type == "meta"
    assert exl3_linear_fake(x, trellis, None, None, False, True, False).dtype == torch.half
