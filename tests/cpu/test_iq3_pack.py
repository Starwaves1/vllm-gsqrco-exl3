"""quantization/iq3_pack.py on the CPU: pack is a bijection (unpack(pack(w)) == w) on real GGUF
rows (tests/fixtures/dequant) and on random bytes, keeps every 16-row tile in place, and pack_
(in place, 256 rows per step) equals pack, also through a padded multi-shard view and with a
row count that is not a multiple of 256. The kernels' side is tests/gpu/test_kernel_parity.py."""

import glob
import os

import numpy as np
import pytest
import torch
from conftest import FIXTURES
from gguf import GGML_QUANT_SIZES
from gguf import GGMLQuantizationType as T

from vllm_gguf_plugin.quantization import iq3_pack

TYPES = [T.IQ3_S, T.IQ3_XXS]


def _random(rows, blocks, qt, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (rows, blocks * GGML_QUANT_SIZES[qt][1]), generator=g, dtype=torch.uint8)


@pytest.mark.parametrize("qt", TYPES, ids=lambda q: q.name)
def test_roundtrip_fixture_rows(qt):
    files = sorted(glob.glob(os.path.join(FIXTURES, f"{qt.name}__*.npz")))
    if not files:
        pytest.skip(f"no {qt.name} fixtures")
    rows = {}
    for f in files:
        raw = np.load(f)["raw"]
        rows.setdefault(raw.size, []).append(raw)
    for same_k in rows.values():  # 16 rows cycled from the fixture rows of one K
        w = torch.from_numpy(np.stack([same_k[i % len(same_k)] for i in range(32)]))
        p = iq3_pack.pack(w, qt)
        assert p.shape == w.shape and not torch.equal(p, w)
        assert torch.equal(iq3_pack.unpack(p, qt), w)


@pytest.mark.parametrize("qt", TYPES, ids=lambda q: q.name)
@pytest.mark.parametrize("rows,blocks", [(16, 1), (32, 2), (272, 20), (48, 68)])
def test_roundtrip_random_bytes(qt, rows, blocks):
    """Every bit pattern, not only what a quantizer writes; tiles stay in their rows' bytes."""
    w = _random(rows, blocks, qt)
    p = iq3_pack.pack(w, qt)
    assert torch.equal(iq3_pack.unpack(p, qt), w)
    for t in range(rows // 16):  # tile t depends only on rows 16t..16t+15
        assert torch.equal(p[16 * t:16 * t + 16], iq3_pack.pack(w[16 * t:16 * t + 16], qt))


@pytest.mark.parametrize("qt", TYPES, ids=lambda q: q.name)
def test_pack_inplace_views(qt):
    """pack_ == pack for 272 rows (a 16-row remainder after 256) and on a run stored inside a
    wider padded buffer (the VLLM_GGUF_LCPP=1 multi-shard layout): bytes outside the run stay."""
    w = _random(272, 3, qt, seed=1)
    v = w.clone()
    iq3_pack.pack_(v, qt)
    assert torch.equal(v, iq3_pack.pack(w, qt))

    buf = torch.full((400, w.shape[1] + 64), 7, dtype=torch.uint8)
    run = buf[16:16 + 272].view(-1)[: w.numel()].view(272, w.shape[1])
    run.copy_(w)
    before = buf.clone()
    iq3_pack.pack_(run, qt)
    assert torch.equal(run, iq3_pack.pack(w, qt))
    outside = torch.ones_like(buf, dtype=torch.bool)
    outside.view(-1)[16 * buf.shape[1]:16 * buf.shape[1] + w.numel()] = False
    assert torch.equal(buf[outside], before[outside])


def test_rejects_other_types_and_shapes():
    with pytest.raises(AssertionError):
        iq3_pack.pack(_random(16, 1, T.IQ3_S), T.Q4_K)
    with pytest.raises(AssertionError):
        iq3_pack.pack(_random(8, 1, T.IQ3_S), T.IQ3_S)  # rows % 16
