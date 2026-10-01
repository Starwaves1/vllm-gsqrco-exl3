# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""quantization/iq3_pack.py on the CPU: pack is a bijection (unpack(pack(w)) ==
w) on quantizer output (the sample GGUFs) and on random bytes, keeps every
16-row tile in place, and pack_ (in place, a group of tiles at a time) equals
pack, also through a padded multi-shard view and with a row count that is not a
multiple of the group. The kernels' side is in tests/test_lcpp_kernels.py."""

import pytest
import torch
from gguf import GGML_QUANT_SIZES
from gguf import GGMLQuantizationType as T

from vllm_gguf_plugin.quantization import iq3_pack

from .kernel_refs import sample_weight

TYPES = [T.IQ3_S, T.IQ3_XXS]


def _random(rows, blocks, qt, seed=0):
    g = torch.Generator().manual_seed(seed)
    size = (rows, blocks * GGML_QUANT_SIZES[qt][1])
    return torch.randint(0, 256, size, generator=g, dtype=torch.uint8)


@pytest.mark.parametrize("qt", TYPES, ids=lambda q: q.name)
def test_roundtrip_sample_blocks(qt):
    w = torch.from_numpy(sample_weight(qt.name, 64, 20))
    p = iq3_pack.pack(w, qt)
    assert p.shape == w.shape and not torch.equal(p, w)
    assert torch.equal(iq3_pack.unpack(p, qt), w)


@pytest.mark.parametrize("qt", TYPES, ids=lambda q: q.name)
@pytest.mark.parametrize("rows,blocks", [(16, 1), (32, 2), (272, 20), (48, 68)])
def test_roundtrip_random_bytes(qt, rows, blocks):
    """Every bit pattern, not only what a quantizer writes; tiles stay in their
    rows' bytes."""
    w = _random(rows, blocks, qt)
    p = iq3_pack.pack(w, qt)
    assert torch.equal(iq3_pack.unpack(p, qt), w)
    for t in range(rows // 16):  # tile t depends only on rows 16t..16t+15
        tile = slice(16 * t, 16 * t + 16)
        assert torch.equal(p[tile], iq3_pack.pack(w[tile], qt))


@pytest.mark.parametrize("qt", TYPES, ids=lambda q: q.name)
def test_pack_inplace_views(qt):
    """pack_ == pack for 272 rows (groups of 80 rows, then 32) and on a run
    stored inside a wider padded buffer (the VLLM_GGUF_LCPP=1 multi-shard
    layout): bytes outside the run stay."""
    w = _random(272, 3, qt, seed=1)
    v = w.clone()
    iq3_pack.pack_(v, qt)
    assert torch.equal(v, iq3_pack.pack(w, qt))

    buf = torch.full((400, w.shape[1] + 64), 7, dtype=torch.uint8)
    run = buf[16 : 16 + 272].view(-1)[: w.numel()].view(272, w.shape[1])
    run.copy_(w)
    before = buf.clone()
    iq3_pack.pack_(run, qt)
    assert torch.equal(run, iq3_pack.pack(w, qt))
    outside = torch.ones_like(buf, dtype=torch.bool)
    start = 16 * buf.shape[1]
    outside.view(-1)[start : start + w.numel()] = False
    assert torch.equal(buf[outside], before[outside])


def test_rejects_other_types_and_shapes():
    with pytest.raises(AssertionError):
        iq3_pack.pack(_random(16, 1, T.IQ3_S), T.Q4_K)
    with pytest.raises(AssertionError):
        iq3_pack.pack(_random(8, 1, T.IQ3_S), T.IQ3_S)  # rows % 16
