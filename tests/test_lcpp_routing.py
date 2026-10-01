# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""linear._lcpp_op's choice per (activation rows n, type, weight rows, K)
at every boundary of the routing table (VLLM_GGUF_LCPP=1). CPU only."""

import pytest
from gguf import GGMLQuantizationType as T

MMVQ, MMQ = "lcpp_mul_mat_vec_q", "lcpp_mul_mat_q"
IQ3, MMA = "lcpp_mul_mat_vec_iq3", "lcpp_mul_mat_vec_iq3_mma"
OWN = "lcpp_mul_mat_vec_own"
MMA_K = "lcpp_mul_mat_mma_k"
PACKED_VEC, PACKED_TILED = "lcpp_mul_mat_vec_iq3_mma_packed", "lcpp_mul_mat_iq3_packed"
BIG = 17408  # rows of a large weight; the owned kernels route above 2048
K = 5120

CASES = [
    # IQ3: dp4a 1..5, mma 6..8, MMQ from 9, at any weight rows
    *[
        (t, n, rows, want)
        for t in (T.IQ3_S, T.IQ3_XXS)
        for rows in (1024, BIG)
        for n, want in ((1, IQ3), (5, IQ3), (6, MMA), (8, MMA), (9, MMQ))
    ],
    # Q4_K above 2048 rows: MMVQ at 1-2, own 3..8, mma_k 9..32, MMQ from 33
    (T.Q4_K, 1, BIG, MMVQ),
    (T.Q4_K, 2, BIG, MMVQ),
    (T.Q4_K, 3, BIG, OWN),
    (T.Q4_K, 8, BIG, OWN),
    (T.Q4_K, 9, BIG, MMA_K),
    (T.Q4_K, 32, BIG, MMA_K),
    (T.Q4_K, 33, BIG, MMQ),
    (T.Q4_K, 3, 2049, OWN),
    (T.Q4_K, 9, 2049, MMA_K),
    # IQ2_S above 2048 rows: own 1..8, mma_k 9..32, MMQ from 33
    (T.IQ2_S, 1, BIG, OWN),
    (T.IQ2_S, 8, BIG, OWN),
    (T.IQ2_S, 9, BIG, MMA_K),
    (T.IQ2_S, 32, BIG, MMA_K),
    (T.IQ2_S, 33, BIG, MMQ),
    (T.IQ2_S, 1, 2049, OWN),
    (T.IQ2_S, 32, 2049, MMA_K),
    # at <= 2048 rows: MMVQ below 8, MMQ from 8
    (T.Q4_K, 4, 2048, MMVQ),
    (T.Q4_K, 8, 2048, MMQ),
    (T.IQ2_S, 1, 2048, MMVQ),
    (T.IQ2_S, 8, 2048, MMQ),
    (T.Q4_K, 9, 2048, MMQ),
    (T.IQ2_S, 32, 2048, MMQ),
    # IQ4_XS above 2048 rows: MMVQ below 8, MMQ at 8, mma_k 9..16, and 17..32
    # only from rows x K = 12288 x 5120; MMQ from 33
    (T.IQ4_XS, 7, BIG, MMVQ),
    (T.IQ4_XS, 8, BIG, MMQ),
    (T.IQ4_XS, 9, 2049, MMA_K),
    (T.IQ4_XS, 16, 2049, MMA_K),
    (T.IQ4_XS, 17, 2049, MMQ),
    (T.IQ4_XS, 17, BIG, MMA_K),
    (T.IQ4_XS, 32, BIG, MMA_K),
    (T.IQ4_XS, 33, BIG, MMQ),
    (T.IQ4_XS, 17, 12288, MMA_K),
    (T.IQ4_XS, 17, 12032, MMQ),
    (T.IQ4_XS, 17, (6144, 2 * K), MMA_K),
    (T.IQ4_XS, 9, 2048, MMQ),
    # every other lcpp type: MMVQ below 8, MMQ from 8
    *[
        (t, n, BIG, want)
        for t in (T.Q2_K, T.Q6_K, T.IQ2_XXS, T.IQ2_XS)
        for n, want in ((1, MMVQ), (7, MMVQ), (8, MMQ), (9, MMQ), (32, MMQ))
    ],
    # IQ1_M (no MMQ upstream): MMVQ up to 32 rows (8 per call), then none
    *[
        (T.IQ1_M, n, rows, want)
        for rows in (1024, BIG)
        for n, want in ((1, MMVQ), (8, MMVQ), (9, MMVQ), (32, MMVQ), (33, None))
    ],
]


@pytest.mark.parametrize(
    "qt,n,rows,want", CASES, ids=lambda v: getattr(v, "name", str(v))
)
def test_lcpp_op(qt, n, rows, want):
    """rows: weight rows, or (weight rows, K) where K matters."""
    from vllm_gguf_plugin.quantization.linear import _lcpp_op

    rows, k = rows if isinstance(rows, tuple) else (rows, K)
    assert _lcpp_op(n, int(qt), rows, k) == want
    assert _lcpp_op(n, int(qt), rows, k, False) == want


PACKED_CASES = [
    # packed IQ3 (GGUFLinearMethod._pack_iq3): the packed mma kernel 1..8,
    # the tiled one from 9
    *[
        (t, n, rows, want)
        for t in (T.IQ3_S, T.IQ3_XXS)
        for rows in (1024, BIG)
        for n, want in (
            (1, PACKED_VEC),
            (5, PACKED_VEC),
            (6, PACKED_VEC),
            (8, PACKED_VEC),
            (9, PACKED_TILED),
            (32, PACKED_TILED),
            (128, PACKED_TILED),
            (2048, PACKED_TILED),
        )  # fmt: skip
    ],
    # a packed layer's other runs route as unpacked
    (T.Q4_K, 3, BIG, OWN),
    (T.Q4_K, 9, BIG, MMA_K),
    (T.IQ4_XS, 8, BIG, MMQ),
    (T.IQ1_M, 9, BIG, MMVQ),
]


@pytest.mark.parametrize(
    "qt,n,rows,want", PACKED_CASES, ids=lambda v: getattr(v, "name", str(v))
)
def test_lcpp_op_packed(qt, n, rows, want):
    from vllm_gguf_plugin.quantization.linear import _lcpp_op

    assert _lcpp_op(n, int(qt), rows, K, True) == want


class _Quantizer:
    """Stands in for torch.ops._C_gguf: records the type the shared quantize
    runs with."""

    def __init__(self):
        self.calls = []

    def lcpp_quantize_q8_1(self, x, qt, mmq, vendored):
        self.calls.append((qt, mmq, vendored))
        return "filled"


def _ids(v):
    return (
        "+".join(t.name for t in v)
        if isinstance(v, list) and v and hasattr(v[0], "name")
        else str(v)
    )


FILLS = [
    (4, [T.Q4_K, T.IQ2_XS], [2048, BIG], T.Q4_K),  # the first run that reads q8_1
    (8, [T.Q4_K], [BIG], T.Q4_K),  # the Q4_K kernel reads it at 8 rows
    (8, [T.Q4_K], [2048], None),  # MMQ quantizes for itself
    (8, [T.IQ4_XS, T.Q4_K], [BIG, BIG], T.Q4_K),  # MMQ beside the Q4_K kernel
    (8, [T.Q4_K, T.IQ3_S], [2048, BIG], T.IQ3_S),  # MMQ beside the IQ3 mma kernel
    (9, [T.IQ3_S, T.Q4_K], [BIG, BIG], None),  # MMQ and mma_k quantize for themselves
    (0, [T.IQ3_S], [BIG], None),  # no rows: nothing to quantize
    (16, [T.Q4_K, T.IQ1_M], [BIG, BIG], T.IQ1_M),  # mma_k beside IQ1_M's MMVQ
    (4, [T.IQ1_M], [BIG], T.IQ1_M),  # IQ1_M on MMVQ
    (33, [T.IQ1_M, T.Q4_K], [BIG, BIG], None),  # stock dequantize beside MMQ
]


@pytest.mark.parametrize("n,types,rows,want", FILLS, ids=_ids)
def test_quantize_x_q8_1_fills(monkeypatch, n, types, rows, want, packed=False):
    """apply()'s shared quantize runs (once, MMVQ layout) iff some run's op
    reads q8_1; else it returns an unfilled buffer of the q8_1 size that
    nothing reads."""
    import torch

    from vllm_gguf_plugin.quantization.linear import _quantize_x_q8_1

    q = _Quantizer()
    monkeypatch.setattr(torch.ops, "_C_gguf", q, raising=False)
    x = torch.zeros(n, 512, dtype=torch.bfloat16)
    out = _quantize_x_q8_1(x, [int(t) for t in types], rows, packed)
    if want is None:
        assert q.calls == [] and out.dtype == torch.uint8
        assert out.numel() == n * 512 // 32 * 36
    else:
        assert q.calls == [(int(want), False, False)] and out == "filled"


PACKED_FILLS = [
    (4, [T.IQ3_S, T.Q4_K], [BIG, BIG], T.IQ3_S),  # the packed mma kernel reads it
    (9, [T.IQ3_XXS, T.Q4_K], [BIG, BIG], None),  # tiled + mma_k
    (9, [T.IQ3_S, T.IQ1_M], [BIG, BIG], T.IQ1_M),  # the tiled one quantizes itself
]


@pytest.mark.parametrize("n,types,rows,want", PACKED_FILLS, ids=_ids)
def test_quantize_x_q8_1_fills_packed(monkeypatch, n, types, rows, want):
    test_quantize_x_q8_1_fills(monkeypatch, n, types, rows, want, packed=True)


def test_fused_mul_mat_gguf_zero_rows():
    """0 activation rows return an empty [0, rows] result before any routing."""
    import torch

    from vllm_gguf_plugin.quantization.linear import _fused_mul_mat_gguf

    w = torch.zeros(BIG, 512 // 256 * 110, dtype=torch.uint8)
    y = _fused_mul_mat_gguf(torch.zeros(0, 512, dtype=torch.bfloat16), w, int(T.IQ3_S))
    assert y.shape == (0, BIG) and y.dtype == torch.bfloat16
