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
]


@pytest.mark.parametrize(
    "qt,n,rows,want", PACKED_CASES, ids=lambda v: getattr(v, "name", str(v))
)
def test_lcpp_op_packed(qt, n, rows, want):
    from vllm_gguf_plugin.quantization.linear import _lcpp_op

    assert _lcpp_op(n, int(qt), rows, K, True) == want


def test_fused_mul_mat_gguf_zero_rows():
    """0 activation rows return an empty [0, rows] result before any routing."""
    import torch

    from vllm_gguf_plugin.quantization.linear import _fused_mul_mat_gguf

    w = torch.zeros(BIG, 512 // 256 * 110, dtype=torch.uint8)
    y = _fused_mul_mat_gguf(torch.zeros(0, 512, dtype=torch.bfloat16), w, int(T.IQ3_S))
    assert y.shape == (0, BIG) and y.dtype == torch.bfloat16
