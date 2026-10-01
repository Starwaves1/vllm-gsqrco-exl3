# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""linear._lcpp_op's choice per (activation rows n, type)
at every boundary of the routing table (VLLM_GGUF_LCPP=1). CPU only."""

import pytest
from gguf import GGMLQuantizationType as T

MMVQ, MMQ = "lcpp_mul_mat_vec_q", "lcpp_mul_mat_q"
BIG = 17408
K = 5120

CASES = [
    # every other lcpp type: MMVQ below 8, MMQ from 8
    *[
        (t, n, BIG, want)
        for t in (
            T.Q2_K,
            T.Q6_K,
            T.IQ2_XXS,
            T.IQ2_XS,
            T.IQ3_S,
            T.IQ3_XXS,
            T.Q4_K,
            T.IQ2_S,
            T.IQ4_XS,
        )
        for n, want in ((1, MMVQ), (7, MMVQ), (8, MMQ), (9, MMQ), (32, MMQ))
    ],
]


@pytest.mark.parametrize(
    "qt,n,rows,want", CASES, ids=lambda v: getattr(v, "name", str(v))
)
def test_lcpp_op(qt, n, rows, want):
    from vllm_gguf_plugin.quantization.linear import _lcpp_op

    del rows  # routing depends on n and the type only
    assert _lcpp_op(n, int(qt)) == want


def test_fused_mul_mat_gguf_zero_rows():
    """0 activation rows return an empty [0, rows] result before any routing."""
    import torch

    from vllm_gguf_plugin.quantization.linear import _fused_mul_mat_gguf

    w = torch.zeros(BIG, 512 // 256 * 110, dtype=torch.uint8)
    y = _fused_mul_mat_gguf(torch.zeros(0, 512, dtype=torch.bfloat16), w, int(T.IQ3_S))
    assert y.shape == (0, BIG) and y.dtype == torch.bfloat16
