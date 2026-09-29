"""Route L's op choice per (activation rows n, type, weight rows), linear._lcpp_op, at every
boundary of ROUTE-L.md's routing table. Every op but lcpp_mul_mat_q reads apply()'s shared
q8_1 X (_quantize_x_q8_1 asks the same function)."""

import pytest
from gguf import GGMLQuantizationType as T

IQ3, MMA, OWN = "lcpp_mul_mat_vec_iq3", "lcpp_mul_mat_vec_iq3_mma", "lcpp_mul_mat_vec_own"
MMVQ, MMQ = "lcpp_mul_mat_vec_q", "lcpp_mul_mat_q"
BIG = 17408  # ffn_gate/up rows; 2048 = attn_k + attn_v, one Q4_K run in several blocks

CASES = [
    # IQ3: dp4a 1..5, mma 6..8, MMQ from 9, at any weight rows
    *[(t, n, rows, want) for t in (T.IQ3_S, T.IQ3_XXS) for rows in (1024, BIG)
      for n, want in ((1, IQ3), (5, IQ3), (6, MMA), (8, MMA), (9, MMQ))],
    # Q4_K above 2048 rows: MMVQ at 1-2, own 3..8, MMQ from 9
    (T.Q4_K, 1, BIG, MMVQ), (T.Q4_K, 2, BIG, MMVQ), (T.Q4_K, 3, BIG, OWN), (T.Q4_K, 8, BIG, OWN),
    (T.Q4_K, 9, BIG, MMQ), (T.Q4_K, 3, 2049, OWN),
    # IQ2_S above 2048 rows: own 1..8
    (T.IQ2_S, 1, BIG, OWN), (T.IQ2_S, 8, BIG, OWN), (T.IQ2_S, 9, BIG, MMQ), (T.IQ2_S, 1, 2049, OWN),
    # at <= 2048 rows (a 1024 + 1024 k/v run lands on 2048): MMVQ below 8, MMQ from 8
    (T.Q4_K, 4, 2048, MMVQ), (T.Q4_K, 8, 2048, MMQ), (T.IQ2_S, 1, 2048, MMVQ), (T.IQ2_S, 8, 2048, MMQ),
    # every other Route L type: MMVQ below 8, MMQ from 8
    *[(t, n, BIG, want) for t in (T.Q2_K, T.Q6_K, T.IQ2_XXS, T.IQ2_XS, T.IQ4_XS)
      for n, want in ((1, MMVQ), (7, MMVQ), (8, MMQ), (9, MMQ))],
]


@pytest.mark.parametrize("qt,n,rows,want", CASES, ids=lambda v: getattr(v, "name", str(v)))
def test_lcpp_op(qt, n, rows, want):
    from vllm_gguf_plugin.quantization.linear import _lcpp_op

    assert _lcpp_op(n, int(qt), rows) == want
