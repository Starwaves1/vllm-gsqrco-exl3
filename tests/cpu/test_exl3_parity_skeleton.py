"""bench/parity/exl3_logits.py (skeleton): its chunk plan, CPU only. Every dumped position
must land in a chunk that returns logits, and the prefill chunks must end before the first."""

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "bench/parity"))


@pytest.mark.parametrize("n,first,chunk", [(10000, 9744, 2048), (10000, 0, 2048), (300, 44, 2048),
                                           (4096, 2048, 2048), (4097, 4096, 2048), (100, 99, 7)])
def test_chunks(n, first, chunk):
    from exl3_logits import chunks

    c = chunks(n, first, chunk)
    assert c[0][0] == 0 and c[-1][1] == n
    assert all(a[1] == b[0] for a, b in zip(c, c[1:]))
    assert all(0 < e - s <= chunk for s, e, _ in c)
    for s, e, wants in c:
        assert wants == (e > first)
        if not wants:
            assert e <= first
    assert any(s <= first < e and w for s, e, w in c)
