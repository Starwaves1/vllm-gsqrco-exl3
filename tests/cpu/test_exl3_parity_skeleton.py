"""bench/parity/exl3_logits.py: its chunk plan, CPU only. Chunks are consecutive from 0 to the
last dumped position + 1, none longer than `chunk`; a logits chunk's head rows are exactly
dumped positions (no computed row wasted, none missing), at most `logits_rows` of them."""

import os
import random
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "bench/parity"))


def _prompts_like(n, tail=256, spread=32, seed=0):
    """positions as prompts.py picks them: the last `tail` plus `spread` spread over the rest."""
    r = random.Random(seed)
    head = sorted(r.sample(range(1, max(2, n - tail)), min(spread, max(0, n - tail - 1))))
    return sorted(set(head) | set(range(max(0, n - tail), n)))


CASES = [(10000, _prompts_like(10000), 2048, 256), (120000, _prompts_like(120000, seed=3), 2048, 256),
         (300, _prompts_like(300), 2048, 256), (4097, [4096], 2048, 256), (100, [99], 7, 3),
         (5000, list(range(4000, 4600)), 2048, 256), (5000, [0, 1, 2, 2047, 2048, 4999], 2048, 256)]


@pytest.mark.parametrize("n,pos,chunk,rows", CASES, ids=range(len(CASES)))
def test_plan(n, pos, chunk, rows):
    from exl3_logits import plan

    c = plan(n, pos, chunk, rows)
    assert c[0][0] == 0 and c[-1][1] == max(pos) + 1
    assert all(a[1] == b[0] for a, b in zip(c, c[1:]))
    assert all(0 < e - s <= chunk and 0 <= k <= min(rows, e - s) for s, e, k in c)
    computed = [p for s, e, k in c for p in range(e - k, e)]
    assert computed == sorted(set(pos))  # every dumped row once, nothing else
    for s, e, k in c:
        if not k:
            assert not any(s <= p < e for p in pos)


def test_plan_rejects_bad_positions():
    from exl3_logits import plan

    with pytest.raises(ValueError):
        plan(10, [10], 4)
