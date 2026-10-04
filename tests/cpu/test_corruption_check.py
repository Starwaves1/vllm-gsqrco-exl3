"""bench/corruption_check.py: the per-response classifier against a reference, and the prompt set."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "bench"))

import corruption_check as cc  # noqa: E402


def resp(ids, content, finish="length", reasoning="", max_tokens=512, bad=0, status=200):
    return {"ids": ids, "content": content, "reasoning": reasoning, "finish": finish, "n_ids": len(ids),
            "max_tokens": max_tokens, "bad_sse": bad, "status": status}


REF = resp(list(range(512)), "A long and complete answer.")


def test_clean():
    assert not any(cc.flags(resp(list(range(512)), "Fine answer."), REF, "Explain X").values())


def test_early_eos():
    f = cc.flags(resp(list(range(40)), "The cache uses a", finish="stop"), REF, "Explain X")
    assert f["early_eos"]
    assert not cc.flags(resp(list(range(40)), "Done.", finish="stop"), REF, "Explain X")["early_eos"]
    short_ref = resp(list(range(30)), "Short.", finish="stop")
    assert not cc.flags(resp(list(range(40)), "The cache uses a", finish="stop"), short_ref, "Explain X")["early_eos"]


def test_repeat_only_if_not_in_reference():
    loop = list(range(10)) + [7, 8, 9, 10, 11, 12, 13, 14] * 5
    assert cc.flags(resp(loop, "x"), REF, "p")["repeat"]
    assert not cc.flags(resp(loop, "x"), resp(loop, "x"), "p")["repeat"]


def test_foreign_script():
    assert cc.flags(resp([1], "The answer is 中文 here."), REF, "Explain X")["foreign"]
    assert not cc.flags(resp([1], "Kø og æble."), REF, "Skriv på dansk")["foreign"]
    assert not cc.flags(resp([1], "中文"), resp([1], "也有中文"), "p")["foreign"]


def test_bad_utf8_and_errors():
    assert cc.flags(resp([1], "bad �"), REF, "p")["bad_utf8"]
    assert cc.flags(resp([1], "ok", bad=1), REF, "p")["bad_utf8"]
    assert cc.flags(resp([], "", status=-1), REF, "p")["error"]


def test_prompts():
    ps = cc.prompts(None, 38)
    assert len(ps) == 38 and len(set(ps)) == 38
