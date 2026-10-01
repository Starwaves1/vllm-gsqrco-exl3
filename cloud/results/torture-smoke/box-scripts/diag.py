"""Diagnostics for the first torture smoke's findings, against a running server (stdlib only).
  diag.py BASE_URL API_KEY
1. T=0 greedy newline loop on a 32k follow-up prompt (smoke 1: prefix hit, 128 x "\\n"): the same
   corpus prefix (smoke seed, salt 6ebe5c4d2d2fa8bb) plus three tails, each sent as a prefix-cache
   hit and as a miss (cache_salt); first-token top-5 and the first 24 tokens.
2. NaN logprobs (smoke 1: echo + logprobs -> HTTP 400 "nan"): echo, prompt_logprobs, and where NaN sits.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "bench" / "torture"))
from load import Api, Corpus  # noqa: E402

api = Api(sys.argv[1], sys.argv[2])


def post(body):
    st, txt = api.call("POST", api.path + "/completions", {"model": api.model, "temperature": 0.0, **body}, timeout=600)
    try:
        return st, json.loads(txt)  # json accepts NaN
    except ValueError:
        return st, txt[:300]


corpus = Corpus(api, 20261001, 32064)
base = corpus.prompt(32000, "6ebe5c4d2d2fa8bb")
print("corpus", corpus.sha, "(smoke 1: 72adf105661fcbf5)")
post({"prompt": base, "max_tokens": 1})  # cache the prefix
for tail in ("0badc0de", "12345678", "deadbeef"):
    ids = base + corpus.tail(tail)
    for name, extra in (("hit", {}), ("miss", {"cache_salt": f"diag-miss-{tail}"})):
        st, d = post({"prompt": ids, "max_tokens": 24, "logprobs": 5, **extra})
        c = d["choices"][0] if st == 200 else {}
        top = (c.get("logprobs") or {}).get("top_logprobs") or [{}]
        print(f"newline {tail} {name}: HTTP {st} finish {c.get('finish_reason')} text {c.get('text')!r}")
        print(f"   first-token top-5: {sorted(top[0].items(), key=lambda kv: -kv[1])[:5]}")

short = "The capital of Denmark is"
for name, body in (("echo+logprobs", {"prompt": short, "echo": True, "logprobs": 1, "max_tokens": 4}),
                   ("prompt_logprobs short", {"prompt": short, "prompt_logprobs": 1, "max_tokens": 1}),
                   ("prompt_logprobs 200 ids", {"prompt": base[:200], "prompt_logprobs": 1, "max_tokens": 1}),
                   ("logprobs only", {"prompt": short, "logprobs": 1, "max_tokens": 4})):
    st, d = post(body)
    if st != 200:
        print(f"nan {name}: HTTP {st} {d}")
        continue
    plp = d["choices"][0].get("prompt_logprobs") or []
    nan = [i for i, e in enumerate(plp) if e and any(v["logprob"] != v["logprob"] for v in e.values())]
    print(f"nan {name}: HTTP 200, {len(plp)} prompt positions, NaN at {nan[:20]}{'...' if len(nan) > 20 else ''} of {len(plp)}")
