"""Find soak requests that returned an empty completion (finish stop, 1 completion token = EOS
first) and rebuild their request bodies by replaying soak_load's deterministic Plan (records sorted
by start time = plan order). Prints the count and, with --resend N, sends the N-th such body once
more with logprobs (top 5) and prints the first generated token and its alternatives.
  soak_eos.py RUN_DIR TOKENIZER_DIR [--resend 0 --url U --key K]"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("dir")
ap.add_argument("tokenizer")
ap.add_argument("--resend", type=int)
ap.add_argument("--url", default="http://127.0.0.1:18090")
ap.add_argument("--key", default="gsq-local-test")
ap.add_argument("--seed", type=int, default=20260927)
ap.add_argument("--hit-vs-miss", action="store_true", help="resend twice at T=0: as is (prefix-cache hit) and with a cache_salt (miss)")
a = ap.parse_args()
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "bench"))
import soak_load  # noqa: E402

recs = sorted((json.loads(x) for x in (Path(a.dir) / "load.jsonl").read_text().splitlines() if x.strip()), key=lambda r: r["t"])
eos = [i for i, r in enumerate(recs) if r.get("finish") == "stop" and r.get("completion_tokens") == 1]
print(f"{len(recs)} requests, {len(eos)} empty completions (EOS first): {[(i, recs[i]['kind'], recs[i]['prompt_tokens']) for i in eos]}")
by_kind = {}
for r in recs:
    by_kind.setdefault(r["kind"], 0)
    by_kind[r["kind"]] += 1
print("requests by kind:", by_kind)
if a.resend is None or not eos:
    sys.exit(0)
plan = soak_load.Plan(a.seed, a.tokenizer)
target = eos[a.resend]
for i in range(target + 1):
    q = plan.next()
assert q["kind"] == recs[target]["kind"] and len(q["body"]["prompt"]) == recs[target]["prompt_tokens"], (q["kind"], recs[target])
body = dict(q["body"], logprobs=5)
print("rebuilt:", q["kind"], q["path"], "prompt tokens", len(body.get("prompt", [])), "(logged", recs[target]["prompt_tokens"], ")",
      "max_tokens", body["max_tokens"], "T", body.get("temperature"), "priority", body.get("priority"))
tail = plan.tok.decode(body["prompt"][-40:]) if "prompt" in body else ""
print("prompt tail:", repr(tail))
h = {"Authorization": f"Bearer {a.key}", "Content-Type": "application/json"}


def send(b):
    d = json.loads(urllib.request.urlopen(urllib.request.Request(a.url + q["path"], json.dumps(b).encode(), h), timeout=3600).read())
    c = d["choices"][0]
    lp = c.get("logprobs") or {}
    return c.get("finish_reason"), d["usage"], (c.get("text") or "")[:200], (lp.get("top_logprobs") or [None])[0]


runs = [("resend", body)]
if a.hit_vs_miss:
    g = dict(body, temperature=0.0, max_tokens=1)
    runs = [("T=0 hit", g), ("T=0 miss (cache_salt)", dict(g, cache_salt="soak-eos-miss")), ("T=0 hit again", g)]
for name, b in runs:
    fin, usage, text, top = send(b)
    print(f"{name}: finish {fin} usage {usage} text {text!r}")
    print(f"  first-token top-5: {top}")
