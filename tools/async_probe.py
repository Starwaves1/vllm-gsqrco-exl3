"""A/B probe for production's async-scheduling switch: same T=0 requests before and after.

  python3 tools/async_probe.py OUT.json [--port 18081]

Runs c=1, 4 and 8 waves of identical chat requests (T=0, ignore_eos, 400 tokens) and records
wall time, decode tok/s per request and the full text, so the two runs can be diffed for
both speed and output drift. Stdlib only.
"""
import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

PROMPTS = [
    "Explain how a bloom filter works and when it gives false positives.",
    "Write a Python function that merges overlapping intervals, with a short docstring.",
    "List the causes of the French Revolution in five bullet points.",
    "What is the difference between TCP and UDP? Answer in one paragraph.",
    "Convert 72 degrees Fahrenheit to Celsius and show the steps.",
    "Write a haiku about a GPU running out of memory.",
    "Describe the water cycle for a ten-year-old.",
    "Give three tips for writing readable SQL.",
]


def one(port, prompt):
    body = {
        "model": "qwen3.8-27b",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 400,
        "ignore_eos": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    t = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=600))
    dt = time.time() - t
    n = r["usage"]["completion_tokens"]
    return {"prompt": prompt, "s": round(dt, 3), "tokens": n, "tok_s": round(n / dt, 1),
            "text": r["choices"][0]["message"]["content"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--port", type=int, default=18081)
    a = ap.parse_args()
    one(a.port, "Say hi.")  # warm-up
    res = {}
    for c in (1, 4, 8):
        t = time.time()
        with ThreadPoolExecutor(c) as ex:
            rows = list(ex.map(lambda p: one(a.port, p), PROMPTS[:c]))
        wall = time.time() - t
        res[c] = {"wall_s": round(wall, 2), "agg_tok_s": round(sum(r["tokens"] for r in rows) / wall, 1),
                  "rows": rows}
        print(f"c={c} wall {wall:.1f}s aggregate {res[c]['agg_tok_s']} tok/s "
              f"per-request {[r['tok_s'] for r in rows]}")
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
