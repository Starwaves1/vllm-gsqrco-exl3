"""Decode ladder against a running vLLM server: concurrency 1/2/4/8, two passes (keep pass 2), T=0.

  python kit/ladder.py --url URL --model NAME --out ladder.json [--conc 1,2,4,8] [--max-tokens 256]
      [--rounds 2] [--label L] [--note TEXT]

Per cohort c: rounds x c chat requests (bench/corruption_check.py's prompts), c in flight at a time,
greedy, ignore_eos so each decodes exactly --max-tokens, streamed. Same definitions as the repo's
ladders (cloud/results/exl3-opt/box-scripts/ladder_summ.py):
  TPOT      (last token time - first token time) / (completion tokens - 1), mean over requests
  tok/s     decode throughput C * 1000 / mean TPOT
  tok/step  1 + accepted draft tokens / drafts, from vllm:spec_decode_* deltas on /metrics (1 without MTP)
  ms/step   C * 1000 / tok/s * tok/step  (time of one engine step: the GSQ metric)
Nothing else may use the server meanwhile (the counters are global).
"""

import argparse
import concurrent.futures as cf
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
from corruption_check import prompts  # noqa: E402


def counters(url):
    d = {}
    with urllib.request.urlopen(f"{url}/metrics", timeout=30) as r:
        for line in r.read().decode().splitlines():
            m = re.match(r"(vllm:spec_decode_num_(?:drafts|accepted_tokens))(?:_total)?(\{[^}]*\})?\s+([0-9.eE+-]+)$", line.strip())
            if m and "position" not in (m.group(2) or ""):
                d[m.group(1)] = d.get(m.group(1), 0.0) + float(m.group(3))
    return d


def one(url, model, prompt, max_tokens):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0.0, "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(f"{url}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0, first, last, ntok = time.perf_counter(), None, None, None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                ntok = d["usage"]["completion_tokens"]
            ch = (d.get("choices") or [{}])[0]
            delta = ch.get("delta") or {}
            if delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning"):
                now = time.perf_counter()
                first = first or now
                last = now
    if not ntok or ntok < 2 or first is None or last == first:
        raise RuntimeError(f"short response: {ntok} tokens")
    return {"ttft_ms": (first - t0) * 1e3, "tpot_ms": (last - first) * 1e3 / (ntok - 1), "tokens": ntok}


def cohort(url, model, ps, c, rounds, max_tokens):
    before = counters(url)
    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(c) as ex:
        res = list(ex.map(lambda i: one(url, model, ps[i % len(ps)], max_tokens), range(rounds * c)))
    wall = time.perf_counter() - t0
    after = counters(url)
    drafts = after.get("vllm:spec_decode_num_drafts", 0) - before.get("vllm:spec_decode_num_drafts", 0)
    acc = after.get("vllm:spec_decode_num_accepted_tokens", 0) - before.get("vllm:spec_decode_num_accepted_tokens", 0)
    tpot = sum(r["tpot_ms"] for r in res) / len(res)
    tok_s = c * 1000 / tpot
    tok_step = 1 + acc / drafts if drafts else 1.0
    return {"requests": len(res), "tokens": sum(r["tokens"] for r in res), "wall_s": round(wall, 2),
            "e2e_tok_s": round(sum(r["tokens"] for r in res) / wall, 1), "mean_tpot_ms": round(tpot, 3),
            "mean_ttft_ms": round(sum(r["ttft_ms"] for r in res) / len(res), 1), "tok_s": round(tok_s, 1),
            "tok_step": round(tok_step, 3), "ms_step": round(c * 1000 / tok_s * tok_step, 3)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--conc", default="1,2,4,8")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--label", default="")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    ps = prompts(None, 38)
    out = {"model": a.label or a.model, "note": a.note, "max_tokens": a.max_tokens, "rounds": a.rounds}
    for p in (1, 2):
        out[f"pass{p}"] = {}
        for c in [int(x) for x in a.conc.split(",")]:
            r = cohort(a.url, a.model, ps, c, a.rounds, a.max_tokens)
            out[f"pass{p}"][str(c)] = r
            print(f"pass {p} c={c}: {r['tok_s']} tok/s, {r['ms_step']} ms/step, {r['tok_step']} tok/step", flush=True)
            a.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
