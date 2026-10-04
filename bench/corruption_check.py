"""Generation-corruption check against an OpenAI-compatible vLLM server (EXL3 opt job 20).

  python bench/corruption_check.py run --url URL --model NAME --label L --out DIR [--conc 1,2,4,8]
        [--temps 0,0.7] [--max-tokens 512] [--prompts FILE] [--n-prompts 38]
  python bench/corruption_check.py compare --ref DIR/ref.jsonl DIR/a.jsonl DIR/b.jsonl ...

run: chat completions with reasoning on (the served template's default), streaming; the raw SSE bytes
are kept and every token id comes back through vLLM's return_tokens_as_token_ids. Each concurrency
level sends every prompt at every temperature (T=0.7 with a per-request seed), so 38 prompts x 2
temperatures x 4 levels = 304 requests per config. One JSON line per response.

compare: per config, against the reference (an --enforce-eager server, T=0) on the same prompt:
  early_eos  finish "stop" before max_tokens, the text not ending a sentence, while the reference
             for that prompt ran to max_tokens
  repeat     a token 8-gram occurring >= 4 times that occurs < 2 times in the reference
  foreign    Han / kana / Hangul / Cyrillic / Arabic / Thai characters where the reference for that
             prompt and the prompt itself have none
  bad_utf8   U+FFFD in the text or undecodable SSE bytes
  mismatch   (T=0) token ids differ from the reference before either ends: reported with the first
             divergence position, not counted as corruption (batching changes the numerics)
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

TOPICS = ["B-trees", "TCP congestion control", "the Rust borrow checker", "Kubernetes pod scheduling",
          "JPEG compression", "Raft consensus", "garbage collection in the JVM", "SQL query planning",
          "the CAP theorem", "CUDA warp divergence", "photosynthesis", "the French Revolution",
          "compound interest", "how vaccines train the immune system", "plate tectonics",
          "public-key cryptography", "Bloom filters", "the Fourier transform", "git rebase vs merge"]
TASKS = ["Explain {t} to a senior engineer in about 300 words, with one concrete example.",
         "Write a short Python script that demonstrates {t}, with comments, then explain the output."]
FOREIGN = re.compile("[Ѐ-ӿ؀-ۿ฀-๿぀-ヿ㐀-䶿一-鿿가-힯]")


def prompts(path: str | None, n: int) -> list[str]:
    out = []
    if path and os.path.exists(path):
        out = [json.loads(line)["prompt"] for line in open(path) if line.strip()]
    out += [task.format(t=t) for t in TOPICS for task in TASKS]
    return out[:n]


def one(url, key, model, prompt, temp, seed, max_tokens, stream=True, top_k=None, top_p=None):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": temp, "stream": stream, "logprobs": True, "return_tokens_as_token_ids": True}
    if stream:
        body["stream_options"] = {"include_usage": True}
    if temp > 0:
        body["seed"] = seed
        if top_k is not None:
            body["top_k"] = top_k
        if top_p is not None:
            body["top_p"] = top_p
    req = urllib.request.Request(url + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    t0, raw, ids, text, reasoning, finish, bad = time.time(), bytearray(), [], [], [], None, 0
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            if not stream:  # one JSON body: same fields as the stream's deltas, in "message"
                raw += r.read()
                try:
                    body_out = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    body_out, bad = {}, 1
                for ch in body_out.get("choices") or []:
                    m = ch.get("message") or {}
                    text.append(m.get("content") or "")
                    reasoning.append(m.get("reasoning_content") or m.get("reasoning") or "")
                    for c in ((ch.get("logprobs") or {}).get("content") or []):
                        tok = c.get("token", "")
                        ids.append(int(tok.split(":", 1)[1]) if tok.startswith("token_id:") else -1)
                    finish = ch.get("finish_reason") or finish
            for line in (r if stream else []):
                raw += line
                if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                    continue
                try:
                    ev = json.loads(line[6:].decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    bad += 1
                    continue
                for ch in ev.get("choices") or []:
                    d = ch.get("delta") or {}
                    text.append(d.get("content") or "")
                    reasoning.append(d.get("reasoning_content") or d.get("reasoning") or "")
                    for c in ((ch.get("logprobs") or {}).get("content") or []):
                        tok = c.get("token", "")
                        ids.append(int(tok.split(":", 1)[1]) if tok.startswith("token_id:") else -1)
                    finish = ch.get("finish_reason") or finish
        status = 200
    except Exception as e:  # noqa: BLE001
        status, finish = -1, f"error: {e!r}"[:200]
    return {"prompt_idx": None, "temp": temp, "status": status, "finish": finish, "n_ids": len(ids), "ids": ids,
            "content": "".join(text), "reasoning": "".join(reasoning), "bad_sse": bad,
            "raw_bytes": len(raw), "seconds": round(time.time() - t0, 2)}


def run(a):
    ps = prompts(a.prompts, a.n_prompts)
    key = os.environ.get("GSQ_API_KEY", "gsq-local-test")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    temps = [float(t) for t in a.temps.split(",")]
    streams = {"1": [True], "0": [False], "both": [True, False]}[a.stream]
    with open(out / f"{a.label}.jsonl", "w") as f:
        for c in [int(x) for x in a.conc.split(",")]:
            jobs = [(i, t, st) for st in streams for t in temps for i in range(len(ps))]
            with cf.ThreadPoolExecutor(c) as ex:
                futs = {ex.submit(one, a.url, key, a.model, ps[i], t, 1000 + i, a.max_tokens, st, a.top_k, a.top_p):
                        (i, t, st) for i, t, st in jobs}
                for fu in cf.as_completed(futs):
                    i, t, st = futs[fu]
                    r = fu.result()
                    r.update(prompt_idx=i, conc=c, stream=st, label=a.label, prompt=ps[i][:80], max_tokens=a.max_tokens)
                    f.write(json.dumps(r) + "\n")
                    f.flush()
            print(f"{a.label} c={c}: {len(jobs)} requests done", flush=True)


def ngram_counts(ids, n=8):
    cnt = {}
    for i in range(len(ids) - n + 1):
        g = tuple(ids[i:i + n])
        cnt[g] = cnt.get(g, 0) + 1
    return cnt


def flags(r, ref, prompt_text):
    text = r["reasoning"] + r["content"]
    f = {}
    ends = text.rstrip()[-1:] in tuple(".!?)]}`\"'*:>|") if text.strip() else False
    f["early_eos"] = (r["finish"] == "stop" and r["n_ids"] < r["max_tokens"] and not ends
                      and ref is not None and ref["finish"] == "length")
    rc = ngram_counts(ref["ids"]) if ref else {}
    f["repeat"] = any(v >= 4 and rc.get(g, 0) < 2 for g, v in ngram_counts(r["ids"]).items())
    ref_text = (ref["reasoning"] + ref["content"]) if ref else ""
    f["foreign"] = bool(FOREIGN.search(text)) and not FOREIGN.search(ref_text) and not FOREIGN.search(prompt_text)
    f["bad_utf8"] = "�" in text or r["bad_sse"] > 0
    f["error"] = r["status"] != 200
    return f


def compare(a):
    ref = {}
    for line in open(a.ref):
        r = json.loads(line)
        if r["temp"] == 0 and r["status"] == 200:
            ref.setdefault(r["prompt_idx"], r)
    ps = {}
    for path in a.runs:
        rows = [json.loads(line) for line in open(path)]
        for r in rows:
            ps.setdefault(r["prompt_idx"], r["prompt"])
        label = rows[0]["label"] if rows else Path(path).stem
        tot, kinds, mism, divpos, examples = 0, {}, 0, [], []
        by_temp, by_conc = {}, {}
        for r in rows:
            rf = ref.get(r["prompt_idx"])
            fl = flags(r, rf, r["prompt"])
            bad = any(fl.values())
            tot += 1
            bt = by_temp.setdefault(r["temp"], [0, 0])
            bt[0] += 1
            bt[1] += bad
            bc = by_conc.setdefault(r["conc"], [0, 0])
            bc[0] += 1
            bc[1] += bad
            for k, v in fl.items():
                kinds[k] = kinds.get(k, 0) + bool(v)
            if bad and len(examples) < 8:
                examples.append({"prompt_idx": r["prompt_idx"], "temp": r["temp"], "conc": r["conc"],
                                 "stream": r.get("stream", True), "flags":
                                 [k for k, v in fl.items() if v], "tail": (r["reasoning"] + r["content"])[-160:]})
            if r["temp"] == 0 and rf is not None:
                n = min(len(r["ids"]), len(rf["ids"]))
                d = next((i for i in range(n) if r["ids"][i] != rf["ids"][i]), None)
                if d is not None:
                    mism += 1
                    divpos.append(d)
        n_bad = sum(by_temp[t][1] for t in by_temp)
        divpos.sort()
        print(json.dumps({"config": label, "requests": tot, "corrupt": n_bad, "rate": round(n_bad / max(tot, 1), 4),
                          "by_temp": {str(t): {"n": v[0], "corrupt": v[1]} for t, v in sorted(by_temp.items())},
                          "by_conc": {str(c): {"n": v[0], "corrupt": v[1]} for c, v in sorted(by_conc.items())},
                          "by_kind": kinds, "t0_token_mismatch": mism,
                          "t0_first_divergence_median": divpos[len(divpos) // 2] if divpos else None,
                          "examples": examples}, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--url", required=True)
    r.add_argument("--model", default="qwen3.8-27b")
    r.add_argument("--label", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--conc", default="1,2,4,8")
    r.add_argument("--temps", default="0,0.7")
    r.add_argument("--max-tokens", type=int, default=512)
    r.add_argument("--prompts")
    r.add_argument("--n-prompts", type=int, default=38)
    r.add_argument("--stream", default="1", choices=["1", "0", "both"])
    r.add_argument("--top-k", type=int)
    r.add_argument("--top-p", type=float)
    c = sub.add_parser("compare")
    c.add_argument("--ref", required=True)
    c.add_argument("runs", nargs="+")
    a = ap.parse_args()
    run(a) if a.cmd == "run" else compare(a)


if __name__ == "__main__":
    sys.exit(main())
