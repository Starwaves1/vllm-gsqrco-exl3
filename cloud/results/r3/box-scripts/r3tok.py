"""R3-28 token-corruption hunt (stdlib + `tokenizers`): greedy chat requests whose outputs are checked
three ways, on raw response bytes (no client-side decoding before the checks):
  utf8   the raw body (SSE or JSON) is valid UTF-8 and holds no U+FFFD
  text   the emitted text (reasoning + content, think tags dropped) equals the local decode of the
         generated token ids (return_tokens_as_token_ids): catches dropped / duplicated / garbled
         characters added after sampling (detokenizer, reasoning parser, streaming deltas, proxy)
  ids    token ids equal across stream vs non-stream at the same concurrency, and vs the c=1
         non-stream reference of the same server (greedy; under MTP batching can legitimately change
         numerics at c > 1, so c=1 is the strict check)
Streamed chunk boundaries (one SSE chunk per engine step) are kept, so a mismatch position can be
compared with step (accepted-draft) boundaries.

  r3tok.py run    --url U --out DIR --tag T [--conc 1,2,4,8] [--temps 0,0.7] [--max-tokens 400] [--plp-pass]
  r3tok.py report DIR... [--ref eager]   (tables across runs; DIR/<tag>.jsonl)
Corruption flags per request: HTTP error, invalid UTF-8 / U+FFFD, text != decode(ids), EOS while still
in reasoning, characters outside Latin/Greek/punctuation/math/emoji (prompts are English), a fragment
of 12-80 chars repeated 4+ times back to back; plus token ids vs the eager server's T=0 c=1 run.
"""

import argparse
import concurrent.futures as cf
import json
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r3load  # noqa: E402

SUBJECTS = ["the 1914–1918 war and its causes", "Zurich's “old town” and its history", "the URL https://example.com/a?b=1&c=2",
            "pre-war Berlin in the 1920s—1930s", "cafe prices: $3.50, €2.99 and £4.10", "the em dash — vs the en dash –",
            "naïve résumé writing tips", "co-op re-entry rules for pre-war buildings", "the years 1994, 1999 and 2026",
            "IPv4 192.168.0.1 and port 8080", "the formula E = mc² and ½ + ¼", "Ångström units in optics",
            "file paths like C:\\Users\\Admin\\file.txt", "§12(b) of the “Act” on taxes", "Krakow and Lodz train routes",
            "a 3×3 matrix with −1 entries", "100°C and 212°F boiling points", "“nested ‘quotes’ in dialogue”",
            "the year 2026—and beyond", "x ≤ y ≥ z ≠ w inequalities", "the history of the printing press",
            "how TCP congestion control works", "the rules of chess castling", "photosynthesis step by step",
            "the Treaty of Versailles (1919)", "binary search with an off-by-one bug", "the 1969 Moon landing timeline",
            "sourdough starter maintenance", "the Python GIL and threads", "compound interest at 4.5% over 30 years"]
TEMPLATES = ["Write a careful answer about {}, using em dashes, curly quotes and exact numbers.",
             "Explain {} in detail; include a numbered list and one URL."]


def prompts():
    return [TEMPLATES[i % 2].format(s) for i, s in enumerate(SUBJECTS)]


_tok = None


def decode(ids):
    global _tok
    if _tok is None:
        import prompts as P
        from tokenizers import Tokenizer
        _tok = Tokenizer.from_file(str(P.HF_CONFIG / "tokenizer.json"))
    return _tok.decode(ids, skip_special_tokens=True)


def norm(t: str) -> str:
    return "".join(t.replace("<think>", "").replace("</think>", "").split())


ALLOWED = [(0x0000, 0x024F), (0x0370, 0x03FF), (0x2000, 0x206F), (0x2070, 0x209F), (0x20A0, 0x20CF), (0x2100, 0x21FF),
           (0x2200, 0x22FF), (0x2300, 0x23FF), (0x2460, 0x24FF), (0x2500, 0x27BF), (0x1F300, 0x1FAFF), (0xFE0F, 0xFE0F)]
REPEAT = None


def odd_chars(t: str) -> str:
    return "".join(ch for ch in t if not any(a <= ord(ch) <= b for a, b in ALLOWED))


def repeated(t: str):
    global REPEAT
    import re
    if REPEAT is None:
        REPEAT = re.compile(r"(.{12,80}?)\1{3,}", re.S)
    m = REPEAT.search(t)
    return m.group(1)[:40] if m else None


def one(srv, i, text, temp, max_tokens, stream=True, seed=None):
    body = {"model": r3load.MODEL, "messages": [{"role": "user", "content": text}], "max_tokens": max_tokens,
            "temperature": temp, "seed": 7 + i if seed is None else seed, "logprobs": True,
            "return_tokens_as_token_ids": True, "stream": stream}
    if stream:
        body["stream_options"] = {"include_usage": True}
    if temp > 0:  # the model's generation_config sampling (production's bench traffic)
        body.update(top_k=20, top_p=0.95)
    c = srv.conn(timeout=900)
    try:
        c.request("POST", "/v1/chat/completions", body=json.dumps(body), headers=srv.hdr)
        r = c.getresponse()
        raw = r.read()
        status = r.status
    except Exception as e:  # noqa: BLE001
        raw, status = repr(e).encode(), -1
    finally:
        c.close()
    rec = {"i": i, "temp": temp, "status": status, "fffd": raw.count(b"\xef\xbf\xbd")}
    try:
        s = raw.decode("utf-8")
        rec["utf8_ok"] = True
    except UnicodeDecodeError as e:
        rec["utf8_ok"] = False
        rec["utf8_err"] = str(e)[:120]
        s = raw.decode("utf-8", "replace")
    ids, content, reasoning, chunks, finish = [], [], [], [], None
    try:
        if not stream:
            o = json.loads(s)
            ch = o["choices"][0]
            m = ch["message"]
            content.append(m.get("content") or "")
            reasoning.append(m.get("reasoning") or m.get("reasoning_content") or "")
            finish = ch.get("finish_reason")
            ids = [int(t["token"].split(":", 1)[1]) for t in (ch.get("logprobs") or {}).get("content") or []]
        for line in (s.splitlines() if stream else []):
            if not line.startswith("data: ") or line[6:].strip() == "[DONE]":
                continue
            o = json.loads(line[6:])
            for ch in o.get("choices") or []:
                d = ch.get("delta") or {}
                content.append(d.get("content") or "")
                reasoning.append(d.get("reasoning") or d.get("reasoning_content") or "")
                finish = ch.get("finish_reason") or finish
                lp = ((ch.get("logprobs") or {}).get("content")) or []
                if lp:
                    chunks.append(len(ids))
                    ids += [int(t["token"].split(":", 1)[1]) for t in lp]
    except Exception as e:  # noqa: BLE001
        rec["parse_err"] = repr(e)[:160]
    rtext, ctext = "".join(reasoning), "".join(content)
    emitted = rtext + ctext
    ref = decode(ids) if ids else ""
    a, b = norm(ref), norm(emitted)
    rec.update({"ids": ids, "chunks": chunks, "finish": finish, "n_tokens": len(ids),
                "reasoning_chars": len(rtext), "content_chars": len(ctext),
                "eos_in_reasoning": finish == "stop" and len(ids) < max_tokens and not ctext.strip(),
                "odd": odd_chars(emitted)[:20], "repeat": repeated(emitted), "text_ok": a == b})
    if a != b:
        k = next((j for j, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        rec["text_diff"] = {"at": k, "decoded": a[max(0, k - 20):k + 20], "emitted": b[max(0, k - 20):k + 20]}
    rec["text_tail"] = emitted[-160:]
    return rec


def plp_client(srv, stop):
    """Short prompt_logprobs requests back to back (the hotfix NaN path) beside the main load."""
    n = 0
    while not stop.is_set():
        body = {"model": r3load.MODEL, "prompt": "The capital of Denmark is", "echo": True, "logprobs": 1,
                "max_tokens": 4, "temperature": 0}
        try:
            srv.post("/v1/completions", body, timeout=60)
        except Exception:  # noqa: BLE001  (400 nan on an unpatched server is expected)
            pass
        n += 1
    return n


def flagged(r):
    return (r["status"] != 200 or not r["utf8_ok"] or r["fffd"] or not r["text_ok"] or r["eos_in_reasoning"]
            or r["odd"] or r["repeat"])


def cmd_run(a):
    srv, out = r3load.Server(a.url), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    P = prompts()

    def batch(f, c, temp, label):
        t0 = time.time()
        nreq = max(len(P), a.per_conc * c)  # enough requests that c stay running most of the batch
        with cf.ThreadPoolExecutor(c) as ex:
            recs = list(ex.map(lambda j: one(srv, j % len(P), P[j % len(P)], temp, a.max_tokens,
                                             not a.nonstream, 7 + j), range(nreq)))
        for r in recs:
            r.update({"tag": a.tag, "conc": c, "pass": label})
            f.write(json.dumps(r) + "\n")
        f.flush()
        print(f"{a.tag} {label} c={c} T={temp}: {len(recs)} requests in {time.time() - t0:.0f}s, "
              f"{sum(map(flagged, recs))} flagged", flush=True)

    with open(out / f"{a.tag}.jsonl", "w") as f:
        for temp in [float(t) for t in a.temps.split(",")]:
            for c in [int(x) for x in a.conc.split(",")]:
                batch(f, c, temp, "main")
        if a.plp_pass:
            stop = threading.Event()
            plp = cf.ThreadPoolExecutor(1).submit(plp_client, srv, stop)
            batch(f, 4, 0.0, "plp")
            stop.set()
            print(f"plp client sent {plp.result()} requests")


def first_div(x, y):
    return next((j for j, (p, q) in enumerate(zip(x, y)) if p != q), None if len(x) == len(y) else min(len(x), len(y)))


def cmd_report(a):
    runs = {}
    for d in a.dirs:
        for p in sorted(Path(d).glob("*.jsonl")):
            for line in open(p):
                r = json.loads(line)
                runs.setdefault(r["tag"], []).append(r)
    ref = {}
    if a.ref in runs:  # T=0, c=1 of the eager server
        ref = {r["i"]: r for r in runs[a.ref] if r["temp"] == 0 and r["conc"] == 1 and r["pass"] == "main"}
    hdr = ("tag          pass  T  c   n  flagged  http  utf8  fffd  text  eos_reason  odd_script  repeat  "
           "ids!=eager(T0)  stop_where_eager_ran")
    print(hdr)
    details = []
    for tag, rs in runs.items():
        for (ps, temp, cc) in sorted({(r["pass"], r["temp"], r["conc"]) for r in rs}):
            sel = [r for r in rs if r["pass"] == ps and r["temp"] == temp and r["conc"] == cc]
            div = early = 0
            if temp == 0 and ref:
                for r in sel:
                    e = ref.get(r["i"])
                    if e and r["i"] == r.get("i") and r["ids"] != e["ids"]:
                        div += 1
                    if e and r["finish"] == "stop" and e["finish"] == "length":
                        early += 1
            print(f"{tag:12s} {ps:5s} {temp:3.1f} {cc:2d} {len(sel):3d}  {sum(map(flagged, sel)):7d}  "
                  f"{sum(r['status'] != 200 for r in sel):4d}  {sum(not r['utf8_ok'] for r in sel):4d}  "
                  f"{sum(r['fffd'] > 0 for r in sel):4d}  {sum(not r['text_ok'] for r in sel):4d}  "
                  f"{sum(r['eos_in_reasoning'] for r in sel):10d}  {sum(bool(r['odd']) for r in sel):10d}  "
                  f"{sum(bool(r['repeat']) for r in sel):6d}  {div:14d}  {early:20d}")
            for r in sel:
                if flagged(r):
                    details.append(f"  {tag} {ps} T={temp} c={r['conc']} prompt {r['i']}: finish={r['finish']} "
                                   f"n={r['n_tokens']} odd={r['odd']!r} repeat={r['repeat']!r} text_diff={r.get('text_diff')} "
                                   f"tail={r['text_tail'][-80:]!r}")
    print("\n".join(details[: a.details]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--url", default=os.environ.get("GSQ_URL", "http://127.0.0.1:18090"))
    p.add_argument("--out", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--conc", default="1,2,4,8")
    p.add_argument("--temps", default="0,0.7")
    p.add_argument("--max-tokens", type=int, default=400)
    p.add_argument("--plp-pass", action="store_true")
    p.add_argument("--nonstream", action="store_true")
    p.add_argument("--per-conc", type=int, default=0, help="requests per batch = max(30, per_conc x c)")
    p = sub.add_parser("report")
    p.add_argument("dirs", nargs="+")
    p.add_argument("--ref", default="eager")
    p.add_argument("--details", type=int, default=60)
    a = ap.parse_args()
    {"run": cmd_run, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
