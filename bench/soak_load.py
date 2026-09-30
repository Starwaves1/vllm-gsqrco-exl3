"""Load generator and report for bench/soak.sh.

  soak_load.py load   --url U --api-key K --tokenizer DIR --conc 2 --hours 24 --out load.jsonl
  soak_load.py report --dir SOAK_DIR --restarts N --hours H
  soak_load.py load --dry-run --tokenizer DIR     print 20 planned requests, send nothing

The mix exercises the paths production's IMA history points at: MTP verify at batch 1-2,
prefix-cache hits on long prompts (align mode), aborted streams, tool calls through
qwen3_coder, priority scheduling, and many batch shapes (CUDA graph sizes <= 32).
Every request is checked (HTTP 200, finish_reason, non-empty output) and logged as one
JSON line. Deterministic given --seed.
"""

import argparse
import json
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "bench" / "parity"))
if os.environ.get("GSQ_ALLOW_GPU") != "1":
    import no_gpu  # noqa: F401

MIX = [("chat", 35), ("tool", 15), ("long_new", 10), ("long_hit", 20), ("abort", 10), ("greedy", 10)]
TOOLS = [{"type": "function", "function": {
    "name": "read_file", "description": "Read a file from the repository",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "start_line": {"type": "integer"}},
                   "required": ["path"]}}}]
SHORT = [
    "Write a Python function that merges overlapping intervals, with tests.",
    "Explain the difference between a mutex and a semaphore with a C example.",
    "Summarize how TCP congestion control works, in five bullet points.",
    "Refactor this into idiomatic Rust: for i in 0..v.len() { if v[i] > 3 { out.push(v[i] * 2) } }",
    "What does `git rebase --onto A B C` do? Show a before/after graph.",
    "Draft a SQL schema for a library lending system with constraints.",
]


class Plan:
    def __init__(self, seed: int, tokenizer: str):
        from tokenizers import Tokenizer

        import prompts

        self.rng = random.Random(seed)
        self.tok = Tokenizer.from_file(str(Path(tokenizer) / "tokenizer.json"))
        self.prompts = prompts
        self.long_cache: list[list[int]] = []
        self.lock = threading.Lock()

    def next(self) -> dict:
        with self.lock:
            r = self.rng
            kind = r.choices([k for k, _ in MIX], [w for _, w in MIX])[0]
            base = {"model": "qwen3.8-27b", "priority": r.randint(-10, 10)}
            if kind in ("chat", "abort", "greedy"):
                body = {**base, "messages": [{"role": "user", "content": r.choice(SHORT) + f" (variant {r.getrandbits(32):08x})"}],
                        "max_tokens": r.choice([16, 64, 256, 1024, 2048])}
                body.update({"temperature": 0.0} if kind == "greedy" else {"temperature": 1.0, "top_p": 0.95, "top_k": 20})
                return {"kind": kind, "path": "/v1/chat/completions", "body": body, "stream": kind == "abort",
                        "abort_after": r.randint(1, 40)}
            if kind == "tool":
                body = {**base, "messages": [{"role": "user", "content": "Open src/main.py and tell me what the entry point does."}],
                        "tools": TOOLS, "tool_choice": "auto", "max_tokens": 512, "temperature": 1.0, "top_p": 0.95, "top_k": 20}
                return {"kind": kind, "path": "/v1/chat/completions", "body": body, "stream": False}
            if kind == "long_hit" and self.long_cache:
                ids = r.choice(self.long_cache)
                ids = ids + self.tok.encode(f"\n# follow-up {r.getrandbits(32):08x}: what does the code above do?\n",
                                            add_special_tokens=False).ids
            else:
                n = r.choice([8192, 16384, 32768, 65536, 120000])
                ids = self.prompts.build_text(r.choice(["code", "mixed"]), n, random.Random(r.getrandbits(64)), self.tok)
                self.long_cache = (self.long_cache + [ids])[-4:]
                kind = "long_new"
            body = {**base, "prompt": ids, "max_tokens": r.choice([32, 256, 1024]), "temperature": 1.0, "top_p": 0.95, "top_k": 20}
            return {"kind": kind, "path": "/v1/completions", "body": body, "stream": False}


def send(url: str, key: str, req: dict) -> dict:
    t0 = time.time()
    rec = {"t": t0, "kind": req["kind"]}
    body = dict(req["body"])
    if req["stream"]:
        body["stream"] = True
    h = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    try:
        r = urllib.request.urlopen(urllib.request.Request(url + req["path"], json.dumps(body).encode(), h), timeout=3600)
        if req["stream"]:
            n = 0
            for line in r:
                if line.startswith(b"data: ") and line.strip() != b"data: [DONE]":
                    n += 1
                    if n >= req["abort_after"]:
                        break
            r.close()
            rec.update(status="aborted", chunks=n)
        else:
            d = json.loads(r.read())
            c = d["choices"][0]
            fin = c.get("finish_reason")
            msg = c.get("message", {})
            # a short max_tokens can end inside the reasoning: vLLM's qwen3 parser then returns
            # content None and the tokens in message.reasoning
            text = c.get("text") or msg.get("content") or msg.get("reasoning") or msg.get("reasoning_content") or ""
            tool = msg.get("tool_calls") or []
            ok = fin in ("stop", "length", "tool_calls") and (text.strip() or tool)
            rec.update(status="ok" if ok else "bad_output", finish=fin, tool_calls=len(tool),
                       prompt_tokens=d["usage"]["prompt_tokens"], completion_tokens=d["usage"]["completion_tokens"])
    except urllib.error.HTTPError as e:
        rec.update(status=f"http_{e.code}", error=e.read()[:300].decode(errors="replace"))
    except Exception as e:  # connection refused/reset = server trouble
        rec.update(status="error", error=repr(e)[:300])
    rec["latency"] = time.time() - t0
    return rec


def load(a) -> None:
    plan = Plan(a.seed, a.tokenizer)
    if a.dry_run:
        for _ in range(20):
            q = plan.next()
            b = q["body"]
            print(q["kind"], q["path"], len(b.get("prompt", [])) or "", b["max_tokens"], "stream" if q["stream"] else "")
        return
    if ":18080" in a.url or ":18081" in a.url:
        raise SystemExit("refusing production ports")
    end = time.time() + a.hours * 3600
    out = open(a.out, "a", buffering=1)
    wlock = threading.Lock()

    def worker():
        while time.time() < end:
            rec = send(a.url, a.api_key, plan.next())
            with wlock:
                out.write(json.dumps(rec) + "\n")
            if rec["status"] == "error":
                time.sleep(10)

    ts = [threading.Thread(target=worker, daemon=True) for _ in range(a.conc)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()


def slope_per_hour(ts, ys):
    import numpy as np

    if len(ts) < 10:
        return None
    return float(np.polyfit((np.asarray(ts) - ts[0]) / 3600.0, np.asarray(ys, float), 1)[0])


def report(a) -> None:
    d = Path(a.dir)
    recs = [json.loads(line) for line in (d / "load.jsonl").read_text().splitlines() if line.strip()] if (d / "load.jsonl").exists() else []
    rows = [r.split(",") for r in (d / "monitor.csv").read_text().splitlines()[1:]]
    faults = (d / "faults.log").read_text().splitlines() if (d / "faults.log").exists() else []
    device = [f for f in faults if any(k in f for k in ("illegal memory access", "misaligned address", "unspecified launch failure", "CUDA error"))]
    t = [float(r[0]) for r in rows]
    warm = [i for i, x in enumerate(t) if x - t[0] >= 3600] if t else []  # ignore the first hour (graph capture, caches)
    gpu = slope_per_hour([t[i] for i in warm], [float(rows[i][3]) for i in warm]) if warm else None
    rss = slope_per_hour([t[i] for i in warm], [float(rows[i][4]) for i in warm]) if warm else None
    by = {}
    for r in recs:
        by.setdefault(r["status"], 0)
        by[r["status"]] += 1
    dur_h = (t[-1] - t[0]) / 3600 if len(t) > 1 else 0
    # TODO(GPU): calibrate the growth limits against a baseline (W4A16) soak.
    rep = {"hours_planned": a.hours, "hours_monitored": dur_h, "restarts": a.restarts, "device_faults": device[:20],
           "other_fault_lines": len(faults) - len(device), "requests": by,
           "gpu_mib_per_hour": gpu, "rss_kib_per_hour": rss,
           "limits": {"gpu_mib_growth_total": 256, "rss_kib_growth_total": 1 << 20}}
    ok = (not device and a.restarts == 0 and dur_h >= 0.98 * a.hours
          and all(r[1] == "1" for r in rows)
          and by.get("error", 0) == 0 and by.get("bad_output", 0) == 0 and not any(k.startswith("http_5") for k in by)
          and (gpu is None or gpu * dur_h <= 256) and (rss is None or rss * dur_h <= (1 << 20)))
    rep["pass"] = bool(ok)
    (d / "report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    sys.exit(0 if ok else 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    l = sub.add_parser("load")
    l.add_argument("--url", default=os.environ.get("GSQ_URL", "http://127.0.0.1:18090"))
    l.add_argument("--api-key", default=os.environ.get("GSQ_API_KEY", "gsq-local-test"))
    l.add_argument("--tokenizer", required=True)
    l.add_argument("--conc", type=int, default=2)
    l.add_argument("--hours", type=float, default=24)
    l.add_argument("--seed", type=int, default=20260927)
    l.add_argument("--out", default="load.jsonl")
    l.add_argument("--dry-run", action="store_true")
    r = sub.add_parser("report")
    r.add_argument("--dir", required=True)
    r.add_argument("--restarts", type=int, default=0)
    r.add_argument("--hours", type=float, default=24)
    a = ap.parse_args()
    load(a) if a.cmd == "load" else report(a)


if __name__ == "__main__":
    main()
