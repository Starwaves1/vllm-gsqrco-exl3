"""Torture-smoke reproductions: echo+logprobs (HTTP 400 "nan") and prompt_logprobs on 2-4k prompts
(EngineCore OOM in the lm_head). Stdlib only; sends one request at a time and records status,
error text, NaN/inf presence in the returned logprobs, and GPU memory before/after.

  r3plp.py --url U --out FILE [--long 512,1024,2048,2600,3936]
Exit 0 always (the caller compares servers); every result line is a JSON record in FILE.
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r3load  # noqa: E402


def gpu_mib():
    try:
        return int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                  capture_output=True, text=True, timeout=10).stdout.split()[0])
    except Exception:  # noqa: BLE001
        return -1


def bad_floats(obj):
    """Count NaN/inf floats anywhere in a parsed JSON value (json.loads accepts NaN/Infinity)."""
    if isinstance(obj, float):
        return int(math.isnan(obj) or math.isinf(obj))
    if isinstance(obj, dict):
        return sum(bad_floats(v) for v in obj.values())
    if isinstance(obj, list):
        return sum(bad_floats(v) for v in obj)
    return 0


def send(srv, body, timeout=600):
    c = srv.conn(timeout)
    t = time.time()
    try:
        c.request("POST", "/v1/completions", body=json.dumps(body), headers=srv.hdr)
        r = c.getresponse()
        raw = r.read().decode(errors="replace")
        return r.status, raw, time.time() - t
    except Exception as e:  # noqa: BLE001
        return -1, repr(e), time.time() - t
    finally:
        c.close()


def alive(srv):
    try:
        srv.get("/health", timeout=10)
        return True
    except Exception:  # noqa: BLE001
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=os.environ.get("GSQ_URL", "http://127.0.0.1:18090"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--long", default="512,1024,2048,2600,3936")
    a = ap.parse_args()
    srv = r3load.Server(a.url)
    cases = [
        ("echo-T1", {"prompt": "The capital of Denmark is", "echo": True, "logprobs": 1, "max_tokens": 4, "temperature": 1.0}),
        ("echo-T0", {"prompt": "The capital of Denmark is", "echo": True, "logprobs": 1, "max_tokens": 4, "temperature": 0.0}),
        ("echo-nolp", {"prompt": "The capital of Denmark is", "echo": True, "max_tokens": 4, "temperature": 0.0}),
        ("plp1-short", {"prompt": "The capital of Denmark is", "prompt_logprobs": 1, "max_tokens": 1, "temperature": 0.0}),
        ("lp-only", {"prompt": "The capital of Denmark is", "logprobs": 1, "max_tokens": 4, "temperature": 0.0}),
    ]
    for n in (40, 200):
        cases.append((f"echo-{n}tok", {"prompt": r3load.prompt_ids(f"plp-echo{n}", "chat", n), "echo": True,
                                       "logprobs": 1, "max_tokens": 4, "temperature": 0.0}))
    for n in [int(x) for x in a.long.split(",") if x]:
        cases.append((f"plp-{n}", {"prompt": r3load.prompt_ids(f"plp{n}", "chat", n), "prompt_logprobs": 1,
                                   "max_tokens": 1, "temperature": 0.0}))
    with open(a.out, "w") as f:
        for name, body in cases:
            body = {"model": r3load.MODEL, **body}
            m0 = gpu_mib()
            st, raw, dt = send(srv, body)
            m1 = gpu_mib()
            rec = {"case": name, "status": st, "seconds": round(dt, 2), "gpu_mib_before": m0, "gpu_mib_after": m1}
            try:
                obj = json.loads(raw)
                rec["bad_floats"] = bad_floats(obj)
                if st != 200:
                    rec["error"] = (obj.get("error") or {}).get("message", raw[:300])
                else:
                    ch = obj["choices"][0]
                    plp = ch.get("prompt_logprobs")
                    rec["prompt_logprobs_len"] = len(plp) if plp else 0
                    lp = ch.get("logprobs") or {}
                    rec["token_logprobs_head"] = (lp.get("token_logprobs") or [])[:8]
            except Exception:  # noqa: BLE001
                rec["raw"] = raw[:300]
            rec["alive_after"] = alive(srv)
            print(json.dumps(rec), flush=True)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if not rec["alive_after"]:
                print("server died; stopping this server's cases", flush=True)
                break


if __name__ == "__main__":
    main()
