"""prompt_logprobs / echo+logprobs against a running EXL3 server (job 18): a ~N-token prompt
(code from the venv's vLLM sources, deterministic), POST /v1/completions with
  1. prompt_logprobs=1, max_tokens=1        2. echo=true, logprobs=1, max_tokens=1
Checks HTTP 200, the number of prompt positions with a logprob, NaN / inf / None counts, and that
/health answers afterwards. Prints one JSON line per request.
  python bench/exl3_logprobs_check.py URL MODEL TOKENIZER_DIR N_TOKENS"""
import json
import math
import sys
import urllib.request
from pathlib import Path

url, model, tok_dir, n_tok = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(tok_dir)
import vllm  # noqa: E402

src = "".join(p.read_text() for p in sorted(Path(vllm.__file__).parent.glob("v1/core/*.py")))
ids = tok(src)["input_ids"][:n_tok]
prompt = tok.decode(ids)
key = {"Authorization": "Bearer " + __import__("os").environ.get("GSQ_API_KEY", "gsq-local-test"),
       "Content-Type": "application/json"}


def post(body):
    req = urllib.request.Request(url + "/v1/completions", data=json.dumps(body).encode(), headers=key)
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode()[:500]}
    except Exception as e:  # noqa: BLE001
        return -1, {"error": repr(e)[:500]}


def stats(vals):
    vals = list(vals)
    none = sum(v is None for v in vals)
    f = [v for v in vals if v is not None]
    return {"n": len(vals), "none": none, "nan": sum(math.isnan(v) for v in f), "inf": sum(math.isinf(v) for v in f),
            "min": min(f) if f else None}


def healthy():
    try:
        with urllib.request.urlopen(urllib.request.Request(url + "/health", headers=key), timeout=30) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


base = {"model": model, "prompt": prompt, "max_tokens": 1, "temperature": 0}
code, r = post({**base, "prompt_logprobs": 1})
plp = (r.get("choices") or [{}])[0].get("prompt_logprobs") if code == 200 else None
vals = [] if not plp else [None if d is None else max(v["logprob"] for v in d.values()) for d in plp[1:]]
print(json.dumps({"request": "prompt_logprobs", "prompt_tokens": len(ids), "http": code, **stats(vals),
                  "error": r.get("error"), "healthy_after": healthy()}), flush=True)
code, r = post({**base, "echo": True, "logprobs": 1})
lp = (r.get("choices") or [{}])[0].get("logprobs") if code == 200 else None
vals = [] if not lp else lp.get("token_logprobs", [])[1:]
print(json.dumps({"request": "echo_logprobs", "prompt_tokens": len(ids), "http": code, **stats(vals),
                  "error": r.get("error"), "healthy_after": healthy()}), flush=True)
