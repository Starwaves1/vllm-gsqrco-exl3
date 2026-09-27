"""200k context fits: production's flags (gpu-memory-utilization 0.94, fp8 KV,
max-model-len 200000, MTP k=3, CUDA graphs) on the GGUF, HANDOFF §2 "Fits".

1. The server started (vLLM refuses to start when max-model-len does not fit) and its log
   reports KV capacity >= 200,000 tokens, i.e. "Maximum concurrency for 200,000 tokens per
   request" >= 1.0.
2. A real ~195k-token request completes, and the server is still healthy with no device
   fault in its log afterwards.

Uses the session server from conftest (serve-gsq.sh, so the argv is production's; the
script prints the diff). Reusing a running server needs GSQ_SERVER_LOG for step 1.
TODO(GPU): read the startup log once and pin the exact KV-capacity numbers in STATUS.md
(expected well above production's 267 blocks: the weights are 12.1 GB vs 15.8 GB).
"""

import json
import re
import urllib.request

import pytest

from gsq_gpu import API_KEY

LONG_TOKENS = 195_000


def _post(url, body, timeout=3600):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def test_kv_capacity_200k(gsq_server):
    _, log = gsq_server
    if log is None:
        pytest.skip("reused server without GSQ_SERVER_LOG")
    text = log.read_text(errors="replace")
    m = re.search(r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x", text)
    assert m, "no KV capacity line in the server log"
    assert int(m.group(1).replace(",", "")) == 200_000, f"max-model-len is {m.group(1)}, not production's 200,000"
    assert float(m.group(2)) >= 1.0, f"200k does not fit: concurrency {m.group(2)}x"
    kv = re.search(r"GPU KV cache size: ([\d,]+) tokens", text)
    print(f"\nKV: {kv.group(0) if kv else '?'}; {m.group(0)}")


def test_long_request_195k(gsq_server):
    import prompts  # bench/parity/prompts.py (deterministic, salted)
    import random

    from tokenizers import Tokenizer

    url, log = gsq_server
    tok = Tokenizer.from_file(str(prompts.HF_CONFIG / "tokenizer.json"))
    ids = prompts.build_text("mixed", LONG_TOKENS, random.Random("fit-200k"), tok)
    r = _post(f"{url}/v1/completions", {"model": "qwen3.8-27b", "prompt": ids, "max_tokens": 64, "temperature": 0})
    assert r["usage"]["prompt_tokens"] == len(ids)
    assert r["choices"][0]["text"].strip(), "empty completion"
    with urllib.request.urlopen(f"{url}/health", timeout=10) as h:
        assert h.status == 200
    if log is not None:
        bad = [l for l in log.read_text(errors="replace").splitlines() if "illegal memory access" in l or "CUDA error" in l]
        assert not bad, bad[:5]
