"""MTP acceptance readout, the same workload against vLLM or llama-server.

vLLM:  counters vllm:spec_decode_num_{drafts,draft_tokens,accepted_tokens}[_per_pos] from
       /metrics, before and after the workload (nothing else may use the server meanwhile).
llama: per-response timings.draft_n / timings.draft_n_accepted (llama-server b11211,
       server-common.cpp:100).
Acceptance rate = accepted draft tokens / drafted tokens; HANDOFF §2 wants vLLM within
2 points of llama.cpp on the same file, both at k=3.

Workload: production's prompts_real.jsonl (8 chat prompts), one request at a time, at
the model's sampling (temperature 1.0, top_p 0.95, top_k 20) and greedy, fixed seeds.

  python bench/speed/mtp_acceptance.py --engine vllm|llama --url URL [--out r.json]
      [--reference llama.json]   exit 1 if |rate - reference rate| > 0.02 at any temperature
"""

import argparse
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
if os.environ.get("GSQ_ALLOW_GPU") != "1":
    import no_gpu  # noqa: F401  (client only; keeps the rule uniform)

TEMPS = {"default": dict(temperature=1.0, top_p=0.95, top_k=20), "greedy": dict(temperature=0.0)}


def http(url: str, key: str, body: dict | None = None, timeout: float = 1800):
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body else None,
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read().decode()
    return json.loads(data) if body else data


def vllm_counters(url: str, key: str) -> dict:
    d: dict = {}
    for line in http(f"{url}/metrics", key).splitlines():
        m = re.match(r'(vllm:spec_decode_\w+?)(?:_total)?(\{[^}]*\})?\s+([0-9.eE+-]+)$', line.strip())
        if m:
            p = re.search(r'position="(\d+)"', m.group(2) or "")
            k = (m.group(1), p.group(1) if p else None)
            d[k] = d.get(k, 0.0) + float(m.group(3))
    return d


def run(engine: str, url: str, key: str, prompts: list[str], max_tokens: int) -> dict:
    res = {}
    for tname, samp in TEMPS.items():
        before = vllm_counters(url, key) if engine == "vllm" else None
        drafted = accepted = gen = 0
        for i, p in enumerate(prompts):
            body = {"model": "qwen3.8-27b", "messages": [{"role": "user", "content": p}],
                    "max_tokens": max_tokens, "seed": 1000 + i, **samp}
            r = http(f"{url}/v1/chat/completions", key, body)
            gen += r["usage"]["completion_tokens"]
            if engine == "llama":
                t = r.get("timings", {})
                drafted += t.get("draft_n", 0); accepted += t.get("draft_n_accepted", 0)
        out = {"requests": len(prompts), "completion_tokens": gen}
        if engine == "vllm":
            after = vllm_counters(url, key)
            g = lambda k, p=None: after.get((k, p), 0.0) - before.get((k, p), 0.0)
            drafts = g("vllm:spec_decode_num_drafts")
            drafted, accepted = g("vllm:spec_decode_num_draft_tokens"), g("vllm:spec_decode_num_accepted_tokens")
            out["per_position"] = [g("vllm:spec_decode_num_accepted_tokens_per_pos", str(j)) / drafts
                                   for j in range(8) if drafts and ("vllm:spec_decode_num_accepted_tokens_per_pos", str(j)) in after]
            out["mean_accepted_length"] = 1 + accepted / drafts if drafts else None
        out.update(drafted=drafted, accepted=accepted, acceptance_rate=accepted / drafted if drafted else None)
        res[tname] = out
        print(f"{engine} {tname}: acceptance {out['acceptance_rate']}  drafted {drafted:.0f}  accepted {accepted:.0f}", flush=True)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", choices=["vllm", "llama"], required=True)
    ap.add_argument("--url", default=os.environ.get("GSQ_URL", "http://127.0.0.1:18090"))
    ap.add_argument("--api-key", default=os.environ.get("GSQ_API_KEY", "gsq-local-test"))
    ap.add_argument("--prompts", type=Path, default=Path(os.environ.get("GSQ_DEPLOY_REPO", Path.home() / "qwen38-27b-rtx3090")) / "bench/prompts_real.jsonl")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--reference", type=Path, help="llama.cpp readout JSON to compare against")
    a = ap.parse_args()
    if re.search(r":1808[01]\b", a.url):
        raise SystemExit("refusing production ports 18080/18081")
    prompts = [json.loads(line)["prompt"] for line in a.prompts.read_text().splitlines() if line.strip()]
    res = {"engine": a.engine, "url": a.url, "prompts": str(a.prompts), "results": run(a.engine, a.url, a.api_key, prompts, a.max_tokens)}
    ok = True
    if a.reference:
        ref = json.loads(a.reference.read_text())["results"]
        for t, r in res["results"].items():
            d = (r["acceptance_rate"] or 0) - (ref[t]["acceptance_rate"] or 0)
            r["delta_vs_reference"] = d
            ok &= abs(d) <= 0.02
            print(f"{t}: delta vs reference {d:+.4f} ({'ok' if abs(d) <= 0.02 else 'FAIL'})")
    if a.out:
        a.out.write_text(json.dumps(res, indent=1))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
