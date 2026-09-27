"""MTP acceptance readout on the GGUF server (k=3, production sampling and greedy),
via bench/speed/mtp_acceptance.py. HANDOFF §2 wants it within 2 points of llama.cpp's on
the same file: pass that readout (bench/speed/mtp_acceptance.py --engine llama against
bench/speed/serve-llamacpp.sh) as GSQ_LLAMACPP_MTP=<json>. Without it the test records
the vLLM readout and passes with a warning.

The session server must be idle otherwise: the vLLM numbers are /metrics deltas.
TODO(GPU): per-position acceptance should decay with position (pos0 > pos1 > pos2); a flat
or zero pos0 means the MTP head did not load from blk.64.nextn.* (check the adapter).
"""

import json
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

from gsq_gpu import API_KEY, ROOT, RUNS


def test_mtp_acceptance(gsq_server):
    url, _ = gsq_server
    out = RUNS / time.strftime("%Y%m%d-%H%M%S-mtp-vllm.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(ROOT / "bench/speed/mtp_acceptance.py"), "--engine", "vllm", "--url", url,
           "--api-key", API_KEY, "--out", str(out)]
    ref = os.environ.get("GSQ_LLAMACPP_MTP")
    if ref:
        cmd += ["--reference", ref]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    print(p.stdout, p.stderr[-2000:])
    res = json.loads(out.read_text())["results"]
    for t, r in res.items():
        assert r["drafted"] > 0, f"{t}: no drafts; speculative decoding is not active"
        pp = r.get("per_position") or []
        assert len(pp) == 3, f"{t}: expected 3 draft positions (k=3), got {pp}"
    if not ref:
        warnings.warn(f"no llama.cpp reference (GSQ_LLAMACPP_MTP); readout only: {out}")
    assert p.returncode == 0, "acceptance differs from llama.cpp by more than 2 points"
