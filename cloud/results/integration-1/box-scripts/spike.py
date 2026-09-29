"""Per-position KLD for one parity run: the 5 largest positions per sequence with both engines'
top tokens (decoded), plus the full per-position KLD list (JSON) for cross-run comparison.
usage: spike.py PROMPTS LLAMA VLLM HF_CONFIG OUT.json seq_009 seq_010"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "bench/parity"))
from compare import log_softmax, read_rows  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

prompts, llama, vllm, hf, out = map(Path, sys.argv[1:6])
tok = AutoTokenizer.from_pretrained(str(hf))
res = {}
for name in sys.argv[6:]:
    lpos, L = read_rows(llama / f"{name}.llama.f32")
    _, V = read_rows(vllm / f"{name}.vllm.f32")
    n = min(L.shape[1], V.shape[1])
    kld, tops = [], []
    for i in range(len(lpos)):
        lp, lq = log_softmax(L[i, :n]), log_softmax(V[i, :n])
        p = np.exp(lp)
        kld.append(float(np.sum(np.where(p > 0, p * (lp - np.where(np.isfinite(lq), lq, -1e30)), 0.0))))
        a, b = int(np.argmax(lp)), int(np.argmax(lq))
        tops.append((a, float(np.exp(lp[a])), float(np.exp(lq[a])), b, float(np.exp(lq[b])), float(np.exp(lp[b]))))
    res[name] = {"positions": [int(x) for x in lpos], "kld": kld}
    print(f"{name}: mean {np.mean(kld):.5f} max {np.max(kld):.4f} top1 "
          f"{np.mean([t[0] == t[3] for t in tops]):.4f}")
    for i in np.argsort(kld)[::-1][:5]:
        a, pa_l, pa_v, b, pb_v, pb_l = tops[i]
        print(f"  pos {int(lpos[i])} (probe {i}): KLD {kld[i]:.4f}  llama top {tok.decode([a])!r} "
              f"p={pa_l:.3f} (vLLM {pa_v:.3f})  vLLM top {tok.decode([b])!r} p={pb_v:.3f} (llama {pb_l:.3f})")
out.write_text(json.dumps(res))
