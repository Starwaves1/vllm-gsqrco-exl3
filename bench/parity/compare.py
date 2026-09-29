"""Compare llama.cpp logits with vLLM logprobs: mean KLD and top-1 agreement.

KLD is KL(P_llama || Q_vllm) per position, in nats, over the token ids both engines
cover (each side renormalized over that set). Pass (HANDOFF definition of done §2):
mean KLD <= 0.001 and top-1 agreement >= 99%, overall and for the 100k+ sequences alone.

  python bench/parity/compare.py -d PROMPT_DIR -l LLAMA_OUT -v VLLM_OUT [--json out.json]

--json keeps each sequence's per-position KLD and top-1 hits, so runs can be compared position
by position after the logit dumps are deleted. Exit code 0 = pass, 1 = fail. CPU only.
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
if os.environ.get("GSQ_ALLOW_GPU") != "1":
    import no_gpu  # noqa: F401

import numpy as np  # noqa: E402

MAGIC = 0x4C4F4731
KLD_MAX, TOP1_MIN, LONG = 0.001, 0.99, 100_000


def read_rows(path: Path):
    with open(path, "rb") as f:
        magic, n_pos, n_vocab = np.fromfile(f, np.int32, 3)
        if magic != MAGIC:
            raise SystemExit(f"{path}: bad magic")
        pos = np.fromfile(f, np.int32, n_pos)
        rows = np.fromfile(f, np.float32, n_pos * n_vocab).reshape(n_pos, n_vocab)
    return pos, rows


def log_softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    m = np.max(x, axis=-1, keepdims=True)
    return x - (m + np.log(np.sum(np.exp(x - m), axis=-1, keepdims=True)))


def compare_seq(lpath: Path, vpath: Path) -> dict:
    lpos, L = read_rows(lpath)
    vpos, V = read_rows(vpath)
    if not np.array_equal(lpos, vpos):
        raise SystemExit(f"{lpath.name}/{vpath.name}: position lists differ")
    n = min(L.shape[1], V.shape[1])
    kld, top1, top5 = [], [], []
    for i in range(len(lpos)):  # row by row: a row is ~1 MB, a sequence can be 300 MB
        lp = log_softmax(L[i, :n])
        lq = log_softmax(V[i, :n])
        p = np.exp(lp)
        kld.append(float(np.sum(np.where(p > 0, p * (lp - np.where(np.isfinite(lq), lq, -1e30)), 0.0))))
        a = int(np.argmax(lp))
        top1.append(a == int(np.argmax(lq)))
        top5.append(a in np.argpartition(-lq, 5)[:5])
    return {"n_pos": len(lpos), "vocab_llama": L.shape[1], "vocab_vllm": V.shape[1],
            "kld_mean": float(np.mean(kld)), "kld_p99": float(np.percentile(kld, 99)),
            "kld_max": float(np.max(kld)), "top1": float(np.mean(top1)), "llama_top1_in_vllm_top5": float(np.mean(top5)),
            "positions": [int(x) for x in lpos], "kld": kld, "top1_hits": [bool(x) for x in top1]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-d", "--prompts", type=Path, required=True)
    ap.add_argument("-l", "--llama", type=Path, required=True)
    ap.add_argument("-v", "--vllm", type=Path, required=True)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    man = json.loads((a.prompts / "manifest.json").read_text())
    res, all_k, all_t, long_k, long_t = {}, [], [], [], []
    for s in man["sequences"]:
        lp, vp = a.llama / f"{s['name']}.llama.f32", a.vllm / f"{s['name']}.vllm.f32"
        if not (lp.exists() and vp.exists()):
            print(f"{s['name']}: missing output, skipped"); continue
        r = compare_seq(lp, vp)
        all_k += r["kld"]; all_t += r["top1_hits"]
        if s["n_tokens"] >= LONG:
            long_k += r["kld"]; long_t += r["top1_hits"]
        res[s["name"]] = {"kind": s["kind"], "n_tokens": s["n_tokens"], **r}  # per-position lists kept
        print(f"{s['name']} {s['kind']:6s} {s['n_tokens']:7d} tok  KLD mean {r['kld_mean']:.5f} "
              f"p99 {r['kld_p99']:.5f} max {r['kld_max']:.4f}  top1 {r['top1']:.4f}")
    if not all_k:
        raise SystemExit("nothing compared")
    summ = {"kld_mean": float(np.mean(all_k)), "top1": float(np.mean(all_t)), "positions": len(all_k),
            "long_kld_mean": float(np.mean(long_k)) if long_k else None,
            "long_top1": float(np.mean(long_t)) if long_t else None,
            "thresholds": {"kld_mean_max": KLD_MAX, "top1_min": TOP1_MIN}}
    ok = summ["kld_mean"] <= KLD_MAX and summ["top1"] >= TOP1_MIN and bool(long_k) \
        and summ["long_kld_mean"] <= KLD_MAX and summ["long_top1"] >= TOP1_MIN
    summ["pass"] = ok
    print(json.dumps(summ, indent=1))
    if a.json:
        a.json.write_text(json.dumps({"summary": summ, "sequences": res}, indent=1))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
