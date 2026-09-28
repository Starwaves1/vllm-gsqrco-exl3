"""Per-seq KLD / top-1: llama.cpp CUDA vs vLLM stock (phase 1b) vs vLLM Route L, and the
llama.cpp CUDA-vs-CPU floor (seq_000-005 only)."""
import os, sys
import numpy as np
sys.path.insert(0, "/workspace/gsq-vllm/bench/parity")
from compare import read_rows, log_softmax
P, D, O = "/workspace/runs/p1b-parity", "/workspace/runs/p1b-parity-diag", "/workspace/runs/p2-parity"

def stats(a, b):
    (pa, A), (pb, B) = read_rows(a), read_rows(b)
    assert (pa == pb).all()
    k, t = [], []
    for i in range(len(pa)):
        x, y = log_softmax(A[i]), log_softmax(B[i])
        k.append(float(np.sum(np.exp(x) * (x - y)))); t.append(np.argmax(x) == np.argmax(y))
    return np.mean(k), np.mean(t)

print(f"{'seq':8} {'CUDA||stock':>12} {'CUDA||L':>10} {'CUDA||CPU':>10} {'CPU||L':>9}   top1 stock/L/CPU    gate L<=floor")
for s in sorted(f.split(".")[0] for f in os.listdir(f"{O}/vllm") if f.endswith(".f32")):
    cu = f"{P}/llama/{s}.llama.f32"
    ks, ts = stats(cu, f"{P}/vllm/{s}.vllm.f32")
    kl, tl = stats(cu, f"{O}/vllm/{s}.vllm.f32")
    cp = f"{D}/llamacpu/{s}.llama.f32"
    if os.path.exists(cp):
        kf, tf = stats(cu, cp); kcl, _ = stats(cp, f"{O}/vllm/{s}.vllm.f32")
        print(f"{s:8} {ks:12.5f} {kl:10.5f} {kf:10.5f} {kcl:9.5f}   {ts:.4f}/{tl:.4f}/{tf:.4f}  {'PASS' if kl <= kf else 'FAIL'}")
    else:
        print(f"{s:8} {ks:12.5f} {kl:10.5f} {'n/a':>10} {'n/a':>9}   {ts:.4f}/{tl:.4f}/  n/a    not measured")
