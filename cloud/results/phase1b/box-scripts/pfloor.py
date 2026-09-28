"""Noise floor: llama.cpp CUDA vs llama.cpp CPU (same b11211 libs, same GGUF) next to vLLM vs each."""
import numpy as np, sys, os
sys.path.insert(0, "/workspace/gsq-vllm/bench/parity")
from compare import read_rows, log_softmax
P = "/workspace/runs/p1b-parity"; D = "/workspace/runs/p1b-parity-diag"
def stats(a, b):
    (pa, A), (pb, B) = read_rows(a), read_rows(b); assert (pa == pb).all()
    k, t = [], []
    for i in range(len(pa)):
        x = log_softmax(A[i]); y = log_softmax(B[i]); k.append(float(np.sum(np.exp(x) * (x - y)))); t.append(np.argmax(x) == np.argmax(y))
    return f"KLD mean {np.mean(k):.5f} median {np.median(k):.5f} top1 {np.mean(t):.4f}"
print("seq      pair                     stats")
for s in sorted(f.split(".")[0] for f in os.listdir(f"{D}/llamacpu") if f.endswith(".f32")):
    cu, cp, v = f"{P}/llama/{s}.llama.f32", f"{D}/llamacpu/{s}.llama.f32", f"{P}/vllm/{s}.vllm.f32"
    print(f"{s}  llamaCUDA || llamaCPU   {stats(cu, cp)}")
    print(f"{s}  llamaCUDA || vLLM       {stats(cu, v)}")
    print(f"{s}  llamaCPU  || vLLM       {stats(cp, v)}")
