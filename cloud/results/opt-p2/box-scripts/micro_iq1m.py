"""IQ1_M blk.13.ffn_gate (17408 x 5120) product, us per call inside a CUDA graph (50 calls,
median of 5 replays), n activation rows (bf16): stock MMVQ (ggml_mul_mat_vec_a8, the path at
<= 8 rows today), vendored MMVQ (lcpp_mul_mat_vec_q, 8-row calls above 8 rows), dequantize +
x @ W.T (the path above 8 rows today), and x @ W.T on a cached bf16 copy.
Also max relative difference of vendored MMVQ vs the dequantized product (fp32 reference)."""
import gguf, numpy as np, torch
from vllm_gguf_plugin import ops

C = torch.ops._C_gguf
r = gguf.GGUFReader("/workspace/models/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf")
t = next(t for t in r.tensors if t.name == "blk.13.ffn_gate.weight")
w = torch.from_numpy(np.ascontiguousarray(t.data)).cuda()
qt, rows, k = int(t.tensor_type), w.shape[0], 5120
wd = ops.ggml_dequantize(w, qt, rows, k, torch.bfloat16)


def us(fn, iters=50):
    fn(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(iters):
            fn()
    ts = []
    for _ in range(5):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); g.replay(); b.record(); torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1000 / iters)
    return sorted(ts)[2]


def lcpp(x):
    return torch.cat([C.lcpp_mul_mat_vec_q(w, x[i:i + 8], qt, rows) for i in range(0, x.shape[0], 8)])


print(f"{'n':>4} {'stock_mmvq':>10} {'lcpp_mmvq':>10} {'dequant+mm':>10} {'bf16_mm':>10}  lcpp_err")
for n in (1, 2, 4, 8, 9, 12, 16, 24, 32, 40, 48, 64, 128):
    x = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    ref = x.float() @ wd.float().T
    err = ((lcpp(x).float() - ref).norm() / ref.norm()).item()
    stock = us(lambda: ops.ggml_mul_mat_vec_a8(w, x, qt, rows)) if n <= 16 else float("nan")
    print(f"{n:4d} {stock:10.1f} {us(lambda: lcpp(x)):10.1f} "
          f"{us(lambda: x @ ops.ggml_dequantize(w, qt, rows, k, x.dtype).T):10.1f} {us(lambda: x @ wd.T):10.1f}  {err:.2e}")
