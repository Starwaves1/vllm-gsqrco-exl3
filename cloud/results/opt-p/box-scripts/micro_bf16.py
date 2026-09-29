"""Tiny-N BF16 products (GDN in_proj_ba: [n, 5120] x [96, 5120]^T) under CUDA graphs: us per call
and max abs error vs fp64, for formulations of x @ w.T. usage: micro_bf16.py"""
import torch
torch.manual_seed(0)
def t(fn, reps=50):
    g = torch.cuda.CUDAGraph(); fn(); torch.cuda.synchronize()
    with torch.cuda.graph(g):
        for _ in range(reps): fn()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    g.replay(); torch.cuda.synchronize(); s.record()
    for _ in range(5): g.replay()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) * 1e3 / (5 * reps)
for rows, k in [(96, 5120), (48, 5120)]:
    w = (torch.randn(rows, k, device="cuda") * 0.02).bfloat16()
    for n in (1, 2, 4, 8, 16):
        x = torch.randn(n, k, device="cuda").bfloat16()
        ref = x.double() @ w.double().T
        V = {"x @ w.T": lambda: x @ w.T,
             "(w @ x.T).T.contiguous()": lambda: (w @ x.T).T.contiguous(),
             "F.linear": lambda: torch.nn.functional.linear(x, w),
             "addmm(0-bias)": lambda: torch.addmm(torch.zeros(rows, device="cuda", dtype=torch.bfloat16), x, w.T),
             "fp32 mm": lambda: (x.float() @ w.float().T).bfloat16()}
        for name, fn in V.items():
            y = fn(); err = (y.double() - ref).abs().max().item()
            print(f"{rows}x{k} n={n:2d} {name:28s} {t(fn):7.2f} us  maxerr {err:.2e}")
