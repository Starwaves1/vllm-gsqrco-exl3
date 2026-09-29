"""Tiny-N BF16 products, round 2: GDN in_proj_ba [n, 5120] x [96, 5120]^T (and 48 rows), n = 1..8,
under CUDA graphs: us per call and max abs error vs fp64. usage: micro_bf16b.py"""
import torch
import torch.nn.functional as F
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
    wt = w.T.contiguous()
    for n in range(1, 9):
        x = torch.randn(n, k, device="cuda").bfloat16()
        ref = x.double() @ w.double().T
        V = {"x @ w.T": lambda: x @ w.T,
             "x @ wt (K-major weight)": lambda: x @ wt,
             "pad8(x) @ w.T [:n]": lambda: (F.pad(x, (0, 0, 0, 8 - n)) @ w.T)[:n],
             "pad16(x) @ w.T [:n]": lambda: (F.pad(x, (0, 0, 0, 16 - n)) @ w.T)[:n],
             "bmm batch n": lambda: torch.bmm(x.unsqueeze(1), w.T.unsqueeze(0).expand(n, -1, -1)).squeeze(1),
             "gemv per row + stack": lambda: torch.stack([w @ x[i] for i in range(n)]),
             "mul+sum fp32": lambda: (x.float().unsqueeze(1) * w.float()).sum(-1).bfloat16()}
        for name, fn in V.items():
            y = fn(); err = (y.double() - ref).abs().max().item()
            print(f"{rows}x{k} n={n} {name:26s} {t(fn):7.2f} us  maxerr {err:.2e}")
