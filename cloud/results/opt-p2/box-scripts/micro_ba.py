"""GDN in_proj_ba (BF16 96 x 5120) at 9..32 activation rows (c=4 / c=8 target passes: 16 / 32),
us per call inside a CUDA graph (50 calls, median of 5 replays): F.linear (today above 8 rows),
batched gemv (today at <= 8 rows), torch.mm on a K-major copy, fp32-out mm, and x @ W.T split in
8-row gemv chunks. Max abs difference of each vs F.linear."""
import torch
import torch.nn.functional as F

g = torch.Generator(device="cuda").manual_seed(0)
w = (torch.randn(96, 5120, device="cuda", generator=g) * 0.02).bfloat16()
wk = w.t().contiguous()


def us(fn, iters=50):
    fn(); torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        for _ in range(iters):
            fn()
    ts = []
    for _ in range(5):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); gr.replay(); b.record(); torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1000 / iters)
    return sorted(ts)[2]


def bmm(x):
    return torch.bmm(x.unsqueeze(1), w.T.expand(x.shape[0], -1, -1)).squeeze(1)


cands = {
    "F.linear": lambda x: F.linear(x, w),
    "bmm gemv": bmm,
    "mm K-major": lambda x: torch.mm(x, wk),
    "mm fp32 out": lambda x: torch.mm(x, w.t(), out_dtype=torch.float32).to(x.dtype),
    "gemv 8-row chunks": lambda x: torch.cat([bmm(x[i:i + 8]) for i in range(0, x.shape[0], 8)]),
    "w @ x.T": lambda x: (w @ x.T).T,
}
print(f"{'n':>3} " + " ".join(f"{k:>18}" for k in cands))
for n in (8, 9, 12, 16, 24, 32):
    x = torch.randn(n, 5120, device="cuda", generator=g).bfloat16()
    ref = F.linear(x, w)
    cells = []
    for k, fn in cands.items():
        try:
            cells.append(f"{us(lambda: fn(x)):8.1f} ({(fn(x).float() - ref.float()).abs().max().item():.0e})")
        except Exception as e:  # noqa: BLE001
            cells.append(f"{'err':>18}")
    print(f"{n:3d} " + " ".join(f"{c:>18}" for c in cells))
