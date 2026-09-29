"""K3: DRAM streaming floor on this GPU. Times a read-only pass (torch.sum over a float32 view)
and a device-to-device copy (read + write) over buffers the size of each weight shape, in CUDA
graphs like bench/micro/gemm.py. The GEMM floor per shape = its weight bytes / the read rate."""
import torch, time
BYTES = {"Q4_K": 4.5, "IQ4_XS": 4.25, "IQ2_S": 2.5625, "IQ3_S": 3.4375}
SHAPES = [(17408, 5120), (5120, 17408), (10240, 5120)]
def t_graph(fn, reps=50):
    fn(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(10):
            fn()
    g.replay(); torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps):
        g.replay()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) * 1e3 / (reps * 10)
print(torch.cuda.get_device_name())
for mb in (16, 32, 64, 256, 1024):
    n = mb * 2**20 // 4
    a = torch.randn(n, device="cuda"); b = torch.empty_like(a); out = torch.empty((), device="cuda")
    tr = t_graph(lambda: torch.sum(a, dim=0, out=out))
    tc = t_graph(lambda: b.copy_(a))
    print(f"{mb:5d} MiB: read (sum) {a.numel()*4/tr/1e3:6.0f} GB/s   copy {2*a.numel()*4/tc/1e3:6.0f} GB/s (read+write)")
for name, bpw in BYTES.items():
    for rows, k in SHAPES:
        nbytes = int(rows * k * bpw / 8)
        a = torch.randn(nbytes // 4, device="cuda"); out = torch.empty((), device="cuda")
        tr = t_graph(lambda: torch.sum(a, dim=0, out=out))
        print(f"floor {name} {rows}x{k}: {nbytes/1e6:.1f} MB, read {tr:.1f} us ({nbytes/tr/1e3:.0f} GB/s)")
