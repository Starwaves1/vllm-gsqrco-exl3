"""iq3_pack memory and time on the GPU (where the load path packs), pre-fix (iq3_pack_old.py =
99f4049) vs this worktree's: every IQ3 tensor of the GGUF, pack_ scratch (max_memory_allocated above
the tensor), time, new bytes == old bytes; then whole-tensor pack() on the largest tensor."""
import importlib.util
import os
import time

import gguf
import numpy as np
import torch

from vllm_gguf_plugin.quantization import iq3_pack as new

spec = importlib.util.spec_from_file_location("iq3_pack_old", os.path.join(os.path.dirname(__file__), "iq3_pack_old.py"))
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
T = (gguf.GGMLQuantizationType.IQ3_S, gguf.GGMLQuantizationType.IQ3_XXS)
ts = [t for t in gguf.GGUFReader(os.environ["GSQ_GGUF"]).tensors if t.tensor_type in T]
MB = 1 << 20

tot = {"old": [0.0, 0.0], "new": [0.0, 0.0]}  # seconds, max scratch / tensor
worst = {}
nbytes = 0
for t in ts:
    src = torch.from_numpy(np.ascontiguousarray(t.data)).cuda()
    nbytes += src.numel()
    outs = {}
    for name, mod in (("old", old), ("new", new)):
        w = src.clone()
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        mod.pack_(w, int(t.tensor_type))
        torch.cuda.synchronize()
        tot[name][0] += time.perf_counter() - t0
        r = (torch.cuda.max_memory_allocated() - base) / src.numel()
        if r > tot[name][1]:
            tot[name][1] = r
            worst[name] = (t.name, t.tensor_type.name, tuple(src.shape), round(src.numel() / MB, 1))
        outs[name] = w
    assert torch.equal(outs["old"], outs["new"]), t.name
print(f"{len(ts)} IQ3 tensors, {nbytes / MB:.0f} MiB, new == old bytes on every tensor")
for name in ("old", "new"):
    print(f"{name}: pack_ total {tot[name][0]:.2f} s; max scratch {tot[name][1]:.3f}x tensor ({worst[name]})")
# whole-tensor pack() (what the parity tests call) on the largest tensor
t = max(ts, key=lambda t: t.data.size)
for name, mod in (("old", old), ("new", new)):
    src = torch.from_numpy(np.ascontiguousarray(t.data)).cuda()
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    mod.pack(src, int(t.tensor_type))
    torch.cuda.synchronize()
    print(f"{name}: whole-tensor pack() on {t.name} ({src.numel() / MB:.1f} MiB): peak above input "
          f"{(torch.cuda.max_memory_allocated() - base) / MB:.0f} MiB = {(torch.cuda.max_memory_allocated() - base) / src.numel():.2f}x")
