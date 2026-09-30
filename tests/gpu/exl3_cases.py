"""EXL3 kernel cases shared by the two sides of the phase-1 kernel parity check:

  tests/gpu/exl3_ref_dump.py     exllamav3 d3739fd itself (the reference venv, GSQ_EXL3_VENV):
                                 dequant hashes + gemm error stats, written to EXL3_REF_DIR
  tests/gpu/test_exl3_kernels.py the plugin's torch.ops._C_exl3 (vLLM's venv), compared with them

Torch-free at import (both venvs import it, and collection must not touch the GPU). Tensors
are real checkpoint tensors, one or two per bit width the 3.50bpw checkpoint uses (all mul1):
names and shapes are pinned against hf-config/.../safetensors_headers.json by
tests/cpu/test_exl3_gpu_cases.py. Activations are seeded randn on the CPU (same torch, same
numbers in both venvs), cast to fp16.

Error statistics are against an fp64 product of the same activations with the original-basis
weight (exl3_dequant had=True, itself checked bit-exact): rel_rms = ||y - ref|| / ||ref||,
max_rel = max|y - ref| / rms(ref).
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

MODEL = Path(os.environ.get("EXL3_MODEL", "/workspace/models/Qwen3.8-27B-exl3-3.50bpw"))
REF_DIR = Path(os.environ.get("EXL3_REF_DIR", "/workspace/runs/exl3/kernel-ref"))

# id -> (checkpoint prefix, K, k = in_features, n = out_features); codebook mul1 throughout
TENSORS = {
    "K3-down": ("model.language_model.layers.10.mlp.down_proj", 3, 17408, 5120),
    "K3-up": ("model.language_model.layers.9.mlp.up_proj", 3, 5120, 17408),
    "K4-kproj": ("model.language_model.layers.11.self_attn.k_proj", 4, 5120, 1024),
    "K4-mtp-up": ("mtp.layers.0.mlp.up_proj", 4, 5120, 17408),
    "K5-oproj": ("model.language_model.layers.63.self_attn.o_proj", 5, 6144, 5120),
    "K6-lmhead": ("lm_head", 6, 5120, 248320),
}
# activation rows: every GEMM bucket and the GEMV limits (1..17), the largest capture size
# (48), the first reconstruct row (145) and the fused reconstruct (1024)
ROWS = list(range(1, 18)) + [48, 145, 1024]
SLICE_N = 32768  # dequant slice (exllamav3 MAX_RECONSTRUCT_SLICE_N)
X_SCALE = 1.0


def seed(tid: str, m: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{tid}/{m}".encode()).digest()[:4], "little")


def make_x(torch, tid: str, m: int, device="cuda"):
    k = TENSORS[tid][2]
    g = torch.Generator().manual_seed(seed(tid, m))
    return (torch.randn(m, k, generator=g) * X_SCALE).half().to(device)


def slices(n: int):
    return [(s, min(SLICE_N, n - s)) for s in range(0, n, SLICE_N)]


def load(torch, tid: str, device="cuda") -> dict:
    """trellis/suh/svh (+ mcg/mul1 flags) of one checkpoint tensor, from the model's index."""
    import json

    from safetensors import safe_open

    prefix = TENSORS[tid][0]
    wm = json.loads((MODEL / "model.safetensors.index.json").read_text())["weight_map"]
    out = {"mcg": f"{prefix}.mcg" in wm, "mul1": f"{prefix}.mul1" in wm}
    for t in ("trellis", "suh", "svh"):
        with safe_open(str(MODEL / wm[f"{prefix}.{t}"]), framework="pt") as f:
            out[t] = f.get_tensor(f"{prefix}.{t}").to(device).contiguous()
    return out


def hash_slices(dequant, n: int) -> str:
    """sha256 over the fp16 bytes of dequant(n_start, n_count) for every slice, in order."""
    h = hashlib.sha256()
    for s, c in slices(n):
        h.update(dequant(s, c).contiguous().cpu().numpy().tobytes())
    return h.hexdigest()


def fp64_ref(torch, x, dequant, n: int):
    """x (fp16 [m, k]) @ W in fp64, W from dequant(n_start, n_count) -> fp16 [k, c] slices."""
    xd = x.double()
    return torch.cat([xd @ dequant(s, c).double() for s, c in slices(n)], dim=1)


def err_stats(torch, y, ref) -> dict:
    d = y.double() - ref
    rms = ref.pow(2).mean().sqrt().item()
    return {"rel_rms": (d.pow(2).mean().sqrt().item() / rms) if rms else float("inf"),
            "max_rel": (d.abs().max().item() / rms) if rms else float("inf"),
            "finite": bool(torch.isfinite(y).all().item()), "ref_rms": rms}


def sha_tensor(t) -> str:
    return hashlib.sha256(t.contiguous().cpu().numpy().tobytes()).hexdigest()
