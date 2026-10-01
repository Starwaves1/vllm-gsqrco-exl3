"""exl3_gemm_mr kernel phases by trellis-serve's timing probes (job 17; needs a TRELLIS_PROBES=1 build):
set_debug_flags(f) cuts the GEMM kernel short: 4 = return at once (launch floor), 8 = after slice setup +
pipeline start, 16 = after the first slice's tiles (no reduce / barrier / write), 32 = no global reduce,
64 = no result write, 128 = no in-block reduce (wrong results except 0). GPU us per call (graph replay,
weight copies cycled past L2) per flag at each row count, for two large shapes.
  GSQ_ALLOW_GPU=1 python bench/micro/exl3_mr_probe.py"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "gpu"))
sys.path.insert(0, str(ROOT / "plugin-exl3"))
sys.path.insert(0, str(ROOT / "bench" / "micro"))
from gsq_gpu import GPU  # noqa: E402,F401

import torch  # noqa: E402

import exl3_cases as C  # noqa: E402
from exl3_mr import time_graph  # noqa: E402
from vllm_exl3_plugin import _C_exl3_mr as M  # noqa: E402
from vllm_exl3_plugin import ops  # noqa: E402

FLAGS = [0, 128, 32, 64, 32 | 64 | 128, 16, 8, 4]
ROWS = [8, 16, 17, 32, 48]
X = torch.ops._C_exl3
print(f"{'tensor':10} {'rows':>4} " + " ".join(f"{f:>8}" for f in FLAGS), flush=True)
for tid in ("K3-up", "K4-oproj"):
    w = C.load(torch, tid)
    base = [w["trellis"]] + [w["trellis"].clone() for _ in range(2)]
    copies = [ops.repack_k4_(t) for t in base] if w["trellis"].shape[2] == 64 else base
    args = (w["suh"], w["svh"], w["mcg"], w["mul1"])
    X.exl3_mr_warmup(copies[0], *args, ROWS, True)
    for m in ROWS:
        x = torch.randn(m, w["suh"].numel(), device="cuda").half()
        row = []
        for f in FLAGS:
            M.set_debug_flags(f)
            row.append(time_graph(lambda xx, c: X.exl3_gemm_mr(xx, c, *args, True), copies, x))
        M.set_debug_flags(0)
        print(f"{tid:10} {m:>4} " + " ".join(f"{u:8.1f}" for u in row), flush=True)
