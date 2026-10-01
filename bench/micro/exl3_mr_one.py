"""Run exl3_gemm_mr on one real checkpoint tensor at the given row counts, a few calls each (for
Nsight Compute: job 16). GSQ_ALLOW_GPU=1 python bench/micro/exl3_mr_one.py TENSOR_ID ROWS..."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "gpu"))
sys.path.insert(0, str(ROOT / "plugin-exl3"))
from gsq_gpu import GPU  # noqa: E402,F401

import torch  # noqa: E402

import exl3_cases as C  # noqa: E402
from vllm_exl3_plugin import ops  # noqa: E402

tid, rows = sys.argv[1], [int(r) for r in sys.argv[2:]]
w = C.load(torch, tid)
b = ops.repack_k4_(w["trellis"]) if w["trellis"].shape[2] == 64 else w["trellis"]
args = (b, w["suh"], w["svh"], w["mcg"], w["mul1"], True)
for m in rows:
    x = C.make_x(torch, tid, m)
    for _ in range(3):
        torch.ops._C_exl3.exl3_gemm_mr(x, *args)
torch.cuda.synchronize()
print("ok", tid, rows)
