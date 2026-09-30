"""R2: run lcpp_mul_mat_iq3_packed (and MMQ) a few times on one real-block shape, for ncu.
  python r2_prof.py TYPE ROWS K N"""
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("WT", "/workspace/wt-r2"))
sys.path.insert(0, str(ROOT / "bench" / "micro"))
from gemm import GGUF, weight  # noqa: E402

import gguf  # noqa: E402
import torch  # noqa: E402

from vllm_gguf_plugin.quantization import iq3_pack  # noqa: E402

name, rows, k, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
C = torch.ops._C_gguf
w, qt = weight(gguf.GGUFReader(str(GGUF)), name, rows, k)
p = iq3_pack.pack(w, qt)
x = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
for _ in range(3):
    C.lcpp_mul_mat_iq3_packed(p, x, qt, rows)
    C.lcpp_mul_mat_q(w, x, qt, rows)
torch.cuda.synchronize()
