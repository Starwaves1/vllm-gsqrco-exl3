"""K3 quick check under each LCPP_MMAK_CFG in argv[1] (';'-separated): mma_k vs MMQ, fp32 X."""
import os, sys
import gguf, numpy as np, torch
import vllm_gguf_plugin.ops  # noqa: F401
R = gguf.GGUFReader(os.environ.get("GSQ_GGUF", "/workspace/models/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf"))
C = torch.ops._C_gguf
torch.manual_seed(0)
def rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm()).item()
worst = 0.0
for cfg in sys.argv[1].split(";"):
    os.environ["LCPP_MMAK_CFG"] = cfg
    for name in sys.argv[2].split(",") if len(sys.argv) > 2 else ["Q4_K", "IQ4_XS", "IQ2_S"]:
        qt = int(gguf.GGMLQuantizationType[name])
        big = next(t for t in R.tensors if t.tensor_type.name == name and len(t.shape) == 2
                   and int(t.shape[1]) == 17408 and int(t.shape[0]) == 5120)
        for rows in (512, 202, 17408):
            w = torch.from_numpy(np.array(big.data[:rows])).cuda()
            for n in (9, 16, 33, 64):
                x = torch.randn(n, 5120, device="cuda")
                x[:, torch.randperm(5120)[:20]] *= 20
                y = C.lcpp_mul_mat_mma_k(w, x, qt, rows)
                ref = C.lcpp_mul_mat_q(w, x, qt, rows)
                torch.cuda.synchronize()
                e = rel(y, ref)
                worst = max(worst, e)
                if e > 1e-6 or not torch.isfinite(y).all():
                    print(f"BAD cfg={cfg} {name} rows={rows} n={n}: rel {e:.2e}", flush=True)
    print(f"cfg={cfg}: worst rel vs MMQ so far {worst:.2e}", flush=True)
