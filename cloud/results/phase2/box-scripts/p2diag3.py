import sys, torch, numpy as np, gguf
sys.path.insert(0, "/workspace/gsq-vllm/tests/gpu")
import _refs
from test_kernel_parity import _x
from vllm_gguf_plugin import ops
r = gguf.GGUFReader("/workspace/models/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf")
C = torch.ops._C_gguf
for name, n, dt in [("IQ3_S", 128, "float16"), ("Q2_K", 128, "float16"), ("Q4_K", 2048, "float16"), ("Q2_K", 2048, "bfloat16")]:
    t = next(t for t in r.tensors if t.tensor_type.name == name and len(t.shape) == 2)
    raw = np.ascontiguousarray(t.data[:512]); qt = int(t.tensor_type)
    x = _x(n, int(t.shape[0]), dt, seed=400 + n)
    R = _refs.refs(raw, name, x, True, True)
    key = {"Q2_K": "d2s6", "Q4_K": "xsum"}.get(name, "q81")
    ref = R[key]
    y = C.lcpp_mul_mat_q(torch.from_numpy(raw).cuda(), x.cuda(), qt, 512).double().cpu()
    e = (y - ref).norm(dim=-1) / ref.norm(dim=-1)
    i = int(e.argmax()); d = (y[i] - ref[i]); j = int(d.abs().argmax())
    # near ties in token i (layout block size)
    qk = 64 if name == "Q2_K" else 32
    b = x[i].float().view(-1, qk); amax = b.abs().amax(-1, keepdim=True); tt = (b * (127.0 / amax)).abs()
    fr = (tt - tt.floor() - 0.5).abs().view(-1)
    ties = (fr < 1e-5).nonzero().view(-1).tolist()
    W = _refs.dequant(raw, name)
    dvec = (amax / 127).expand(-1, qk).reshape(-1).double()
    contrib = [(k, float(dvec[k] * W[j, k])) for k in ties]
    e2 = d.clone(); 
    print(name, n, dt, key, "row err", float(e[i]), "median", float(e.median()), "tok", i, "col", j, "diff", float(d[j]), "near-ties", contrib[:6])
    # rel err with worst element removed
    d[j] = 0; print("   row err without that element", float(d.norm() / ref[i].norm()))
