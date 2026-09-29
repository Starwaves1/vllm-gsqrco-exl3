"""Host time per eager call (us; 2000 back-to-back calls, one sync at the end; the GPU work is
a few us, so the loop is host-bound) of the pieces of the eager embedding and lm_head paths:
token_embd IQ2_S (248320 x 5120) for 4 token ids, output Q4_K (first 4096 rows) for 4 rows."""
import time, gguf, numpy as np, torch
from vllm_gguf_plugin import ops
from vllm_gguf_plugin.quantization import linear, vocal_embeds

C = torch.ops._C_gguf
r = gguf.GGUFReader("/workspace/models/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf")
T = {t.name: t for t in r.tensors}
emb = torch.from_numpy(np.ascontiguousarray(T["token_embd.weight"].data)).cuda()
out = torch.from_numpy(np.ascontiguousarray(T["output.weight"].data[:4096])).cuda()  # host cost only: a 4096-row slice
ET, OT = int(T["token_embd.weight"].tensor_type), int(T["output.weight"].tensor_type)
ids = torch.tensor([11, 2000, 30000, 200000], device="cuda")
x = torch.randn(4, 5120, device="cuda", dtype=torch.bfloat16)
rows = torch.index_select(emb, 0, ids)


def us(fn, n=2000):
    for _ in range(50):
        fn()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1e6


cases = {
    "embed: torch.ops.vllm._apply_gguf_embedding": lambda: vocal_embeds.apply_gguf_embedding(ids, emb, ET, 5120, dtype=torch.bfloat16),
    "embed: _apply_gguf_embedding (python, no custom op)": lambda: vocal_embeds._apply_gguf_embedding(ids, emb, ET, 5120, dtype=torch.bfloat16),
    "embed: index_select + _C_gguf.ggml_dequantize": lambda: C.ggml_dequantize(torch.index_select(emb, 0, ids), ET, 5120, 4, torch.bfloat16),
    "embed:   index_select": lambda: torch.index_select(emb, 0, ids),
    "embed:   _C_gguf.ggml_dequantize": lambda: C.ggml_dequantize(rows, ET, 5120, 4, torch.bfloat16),
    "embed:   ops.ggml_dequantize (python wrapper)": lambda: ops.ggml_dequantize(rows, ET, 5120, 4, torch.bfloat16),
    "lm_head: torch.ops.vllm._fused_mul_mat_gguf": lambda: linear.fused_mul_mat_gguf(x, out, OT),
    "lm_head: _fused_mul_mat_gguf (python)": lambda: linear._fused_mul_mat_gguf(x, out, OT),
    "lm_head:   _C_gguf.lcpp_mul_mat_vec_own": lambda: C.lcpp_mul_mat_vec_own(out, x, OT, out.shape[0]),
    "lm_head:   _C_gguf.lcpp_mul_mat_vec_own fp32 X": lambda: C.lcpp_mul_mat_vec_own(out, x.float(), OT, out.shape[0]),
    "ref: torch.empty": lambda: torch.empty(4, 5120, device="cuda"),
    "ref: x.to(float32)": lambda: x.to(torch.float32),
}
for k, fn in cases.items():
    print(f"{us(fn):8.1f} us  {k}")
