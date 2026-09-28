import numpy as np, sys
sys.path.insert(0, "/workspace/gsq-vllm/bench/parity")
from compare import read_rows, log_softmax
from tokenizers import Tokenizer
t = Tokenizer.from_file("/workspace/gsq-vllm/hf-config/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp/tokenizer.json")
P = "/workspace/runs/p1b-parity"; D = "/workspace/runs/p1b-parity-diag"
pos, A = read_rows(f"{P}/llama/seq_000.llama.f32"); _, B = read_rows(f"{D}/llamacpu/seq_000.llama.f32"); _, C = read_rows(f"{P}/vllm/seq_000.vllm.f32")
ids = np.fromfile(f"{P}/prompts/seq_000.ids", np.int32)
for p in [843, 897, 900, 987, 1001]:
    i = int(np.where(pos == p)[0][0])
    print(f"pos {p} ctx={t.decode(ids[p-6:p+1].tolist())!r} next={t.id_to_token(int(ids[p+1])) if p+1 < len(ids) else None!r}")
    for nm, X in (("llamaCUDA", A), ("llamaCPU", B), ("vLLM", C)):
        lp = log_softmax(X[i]); top = np.argsort(-lp)[:3]; H = float(-(np.exp(lp) * lp).sum())
        print(f"   {nm:9s} H={H:.2f} maxlogit={X[i].max():.1f} " + "  ".join(f"{t.id_to_token(int(k))!r}:{np.exp(lp[k]):.3f}" for k in top))
