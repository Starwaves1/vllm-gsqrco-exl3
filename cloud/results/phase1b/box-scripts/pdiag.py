import numpy as np, sys, json
sys.path.insert(0, "/workspace/gsq-vllm/bench/parity")
from compare import read_rows, log_softmax
P = "/workspace/runs/p1b-parity"; name = sys.argv[1] if len(sys.argv) > 1 else "seq_000"
ids = np.fromfile(f"{P}/prompts/{name}.ids", np.int32)
lpos, L = read_rows(f"{P}/llama/{name}.llama.f32")
vpos, V = read_rows(f"{P}/vllm/{name}.vllm.f32")
print("vocab", L.shape, V.shape, "finite per vllm row min/max", np.isfinite(V).sum(1).min(), np.isfinite(V).sum(1).max())
n = min(L.shape[1], V.shape[1])
rows = []
for i, p in enumerate(lpos):
    lp = log_softmax(L[i, :n]); lq = V[i, :n].astype(np.float64)
    pr = np.exp(lp); k = float(np.sum(pr * (lp - np.where(np.isfinite(lq), lq, -1e30))))
    nxt = ids[p + 1] if p + 1 < len(ids) else -1
    rows.append((int(p), k, int(np.argmax(lp)), int(np.argmax(lq)), int(nxt),
                 float(-lp[nxt]) if nxt >= 0 else np.nan, float(-lq[nxt]) if nxt >= 0 else np.nan,
                 float(np.exp(lq).sum())))
r = np.array(rows, dtype=float)
print("vllm row prob mass min/max", r[:, 7].min(), r[:, 7].max())
ok = ~np.isnan(r[:, 5])
print("NLL of true next token: llama %.4f vllm %.4f" % (r[ok, 5].mean(), r[ok, 6].mean()))
print("top1==next: llama %.3f vllm %.3f" % ((r[ok, 2] == r[ok, 4]).mean(), (r[ok, 3] == r[ok, 4]).mean()))
bad = r[r[:, 1] > 0.1]
print("positions with KLD>0.1:", len(bad), "of", len(r))
for b in bad[:25]: print("  pos %d kld %.3f l_top %d v_top %d next %d nll_l %.2f nll_v %.2f" % tuple(b[:7]))
print("KLD by quartile of position:", [round(float(np.mean(q[:, 1])), 4) for q in np.array_split(r, 4)])
print("median KLD", np.median(r[:, 1]))
