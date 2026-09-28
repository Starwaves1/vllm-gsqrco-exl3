import numpy as np, sys
sys.path.insert(0, "/workspace/gsq-vllm/bench/parity")
from compare import read_rows, log_softmax
lp_path, vp_path = sys.argv[1], sys.argv[2]
lpos, L = read_rows(lp_path); vpos, V = read_rows(vp_path)
assert (lpos == vpos).all()
k = []
for i in range(len(lpos)):
    a = log_softmax(L[i]); b = log_softmax(V[i]); p = np.exp(a); k.append(float(np.sum(p * (a - b))))
k = np.array(k)
for lo in range(0, int(lpos.max()) + 1, 32):
    m = (lpos >= lo) & (lpos < lo + 32)
    if m.any(): print(f"pos {lo:5d}-{lo+31:5d} n={m.sum():3d} KLD mean {k[m].mean():.5f} median {np.median(k[m]):.5f} max {k[m].max():.3f}")
print("overall mean %.5f median %.5f" % (k.mean(), np.median(k)))
