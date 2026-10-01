"""Per-sequence KLD of a 05 compare JSON with the reference's own glitches excluded: a position with
KLD > 0.1 where the exllamav3 reference gives the actual next token < 1 % (job 05, bf16 KV: exllamav3
predicted ' super' after 'hidden_states = hidden'). Prints mean, the excluded count and mean without
them per sequence, then the overall mean.
  python parity_glitch.py PARITY_JSON EXL3_DUMP_DIR PROMPT_DIR"""
import json
import sys

import numpy as np

j = json.load(open(sys.argv[1]))["sequences"]
ref_dir, prompt_dir = sys.argv[2], sys.argv[3]


def rows(path):
    b = open(path, "rb").read()
    h = np.frombuffer(b[:12], np.int32)
    n_pos, n_vocab = int(h[1]), int(h[2])
    pos = np.frombuffer(b[12:12 + 4 * n_pos], np.int32)
    return pos, np.frombuffer(b[12 + 4 * n_pos:], np.float32).reshape(n_pos, n_vocab)


tot = []
for name, s in j.items():
    kld, pos = np.array(s["kld"]), s["positions"]
    rp, r = rows(f"{ref_dir}/{name}.exl3.f32")
    ids = np.fromfile(f"{prompt_dir}/{name}.ids", dtype=np.int32)
    glitch = []
    for i in np.where(kld > 0.1)[0]:
        k = int(np.where(rp == pos[i])[0][0])
        x = r[k].astype(np.float64)
        x -= x.max()
        if np.exp(x[int(ids[pos[i] + 1])]) / np.exp(x).sum() < 0.01:
            glitch.append(i)
    keep = np.ones(len(kld), bool)
    keep[glitch] = False
    tot.append(kld[keep])
    print(f"{name} {s['kind']:6} {s['n_tokens']:6d} tok  KLD mean {kld.mean():.5f}  ref glitches {len(glitch)}  "
          f"mean without {kld[keep].mean():.5f}  top1 {s['top1']:.4f}")
a = np.concatenate(tot)
print(f"all: KLD mean without reference glitches {a.mean():.5f} over {len(a)} positions")
