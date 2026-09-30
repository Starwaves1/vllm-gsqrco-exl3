#!/usr/bin/env python3
"""Weight inventory for the model matrix (CPU only, headers only).

GGUF: per-tensor type/shape/bytes, Swift vs Base diff, metadata diff (tokenizer arrays skipped),
embedded chat template sha256. Safetensors dirs: bytes by class.
Bytes "read per decode step" = every weight the target model's forward streams: all linear and
norm weights of the 64 layers plus the output head; token_embd/embed_tokens is a row gather (not
counted), the vision tower is not run, the MTP block (blk.64 / mtp.*) is listed separately.
usage: inventory.py SWIFT.gguf BASE.gguf [SAFETENSORS_DIR ...]
"""
import collections, hashlib, json, re, struct, sys
from pathlib import Path

import gguf


def gguf_inv(path):
    r = gguf.GGUFReader(path)
    t = {x.name: (x.tensor_type.name, tuple(int(d) for d in x.shape), int(x.n_bytes)) for x in r.tensors}
    meta = {k: f.contents() for k, f in r.fields.items() if not k.startswith("tokenizer.ggml.")}
    return t, meta


def cls_gguf(name):
    if name == "token_embd.weight":
        return "embed (gather)"
    if name.startswith("blk.64."):
        return "mtp block"
    if name.startswith("output"):
        return "output head"
    return "layers 0-63"


def st_inv(d):
    t = {}
    for f in sorted(Path(d).glob("*.safetensors")):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            h = json.loads(fh.read(n))
        for k, v in h.items():
            if k != "__metadata__":
                a, b = v["data_offsets"]
                t[k] = (v["dtype"], tuple(v["shape"]), b - a)
    return t


def cls_st(name):
    if "visual" in name:
        return "vision (not run)"
    if name.startswith("mtp.") or ".mtp." in name:
        return "mtp block"
    if "embed_tokens" in name:
        return "embed (gather)"
    if "lm_head" in name:
        return "output head"
    return "layers 0-63"


def summary(label, t, cls):
    by = collections.Counter()
    for k, (_, _, n) in t.items():
        by[cls(k)] += n
    tot = sum(by.values())
    step = by["layers 0-63"] + by["output head"]
    print(f"== {label}: {len(t)} tensors, {tot:,} bytes total")
    for c, n in sorted(by.items()):
        print(f"   {c:18s} {n:>15,} B  {n / 1e9:7.3f} GB")
    print(f"   READ/STEP (layers + output head) {step:,} B = {step / 1e9:.3f} GB; + mtp block {by['mtp block'] / 1e9:.3f} GB")
    return by


sw, bs = sys.argv[1], sys.argv[2]
(ts, ms), (tb, mb) = gguf_inv(sw), gguf_inv(bs)
for label, t in (("Swift GGUF", ts), ("Base GGUF", tb)):
    summary(label, t, cls_gguf)
    types = collections.Counter()
    for k, (ty, _, n) in t.items():
        types[ty] += n
    print("   bytes by type:", ", ".join(f"{k} {v / 1e9:.3f}" for k, v in types.most_common()))
print("== tensor names equal:", set(ts) == set(tb), "| only Swift:", sorted(set(ts) - set(tb))[:5], "| only Base:", sorted(set(tb) - set(ts))[:5])
dt = [(k, ts[k][0], tb[k][0]) for k in sorted(set(ts) & set(tb)) if ts[k][0] != tb[k][0]]
ds = [k for k in set(ts) & set(tb) if ts[k][1] != tb[k][1]]
print(f"== per-tensor type differences: {len(dt)}; shape differences: {len(ds)}")
for x in dt[:20]:
    print("   ", *x)
print(f"== metadata keys only Swift: {sorted(set(ms) - set(mb))}; only Base: {sorted(set(mb) - set(ms))}")
for k in sorted(set(ms) & set(mb)):
    a, b = ms[k], mb[k]
    if a != b:
        sa, sb = str(a), str(b)
        if k == "tokenizer.chat_template":
            sa, sb = hashlib.sha256(a.encode()).hexdigest(), hashlib.sha256(b.encode()).hexdigest()
        print(f"   {k}: Swift={sa[:100]} | Base={sb[:100]}")
for label, m in (("Swift", ms), ("Base", mb)):
    ct = m.get("tokenizer.chat_template")
    print(f"== {label} embedded chat template sha256:", hashlib.sha256(ct.encode()).hexdigest() if ct else None)
    print(f"   {label} nextn_predict_layers={m.get('qwen35.nextn_predict_layers')} block_count={m.get('qwen35.block_count')} general.name={m.get('general.name')}")
for d in sys.argv[3:]:
    summary(d, st_inv(d), cls_st)
