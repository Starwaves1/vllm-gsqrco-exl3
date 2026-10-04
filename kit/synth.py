"""Synthetic weights at the Qwen3.8-27B shapes, so tier 1 needs no model download.

  python kit/synth.py gguf OUT.gguf [--max-rows N]
  python kit/synth.py exl3 OUT_DIR  [--max-rows N]
  python kit/synth.py check-gguf OUT.gguf      (CPU: every tensor dequantizes to finite values)

gguf: one tensor per distinct (type, rows, K) of kit/data/swift-27b-gguf-tensors.tsv, under the
real tensor's name, plus the ffn_gate/ffn_up and attn_qkv/attn_gate pairs of the first block for
every distinct pair of types (tests/gpu/test_kernel_parity.py looks those up by blk.N name).
token_embd (a gather, no GEMM) and output.weight (248320 rows: the whole-tensor fp64 references
would need ~5 GB of VRAM) are left out; the micro-benchmark builds the lm_head shape itself.
Blocks are random bytes with every fp16 scale replaced by a finite value in [2^-13, 2^-9], so
every grid index, sign and sub-scale is valid and no block decodes to inf/NaN.

exl3: a checkpoint directory named like erlidev's (tests/gpu/exl3_cases.py picks its tensor table
by the directory name), holding the 8 named case tensors plus one tensor per other distinct
trellis shape of the real checkpoint (hf-config/.../safetensors_headers.json): random int16
trellis (every 16-bit pattern decodes), suh = random signs, svh = random signs x 0.05, mul1
codebook. Kernel speed does not depend on the values.
--max-rows N caps every tensor's rows (out_features) for a quick CPU dry run.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GGUF_TSV = ROOT / "kit/data/swift-27b-gguf-tensors.tsv"
EXL3_NAME = "Swift-1.5-Qwen3.8-27B-exl3-SC_3.50bpw_H4_V6"
EXL3_HEADERS = ROOT / "hf-config" / EXL3_NAME / "safetensors_headers.json"
SKIP = ("token_embd.weight", "output.weight")
PAIRS = [("ffn_gate", "ffn_up"), ("attn_qkv", "attn_gate")]
# fp16 scale fields per block (byte offsets), ggml-common.h; IQ1_M keeps its scale in the
# top nibbles of its 4 uint16 sub-scale words (offset 48), handled separately
F16_OFFSETS = {"Q2_K": (80, 82), "Q4_K": (0, 2), "Q6_K": (208,), "IQ2_XXS": (0,), "IQ2_XS": (0,),
               "IQ2_S": (0,), "IQ3_XXS": (0,), "IQ3_S": (0,), "IQ4_XS": (0,)}
SEED = 20261004


def gguf_tensors():
    rows = [r for r in csv.DictReader((line for line in GGUF_TSV.open() if not line.startswith("#")),
                                      delimiter="\t")]
    return [(r["name"], r["type"], int(r["rows"]), int(r["K"])) for r in rows]


def gguf_selection(tensors):
    """The tensors the synthetic GGUF holds (see the module docstring), in file order."""
    by_name = {t[0]: t for t in tensors}
    keep, shapes, pairs = [], set(), set()
    for name, typ, rows, k in tensors:
        if name in SKIP:
            continue
        if (typ, rows, k) not in shapes:
            shapes.add((typ, rows, k))
            keep.append(name)
    for i in range(64):
        for a, b in PAIRS:
            ta, tb = by_name.get(f"blk.{i}.{a}.weight"), by_name.get(f"blk.{i}.{b}.weight")
            if ta and tb and ta[1] != tb[1] and (a, ta[1], tb[1]) not in pairs:
                pairs.add((a, ta[1], tb[1]))
                keep += [ta[0], tb[0]]
    order = {t[0]: i for i, t in enumerate(tensors)}
    return [by_name[n] for n in sorted(set(keep), key=order.get)]


def random_blocks(rng, typ, rows, k):
    import gguf

    qt = gguf.GGMLQuantizationType[typ]
    bs, bsz = gguf.GGML_QUANT_SIZES[qt]
    nb = rows * (k // bs)
    b = rng.integers(0, 256, size=(nb, bsz), dtype=np.uint8)
    scale = lambda: np.exp2(rng.uniform(-13, -9, nb)).astype(np.float16)  # noqa: E731
    if typ == "IQ1_M":
        bits = scale().view(np.uint16).astype(np.uint16)
        sc = b[:, 48:56].copy().view(np.uint16)  # 4 words; top nibble of word j = scale bits 4j..4j+3
        for j in range(4):
            sc[:, j] = (sc[:, j] & 0x0FFF) | (((bits >> (4 * j)) & 0xF) << 12)
        b[:, 48:56] = sc.view(np.uint8)
    else:
        for off in F16_OFFSETS[typ]:
            b[:, off:off + 2] = scale().view(np.uint8).reshape(nb, 2)
    return b.reshape(rows, -1), qt


def write_gguf(out: Path, max_rows: int | None):
    import gguf

    rng = np.random.default_rng(SEED)
    w = gguf.GGUFWriter(str(out), "kit-synthetic")
    w.add_string("kit.source", f"{GGUF_TSV.name}: synthetic blocks, seed {SEED}")
    sel = gguf_selection(gguf_tensors())
    total = 0
    for name, typ, rows, k in sel:
        rows = min(rows, max_rows) if max_rows else rows
        data, qt = random_blocks(rng, typ, rows, k)
        w.add_tensor(name, data, raw_dtype=qt)
        total += data.nbytes
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"{out}: {len(sel)} tensors, {total / 2**30:.2f} GiB")


def check_gguf(path: Path):
    import gguf

    r = gguf.GGUFReader(str(path))
    bad = 0
    for t in r.tensors:
        v = gguf.quants.dequantize(np.asarray(t.data[:64]), t.tensor_type)
        ok = bool(np.isfinite(v).all()) and float(np.abs(v).max()) > 0
        bad += not ok
        print(f"{t.name:28s} {t.tensor_type.name:8s} {int(t.shape[1]):6d}x{int(t.shape[0]):<6d} "
              f"absmax {float(np.abs(v).max()):.3e} {'ok' if ok else 'BAD'}")
    sys.exit(1 if bad else 0)


def write_exl3(out: Path, max_rows: int | None):
    import torch
    from safetensors.torch import save_file

    sys.path.insert(0, str(ROOT / "tests/gpu"))
    sys.path.insert(0, str(ROOT / "plugin-exl3"))
    from vllm_exl3_plugin.format import MUL1_MULT

    if out.name != EXL3_NAME:
        sys.exit(f"the directory must be named {EXL3_NAME} (exl3_cases.py keys its table on it)")
    heads = json.loads(EXL3_HEADERS.read_text())
    shapes = {}
    for h in heads.values():
        for key, v in h["tensors"].items():
            if key.endswith(".trellis") and "visual" not in key:
                shapes[key[: -len(".trellis")]] = tuple(v["shape"])
    import os

    os.environ.setdefault("EXL3_MODEL", str(out))
    import exl3_cases

    keep = [v[0] for v in exl3_cases.CHECKPOINTS[EXL3_NAME].values()]
    seen = {shapes[p] for p in keep}
    for p, s in sorted(shapes.items()):
        if s not in seen:
            seen.add(s)
            keep.append(p)
    g = torch.Generator().manual_seed(SEED)
    tensors = {}
    for p in keep:
        kt, nt, w16 = shapes[p]
        if max_rows:
            nt = min(nt, max(1, max_rows // 16))
        tensors[f"{p}.trellis"] = torch.randint(-32768, 32768, (kt, nt, w16), generator=g, dtype=torch.int32).to(torch.int16)
        tensors[f"{p}.suh"] = (torch.randint(0, 2, (kt * 16,), generator=g) * 2 - 1).half()
        tensors[f"{p}.svh"] = ((torch.randint(0, 2, (nt * 16,), generator=g) * 2 - 1) * 0.05).half()
        tensors[f"{p}.mul1"] = torch.tensor(MUL1_MULT - 2**32 if MUL1_MULT >= 2**31 else MUL1_MULT, dtype=torch.int32)
    out.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out / "model.safetensors"), metadata={"kit": f"synthetic, seed {SEED}"})
    (out / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {}, "weight_map": {k: "model.safetensors" for k in tensors}}, indent=1))
    nbytes = sum(t.numel() * t.element_size() for t in tensors.values())
    print(f"{out}: {len(keep)} EXL3 tensors, {nbytes / 2**30:.2f} GiB")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["gguf", "exl3", "check-gguf"])
    ap.add_argument("out", type=Path)
    ap.add_argument("--max-rows", type=int)
    a = ap.parse_args()
    if a.what == "gguf":
        write_gguf(a.out, a.max_rows)
    elif a.what == "exl3":
        write_exl3(a.out, a.max_rows)
    else:
        check_gguf(a.out)


if __name__ == "__main__":
    main()
