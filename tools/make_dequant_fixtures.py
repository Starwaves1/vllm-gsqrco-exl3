"""Build dequant fixtures from the Swift GSQ-RCO IQ3_S-mtp GGUF.

For each quantized ggml type in the file (10 types), take up to 3 tensors
(first/middle/last by name order) and 3 rows each (first/middle/last), and save
  tests/fixtures/dequant/<TYPE>__<tensor>__r<row>.npz
with keys raw (uint8 [row_bytes]), ref (float32 [ne0], gguf-py dequant),
ggml_type (int), tensor (str), rows (int64 [1]: the row index), shape
(int64 [2]: logical (rows, cols) of the whole tensor), plus manifest.json.

Every ref is cross-checked bit-exactly against libggml-base (tools/ggml_ref.py)
at generation time. Only the selected rows are read (the GGUF is memory-mapped),
so memory stays small. Run light:
  GSQ_LIGHT=1 tools/capped .venv/bin/python tools/make_dequant_fixtures.py
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import no_gpu  # noqa: F401,E402  (before anything that might pull torch)

import numpy as np  # noqa: E402
import gguf  # noqa: E402
from gguf.quants import dequantize  # noqa: E402

import ggml_ref  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GGUF = os.path.expanduser(
    "~/qwen38-27b-rtx3090/models/Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF/"
    "Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf"
)
UNQUANT = {"F32", "F16", "BF16"}


def _key(name):
    parts = name.split(".")
    return (int(parts[1]) if parts[0] == "blk" else -1, name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default=GGUF)
    ap.add_argument("--out", default=os.path.join(ROOT, "tests/fixtures/dequant"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    r = gguf.GGUFReader(a.gguf)
    by_type = {}
    for t in r.tensors:
        if t.tensor_type.name not in UNQUANT:
            by_type.setdefault(t.tensor_type.name, []).append(t)

    manifest = {"gguf": os.path.basename(a.gguf), "types": sorted(by_type), "fixtures": []}
    for tname, ts in sorted(by_type.items()):
        ts.sort(key=lambda t: _key(t.name))
        picks = list(dict.fromkeys([0, len(ts) // 2, len(ts) - 1]))
        for t in (ts[i] for i in picks):
            ne0, nrows = int(t.shape[0]), int(np.prod(t.shape[1:]))
            data = t.data.reshape(nrows, -1)
            for row in dict.fromkeys([0, nrows // 2, nrows - 1]):
                raw = np.array(data[row], dtype=np.uint8)
                ref = dequantize(raw, t.tensor_type).reshape(-1).astype(np.float32)
                c = ggml_ref.dequantize(raw, int(t.tensor_type), ne0)
                if ref.shape != (ne0,) or ref.tobytes() != c.tobytes():
                    raise SystemExit(f"gguf-py != libggml-base: {t.name} row {row}")
                fn = f"{tname}__{t.name}__r{row}.npz"
                np.savez_compressed(
                    os.path.join(a.out, fn),
                    raw=raw, ref=ref, ggml_type=np.int64(int(t.tensor_type)),
                    tensor=np.str_(t.name), rows=np.array([row], dtype=np.int64),
                    shape=np.array([nrows, ne0], dtype=np.int64),
                )
                manifest["fixtures"].append(
                    {"file": fn, "type": tname, "tensor": t.name, "row": row,
                     "shape": [nrows, ne0]}
                )
    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"{len(manifest['fixtures'])} fixtures, types {manifest['types']}")


if __name__ == "__main__":
    main()
