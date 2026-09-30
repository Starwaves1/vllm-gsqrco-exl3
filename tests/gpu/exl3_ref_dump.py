"""Reference side of the EXL3 kernel parity check: exllamav3 d3739fd on the same checkpoint
tensors and activations as tests/gpu/test_exl3_kernels.py (cases in exl3_cases.py). Runs in
the exllamav3 reference venv (/workspace/venv-exl3ref: exllamav3_ext built for sm86,
EXL3_INT8_GEMV=0 forced at interpreter start, as the plugin's shim does), never in vLLM's.

For each tensor, into EXL3_REF_DIR/<tid>.json:
  dq_had_sha256  sha256 of ext.reconstruct_had_slice (original basis, both Hadamards and
                 suh/svh folded in) over 32768-column slices, in order
  dq_rot_sha256  same for ext.reconstruct_slice (rotated basis)
  gemm[m][fp16|fp32]  LinearEXL3.forward (exllamav3's own dispatch: exl3_gemm up to 144 rows,
                 reconstruct + hgemm above, fused from 1024) at out_dtype fp16/fp32:
                 sha256 of the output and its error stats against fp64
Share EXLLAMAV3_TUNE_CACHE with the plugin run so both pick the same autotuned kernels.

  GSQ_ALLOW_GPU=1 /workspace/venv-exl3ref/bin/python tests/gpu/exl3_ref_dump.py [--only TID,..]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
if os.environ.get("GSQ_ALLOW_GPU") != "1":
    raise SystemExit("GPU job: set GSQ_ALLOW_GPU=1")
os.environ["EXL3_INT8_GEMV"] = "0"  # before exllamav3_ext loads (the venv's .pth does it too)

import exl3_cases as C  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only")
    a = ap.parse_args()
    import torch
    from exllamav3.ext import exllamav3_ext as ext
    from exllamav3.version import __version__ as exl3_version
    from exllamav3.modules.quant.exl3 import LinearEXL3

    C.REF_DIR.mkdir(parents=True, exist_ok=True)
    tids = a.only.split(",") if a.only else list(C.TENSORS)
    meta = {"exllamav3": exl3_version, "torch": torch.__version__,
            "device": torch.cuda.get_device_name(0), "EXL3_INT8_GEMV": os.environ["EXL3_INT8_GEMV"],
            "tune_cache": os.environ.get("EXLLAMAV3_TUNE_CACHE")}
    (C.REF_DIR / "meta.json").write_text(json.dumps(meta, indent=1))
    for tid in tids:
        t0 = time.time()
        _, K, k, n = C.TENSORS[tid]
        w = C.load(torch, tid)
        flags = {"mcg": w["mcg"], "mul1": w["mul1"]}
        one = torch.ones((), dtype=torch.int32, device="cuda")
        lin = LinearEXL3(None, k, n, suh=w["suh"], svh=w["svh"], trellis=w["trellis"],
                         mcg=one if w["mcg"] else None, mul1=one if w["mul1"] else None, key=tid)
        Kx = lin.K

        def dq_had(s, c):
            out = torch.empty((k, c), dtype=torch.half, device="cuda")
            ext.reconstruct_had_slice(out, w["trellis"], w["suh"], w["svh"][s:], Kx, flags["mcg"], flags["mul1"], s)
            return out

        def dq_rot(s, c):
            out = torch.empty((k, c), dtype=torch.half, device="cuda")
            ext.reconstruct_slice(out, w["trellis"], Kx, flags["mcg"], flags["mul1"], s)
            return out

        rec = {"tid": tid, "K": K, "k": k, "n": n, **flags,
               "dq_had_sha256": C.hash_slices(dq_had, n), "dq_rot_sha256": C.hash_slices(dq_rot, n), "gemm": {}}
        for m in C.ROWS:
            x = C.make_x(torch, tid, m)
            ref = C.fp64_ref(torch, x, dq_had, n)
            rec["gemm"][str(m)] = {}
            for name, dt in (("fp16", torch.half), ("fp32", torch.float)):
                y = lin.forward(x, {}, out_dtype=dt)
                torch.cuda.synchronize()
                rec["gemm"][str(m)][name] = {"sha256": C.sha_tensor(y), **C.err_stats(torch, y, ref)}
            del ref
        (C.REF_DIR / f"{tid}.json").write_text(json.dumps(rec, indent=1))
        worst = max(v["rel_rms"] for g in rec["gemm"].values() for v in g.values())
        print(f"{tid}: K={K} k={k} n={n} dq_had {rec['dq_had_sha256'][:12]} worst rel_rms {worst:.2e} "
              f"({time.time() - t0:.0f} s)", flush=True)
        del w, lin
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
