"""R2 microbench: lcpp_mul_mat_iq3_packed (tiled, packed W) vs vendored MMQ on the GGUF bytes
(+ the unpack it needed on packed W before R2, on builds that still have lcpp_iq3_unpack) and R1's
packed decode kernel, on real GGUF blocks, bf16 X.

  GSQ_ALLOW_GPU=1 python r2_bench.py [--tokens 32,128] [--shapes 17408x5120] [--out f.tsv]

Times: GPU time per call, 10 calls per CUDA graph (bench/micro/gemm.py's time_graph), incl.
the q8_1 quantize and (MMQ) the output cast. floor_dram: torch.sum over the weight bytes, same
timing; floor_int8: 2 * rows * K * n / 284 TOPS (3090 dense int8). Each case also checks the
new op against MMQ (max |diff| / max |ref|, fp32 X).
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("WT", "/workspace/wt-r2"))
sys.path.insert(0, str(ROOT / "bench" / "micro"))
from gemm import GGUF, time_graph, weight  # noqa: E402

import gguf  # noqa: E402
import torch  # noqa: E402

from vllm_gguf_plugin.quantization import iq3_pack  # noqa: E402

C = torch.ops._C_gguf
SHAPES = {"17408x5120": (17408, 5120), "5120x17408": (5120, 17408), "10240x5120": (10240, 5120)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--types", default="IQ3_S,IQ3_XXS")
    ap.add_argument("--shapes", default=",".join(SHAPES))
    ap.add_argument("--tokens", default="8,16,32,64,128,512,2048")
    ap.add_argument("--variants", default="new,r1,mmq,unpack_mmq")
    ap.add_argument("--tn", default="", help="DEV: also time the new op at these forced tile widths (R2_TN)")
    ap.add_argument("--g", default="", help="DEV: also time the new op at these forced CTA counts (R2_G)")
    ap.add_argument("--abl", default="", help="DEV: also time these ablations of the new op (R2_ABL)")
    ap.add_argument("--out")
    a = ap.parse_args()
    variants = a.variants.split(",")
    reader = gguf.GGUFReader(str(GGUF))
    lines = ["type\tshape\tn\tvariant\tus\tpct_dram\tpct_int8\trel_vs_mmq"]
    for name in a.types.split(","):
        for sh in a.shapes.split(","):
            rows, k = SHAPES[sh]
            w, qt = weight(reader, name, rows, k)
            p = iq3_pack.pack(w, qt)
            buf = w.view(torch.float32).view(-1)  # any bytes: the reduction is bandwidth-bound
            floor = time_graph(lambda w_, x_: buf.sum(), w, None)
            print(f"\n{name} {sh}: {w.numel() / 1e6:.1f} MB, DRAM floor {floor:.1f} us", flush=True)
            for n in [int(t) for t in a.tokens.split(",")]:
                x = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
                xf = x.float()
                ref = C.lcpp_mul_mat_q(w, xf, qt, rows)
                got = C.lcpp_mul_mat_iq3_packed(p, xf, qt, rows)
                rel = ((got - ref).abs().max() / ref.abs().max()).item()
                exact = torch.equal(got, ref)
                fl8 = 2 * rows * k * n / 284e12 * 1e6
                fns = {}
                if "new" in variants:
                    fns["new"] = lambda w_, x_: C.lcpp_mul_mat_iq3_packed(p, x_, qt, rows)
                for tn in [t for t in a.tn.split(",") if t]:
                    def forced(w_, x_, tn=tn):
                        os.environ["R2_TN"] = tn
                        try:
                            return C.lcpp_mul_mat_iq3_packed(p, x_, qt, rows)
                        finally:
                            del os.environ["R2_TN"]
                    fns[f"new_tn{tn}"] = forced
                for gg in [t for t in a.g.split(",") if t]:
                    def forced_g(w_, x_, gg=gg):
                        os.environ["R2_G"] = gg
                        try:
                            return C.lcpp_mul_mat_iq3_packed(p, x_, qt, rows)
                        finally:
                            del os.environ["R2_G"]
                    fns[f"new_g{gg}"] = forced_g
                for ab in [t for t in a.abl.split(",") if t]:
                    def forced_a(w_, x_, ab=ab):
                        os.environ["R2_ABL"] = ab
                        try:
                            return C.lcpp_mul_mat_iq3_packed(p, x_, qt, rows)
                        finally:
                            del os.environ["R2_ABL"]
                    fns[f"new_abl{ab}"] = forced_a
                if "r1" in variants and n <= 32:
                    fns["r1"] = lambda w_, x_: C.lcpp_mul_mat_vec_iq3_mma_packed(p, x_, qt, rows)
                if "mmq" in variants:
                    fns["mmq"] = lambda w_, x_: C.lcpp_mul_mat_q(w_, x_, qt, rows)
                if "unpack_mmq" in variants and hasattr(C, "lcpp_iq3_unpack"):  # R1's path (removed in R2)
                    fns["unpack_mmq"] = lambda w_, x_: C.lcpp_mul_mat_q(C.lcpp_iq3_unpack(p, qt), x_, qt, rows)
                row = []
                for v, fn in fns.items():
                    us = time_graph(fn, w, x)
                    lines.append(f"{name}\t{sh}\t{n}\t{v}\t{us:.1f}\t{100 * floor / us:.0f}\t{100 * fl8 / us:.0f}\t{rel:.2e}")
                    row.append(f"{v} {us:8.1f}")
                print(f"  n={n:5d} " + " | ".join(row) + f"   floors dram {floor:.0f} int8 {fl8:.0f}  rel {rel:.1e}"
                      + (" exact" if exact else ""), flush=True)
            del w, p, buf
            torch.cuda.empty_cache()
    if a.out:
        Path(a.out).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
