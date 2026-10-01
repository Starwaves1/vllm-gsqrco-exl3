"""R3 GEMM sweep at production's row counts: current route vs every applicable Route L kernel.

Rows (activation tokens) production's main schedule produces: k=5 at 1-4 running -> 6/12/18/24,
k=3 at 5-8 -> 20/24/28/32, k=2 at 9-16 -> 27..48, plus prefill chunks (128-token long-prefill
threshold + decode rows -> 128..160) and the MTP draft passes (1 row per sequence).
For each (type, shape, n): the op linear._lcpp_op routes to ("route"), and the alternatives:
MMQ, mma_k (any n; the kernel's own guard decides), packed tiled IQ3, packed IQ3 decode (<= 32),
MMVQ / owned decode kernels (<= 8). Time = GPU time per call inside a CUDA graph (bench/micro/gemm.py's
time_graph). Floor = weight bytes / 936 GB/s; also reported: effective GB/s and int8 TOPS.

  GSQ_ALLOW_GPU=1 python r3gemm.py --out sweep.tsv [--types ...] [--tokens ...] [--shapes RxK,...]
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "bench" / "micro"))
import gemm  # noqa: E402  (bench/micro/gemm.py: weight(), time_graph(), GGUF reader)

import gguf  # noqa: E402
import torch  # noqa: E402
from vllm_gguf_plugin.quantization import iq3_pack  # noqa: E402
from vllm_gguf_plugin.quantization import linear  # noqa: E402

TYPES = ["IQ3_S", "IQ3_XXS", "IQ4_XS", "Q4_K", "IQ2_S", "Q6_K", "Q2_K", "IQ2_XS", "IQ2_XXS"]
SHAPES = [(17408, 5120), (5120, 17408), (10240, 5120), (6144, 5120)]
TOKENS = [6, 12, 18, 20, 24, 27, 28, 32, 36, 48, 128, 136, 160]
HBM = 936e9


def candidates(name, qt, rows, k, n, packed):
    C = torch.ops._C_gguf
    v = {"mmq": lambda w, x: C.lcpp_mul_mat_q(w, x, qt, rows)}
    if hasattr(C, "lcpp_mul_mat_mma_k"):
        v["mma_k"] = lambda w, x: C.lcpp_mul_mat_mma_k(w, x, qt, rows)
        if n > 64:  # the kernel takes <= 64 rows (8 x MAX_NT): chunked calls, W read once per chunk
            v["mma_k_chunked64"] = lambda w, x: torch.cat(
                [C.lcpp_mul_mat_mma_k(w, x[i:i + 64], qt, rows) for i in range(0, x.shape[0], 64)])
    if packed is not None:
        v["iq3_tiled_packed"] = lambda w, x: C.lcpp_mul_mat_iq3_packed(packed, x, qt, rows)
        if n <= 32:
            v["iq3_vec_mma_packed"] = lambda w, x: C.lcpp_mul_mat_vec_iq3_mma_packed(packed, x, qt, rows)
    if n <= 8:
        v["mmvq"] = lambda w, x: C.lcpp_mul_mat_vec_q(w, x, qt, rows)
        v["own_vec"] = lambda w, x: C.lcpp_mul_mat_vec_own(w, x, qt, rows)
    return v


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--types", default=",".join(TYPES))
    ap.add_argument("--tokens", default=",".join(map(str, TOKENS)))
    ap.add_argument("--shapes", default=",".join(f"{r}x{k}" for r, k in SHAPES))
    a = ap.parse_args()
    reader = gguf.GGUFReader(str(gemm.GGUF))
    tokens = [int(t) for t in a.tokens.split(",")]
    shapes = [tuple(int(d) for d in s.split("x")) for s in a.shapes.split(",")]
    lines = ["type\trows\tK\tn\tvariant\tis_route\tus\tfloor_us\tGBps\tTOPS"]
    print(f"{torch.cuda.get_device_name()}  X bf16, graph-replay timing", flush=True)
    for name in a.types.split(","):
        for rows, k in shapes:
            try:
                w, qt = gemm.weight(reader, name, rows, k)
            except Exception as e:  # noqa: BLE001  (type absent from the GGUF)
                print(f"{name} {rows}x{k}: skipped ({e})")
                continue
            packed = None
            if name.startswith("IQ3") and rows % 16 == 0:
                packed = iq3_pack.pack(w, qt)
            floor = w.numel() / HBM * 1e6
            print(f"\n{name} {rows}x{k} ({w.numel() / 1e6:.1f} MB, floor {floor:.1f} us)", flush=True)
            for n in tokens:
                x = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
                route = linear._lcpp_op(n, qt, rows, k, packed is not None)
                row = []
                for vname, fn in candidates(name, qt, rows, k, n, packed).items():
                    try:
                        us = gemm.time_graph(fn, w, x)
                    except Exception as e:  # noqa: BLE001  (kernel guard: unsupported n/type)
                        msg = str(e).splitlines()[0][:60]
                        row.append(f"{vname}: n/a ({msg})")
                        torch.cuda.synchronize()
                        continue
                    op = {"mmq": "lcpp_mul_mat_q", "mma_k": "lcpp_mul_mat_mma_k", "mma_k_chunked64": "-",
                          "iq3_tiled_packed": "lcpp_mul_mat_iq3_packed",
                          "iq3_vec_mma_packed": "lcpp_mul_mat_vec_iq3_mma_packed",
                          "mmvq": "lcpp_mul_mat_vec_q", "own_vec": "lcpp_mul_mat_vec_own"}[vname]
                    is_route = int(op == route)
                    gbps = w.numel() / (us * 1e-6) / 1e9
                    tops = 2.0 * n * rows * k / (us * 1e-6) / 1e12
                    lines.append(f"{name}\t{rows}\t{k}\t{n}\t{vname}\t{is_route}\t{us:.1f}\t{floor:.1f}\t{gbps:.0f}\t{tops:.1f}")
                    row.append(f"{'*' if is_route else ''}{vname} {us:.1f}us ({us / floor:.2f}x floor)")
                print(f"  n={n:4d} route={route}: " + " | ".join(row), flush=True)
            del w, packed
            torch.cuda.empty_cache()
            Path(a.out).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
