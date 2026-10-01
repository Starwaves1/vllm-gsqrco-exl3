"""exl3_gemm_mr launch-config probe at the MTP verify row counts (job 15): per representative shape
and rows 16/17/24/32/48, the default config vs trellis-serve's knobs (set_force_cfg(thread_k,
thread_n), set_blocks_per_sm(2)), GPU us per call in a CUDA graph (weight copies cycled past L2).
Prints one table; names the best knob per (shape, rows) and the model-level gain if each row
count used its best global knob.
  GSQ_ALLOW_GPU=1 EXL3_MODEL=<dir> python bench/micro/exl3_mr_cfg.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "gpu"))
sys.path.insert(0, str(ROOT / "plugin-exl3"))
sys.path.insert(0, str(ROOT / "bench" / "micro"))
from gsq_gpu import GPU  # noqa: E402,F401

import torch  # noqa: E402

import exl3_cases as C  # noqa: E402
from exl3_mr import inventory, load, time_graph  # noqa: E402
from vllm_exl3_plugin import _C_exl3_mr as M  # noqa: E402
from vllm_exl3_plugin import ops  # noqa: E402

ROWS = [16, 17, 24, 32, 48]
KNOBS = [("default", 0, 0, 1), ("k64n128", 64, 128, 1), ("k128n64", 128, 64, 1), ("k128n128", 128, 128, 1),
         ("k64n256", 64, 256, 1), ("bps2", 0, 0, 2), ("k64n128-bps2", 64, 128, 2), ("k128n64-bps2", 128, 64, 2)]


def main():
    X = torch.ops._C_exl3
    inv = inventory(C.MODEL)
    total = {}
    print(f"{'shape':>22} {'rows':>4} " + " ".join(f"{k[0]:>13}" for k in KNOBS), flush=True)
    for (k, n, K), (prefix, n_target, _) in sorted(inv.items()):
        if K not in (3, 4, 5) or n_target == 0 or n > 65536:
            continue
        w = load(C.MODEL, prefix)
        base = [w["trellis"]] + [w["trellis"].clone() for _ in range(max(0, -(-24 * 2**20 // (w["trellis"].numel() * 2)) - 1))]
        copies = [ops.repack_k4_(t) for t in base] if K == 4 else base
        args = (w["suh"], w["svh"], w["mcg"], w["mul1"])
        X.exl3_mr_warmup(copies[0], *args, ROWS, True)
        for m in ROWS:
            x = torch.randn(m, k, device="cuda").half()
            row = []
            for name, tk, tn, bps in KNOBS:
                M.set_force_cfg(tk, tn)
                M.set_blocks_per_sm(bps)
                try:
                    us = time_graph(lambda xx, c: X.exl3_gemm_mr(xx, c, *args, True), copies, x)
                except RuntimeError:
                    us = float("nan")
                row.append(us)
                total.setdefault((m, name), 0.0)
                total[(m, name)] += n_target * (us if us == us else 1e9)
            M.set_force_cfg(0, 0)
            M.set_blocks_per_sm(1)
            print(f"{f'{k}x{n} K{K} x{n_target}':>22} {m:>4} " + " ".join(f"{u:13.1f}" for u in row), flush=True)
        del base, copies, w
        torch.cuda.empty_cache()
    print("\nmodel sum (target pass, lm_head excluded) per knob, ms:")
    for m in ROWS:
        cells = {name: total[(m, name)] / 1e3 for name, *_ in KNOBS}
        best = min(cells, key=cells.get)
        print(f"  rows {m:3d}: " + " ".join(f"{nm} {v:7.2f}" for nm, v in cells.items()) + f"  | best {best}")


if __name__ == "__main__":
    main()
