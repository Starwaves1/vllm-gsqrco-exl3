"""EXL3 per-shape GEMM microbenchmark: exl3_gemm vs exl3_gemm_mr vs the dequant route, on the
checkpoint's real tensors, plus the model-level sum per row count (job 11-mr-micro).

  GSQ_ALLOW_GPU=1 EXL3_MODEL=<dir> python bench/micro/exl3_mr.py --out DIR [--rows 1,2,...]

Shapes: every distinct (k, n, K) among the checkpoint's text EXL3 tensors (vision skipped), one
real tensor each, with the number of products per target pass (main model + lm_head) and per
draft step (mtp.*). Activations fp16 randn (what EXLLinearMethod.apply passes), output fp32
(a bf16 model). Routes:
  gemm     torch.ops._C_exl3.exl3_gemm (exllamav3; re-streams the weight per 16 rows)
  mr       torch.ops._C_exl3.exl3_gemm_mr (trellis-serve Marlin-EXL3): K3/K5 as stored,
           K4 repacked (exl3_mr_repack); K2 and others: none
  dequant  ops._recon_hgemm, the 145..1023-row route (had, rotated dequant, f16acc hgemm, had)
gemm and mr are timed as 10 calls captured in one CUDA graph and replayed (GPU time, no launch
cost: decode runs under graphs), cycling through enough copies of the weight to defeat the
6 MB L2; dequant eager (it allocates; above 144 rows nothing is captured). GB/s = trellis bytes
/ time; floor % = the trellis bytes at 936 GB/s (3090 DRAM spec) / time.

Writes DIR/shapes.tsv (per shape, rows, route) and DIR/model.tsv + DIR/summary.txt: per row
count, ms per target pass and per draft step for EXL3_MR=0/1/2 routing, the per-shape best of
the three routes, and the byte floor. Above 144 rows (where every mode uses the dequant route)
only mr and dequant are timed: the crossover is the candidate new threshold.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "gpu"))
sys.path.insert(0, str(ROOT / "plugin-exl3"))
from gsq_gpu import GPU  # noqa: E402,F401  (imports no_gpu unless GSQ_ALLOW_GPU=1)

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402

import exl3_cases as C  # noqa: E402
from vllm_exl3_plugin import ops  # noqa: E402

DRAM_GBPS = 936.0
PER_GRAPH = 10
L2_BYTES = 6 * 2**20
ROWS = [1, 2, 4, 6, 8, 12, 16, 17, 24, 32, 48, 64, 96, 128, 144, 192, 256, 384, 512]


def inventory(model: Path):
    """(k, n, K) -> [example tensor prefix, products per target pass, products per draft step]."""
    wm = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    shapes: dict = {}
    for key, fname in wm.items():
        if not key.endswith(".trellis") or ".visual." in key or key.startswith(("visual.", "model.visual")):
            continue
        with safe_open(str(model / fname), framework="pt") as f:
            kt, nt, width = f.get_slice(key).get_shape()
        sk = (kt * 16, nt * 16, width // 16)
        e = shapes.setdefault(sk, [key[: -len(".trellis")], 0, 0])
        e[2 if key.startswith("mtp.") else 1] += 1
    return shapes


def load(model: Path, prefix: str):
    wm = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    out = {"mcg": f"{prefix}.mcg" in wm, "mul1": f"{prefix}.mul1" in wm}
    for t in ("trellis", "suh", "svh"):
        with safe_open(str(model / wm[f"{prefix}.{t}"]), framework="pt") as f:
            out[t] = f.get_tensor(f"{prefix}.{t}").cuda().contiguous()
    return out


def time_graph(call, copies, x):
    """GPU us per call: PER_GRAPH calls cycling the weight copies, in one graph, replayed."""
    for c in copies:
        call(x, c)
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        call(x, copies[0])
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(PER_GRAPH):
            call(x, copies[i % len(copies)])
    g.replay()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    best = float("inf")
    for _ in range(5):
        a.record()
        for _ in range(4):
            g.replay()
        b.record()
        b.synchronize()
        best = min(best, a.elapsed_time(b) * 1e3 / (4 * PER_GRAPH))
    del g
    return best


def time_eager(call, copies, x, calls=12):
    for c in copies:
        call(x, c)
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for i in range(calls):
        call(x, copies[i % len(copies)])
    b.record()
    b.synchronize()
    return a.elapsed_time(b) * 1e3 / calls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", default=",".join(map(str, ROWS)))
    a = ap.parse_args()
    rows = [int(r) for r in a.rows.split(",")]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    X = torch.ops._C_exl3
    inv = inventory(C.MODEL)
    print(f"{torch.cuda.get_device_name()} | {C.MODEL.name} | {len(inv)} shapes, "
          f"{sum(v[1] for v in inv.values())} target / {sum(v[2] for v in inv.values())} draft products", flush=True)
    lines = ["k\tn\tK\ttarget\tdraft\trows\troute\tus\tGBps\tfloor_pct"]
    res: dict = {}  # (shape, rows) -> {route: us}
    for (k, n, K), (prefix, n_target, n_draft) in sorted(inv.items()):
        w = load(C.MODEL, prefix)
        nbytes = w["trellis"].numel() * 2
        ncopy = max(1, -(-4 * L2_BYTES // nbytes))
        base = [w["trellis"]] + [w["trellis"].clone() for _ in range(ncopy - 1)]
        mr_copies = None
        if w["mul1"] and K in (3, 5):
            mr_copies = base
        elif w["mul1"] and K == 4:
            mr_copies = [X.exl3_mr_repack(t) for t in base]
        args = (w["suh"], w["svh"], w["mcg"], w["mul1"])
        if mr_copies is not None:
            X.exl3_mr_warmup(mr_copies[0], *args, rows, True)
        X.exl3_warmup(w["trellis"], *args, [1, 2, 4, 8, 16], True)
        floor_us = nbytes / (DRAM_GBPS * 1e3)
        print(f"\n{prefix} k={k} n={n} K={K} ({nbytes / 2**20:.1f} MiB, x{n_target} target, x{n_draft} draft, "
              f"floor {floor_us:.1f} us)", flush=True)
        for m in rows:
            x = torch.randn(m, k, device="cuda").half()
            t = {}
            if m <= ops.GEMM_MAX_ROWS:
                t["gemm"] = time_graph(lambda xx, c: X.exl3_gemm(xx, c, *args, True), base, x)
            if mr_copies is not None:
                t["mr"] = time_graph(lambda xx, c: X.exl3_gemm_mr(xx, c, *args, True), mr_copies, x)
            t["dequant"] = time_eager(lambda xx, c: ops._recon_hgemm(xx, c, w["suh"], w["svh"], w["mcg"], w["mul1"],
                                                                     False), base, x)
            res[((k, n, K), m)] = t
            for route, us in t.items():
                lines.append(f"{k}\t{n}\t{K}\t{n_target}\t{n_draft}\t{m}\t{route}\t{us:.1f}\t{nbytes / us / 1e3:.0f}\t"
                             f"{100 * floor_us / us:.0f}")
            print(f"  m={m:3d}  " + " | ".join(f"{r} {us:7.1f} us {100 * floor_us / us:3.0f}%" for r, us in t.items()),
                  flush=True)
        del base, mr_copies, w
        torch.cuda.empty_cache()
    (out / "shapes.tsv").write_text("\n".join(lines) + "\n")

    def route(K, m, mode):  # mode 0, 1, 2 or "2a" (EXL3_MR=2 with EXL3_MR_MIN=1)
        if m > ops.GEMM_MAX_ROWS:
            return "dequant"
        if mode in (2, "2a") and K == 4:
            return "mr"
        if mode != 0 and K in (3, 5) and (m >= 17 or mode == "2a"):
            return "mr"
        return "gemm"

    mlines = ["rows\tpass\tmr0_ms\tmr1_ms\tmr2_ms\tmr2a_ms\tbest_ms\tfloor_ms\tbest_routes"]
    summary = [f"model-level sum of per-shape GEMM time ({C.MODEL.name}); target = main model + lm_head "
               f"per verify pass, draft = mtp.* per draft step; floor = trellis bytes at {DRAM_GBPS:.0f} GB/s",
               f"{'rows':>4} {'pass':6} {'MR=0':>8} {'MR=1':>8} {'MR=2':>8} {'MR=2a':>8} {'best':>8} {'floor':>7}  best route mix"]
    for m in rows:
        for which, idx in (("target", 1), ("draft", 2)):
            tot = Counter()
            mix = Counter()
            for (k, n, K), v in inv.items():
                cnt = v[idx]
                if not cnt:
                    continue
                t = res[((k, n, K), m)]
                for mode in (0, 1, 2, "2a"):
                    tot[f"mr{mode}"] += cnt * t[route(K, m, mode)]
                b = min(t, key=t.get)
                tot["best"] += cnt * t[b]
                mix[b] += cnt
                tot["floor"] += cnt * k * n * K / 8 / (DRAM_GBPS * 1e3)
            if not tot:
                continue
            ms = {key: val / 1e3 for key, val in tot.items()}
            mixs = ",".join(f"{r}:{c}" for r, c in sorted(mix.items()))
            mlines.append(f"{m}\t{which}\t{ms['mr0']:.3f}\t{ms['mr1']:.3f}\t{ms['mr2']:.3f}\t{ms['mr2a']:.3f}\t{ms['best']:.3f}\t"
                          f"{ms['floor']:.3f}\t{mixs}")
            summary.append(f"{m:4d} {which:6} {ms['mr0']:8.3f} {ms['mr1']:8.3f} {ms['mr2']:8.3f} {ms['mr2a']:8.3f} {ms['best']:8.3f} "
                           f"{ms['floor']:7.3f}  {mixs}")
    (out / "model.tsv").write_text("\n".join(mlines) + "\n")
    (out / "summary.txt").write_text("\n".join(summary) + "\n")
    print("\n" + "\n".join(summary))


if __name__ == "__main__":
    main()
