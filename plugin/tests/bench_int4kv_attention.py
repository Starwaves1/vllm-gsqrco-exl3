# SPDX-License-Identifier: Apache-2.0
"""Decode attention time per layer call: CUDA int4 kernel vs vLLM's Triton int4.

    python tests/bench_int4kv_attention.py [--ctx 32768,131072] [--c 1,2,4,8] [--q-len 1,4]

Each cell: median of 15 calls, CUDA events around the attention op only, L2
flushed before every call. Both paths include their Q rotation, split-KV
reduce and output rotation. GB/s counts the K+V+scale bytes read once.
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from test_int4kv_attention import SLOT, KV_HEADS, build_case, run_cuda, run_triton, triton_segments  # noqa: E402


def median_ms(fn, flush, n=15):
    fn()
    times = []
    for _ in range(n):
        flush.sum()  # evicts L2 with clean lines
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b))
    return sorted(times)[n // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", default="8192,32768,65536,131072")
    ap.add_argument("--c", default="1,2,4,8")
    ap.add_argument("--q-len", default="1,4")
    ap.add_argument("--bw", type=float, default=None, help="measured DRAM GB/s for the floor column")
    args = ap.parse_args()
    flush = torch.ones(32 << 20, dtype=torch.int64, device="cuda")
    print(f"# {torch.cuda.get_device_name()}  bw floor {args.bw} GB/s")
    print("ctx\tc\tq_len\ttriton_ms\tcuda_ms\tspeedup\tcuda_GB/s\tof_floor")
    for ctx in map(int, args.ctx.split(",")):
        for c in map(int, args.c.split(",")):
            for q_len in map(int, args.q_len.split(",")):
                case = build_case(c, ctx, q_len)
                segm = triton_segments(case)
                out = torch.empty_like(case.q)
                t_tri = median_ms(lambda: run_triton(case, out, segm), flush)
                t_cu = median_ms(lambda: run_cuda(case, out), flush)
                gbs = case.seq_lens.sum().item() * KV_HEADS * SLOT / t_cu / 1e6
                floor = f"{gbs / args.bw:.0%}" if args.bw else "-"
                print(f"{ctx}\t{c}\t{q_len}\t{t_tri:.3f}\t{t_cu:.3f}\t{t_tri / t_cu:.2f}x\t{gbs:.0f}\t{floor}", flush=True)
                del case, segm, out
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
