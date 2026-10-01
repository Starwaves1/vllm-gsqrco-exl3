# EXL3 optimization (branch `exl3-opt`, from `exl3` 977bafd)

2026-10-01: prepared CPU-only, then measured on the rented 3090 (jobs 10-14, results below).
Labels as in EXL3.md: VERIFIED, DOCUMENTED, INFERRED.

## Results (box, erlidev SC_3.50bpw_H4_V6, production's main argv, max-model-len 196608)

**10-mr-parity** (VERIFIED, b01436b): 294 passed, 28 skipped (K3/K5 below 17 rows in the old
table). Decoded weights equal exl3_dequant for every row and column (K3/K4/K5, lm_head
included). Across 150 (tensor, rows 1..384, bf16/fp16) cases, mr's rms error vs fp64 is at or
below exl3_gemm's in 149 (worst 2.0e-3); max_rel is at or below in 133 (worst 3.4e-2, floor
3e-2 x1.5 slack; run 1 failed one case at 0.0246 vs 0.0244, a bf16-rounding outlier with better
rms). Also: routing (K2 bit-identical), repacked K4 above 384 == stored, determinism, graph
replay, unwarmed capture refused, bf16 io == cast path bit for bit, host embedding gather ==
F.embedding (incl. graph replay). memcheck and initcheck clean on 42 cases. Error table:
`cloud/results/exl3-opt/10-mr-parity-run1/errors.txt`.

**11-mr-micro** (VERIFIED): model sum of the 401 target-pass GEMMs (trellis 11.28 GB, floor
12.05 ms at 936 GB/s), `cloud/results/exl3-opt/11-mr-micro/`:

| rows | exl3_gemm (MR=0) | MR=2, K3/K5 from 17 | MR=2, K3/K5 from 1 | dequant route |
|---|---|---|---|---|
| 6 (c=1) | 24.6 ms, 459 GB/s, 49 % | 22.3 | 18.8 ms, 599 GB/s, 64 % | |
| 12 (c=2) | 26.4 | 24.7 | 23.1 | |
| 24 (c=4) | 51.3 ms, 220 GB/s, 23 % | 35.4 | 35.4 ms, 319 GB/s, 34 % | |
| 32 (c=8) | 53.5 | 36.1 | 36.1 | |
| 48 | 79.4 | 50.0 | 50.0 | |
| 144 | 234.8 | 142.5 | 142.5 | 234.9 |
| 192 | | | 169.6 (mr) | 279.7 |
| 384 | | | 326.0 (mr) | 353.5 |
| 512 | | | 434.5 (mr) | 423.5 |

Neither kernel is near the floor: mr reaches 64 % at 6 rows and 34 % at 24-32 rows. Per shape at
24 rows, K4 spans 13-48 % of floor, K3 24-30 %, K5 (k_proj, 5120x1024) 17 %. The step from 16 to
17 rows (23.1 to 35.4 ms) is the launcher's switch from the 16-row family to thread_m_blocks 2
with 256-thread configs (INFERRED from `exl3_marlin.cu`'s config tables, not profiled).

**12-mr-ladder** (VERIFIED, pass 2, T=0; ms/step = C x 1000 / decode tok/s x tok/step; MR=0 is
phase 1's 06-ladder, same argv):

LADDER_PLACEHOLDER

**Fit**: FIT_PLACEHOLDER

**13-profile**: PROFILE_PLACEHOLDER

**Kept / reverted**: KEPT_PLACEHOLDER

**Found on the way**: an allocating K4 repack left the int16 copies live at load (23.05 GiB
allocated, OOM in the lm_head warmup): the repack is now in place. vLLM's torch.compile cache key
does not cover the plugin's `apply()`/`embedding()` (same aot hash with and without glue), so
graph variants get their own `VLLM_CACHE_ROOT` in the jobs; the two shared entries written while
that was not so are in `/workspace/runs/exl3-opt/moved-compile-cache/`. A bare `torch.cat` of
custom-op outputs (glue on fused layers) trips inductor's split-of-cat simplification (GDN's z
comes back with the part's own stride): glue is on single-part layers only.


## What is here

| path | what |
|---|---|
| `plugin-exl3/vllm_exl3_plugin/csrc/trellis_serve/` | trellis-serve `1ace59c4b43c` (MIT; Marlin parts Apache-2.0), 19 files, byte-identical, `VENDORED.md` with sha256s and licenses |
| `.../csrc/exl3_mr_shim.cu` | ops `exl3_gemm_mr`, `exl3_mr_repack`, `exl3_mr_unpack`, `exl3_mr_warmup` in the `_C_exl3` namespace |
| `plugin-exl3/setup.py` | second extension `_C_exl3_mr` (15 generated instance units: row families 0..4 x mul1 x K 3/4/5); `_C_exl3` is unchanged |
| `.../ops.py`, `.../quantization/linear.py` | `EXL3_MR` routing, the K4 repack at load, the mr warmup |
| `tests/cpu/test_exl3_mr.py` | 96 CPU tests |
| `tests/gpu/test_exl3_mr.py`, `bench/micro/exl3_mr.py` | GPU parity (223 cases) and the per-shape microbenchmark |
| `cloud/results/exl3-opt/box-scripts/` | jobs 10-13, `run-job.sh`, `lib.sh` |

## The kernel and its weight layout

trellis-serve's dense kernel is vLLM's Marlin template with the int4 dequant swapped for EXL3
trellis decode. It covers K 3/4/5/6, codebooks 3INST/MCG/MUL1, and m-tiles up to 64 rows, so
17..64 rows take one weight pass where exllamav3 needs one per 16 rows. MMA accumulates in fp32;
C is stored fp16 before the output Hadamard (VERIFIED in source). It reads (VERIFIED,
`repack_trellis`):

| K | layout vs exllamav3's int16 `[k/16, n/16, 16K]` | here |
|---|---|---|
| 3, 5 | the same bytes viewed as int32 `[k/16, n/64, 4, 24 or 40]` | read in place, no copy |
| 4 | word permutation to `[k/16, n/64, 32, 4]` (lossless, same size) | `exl3_mr_repack` at load (EXL3_MR=2) |
| 6 | bit re-layout, 8 bits/weight (+33 %) | not used |
| 2, 1, 7, 8, x.5 | unsupported | stay on exl3_gemm |

exllamav3's kernels and dequant cannot read the K4 permutation, and the fit has no room for two
copies. So under EXL3_MR=2 the repack is the only K4 copy (that is trellis-serve's own 24 GB
layout). K4 then runs `exl3_gemm_mr` at every row count up to 144, and above 144 the dequant
routes unpack per call. Bytes per target pass (erlidev, VERIFIED from the headers): K3 38.6 %,
K4 54.0 % + lm_head 5.6 %, K5 0.9 %, K2 0.8 %, out of 11.28 GB (12.05 ms at 936 GB/s). EXL3_MR=1
alone therefore reaches 40 % of the bytes, and EXL3_MR=2 reaches 99 %.

## Switches (read once when the plugin imports)

| variable | default | what |
|---|---|---|
| `EXL3_MR` | 2 | 0: phase 1's routing; 1: K3/K5 (mul1) on exl3_gemm_mr; 2: and K4 (lm_head, MTP included), repacked in place at load |
| `EXL3_MR_MIN` | 1 | first row count K3/K5 take exl3_gemm_mr at (K4 under 2: always from 1) |
| (constant) `MULTI_ROW_MAX` | 384 | last row count on exl3_gemm_mr; above, the dequant routes (K2 keeps exllamav3's 144) |
| `EXL3_MR_GLUE` | 0 | with 2: bf16 straight through exl3_gemm_mr on single-part layers (A/B) |
| `EXL3_EMBED_HOST` | 0 | bf16 token embedding in pinned host memory, gathered per step (A/B) |

With bf16 out the kernel writes bf16, widened to fp32 for `out_fp32`, so the result is one
rounding of the fp32 epilogue, as with exl3_gemm. The capture guard is keyed per (k, n, K, rows)
because the launcher picks its config from m. `exl3_mr_warmup` runs every routed row count at
load. The vendored per-device state is 21.5 MB of fp32 reduce scratch plus locks.

## Build and tests

    cd plugin-exl3 && VLLM_EXL3_BUILD=1 python setup.py build_ext --inplace   # both .so; VLLM_EXL3_MR_BUILD=0 skips _C_exl3_mr

Box, `/workspace/wt-exl3-opt`, phase 1's CUDA 13.0 toolchain, `-j8`, nice 19: 192 s for both,
`_C_exl3_mr` 9.2 MB with all 60 kernel instances linked. Local under `tools/capped`: 601 s.
Both VERIFIED. The `.so` loads with `CUDA_VISIBLE_DEVICES=""`, the four ops register, the K4
repack round-trips, and CUDA is never initialized. CPU (`GSQ_LIGHT=1 GSQ_VENV=.venv-main
tools/capped tools/pytest tests/cpu -k exl3`): 263 passed (96 new, plus phase 1's 167). The
phase-1 guard test's `.so` glob was narrowed to `_C_exl3.*.so`.

## GPU jobs (not run)

    gpuq submit exl3opt-10 -- bash /workspace/wt-exl3-opt/cloud/results/exl3-opt/box-scripts/run-job.sh 10-mr-parity

Logs go to `/workspace/logs/exl3-opt/`, runs to `/workspace/runs/exl3-opt/`, and results to
`cloud/results/exl3-opt/`. The jobs use a copy of phase 1's autotune cache, their own KV tier,
and max-model-len 196608 as 06-ladder. They need phase 1's checkpoint and draft head (02). Jobs
12 and 13 are skipped unless 10 passed.

| job | what | est. |
|---|---|---|
| 10-mr-parity | decoded weights == exl3_dequant for every row and column (K3/K5/K4, lm_head included); op vs fp64 at 17/24/32/48/64/96/144 rows (+1/8/16 for K4) inside exl3_gemm's error x1.5, bf16 and fp16 outputs, dequant+fp16 GEMM printed alongside; routing (K2 bit-identical to phase 1); determinism; graph replay; unwarmed capture refused; memcheck + initcheck on 41 cases | 30-45 min |
| 11-mr-micro | each of the 21 (k, n, K) shapes at rows 1..512: exl3_gemm, exl3_gemm_mr, dequant route, in µs, GB/s and % of the DRAM floor; model sum per target pass and per draft step for MR 0/1/2, best-of and floor | 15-25 min |
| 12-mr-ladder | production's decode cohorts c=1/2/4/8, pass 2 T=0, ms/step for EXL3_MR=0, 1, 2 (+8k prefill) | ~60 min |
| 13-profile | torch profiler, 6 decode steps at c=1 and c=4 for MR 0 and 2: launches and ms per class (gemm, Hadamard, casts/cat, dequant, attention, GDN), idle, host gaps, load+warmup time | ~30 min |

## Levers, ranked (INFERRED until 11-13 run)

At production's schedule a verify pass carries 6 rows at c=1, 12 at c=2, 24 at c=4, 32 at c=8
and 48 at c=16 (k = 5/3/2 by batch size). One weight pass is about 15 ms at 80 % of DRAM.

| # | lever | expected | measure |
|---|---|---|---|
| 1 | multi-row kernel with the K4 repack (EXL3_MR=2) | c=4 and c=8: exl3_gemm pays 2 weight passes, mr pays 1. About -12 to -15 ms/step (EXL3_MR=1: -5 to -6). c=16: -25. c=1/2: ±1-2 (exl3_gemm's K4 GEMV vs mr at 6/12 rows) | 10, then 11 (% floor at 24/32/48), then 12 |
| 2 | dequant threshold for the mr route (144 to the crossover, likely 300-600) | production's 128-token prefill chunks plus decode rows give 129-224-row steps. Those take the dequant route, which costs about as much as 9 exllamav3 passes, against 3-4 for mr: about -50 to -80 ms per mixed step (prefill under load, decode stalls). Steady decode: 0 | 11 (rows 192-512), then a prefill-under-decode run |
| 3 | glue: bf16 output into the fused output, one input Hadamard per fused group with the bf16 cast inside it (trellis-serve's `multi_out`, bf16 x), fewer cat/cast kernels | about 1,900 small launches per verify pass down to about 650: -2 to -4 ms/step at every c, and the only lever at c=1 | 13 (launches, cast/cat/Hadamard ms, idle) |
| 4 | routing below 17 rows, per shape | ±1-2 ms at c=1/2 | 11's per-shape best |
| 5 | draft head | bf16 0.42 GB read on each of up to 5 draft steps is about 2.2 ms/step of floor at c=1. An fp8/int8 head, or EXL3 on the gathered rows: about -1 ms | 13 (draft-step kernels) |
| 6 | Hadamard in the GEMM prologue (`set_in_had_inlaunch`, cooperative) | -0.5 ms; experimental upstream, graph capture untested | after 3 |

Order: decide 1 (and 2) on 11 and 12's data, then 3, which is owned glue around the vendored
calls with no kernel edits. Upstreamability holds as long as vendored files stay unmodified.

## Risks

- Fit: 196608 tokens had 0.01 GiB to spare at 200k. MR adds 21.5 MB of scratch, and the load
  briefly holds two copies of the largest K4 tensor (lm_head, 0.64 GB) before vLLM profiles.
- lm_head at n=248320 on the Marlin kernel: upstream keeps n > 65536 on exllamav3 by default (for
  transients, not correctness). Job 10 covers it.
- 128-token prefill chunks at 129..144 rows: the mr route takes them eagerly. Repacked K4 above
  144 rows pays one unpack copy per call (about 1 % of a 2048-row chunk).
