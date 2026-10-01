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

| mode | c=1 | c=2 | c=4 | c=8 | acc. rate |
|---|---|---|---|---|---|
| EXL3_MR=0 (06-ladder) | 74.7 tok/s, 39.9 ms | 136.6, 43.0 | 171.2, 69.9 | 308.6, 72.1 | 0.406 |
| 2 (K3/K5 from 17) | 77.0, 37.0 | 145.1, 40.4 | 226.5, 51.9 | 411.5, 52.7 | 0.394 |
| 2a (K3/K5 from 1) | 85.7, 33.7 | 155.8, 38.9 | 226.0, 52.0 | 409.2, 52.4 | 0.390 |
| 2ga (+ glue) | 89.1, 33.1 | 148.5, 38.4 | 216.2, 51.3 | 397.4, 51.9 | |
| **2gah (+ host embedding) = defaults** | **90.4, 33.2** | **151.9, 38.5** | **234.5, 51.3** | **408.4, 51.7** | |
| GGUF Integration-2 | 27.9 | 31.6 | 35.7 | 44.2 | |
| prod W4A16 | 27.6 | 27.3 | 30.0 | 41.3 | |

Defaults vs EXL3_MR=0: -6.7 / -4.6 / -18.5 / -20.4 ms/step. Still 5.6-21 ms/step behind
production and GGUF. Acceptance moved within noise (the whole-run rate includes sampled
cohorts). 8k prefill unchanged (1070 vs 1078 tok/s: 2048-row chunks stay on the dequant route).

**Fit**: at 196,608 (production argv as 06): EXL3_MR=0 199,716 KV tokens, defaults without the host
embedding 197,385-198,162 (MR scratch and warmup), with it 264,993. **200,000 (job 14, defaults):
GPU KV 265,369 tokens (1.33x), a real ~195k-token request completes (2 passed), VRAM 23.9 of
24.6 GB after it**: the host embedding restores production's 200k context with 65k tokens to spare.

**13-profile**: torch profiler, 5 complete steps each (`cloud/results/exl3-opt/13-profile/`):

| per step | MR=0 c=1 | defaults c=1 | MR=0 c=4 | defaults c=4 |
|---|---|---|---|---|
| span / busy / GPU idle (ms) | 43.3 / 32.6 / 10.6 | 39.4 / 26.9 / 12.6 | 74.2 / 63.7 / 10.5 | 55.5 / 44.8 / 10.7 |
| EXL3 GEMM (ms, launches) | exl3_gemm 25.7, 441 | mr 18.3, 437 | exl3_gemm 54.1, 441 | mr 32.7, 437 |
| mr input Hadamard | | 0.80, 479 | | 1.35, 610 |
| casts/copies | 0.43, 183 | 1.25, 474 | 0.54, 183 | 1.37, 474 |
| dense GEMM (draft head, unquantized) | 2.9 | 2.9 | 2.9 | 2.8 |
| GDN delta-rule kernel | 0.72 | 0.72 | 2.78 | 2.79 |

Top costs at the defaults: (1) the EXL3 GEMMs, 18.3 ms at c=1 and 32.7 at c=4 (34 % of floor
at 24 rows, job 11); (2) GPU idle 10.7-12.6 ms/step, the largest part (6.2 ms at c=1) not
under any op (Python between graph replays in the MTP loop), then GDN core 0.9, index 0.7,
pre-graph bytecode 0.6, pin_memory 0.5, embed gather 0.2; (3) per-part glue on fused layers:
320 fp32 widen copies (0.9-1.0 ms) and 479-610 input Hadamards (0.8-1.35 ms); (4) the bf16
draft head 2.9 ms; (5) GDN's delta-rule kernel 2.8 ms at c=4. Load: init engine 21 s with a
warm compile cache.

**Kept / reverted**: kept (defaults, all parity-gated by job 10 and measured by job 12): EXL3_MR=2 with the
in-place K4 repack; K3/K5 on mr from 1 row; mr range to 384 rows (job 11); glue on single-part
layers; host embedding. Reverted or narrowed: the allocating repack (OOM), glue on fused layers
(inductor stride bug), EXL3_MR=1 (dominated by 2: -18 ms/step at c=4 needs K4).

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
| `.../csrc/exl3_mr_shim.cu` | ops `exl3_gemm_mr`, `exl3_mr_repack`, `exl3_mr_unpack`, `exl3_mr_warmup`, `exl3_embed_host` in the `_C_exl3` namespace |
| `plugin-exl3/setup.py` | second extension `_C_exl3_mr` (15 generated instance units: row families 0..4 x mul1 x K 3/4/5); `_C_exl3` is unchanged |
| `.../ops.py`, `.../quantization/linear.py` | `EXL3_MR` routing, the in-place K4 repack at load, the mr warmup, glue |
| `.../quantization/embedding.py` | the host-pinned token embedding (`EXL3_EMBED_HOST`) |
| `tests/cpu/test_exl3_mr.py` | 121 CPU tests |
| `tests/gpu/test_exl3_mr.py`, `bench/micro/exl3_mr.py`, `bench/micro/exl3_mr_cfg.py` | GPU parity (294 cases), the per-shape microbenchmark, the launch-config probe |
| `cloud/results/exl3-opt/box-scripts/` | jobs 10-15, `run-job.sh`, `lib.sh` |

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
layout), done in place on the tensor's storage. K4 then runs `exl3_gemm_mr` at every row count
up to 384, and above that the dequant routes unpack per call. Bytes per target pass (erlidev, VERIFIED from the headers): K3 38.6 %,
K4 54.0 % + lm_head 5.6 %, K5 0.9 %, K2 0.8 %, out of 11.28 GB (12.05 ms at 936 GB/s). EXL3_MR=1
alone therefore reaches 40 % of the bytes, and EXL3_MR=2 reaches 99 %.

## Switches (read once when the plugin imports)

| variable | default | what |
|---|---|---|
| `EXL3_MR` | 2 | 0: phase 1's routing; 1: K3/K5 (mul1) on exl3_gemm_mr; 2: and K4 (lm_head, MTP included), repacked in place at load |
| `EXL3_MR_MIN` | 1 | first row count K3/K5 take exl3_gemm_mr at (K4 under 2: always from 1) |
| (constant) `MULTI_ROW_MAX` | 384 | last row count on exl3_gemm_mr; above, the dequant routes (K2 keeps exllamav3's 144) |
| `EXL3_MR_GLUE` | 1 | with 2: bf16 straight through exl3_gemm_mr on single-part layers (same bits) |
| `EXL3_EMBED_HOST` | 1 | bf16 token embedding in pinned host memory, gathered per step by a UVA kernel |

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
tools/capped tools/pytest tests/cpu -k exl3`): 288 passed (121 new, plus phase 1's 167, which
pin EXL3_MR=0 with a fixture since their tables are phase 1's). The phase-1 guard test's `.so`
glob was narrowed to `_C_exl3.*.so`.

## GPU jobs (run 2026-10-01, results above)

    gpuq submit exl3opt-10 -- bash /workspace/wt-exl3-opt/cloud/results/exl3-opt/box-scripts/run-job.sh 10-mr-parity
    gpuq submit exl3opt-12 -- bash .../run-job.sh 12-mr-ladder 2gah 2ga   # modes as arguments (lib.sh serve_mr)

Logs go to `/workspace/logs/exl3-opt/`, runs to `/workspace/runs/exl3-opt/`, and results to
`cloud/results/exl3-opt/`. The jobs use a copy of phase 1's autotune cache, their own KV tier,
max-model-len 196608 as 06-ladder (14-fit: 200,000), and one `VLLM_CACHE_ROOT` per graph
variant. They need phase 1's checkpoint and draft head (02). Jobs 12-14 are skipped unless 10
passed. Measured run times: 10 about 2 min, 11 about 1 min, 12 about 25 min per mode, 13 about 7
min per mode, 14 about 10 min.

| job | what | est. |
|---|---|---|
| 10-mr-parity | decoded weights == exl3_dequant for every row and column (K3/K5/K4, lm_head included); op vs fp64 at 17/24/32/48/64/96/144 rows (+1/8/16 for K4) inside exl3_gemm's error x1.5, bf16 and fp16 outputs, dequant+fp16 GEMM printed alongside; routing (K2 bit-identical to phase 1); determinism; graph replay; unwarmed capture refused; memcheck + initcheck on 41 cases | 30-45 min |
| 11-mr-micro | each of the 21 (k, n, K) shapes at rows 1..512: exl3_gemm, exl3_gemm_mr, dequant route, in µs, GB/s and % of the DRAM floor; model sum per target pass and per draft step for MR 0/1/2, best-of and floor | 15-25 min |
| 12-mr-ladder | production's decode cohorts c=1/2/4/8, pass 2 T=0, ms/step per mode (+8k prefill) | ~25 min/mode |
| 13-profile | torch profiler, 6 decode steps at c=1 and c=4 per mode: launches and ms per class (gemm, Hadamard, casts/cat, dequant, attention, GDN), idle, host gaps, load+warmup time | ~7 min/mode |
| 14-fit | phase 1's 07-fit at 200,000 tokens with a mode (default 2gah) | ~10 min |
| 15-mr-cfg | exl3_gemm_mr launch-config probe at 16..48 rows | ~5 min |

## Next levers, ranked (from jobs 11, 13, 15)

| # | lever | expected | evidence / measure |
|---|---|---|---|
| 1 | mr launch config at 17-48 rows (the verify rows at c=4/8/16): the target pass costs 23.3 ms at 16 rows and 35.3 at 17 (thread_m_blocks 2, 256-thread configs); a per-row-count config pick, or splitting 17-32-row calls into two 16-row launches | up to -10 ms/step at c=4/8 | job 15 (CFG_NOTE), then 12 at c=4/8 |
| 2 | host idle in the MTP loop: 10.7-12.6 ms/step of GPU idle, 6.2 ms not under any op (Python between graph replays), index / pin_memory / pre-graph bytecode 1.8 ms | -3 to -6 ms/step at every c; shared with every vLLM model, check prod's own idle first | 13 (gaps.py) on prod W4A16 for comparison |
| 3 | fused-layer glue: one input Hadamard and one GEMM launch per fused layer for equal-K parts (`exl3_linear_marlin_multi_out`), bf16 written into the fused output's column spans (`had_out_into`): removes the 320 fp32 widen copies and ~140 Hadamard launches | -1.5 to -2 ms/step | 13 (cast/copy, had mr-in) |
| 4 | the remaining mr floor gap at 1-16 rows (64 % of floor at 6 rows) and K4/K5 shapes at 13-17 % (k_proj 5120x1024) | -2 to -4 ms/step at c=1/2 | 11 per shape |
| 5 | draft head 2.9 ms/step (bf16, 40,960 x 5120, read on each draft step) | int8/fp8 head: about -1.4 ms | 13 |
| 6 | prefill: 2048-row chunks run the dequant route at ~1070 tok/s (8k) | owned dense path / f16acc tuning | 06/12 prefill rows |

## Risks

- Compile cache: vLLM's torch.compile cache key misses the plugin's graph variants (EXL3_MR,
  EXL3_MR_GLUE, EXL3_EMBED_HOST change the traced graph or its parameter layouts). The jobs and
  `scripts/serve-exl3.sh` use one `VLLM_CACHE_ROOT` per variant; any other launcher must too.
- lm_head at n=248320 on the Marlin kernel: upstream keeps n > 65536 on exllamav3 by default (for
  transients, not correctness). Job 10 covers it (decode exact, error inside exl3_gemm's).
- Repacked K4 above 384 rows pays one unpack copy per call (about 1 % of a 2048-row chunk).
- The host embedding puts 2.4 GB in pinned host memory per server (and briefly the MTP draft's
  own copy at load).
