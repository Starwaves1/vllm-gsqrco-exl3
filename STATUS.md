# Status: Swift GSQ-RCO IQ3_S-mtp GGUF on production vLLM

Latest: Integration 1 (opt-p + K1 + K2 on main): decode 101.1 / 175.0 / 264.4 / 436.7 tok/s c=1/2/4/8 greedy (W4A16 baseline 94.1 / 194.4 / 345.1 / 505.4), prefill 1152 / 900 / 622 tok/s at 8k / 64k / 180k (baseline 1108 / 868 / 603). See "Integration 1" below.

## Current state (2026-09-29)

Route L (llama.cpp b11211 MMVQ/MMQ behind `lcpp_shim.cu`, `VLLM_GGUF_LCPP=1`) is on main with
phase 3 and Integration 1 (opt-p, K1, K2): three owned 1..8-row kernels, one q8_1 quantize per
fused layer's q8_1-reading runs, a 61,440-row MTP draft head. Measured on a rented 350 W 3090:
c=1 decode 1.07x the production W4A16 baseline, c=2..8 0.77-0.90x, prefill 1.03-1.04x (see
"Integration 1"). In flight on the box: K3 (`opt-k3`, 16-64-row mma for Q4_K/IQ2_S/IQ4_XS) and
R1 (`opt-r1`, IQ3 load-time repack); they merge later. Production is untouched.

The remaining sections are dated history ("Integration 1", then Phase 3, is the newest). Labels: VERIFIED (checked in source or by running it here), DOCUMENTED (read in docs), INFERRED (reasoned, not checked).

2026-09-27 handoff: deliverable 5 (kernel porting) was paused pending a prior-art survey; the survey chose Route L and kernel work resumed in phase 2/3.

## Phase 1 on a rented RTX 3090 (2026-09-28, Vast.ai, cut short)

Box: RTX 3090 24 GB, driver 580.142, **power-capped at 180 W** (default 370 W). Under load
the SM clock sits at 480-615 MHz (decode 480, prefill 540-615; memory 9501 MHz), so every
absolute speed number below is heavily depressed. Raw data: `cloud/results/phase1/`.

- Env: venv equals `env/gsq-freeze.txt`, vLLM overlay ba05ffab verifies, plugin built with
  the cu130 pip toolchain, GGUF sha256 and hf-config verify pass.
- First serve with MTP OOMed in the draft load (fixed in e751e64): the plugin staged each
  unsharded weight on the GPU, and inside vLLM's cumem "weights" pool the freed segments
  could not hold the draft's 2 x 2.37 GiB bf16 embed/lm_head placeholders. Headroom after
  the fix is about 1 GiB at the draft-load peak (22.54 of 23.56 GiB reserved); about
  5.3 GiB stays reserved-free from the sharded path (`_create_padded_weight_param`), so a
  larger GGUF would OOM again. The placeholders cannot simply be dropped: vLLM probes
  `draft.embed_input_ids` before sharing the target's embedding.
- Load: 169-170 s, "Model loading took 12.29 GiB"; 22.4 GB VRAM used after startup.
- Fit (gpu-util 0.94, fp8 KV, MTP k=3): 246,093 KV tokens, 1.23x at 200k (282,031 / 1.41x
  without MTP). A 195k-token request completes (27.5 min) with no device fault.
- Smoke: coherent chat, reasoning split, qwen3_coder tool call parsed.
- Kernel parity at e2b8ad5: 276 pass / 12 skip after calibration (0d8fd9e). Max errors per
  type: `cloud/results/phase1/kernel-parity-errors.txt`. No kernel bug found; Q4_K MMQ is
  7e-2 from full precision but 2.5e-3 from the xsum model (ggml's MMQ min term).
- Guards at e2b8ad5 (expected): x_noncontig silently wrong (rel err ~1.5), w_narrow_view
  NaN, w_misaligned device fault ("misaligned address"), row_too_big and k_mismatch
  accepted silently; x_misaligned and graph_replay pass (all 5 type/op combos each).
- Speed as-is (180 W, SM ~500 MHz): decode 10.7 tok/s c=1, 11.9 c=2 (real prompts, 8 x
  1024, T default), MTP mean acceptance length 2.99; prefill 133 tok/s at 8k c=1, 175 at
  8k c=2. 64k/180k and the W4A16 baseline were not measured (box abandoned for the power
  cap).
- Box-only gotcha: a stopped vLLM leaves its CPU-tier mmap (`/dev/shm/vllm_offload_*.mmap`)
  behind; with a 15 GB /dev/shm the next start fails with EFAULT in
  `shared_offload_region.py`. Clear it between runs when no vLLM is running.

## Integration 1: opt-p + K1 + K2 merged (2026-09-29, same 350 W 3090)

Branch `integrate` = main + `opt-p` + `opt-k1` + `opt-k2` (merged in that order), fast-forwarded
into main. Results, box scripts and profiles: `cloud/results/integration-1/`. One server per
gpuq job; decode = pass 2 of production's `run_benchmarks.sh single`, T=0, decode(C/meanTPOT);
prefill = the salted ladder at c=1 (`bench/speed/run.sh gsq`, unmodified, clocks logged).

| | c=1 | c=2 | c=4 | c=8 | 8k | 64k | 180k |
|---|---|---|---|---|---|---|---|
| Integration 1 | 101.1 | 175.0 | 264.4 | 436.7 | 1152 | 900 | 622 |
| prod W4A16 (phase 1b, pass 2) | 94.1 | 194.4 | 345.1 | 505.4 | 1108 | 868 | 603 |
| ratio | 1.07 | 0.90 | 0.77 | 0.86 | 1.04 | 1.04 | 1.03 |
| ms/step Integration 1 / W4A16 | 30.5 / 27.6 | 34.9 / 27.3 | 46.7 / 30.0 | 55.7 / 41.3 | | | |
| pre-campaign, f6b96bf (opt-p base run; no prefill ladder) | 88.1 | 154.3 | 258.4 | 432.9 | - | - | - |
| ratio | 1.15 | 1.13 | 1.02 | 1.01 | - | - | - |
| ms/step Integration 1 / f6b96bf | 30.5 / 33.7 | 34.9 / 38.5 | 46.7 / 47.2 | 55.7 / 56.4 | | | |
| phase 2 (prefill only) | | | | | 1036 | - | 589 |
| ratio vs phase 2 | | | | | 1.11 | - | 1.06 |

- ms/step = C x 1000 / tok/s x tok/step. tok/step 3.08 / 3.05 / 3.09 / 3.04 here vs the
  W4A16 baseline's 2.60 / 2.65 / 2.59 / 2.61 (its own MTP head; 2.57 is phase 1b's c=1 whole-run
  mean). **The c=1 1.07x is throughput bought with higher MTP acceptance: per engine step this
  build is slower than the W4A16 baseline at every concurrency** (30.5 vs 27.6 ms at c=1, 1.10x;
  1.28x / 1.56x / 1.35x at c=2/4/8).
- MTP acceptance 0.657 over the run (pre-campaign 0.630 with the 40,960-row draft head; opt-p's
  61,440 rows). Decode clocks: median SM 1740-1755 MHz, 344 W, 89% util (both passes).
- Noise: the same code measured 154.2 and 160.4 tok/s at c=2 in two sessions (phase 3 item 5
  vs K2's same-session A/B), so c=2 moves under ~4% are not signal; opt-p saw ~1% at c=1. The
  campaign's gain is at c=1/c=2 (4 / 8 rows per target pass); the c=4/c=8 moves (+2% / +1%,
  target passes on MMQ) are within noise.
- No prefill ladder ran at f6b96bf: the prefill gain vs phase 2 is attributed to phase 3 items 4
  and 4b (INFERRED: none of these branches changes the >= 9-row MMQ path).
- Logit parity vs the phase-1b llama.cpp CUDA dumps (`parity/table.txt`): KLD 0.0237 / top-1
  98.18% overall (phase 2 Route L 0.0284 / 97.90%, stock 0.0404 / 98.02%); >= 100k 0.0055 /
  98.44% (phase 2 0.0045 / 98.09%). Absolute gate still FAIL; relative gate (<= llama.cpp's
  CUDA-vs-CPU floor) PASS on seq_000-005. Worse than phase 2 on two prompts: seq_009 KLD 0.00191
  vs 0.00167 and seq_010 0.00902 vs 0.00740 (max 0.487 vs 0.155), with top-1 up on both; that is
  inside the spread between stock and Route L on other prompts (seq_002 +46%), but no repeat run
  exists to bound single-prompt noise. MTP acceptance vs llama.cpp (8 real prompts, k=3): greedy
  0.664 vs 0.673 (-0.95 pt, within the 2-pt gate); T=1.0 0.656 vs 0.628 (+2.9, FAIL as in phase 1b).
- Merge resolutions (all in `lcpp_shim.cu`, `linear.py`, `setup.py`, the tests):
  - Routing is one function, `linear._lcpp_op(n, type, weight rows)`; the table is in
    ROUTE-L.md. IQ3 1..5 rows dp4a, 6..8 mma; Q4_K from 3 rows and IQ2_S from 1, both only above
    2048 weight rows, up to 8; everything else MMVQ below 8 rows, MMQ from 8 (IQ1_M stock).
  - opt-p's shared quantize (`_quantize_x_q8_1`) now takes the runs' row counts and asks
    `_lcpp_op`: with K1, Q4_K/IQ2_S read q8_1 at 8 rows, where opt-p's type-only predicate said
    MMQ and would have left `x_q8` unfilled for them (new test case).
  - `lcpp_mul_mat_vec_own` and `lcpp_mul_mat_vec_iq3_mma` take opt-p's optional `x_q8`. They
    keep fp32 dst + the shim's output cast (their kernels write `float*`); only the dp4a IQ3
    kernel writes 16-bit directly.
  - Tests: the owned-kernel parity and graph tests run over (type, op) for all three owned ops;
    K2's extra shapes (odd_rows, k_min, few_rows, many_tiles) now cover `lcpp_mul_mat_vec_own`
    too, except fp32 odd_rows (its only fp32 reference, MMVQ, reads past that W).
- Tests on the box (this build; logs in `tests/`): kernel parity 1970 pass / 128 skip / 0 fail with
  `VLLM_GGUF_LCPP=1` (1594 on opt-k2 alone, 1185 on opt-p); 1939 / 159 / 0 with it off, where
  the 31 extra skips need the flag. Stock path: main's 328 non-Route-L parity tests collect
  unchanged here (+6 new: opt-p's gemv and shared-quantize tests) and pass on both builds (main:
  1137 pass / 31 skip, flag off). GPU guards `-k "lcpp or first_call"` 132 pass / 10 skip (x_q8
  cases on MMQ ops) / 0 fail; CPU guards 73 + routing table 54 pass. compute-sanitizer memcheck
  and initcheck 0 errors on 57 cases: 16 per owned op (IQ3_S/IQ3_XXS or Q4_K/IQ2_S, n = 1..8,
  all tail shapes), x_q8 and shared-quantize tests. CPU suite locally 471 pass / 73 skip / 54
  xfail. Vendored files unchanged (VENDORED.md sha256, 39 files).
- Profile (torch profiler, 5 complete decode steps, `runs/profile-c1-c4.txt`):
  - c=1 (4 rows per target pass): 2005 GPU activities per step, busy 25.6 ms, span 34.0 ms
    (8.4 ms idle). GEMM 21.3 ms: dp4a IQ3 10.3 ms (192 launches), MMVQ 8.3 ms (142; IQ4_XS 4.1,
    Q6_K 1.3, Q2_K 0.8, IQ2_XS 0.7, Q4_K 0.7, IQ2_XXS 0.4), owned Q4_K/IQ2_S 2.8 ms
    (41; Q4_K 2.0, IQ2_S 0.8). q8_1 quantize 276 launches 0.40 ms, bf16 casts 182 launches 0.28 ms,
    GDN in_proj_ba gemv 48 x 6 us.
  - c=4 (16 rows per target pass): 3030 activities, busy 42.1 ms, span 49.5 ms (7.4 idle), as in
    opt-p's c=4 profile. Target passes are MMQ for every type: 686 MMQ launches 33.0 ms (IQ3_S
    12.2, IQ3_XXS 7.0, IQ4_XS 5.4, Q4_K 2.9, IQ2_S 1.5, IQ2_XS 1.0, Q2_K 0.8), MMQ quantize 361
    launches 0.54 ms, 374 output casts 0.62 ms, 360 memsets 0.41 ms. The 4-row draft passes are
    not MMQ: MMVQ Q6_K 1.0 ms (10 launches) and the Q4_K kernel 0.66 ms (3 launches, one per
    draft step: the draft head, rows of output.weight, which is Q4_K).
  - Idle is inflated by the profiler: phase 3 item 2 measured ~4.7 ms/step unprofiled.
  - Round 2: c >= 4 is K3's 16-64-row range (MMQ is 78% of busy time at c=4). At c=1 the
    largest non-owned GEMM is IQ4_XS on MMVQ (4.1 ms; K1 dropped IQ4_XS).
- Open, left for round 2 (review and test audit, low severity):
  - 16-bit output from the mma and Q4_K/IQ2_S kernels (a `dst_t` template as in
    `iq3_mul_mat_vec_y`) would drop one cast launch per product: ~41 per c=1 step, ~230 at c=2
    (casts cost ~1.5 us each: ~0.2% / ~1% of the step). Not done here: it edits kernels K3 and R1
    are still changing and would invalidate this ladder.
  - `_fused_mul_mat_gguf` looks the op up by name per call (`getattr(torch.ops._C_gguf, ...)`);
    the branches did the same attribute access, so no regression.

## Phase 3: closing Route L's decode gap (2026-09-28, same 350 W 3090)

Full numbers and method: `cloud/results/phase3/summary.txt`. Decode = pass 2 of production's
`run_benchmarks.sh single`, T=0 cohorts; ms/step = C x 1000 / tok/s x tok/step.

| item | change | c=1 tok/s | c=2 tok/s | ms/step c=1 / c=2 | kept |
|---|---|---|---|---|---|
| phase 2 | Route L as merged | 77.8 | 117.0 | 40.1 / 53.0 | |
| 1 | lcpp MMQ from 8 rows (was >8) | 77.8 (a) | 135.5 | 40.1 / 45.9 | yes |
| 1b | lm_head via MMQ from 4 rows | 76.8 | = | 40.2 / - | no: 1119 vs 1077 us in situ |
| 2 | host gaps: diagnosis only | | | | |
| 3 | draft lm_head row-pruned to 40,960 (production's list and mechanism); no bf16 lm_head placeholder | 79.4 | 139.5 | 37.8 / 43.2 | yes |
| 4 | owned X -> q8_1 quantizer (no input cast) | 81.2 | 142.9 | 36.9 / 42.1 | yes |
| 4b | one GEMM per same-type shard run (433 -> 356 per pass); dequant output not zeroed | 83.5 (b) | 146.7 (b) | 36.6 / 41.6 | yes |
| 5 | owned IQ3_S/IQ3_XXS decode-once kernel (`lcpp_mul_mat_vec_iq3`), routed at 1..8 rows | 90.9 | 154.2 | 33.1 / 37.9 | yes |
| K2 | owned IQ3 int8 tensor-core kernel (`lcpp_mul_mat_vec_iq3_mma`), routed at 6..8 rows (c) | 90.3 | 173.2 | 33.0 / 35.6 | yes |

(c) same-session A/B against this build with item-5 routing: 90.6 / 160.4 tok/s, 33.0 / 37.9 ms/step
(cloud/results/phase3/k2). c=1 (4 rows) is unchanged by design.
(a) item 1 at c=1 is phase 2's path (c=1 never reaches 8 rows). (b) mostly tok/step: acceptance
moved 0.633 -> 0.651 with MMQ numerics; ms/step fell only 0.8% / 1.2%.

- Tests: kernel parity 776 -> 992 -> 1002 pass (new: q8 bytes vs the vendored quantizer,
  bit-identical; same-type run test); item 3 is loader-only (CPU tests + serve smoke). GPU guards
  64/64, CPU guards 42, memcheck/initcheck 0 errors on 18 targeted cases after item 4 and on the
  dequant tests after 4b. Vendored files unchanged (VENDORED.md sha256 pass).
- Host gaps (item 2): vLLM runs both Route L and the W4A16 baseline PIECEWISE (FlashInfer has no
  FULL cudagraph support under spec decode); every Route L GEMM except the eager lm_head calls is
  graph-captured, draft passes included; both models idle the GPU ~4.7 ms/step without the
  profiler. Not the plugin. The rest of the gap is GPU time.
- Item 3 costs acceptance: 0.688 -> 0.630 (mean length 3.06 -> 2.89), since tokens outside the
  40,960 are never drafted. Net positive. VRAM: load peak -2.37 GiB (23,427 -> 21,001 MiB); the
  pruned head adds 0.11 GiB of weights, so KV 250,000 -> 246,875 tokens (1.23x at 200k). Off
  switch: MTP_DRAFT_VOCAB=0 (the head follows the ids file, not VLLM_GGUF_LCPP).
- Item 4 keeps the output cast: MMVQ/MMQ write fp32 only. W rows must now be contiguous (MMVQ
  goes through upstream's q8_1 entry `ggml_cuda_op_mul_mat_vec_q`).
- Item 5: owned IQ3_S/IQ3_XXS decode-once kernel, `iq3_mul_mat_vec` (`lcpp_shim.cu`), routed at
  1..8 activation rows (`linear.py`), MMVQ/MMQ unchanged elsewhere. Design (`cloud/results/phase3/item5/iterations.txt`
  has the full iteration log):
  - Reimplements the vendored `vec_dot_iq3_*_q8_1` decode arithmetic bit for bit (grid lookup,
    sign unpack, scale) but restructures the CTA: 4 warps x 4 weight rows/warp (16 rows/CTA),
    q8_1 chunks staged in shared memory, decode grid read from smem, each warp reads its
    activation slice once for all 4 of its rows instead of once per row.
  - Sign step ported from ninfer-all's byte-negation (4 ops/word) in place of the vendored
    `__vcmpne4`+`__vsub4` pair; identical int8 outputs, fewer instructions.
  - The vendored MMVQ already CSEs the per-column decode across activation rows (SASS
    check, ROUTE-L.md); the actual levers are per-column q8_1 loads, their reuse over only
    2 weight rows/warp in MMVQ, and instruction count, not "decode once" by itself.
  - Iteration swept CTA shape (8x2 warps x rows down to 4x4), staged vs unstaged q8_1, and two
    overlap schemes (register-prefetch, cp.async double buffering); both overlap attempts were
    slower and dropped. Not tried: int8 tensor-core (mma) fragments — a much larger kernel;
    MMQ already uses them and loses to tile overhead at n <= 8. (K2 did it later:
    `lcpp_mul_mat_vec_iq3_mma`, routed at 6..8 rows.)
  - Goal (>=600 GB/s op-level at 4 rows) not reached: 541/520 GB/s (IQ3_S/IQ3_XXS).
  - Tests: kernel parity 1002 -> 1152 pass / 16 skip / 0 fail (`VLLM_GGUF_LCPP=1`; new:
    `test_lcpp_iq3` correctness across n=1..8, 3 dtypes, real/row_tail/k_tail shapes, and
    `test_lcpp_iq3_graph_replay`; log `item5/review/parity.log`). The first full run
    (`item5/parity.log`) had 6 failures, all `test_lcpp_same_type_run` at
    `assert torch.equal(y, whole)`, n = 1, 4, 8: a test bug, not a kernel bug. The test built
    its "whole run" and "per shard" references by calling `lcpp_mul_mat_vec_q` /
    `lcpp_mul_mat_q` directly, while `apply()` now sends IQ3 shards at 1..8 rows to the new
    kernel (~1e-7 rel from MMVQ, ~35% bit-equal, `iq3-vs-mmvq.txt`). Fixed by building both
    references through `_fused_mul_mat_gguf` (production's dispatch); bit-exact at n <= 8
    (MMVQ / IQ3 kernel, rows independent), 1e-3 relative where MMQ's stream-k reorders the sum.
    GPU guards `-k lcpp` 80 pass (`item5/guards.log`, pre-alignment build), CPU guards 52 pass. memcheck/initcheck
    0 errors on 12 `test_lcpp_iq3` cases (`item5/review/`; graph replay excluded, capture is
    unsupported under compute-sanitizer). The first memcheck (`item5/sanitizer-memcheck.log`)
    had 26 errors, all in vendored MMVQ (the test's reference) reading past a 203-row W; the
    row_tail case now uses 202 rows. Vendored files unchanged (VENDORED.md sha256 pass).
  - Microbench (`cloud/results/phase3/item5/micro-final.tsv`; op time incl. q8_1 quantize +
    output cast, CUDA graph, us at 17408 x 5120):

    | n | IQ3_S MMVQ | IQ3_S MMQ | IQ3_S iq3 | IQ3_XXS MMVQ | IQ3_XXS MMQ | IQ3_XXS iq3 |
    |---|---|---|---|---|---|---|
    | 1 | 67.8 | 122.4 | 55.0 | 62.9 | 108.6 | 53.2 |
    | 4 | 103.0 | 123.6 | 70.7 | 94.3 | 109.9 | 71.0 |
    | 8 | 147.1 | 124.5 | 100.5 | 133.9 | 110.2 | 100.6 |

  - Decode (350 W, real prompts, T=0): 83.5 / 146.7 -> 90.9 / 154.2 tok/s c=1 / c=2 (baseline
    94.1 / 194.4); ms/step 36.6 / 41.6 -> 33.1 / 37.9. c=1 MTP acceptance 0.633 (unchanged: the
    IQ3 kernel doesn't touch numerics enough to move it outside noise at c=1).
  - Tried and dropped (see iterations.txt for the numbers): exact int->float via a magic-number
    add instead of I2F (no change); cp.async double-buffered weight + q8_1 tiles in smem and a
    next-chunk register-prefetch scheme (both slower than staged-in-smem-no-overlap); q8_1 read
    straight from global with no staging (within noise, staging kept for a small n=8 win).
- K2 (opt-k2): `lcpp_mul_mat_vec_iq3_mma` (`csrc/lcpp_owned_iq3_mma.cu`): one mma.sync m16n8k32
  s8 per 16 rows x 32 values x 8 activation columns, decoded straight into fragments from a
  per-CTA signed grid table (grid x 16 sign nibbles), weight bytes staged through shared memory
  with coalesced loads, persistent CTAs of 4 warps splitting K. Flat in n: 78.1 / 83.6 us at
  n=4 / 8 on IQ3_S 17408x5120 (dp4a kernel 71.9 / 99.1); wins from 6 rows at every tested
  shape, so it is routed at 6..8 and the dp4a kernel keeps 1..5. Targets (n4 <= 58, n8 <= 75 us)
  not met: the kernel is SM-bound (~65 us with weights cache-resident), not DRAM-bound. Numerics:
  exact int32 per slice and the vendored integer sub-scale; d_w applied once per weight block
  (fp32 order differs from MMVQ, within 1e-5 on fp32). Iteration log with 8 variants:
  cloud/results/phase3/k2/iterations.txt.
- Reviews: /check (Fable) after item 2, after 4b, and after item 5; outcomes in summary.txt.
  Item 5 review: kernel, routing and the test-bug diagnosis held; fixed a latent smem alignment
  assumption (`__align__(16)` on the staged q8_1 tile), the unlogged test claims (logs now
  committed, sanitizer rerun) and the n = 8 failure cause in this write-up.

## Phase 2: Route L on the rented RTX 3090 (2026-09-28, 350 W)

Route L (llama.cpp b11211 MMVQ/MMQ behind `csrc/lcpp_shim.cu`, `VLLM_GGUF_LCPP=1`) is correct and
serves. Full numbers: `cloud/results/phase2/summary.txt`. No shim or kernel bug was found; no plugin
code changed. Test and bench changes: be506ce.

- Build: in place with `VLLM_GGUF_BUILD_LCPP=1` (41 s at 24 jobs); both ops register; import keeps
  CUDA uninitialised.
- Kernel parity (VERIFIED): 464 Route L tests pass on real rows, all 9 types, MMVQ 1..8 and MMQ
  1..2048 rows (incl. 128), bf16/fp16, odd row counts, poisoned scratch, calibrated tolerances
  unchanged. Worst error vs the closest CPU model: bf16 2.9e-3, fp16 4e-4 (K-quant MMQ 6-9e-4).
  Test-side only: a D2S6 reference for Q2_K MMQ, and test inputs nudged off exact q8_1 rounding
  ties (16-bit x hits them often; fast-math division rounds them either way).
  Q2_K via lcpp MMQ is 7.7e-2 from full precision on outlier-heavy inputs (per-64 activation
  scale); that is llama.cpp's own arithmetic.
- ROUTE-L.md's guesses, each confirmed: stream mapping and graph safety (capture + replay bit-exact,
  81/81); the MMQ tail at 1-7 rows (poisoned scratch, memcheck and initcheck clean with one
  cudaMalloc per tensor); guards reject every bad case with no device fault (64/64, memcheck clean);
  the contiguous mixed-shard layout (views, bit-exact through `GGUFLinearMethod.apply`).
- Serve: load 281 s, 12.29 GiB weights, 23.4 of 24 GiB used after the smoke, 250,000 KV tokens
  (1.25x at 200k). Chat and tool-call smoke pass.
- Logit parity vs the phase-1b llama.cpp CUDA dumps: overall KLD 0.028 (stock 0.040), top-1 97.9%,
  per-prompt changes mixed. vLLM-vs-llama.cpp-CUDA is at or below llama.cpp's own CUDA-vs-CPU spread
  on all six prompts where that floor exists, but stock passes that gate too, so it does not
  discriminate. The absolute gate (KLD <= 0.001) still fails, as with stock kernels.
- Speed (350 W): decode 76.6 / 116.1 tok/s at c=1 / c=2 (stock 31.6 / 37.1; W4A16 baseline 89.1 /
  182.6), prefill 1036 tok/s at 8k and 589 at 180k (baseline 1108 / 603). MTP acceptance unchanged.
- Where decode time goes (c=1, one MTP step, 42 ms with the profiler on): GEMM kernels 27.8 ms, shim
  casts + q8_1 quantize 2.5 ms, draft lm_head reads 2.4 ms, host gaps 6.9 ms. IQ3 MMVQ at 4 rows
  runs at ~380 GB/s against 625 for IQ4_XS, which suggests 4-row IQ3 decode is lookup-bound.
  lcpp MMQ is faster than MMVQ from 8 rows (and for the lm_head from 4); the routing is unchanged.
- Not done: ninfer-all's decode-once vector kernel was fetched but not built or timed (the session's
  permission policy refused running that third-party code).

## Branches

| Branch | What |
|---|---|
| `main` | everything: `plugin/` (git subtree of vllm-project/vllm-gguf-plugin at e2b8ad5 plus our commits), tools, HF config dir, env records, this file |
| `route-l` | Route L as first built compile-only (merged into main in phase 2) |
| `opt-p`, `opt-k1`, `opt-k2` | optimization campaign round 1: plumbing, Q4_K/IQ2_S kernel, IQ3 mma kernel; merged via `integrate` |
| `integrate` | Integration 1 merge branch; main was fast-forwarded to it |
| `opt-k3`, `opt-r1` | round-1 work still running on the box (K3: 16-64-row mma; R1: IQ3 load-time repack); not merged |
| `swift-gsq-rco` | the plugin fork itself, `git subtree split --prefix=plugin`: upstream history through e2b8ad5, plus our adapter commit (7794689). Kept as a clean split because Garrett intends to send it upstream later; don't push it or open a PR right now. Regenerate after new plugin commits: `git subtree split --prefix=plugin -b swift-gsq-rco` |

Remote `plugin-upstream` has `pushurl = no_push`. Nothing was pushed anywhere.

## Done (2026-09-27 handoff)

1. **Isolated venv `.venv` = production's package set** (VERIFIED)
   - How it was built, and how to rebuild it:
     1. `uv venv --python ~/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12 .venv`
     2. `uv pip install --python .venv/bin/python --link-mode=copy --no-deps --offline -r env/prod-freeze.txt`. `env/prod-freeze.txt` is production's `uv pip freeze`: 202 pins, all from PyPI, vllm 0.27.1 manylinux wheel, torch 2.13.0+cu130. Everything came from the uv cache in 14 s. **`--link-mode=copy` is required:** production's venv files are hard links into the uv cache, so a hard-linked venv would share inodes with production, and any in-place write would change production's files.
     3. Overlay Garrett's patched vLLM with his own tool, pointed at our venv:
        `SITE_PACKAGES=.venv/lib/python3.12/site-packages STATE_FILE=<tmp>/state BACKUP_ROOT=<tmp>/bk ~/qwen38-27b-rtx3090/scripts/deploy-vllm.sh --init v0.27.1`, then `... deploy-vllm.sh ba05ffababdc`. The fresh wheel matched v0.27.1 on all 2,755 tracked files. After deploy, `--verify` passes on all 2,769 files at ba05ffab, the commit production has deployed. Ignore the restart line the script prints: nothing was restarted.
     4. Copy production's top-level `site-packages/build_backend.py`. Both flashinfer-python and torch-c-dlpack-ext ship this file, so install order decides which copy wins. Nothing imports it at runtime.
     5. `uv pip install --no-deps --link-mode=copy ~/llama.cpp-b11211/gguf-py` (gguf 0.19.0 at tag b11211), then build the plugin (item 2).
   - Evidence:
     - `diff env/prod-freeze.txt <(uv pip freeze --python .venv/bin/python)` shows only `gguf @ file://…/gguf-py` and `-e file://…/plugin` (saved as `env/gsq-freeze.txt`).
     - `rsync -rcni --exclude __pycache__ --exclude '*.pyc' <prod site-packages>/ .venv/…/site-packages/` (full checksums) differs only in:
       - 41 `*.dist-info/RECORD` files, where the only differing lines are entry-point script hashes (the venv path is in the shebang);
       - `build_backend.py`, since fixed by copying production's;
       - production's stray `vllm/v1/worker/gpu/model_runner.py.orig` (a known Aug 21 patch backup).
     - `stat` link count is 1 (real copies).
   - To re-verify: rerun the freeze diff and the `rsync -rcn` above, and run `deploy-vllm.sh --verify` with the same env overrides.
2. **Plugin built from source, sm86 only** (VERIFIED). Commands: `tools/setup-cuda-toolchain.sh`, then `tools/capped tools/build-plugin.sh` (editable install, about 20 s).
   - The venv's own `nvidia/cu13` is unusable for building: it pairs nvcc 13.3.73 with cudart 13.0.96 headers, and CCCL `#error`s on that ("CUDA compiler and CUDA toolkit headers are incompatible").
   - So `build/cu130` holds a matched CUDA 13.0 toolchain (nvcc/crt/nvvm 13.0.88, cccl 13.0.85, runtime 13.0.96). These are the pins of the `cuda-toolkit` metapackage already in production's venv, and they match torch.version.cuda 13.0.
   - CUDA 13.0 headers clash with this host's glibc 2.43, which declares `rsqrt/rsqrtf` noexcept. The script adds `noexcept(true)` to those two declarations in our private header copy, only when glibc ≥ 2.41; CUDA ≥ 13.1 does the same with `_NV_RSQRT_SPECIFIER`. Host compiler is g++-13; `-lcudart` is satisfied by a `lib/libcudart.so` symlink.
   - Result: `_C_gguf.abi3.so` contains only a `sm_86` cubin and NEEDs only `libcudart.so.13` (no CUDA 12/13 runtime mix). All 9 cudart symbols it imports resolve against the venv's libcudart 13.0.96. All 6 ops register via `torch.ops.load_library` with `torch.cuda.is_initialized() == False`.
3. **Adapter fix** (`plugin/…/weights_adapter/qwen3_5.py`, commit 33cbfdd / 7794689 on the fork; VERIFIED in source, gate unit-checked).
   - Before: a multimodal config without mmproj raised. Even without the raise, the text prefix would have been `model.`, which `Qwen3VLForConditionalGeneration.hf_to_vllm_mapper` (qwen3_vl.py:1731, which maps `model.language_model.` to `language_model.model.`) does not map. Neither would the quant config's layouts and unquantized modules (model_loader/utils.py:309-314).
   - Now: the prefix follows `vision_config`, and `build_name_map` requires image and video limits of 0, or `--language-model-only`. Then vLLM's `_mark_tower_model` (interfaces.py:269-306) replaces the vision tower with `StageMissingLayer` instead of leaving it silently uninitialized. The plugin's loader has no "all weights loaded" check.
   - The multimodal config is kept because vLLM 0.27.1 only creates the MTP draft for `model_type qwen3_5` (config/speculative.py:516).
   - Checked: the gate accepts limits of 0 and `--language-model-only`, and rejects the defaults, `image>0`, and a missing multimodal config.
4. **HF config dir** `hf-config/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp/`, built and verified by `tools/make_hf_config.py build|verify` (35 checks pass; VERIFIED).
   - Contents: Swift W4A16 AutoRound's `config.json` minus `quantization_config`; Swift's tokenizer, processor and generation files byte-identical; `chat_template.jinja` = production's `qwen-sharp` template (froggeric v22.1). transformers prefers that file over tokenizer_config's entry (tokenization_utils_base.py:1783-1799). `PROVENANCE.json` holds sha256 of all sources and outputs.
   - Checks: every dimension equals the GGUF metadata (layers, heads, GDN dims, rope/mrope, layer_types incl. the MTP block, vocab, context); all 248,077 HF token strings equal the GGUF's; the 243 extra GGUF ids are `[PAD…]`; merges are identical; the pre-tokenizer regex equals llama.cpp's `qwen35` (llama-vocab.cpp:392-397); eos/bos match; AutoTokenizer loads production's template.
   - Keep this dir path unique to this model: the fs KV-tier namespace hashes `model_config.model`, which the plugin sets to this dir.
5. **CPU-safety tooling:**
   - `tools/capped`: the resource wrapper.
   - `tools/no_gpu.py`: import it first in any Python that touches torch or vLLM. It blocks NVML/driver dlopen through ctypes, so vLLM resolves `UnspecifiedPlatform` and nothing talks to the GPU.
   - `tools/pytest`: pytest living in `build/pytest`, outside the venv.

## Half-done (2026-09-27 handoff)

- **CPU tensor-mapping dry run** (deliverable 3, second half): not written at this point; done later
  the same day (`tools/meta_dry_run.py`, 866/866 tensors mapped). Plan as it was:
  - Run `GGUFModelLoader.load_model` (the plugin's real loader) with vLLM's real `Qwen3_5ForConditionalGeneration`, and separately `Qwen3_5MTP` for `blk.64`, on the meta device, feeding memory-mapped GGUF tensors through the adapter.
  - Assert: no unmapped names (851 main + 15 MTP = 866); every non-vision parameter loaded; GGUF logical shapes equal each shard's partition size; the 48 `linear_attn.out_proj` layers got the GDN layout.
  - Needs:
    - a platform stub (subclass `vllm.platforms.cuda.NonNvmlCudaPlatform`, overriding `get_device_capability`→8.6, `get_device_name`, `get_device_total_memory`; set `vllm.platforms._current_platform` before building configs);
    - a gloo world of size 1 (`init_distributed_environment` + `initialize_model_parallel(1,1)`);
    - `device_config.device = meta`;
    - `vllm.plugins.load_general_plugins()`;
    - running under `tools/capped` (6 GB), not light.
  - The GDN V-row permutes copy about 650 MB of packed rows transiently; everything else stays memory-mapped.
  - Watch production's local `draft_lm_head` patch (qwen3_5_mtp.py:91-105). The GGUF has no `mtp.draft_lm_head`, so that head must stay disabled.
- **Dequant fixtures + GDN round-trip** (deliverable 4): the agent stopped before writing files. Its read-only findings:
  - VERIFIED: the venv's gguf-py equals b11211's.
  - VERIFIED: `libggml-base.so` is CPU-only (no FMA flags), so a ctypes bit-exact cross-check vs gguf-py is feasible.
  - VERIFIED stored GDN shapes: `attn_qkv` 10240 rows (Q 2048, K 2048, V 6144); `attn_gate` 6144; `ssm_alpha/beta` BF16 [48,5120]; `ssm_conv1d` F32 [10240,4]; `ssm_a`/`ssm_dt.bias` [48]; `ssm_norm` [128]. All `ssm_out` are quantized, so the `input_to_gguf` activation-permute path is the one used.
  - VERIFIED from `conversion/qwen.py`: the converter adds +1 to every `*norm.weight` except `linear_attn.norm.weight`, and the plugin's −1 sets cover exactly that set (read, not yet tested).
  - VERIFIED: e2b8ad5 fixed the CUDA table in `csrc/gguf/ggml-common.h`, not a Triton table as the original brief said. The Triton tables are built from gguf-py at runtime, and production has no gguf package, so this venv's 0.19.0 supplies them.
  - INFERRED: the CUDA `iq3xs_grid` equals 4× b11211's `iq3s_grid`, with `(0.5+s)*0.5` compensating. `iq1s_grid_gpu` is uint64 but used truncated to 32 bits. Both still need a test.
  - How to resume: fake the converter with `Qwen3_5TextModel.__new__` plus hparams, `tensor_map=gguf.get_tensor_name_map(QWEN35, 65)`, `fuse_qkv=False`, `fuse_gate_up_exps=False`. Fixture `.npz` keys were fixed as `raw, ref, ggml_type, tensor, rows, shape`, named `tests/fixtures/dequant/<TYPE>__<tensor>__r<row>.npz`, plus `manifest.json`.
- **README.md** (end state, phases, build and test): not written. This file stands in for it.

## Not started (2026-09-27 handoff)

- Deliverable 5, all of it (paused then; resumed as Route L in phases 2-3).
- Deliverable 4 files: `tools/ggml_ref.py`, `tools/make_dequant_fixtures.py`, `tests/cpu/test_{dequant_fixtures,kernel_tables,gdn_roundtrip}.py` (findings above).
- Deliverable 6, all harness files: `tests/gpu/`, `bench/parity/`, `bench/speed/`, `cloud/bootstrap.sh`, `scripts/serve-{gguf,llamacpp}.sh`. The agent stopped while it was still reading source. What it found, all VERIFIED:
  - **q8_1 activation quantization:** float `d = amax/127`, `q = roundf(x/d)`, with `half(d)` and `half(sum x)` stored (`csrc/gguf/gguf_kernel.cu:32-67`). The build uses `--use_fast_math` (`setup.py:42`), so a reference must allow rare ±1 flips in q.
  - **No contiguity, stride or alignment checks** on W or X anywhere; the kernels use `data_ptr()` only (`gguf_kernel.cu:98, 118-285`). Guards are needed.
  - **The min term differs by path:**
    - MMVQ Q4_K (`vecdotq.cuh:328-351`), MMQ Q2_K (`:232-262`) and IQ1_M (`:1707-1750`) use the sum of the quantized q;
    - MMQ Q4_K/Q5_K (`:353-378`) use `half(sum x)`.
  - **Triton fallback:** `ops.ggml_mul_mat_a8` sends IQ types to Triton (`triton/gemm/iq_quant/iq3_s.py`). That path doesn't quantize activations and rounds the weights to the activation dtype, so bf16 is rounded twice.
  - **F32/BF16:** `ggml_dequantize` on these types raises in Triton. They load through `weight_utils.py:194-198` instead.
  - **Production's start script** (`single-user/start_qwen_vision.sh`) sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `VLLM_USE_FLASHINFER_SAMPLER=0`, `PYTHONHASHSEED=0` and an API key from `api_key.txt`, so the bench needs `--api-key`. The argv passes `--limit-mm-per-prompt` twice; the last one (`image 0, video 0`) wins.
- **Resume plan for the harnesses:**
  - Put shared defaults in `scripts/env.sh`: paths, a default port of 18090, and a refusal to use 18080/18081 without an explicit flag.
  - GPU use is opt-in via `GSQ_ALLOW_GPU=1`; without it, conftest imports `no_gpu` and the GPU tests skip.
  - Tight kernel check: accept the result if it matches either the q8_1 emulation (q-sum model, or x-sum for MMQ Q4_K/Q5_K) or the model with weights rounded to the activation dtype. The tolerance covers fp32 accumulation, output rounding, and the flips.
  - Loose check against full precision.
  - Run the out-of-bounds and misalignment tests in subprocesses, because CUDA errors are sticky.
  - Parity: a small C++ tool on b11211's libllama that dumps full logits only at flagged positions (e.g. the last 256-512 tokens of each 8k/32k/100k+ sequence), and the same token ids fed to vLLM with logits captured in-process, run without spec decode. Full logits for every position of a 100k sequence would be ~50 GB.
  - Bench: salt every prefill prompt so the prefix cache and KV tiers can't hit. Read MTP acceptance from vLLM `/metrics` and from llama-server's draft stats.

## Findings for the kernel-route decision

- VERIFIED, plugin CUDA code: `csrc/gguf` is llama.cpp **b2899** (May 2024).
  - MMVQ launches one grid-y slice per token, so an n-token batch reads the weights n times.
  - For IQ types, `linear.py:38-54` uses MMVQ up to 8 tokens (16 when rows ≤ 5120). Above that, since IQ types aren't in `MMQ_QUANT_TYPES`, it runs `ggml_dequantize` of the whole matrix every forward, then `x @ W.T`.
  - The plugin also ships Triton IQ GEMM kernels (`triton/gemm/iq_quant`), but `linear.py` never routes IQ types to them.
- VERIFIED, mixed quant types inside fused layers are common in this GGUF: gate/up (e.g. blk.0 IQ2_XS + IQ2_XXS), GDN qkv/z (IQ3_S + IQ3_XXS), and full-attention q/k/v (Q2_K + Q4_K). `GGUFLinearMethod` pads all shards to the widest row (wasted VRAM). `apply()` then runs `weight[start:end, :offset].contiguous()` every forward, which copies each narrower shard.
- VERIFIED, llama.cpp b11211 `ggml-cuda` has MMQ (int8 MMA, stream-k) for 9 of the 10 quantized types here. IQ1_M is the exception, and it appears only in `blk.13.ffn_gate`. b11211 also has multi-column MMVQ for all of them. Vendoring those files behind a small shim (host launcher + pool) is the low-risk route if the survey finds nothing better. Its q8_1 activation quantization would also match llama.cpp numerics.
- VERIFIED, `--language-model-only` does more than set the limits to 0. It enables a fused QK-norm/RoPE/gate path in Qwen3.5 attention (qwen3_next.py:315-329) that production doesn't use. It's a speed knob to A/B on both W4A16 and GGUF; the serve script mirrors production's `--limit-mm-per-prompt` instead.
- VERIFIED, production serves `Starw1/Qwen3.8-27B-absolute-heresy-W4A16` (local dir `Qwen3.8-27B-W4A16-AutoRound-fast`, `base_model: MuXodious/Qwen3.8-27B-absolute-heresy`), not Swift. The Swift tokenizer differs from production's in its pre-tokenizer regex (`[\p{L}\p{M}]+`).

## Resource caps and interference rules used

- Every heavy command ran under `tools/capped`: a `flock` on `/tmp/gsq-heavy.lock` (one at a time), a wait until MemAvailable ≥ 8 GB, `systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0 -p CPUQuota=200%`, `nice 19`, `ionice -c3`, `CUDA_VISIBLE_DEVICES=""`, `MAX_JOBS=2`. The user cgroup delegates cpu/memory, which I checked in `/sys/fs/cgroup`. `GSQ_LIGHT=1` means 2 GB, 1 CPU, no lock.
- Never touched: production venv, unit, drop-ins, model dirs (read only), the running vLLM (pid 130752, only its `/proc` cmdline was read), ports 18080/18081, VM 192.168.1.84, Proxmox. No sudo, no pushes.
- The GGUF was only memory-mapped and its header read, never loaded whole.
- /tmp is tmpfs (RAM). Scratch was kept tiny and deleted. `build/` holds only `cu130/` (271 MB toolchain) and `pytest/` (3 MB); keep both, the build needs them. `.venv/` is 7.9 GB.

## Open questions for Garrett (2026-09-27 handoff)

- The kernel route: answered, Route L (docs/adr/0001, accepted 2026-09-28).
- Cloud baseline: production's `Starw1/Qwen3.8-27B-absolute-heresy-W4A16`, or `Swift-1.5-Qwen3.8-27B-W4A16-AutoRound`, which has the same Swift weights as the GGUF? The speed is the same architecture either way. Only the Swift checkpoint gives a quality-comparable baseline.
- GGUF download access for a cloud box (HF token?).

## Gotchas

- vLLM's platform detection calls NVML during `import vllm`. Scripts must `import no_gpu` before torch and vLLM. Early in this run, one NVML probe happened (a single `nvmlInit` via `vllm/platforms/cuda.py`, like one nvidia-smi query) before the guard existed. No kernel was loaded.
- `import torch` dlopens libcuda lazily. That's harmless with `CUDA_VISIBLE_DEVICES=""`, so the guard asserts on NVML and `torch.cuda.is_initialized()` instead.
- `uv pip install --target` without `--no-deps` shadows venv packages through PYTHONPATH, which is why `tools/pytest` installs pytest, pluggy and iniconfig only.
- The GGUF HF repo API answered 200 without auth on 2026-09-27, but its README says it's private/gated. Check download access before relying on a cloud bootstrap. `Starwaves1/vllm` @ ba05ffab and llama.cpp `b11211` (d7fb90e8) are publicly readable.

## 2026-09-27 evening: verification checklist

HANDOFF.md §8, worked read-only (no GPU, nothing restarted). Captured 20:20 UTC.

| # | Item | Result |
|---|---|---|
| 1 | GGUF checksum | SKIPPED re-hash (page-cache pressure; already verified). SHA256SUMS line = `9aecf1cd…677e5`, same as HANDOFF; size 12,120,016,896 matches. |
| 2 | Prod argv vs `production-vllm-server-info.json` | VERIFIED: `/proc/130752/cmdline` split on NUL equals JSON `argv` element for element (53 args). Process started 2026-09-25 23:22 EDT. The 4 model-file sha256s and the chat template sha256 match; deploy repo HEAD `2138d1ae8d`; prod venv has `vllm-0.27.1` and `torch-2.13.0` dist-info; GPU 595.91.07, 23,614/24,576 MiB. No drift. |
| 2a | Prod model identity (r03 vs r13) | VERIFIED r03: `-AutoRound-fast/README.md` has `base_model: MuXodious/Qwen3.8-27B-absolute-heresy` and serves as `Starw1/Qwen3.8-27B-absolute-heresy-W4A16`. Prod is the heresy finetune, not base Qwen3.8 and not Swift (Findings wording corrected). |
| 3 | Run 1 live | VERIFIED via `curl http://<bench VM>:8765/api/status`: job `pi-qwen3.8-27b-full113-k1-20260927-175853`, state running, concurrency 2, 1/113 done (1 failed, 0 timeouts), 2 trials in agent phase at ~28 tok/s. |
| 3a | Priority scheduling | VERIFIED in argv (`--scheduling-policy priority`). The dashboard does not expose priority. `/metrics`: `num_requests_running 1`, `waiting 0`, `num_preemptions_total 3`. |
| 4 | Adapter commit 33cbfdd / 7794689 | VERIFIED same patch (only the `plugin/` path prefix differs); `swift-gsq-rco` = upstream `e2b8ad5` + this one commit, 1 file, +23/-8; `git subtree split` still yields 7794689. Logic is right: the mm_proj-present paths are unchanged, and the new gate relies on `MultiModalConfig.get_limit_per_prompt`, which returns 0 under `language_model_only` (config/multimodal.py:350). Upstreamable in substance, with two gaps: no regression test is committed (the gate check was ad hoc; the plugin has no qwen3_5 adapter tests), and the commit message references local STATUS.md and vLLM 0.27.1 line refs. The error now fires in `build_name_map` rather than `patch_hf_config` (later, still clear). |
| 5a | vLLM PR #36226 | VERIFIED: `.diff` touches `csrc/quantization/gguf/{gguf_kernel.cu,mmq.cuh,vecdotq.cuh}`, the vLLM test file and `vllm/.../quantization/gguf.py`. With paths remapped to `vllm_gguf_plugin/csrc/gguf/`, `git apply --check` passes on e2b8ad5, and also on e2b8ad5 plus efschu's csrc. The PR does **not** touch `ggml-common.h`; it only indexes `iq3xs_grid`, so applying it to the plugin keeps e2b8ad5's fixed table automatically. Its dispatch change is in vLLM's `gguf.py` and has to be ported by hand to the plugin's `quantization/linear.py`. Not compiled. |
| 5b | efschu `qwen35-support` | VERIFIED: head `789d132`, 22 behind / 14 ahead of upstream `e2b8ad5` (merge-base `acf0e6d`). `mmvq.cuh` has `mul_mat_vec_q<…, ncols_dst>` launched with 1/2/4/8, and IQ3_S goes through the same `mmvq_launch`. `git merge-tree` onto e2b8ad5: all csrc merges clean; conflicts only in Python (config_parser, loader, quantization/{config,linear,params}, weights_adapter/__init__). It has no `ggml-common.h` change, so rebasing picks up the table fix. |
| 6 | localweights "29 vs 79" | SKIPPED: left to the kernel-survey agent. |
| 7 | Secrets | VERIFIED clean for this repo (excluding .venv/build) and the handoff dir: no hits for HF/OpenAI/GitHub/AWS tokens, private keys, `password=`, `api_key=`, Bearer, or emails; the git history (`git log --all -p`) has none either. The prod vLLM process has no `VLLM_API_KEY` in its environment, so no literal key could be checked. The only author is `Starwaves1 <…@users.noreply.github.com>`. For an upstream PR from `swift-gsq-rco`, a grep for `garrett`, `Starwaves`, `192.168` finds nothing. `main` (not for upstream) has the LAN IP of the bench VM in STATUS.md and `/home/garrett` paths in env/, PROVENANCE.json and make_hf_config.py. |

Repo reproducibility:
- `.gitignore` covers `.venv/ build/ __pycache__ *.pyc *.so *.egg-info /runs/`. No committed binaries: the largest tracked files are tokenizer.json (12.5 MB), vocab.json and merges.txt (text).
- Fork pinned: the b98f4a9 trailer `git-subtree-split: e2b8ad53…`; prod vLLM overlay `ba05ffab`; gguf-py b11211; CUDA wheel pins in `tools/setup-cuda-toolchain.sh`.
- Build recipe: README.md (commit 30dd497) and item 1 above. It is not one command yet: the venv and overlay steps are manual (DoD 7 gap).
- Remote `plugin-upstream`: `pushurl = no_push`, verified in `git config`. `git fetch` of efschu (FETCH_HEAD only, no remote added) and of plugin-upstream (still `e2b8ad5`) was done for 5b.

## 2026-09-27 evening: Phase B harnesses

Commits `b16541f` (harnesses) and `10d6774` (bootstrap pins `b16541f`). **Nothing here has run on a GPU.** Everything was written and checked on the CPU only.

| Entry point | What |
|---|---|
| `scripts/env.sh` | Shared settings and guards. Port 18090. 18080/18081 are refused with no override. `GSQ_ALLOW_GPU=1` gate. HF config dir must be `hf-config/<name>`, and its chat template must match production's sha256. The fs tier root must not be production's `/mnt/kvcache/tier` or anything under it. Production counts as live if :18080/:18081 listen or a production-venv vLLM runs. |
| `scripts/serve-gsq.sh`, `serve-baseline.sh` | Production's argv from `env/prod-serve-argv.txt`. They differ only in venv, model, `--hf-config-path`/`--tokenizer` (gsq), port, host 127.0.0.1, `--chat-template` (repo copy, same bytes) and fs tier `root_dir`/`max_bytes`. Optional: `GSQ_CPU_TIER_BYTES`, `GSQ_MAX_MODEL_LEN`. Plugin on/off via `VLLM_PLUGINS`. The argv diff is printed before launch; `--dry-run` stops there. Both refuse to start while production is live (`GSQ_ALLOW_LOCAL_BASELINE=1` / `GSQ_ALLOW_BESIDE_PROD=1` override). |
| `tests/gpu/` | Kernel parity per quant type on real GGUF rows (dequant, MMVQ 1–16 tokens, MMQ 16–512, production routing on whole tensors). OOB/contiguity/alignment/CUDA-graph cases, each in a subprocess (optional compute-sanitizer). 200k fit. MTP acceptance. 326 tests; all skip without `GSQ_ALLOW_GPU=1`. Shared constants are in `gsq_gpu.py`, not conftest, because `tests/cpu` has its own conftest. |
| `bench/parity/` | `prompts.py`: 11 salted sequences, 1k–120k tokens (two ≥100k), built from the pinned venv's vLLM source. Fingerprint `eb7ac4b4…` is locked in `prompts.lock.json`. `llama_logits.cpp` + `build.sh` (b11211 libllama; `--build-llama` for a cloud box). `vllm_logprobs.py`: in-process, full-vocab logprob probes via prefix cache. `compare.py`: mean KLD and top-1, overall and ≥100k only. `run.sh` ties them together. |
| `bench/speed/` | `run.sh gsq\|baseline [--start]`: runs production's `bench/run_benchmarks.sh single` twice, verbatim and sha256-pinned, from a staging dir with a `venv` symlink to `.venv`, so production's venv is never executed. Adds a salted prefill ladder: 8k/64k at c1+c2, 180k at c1. Records MTP per-position acceptance from `/metrics`. `mtp_acceptance.py` (vLLM `/metrics` or llama-server `timings.draft_n*`) and `serve-llamacpp.sh` (:18091) give the llama.cpp reference. |
| `bench/soak.sh`, `soak_load.py` | 24 h at c2 with production's CUDA-graph argv. Mix: chat, tool calls, long new and prefix-hit prompts up to 120k, aborted streams, greedy, random priorities. Every 60 s it logs liveness, GPU MiB of the server's session, RSS and gauges, and greps the log for faults. A dead server is not restarted unless `GSQ_SOAK_RESTART=1`. Report thresholds: 256 MiB GPU / 1 GiB RSS growth after the first hour. |
| `cloud/bootstrap.sh` | For an sm86 box: preflight checks, clone (URL or `git bundle`) at the pin, venv from `env/prod-freeze.txt`, production's `deploy-vllm.sh` overlay of `ba05ffab`, gguf-py b11211, `tools/build-plugin.sh`, freeze check, GGUF (sha256-checked). Then the llama.cpp build, llama MTP reference, tests, parity (bf16 KV, then fp8 for information), speed for gsq and baseline, optional `--soak`, then tar + rsync. `--baseline prod` (default) uses HF `Starw1/Qwen3.8-27B-absolute-heresy-W4A16`, sha-checked against production's config, quantization config and index. `--baseline swift` needs `GSQ_BASELINE_SRC` (the `-prepared` dir; not on HF) and runs with `--max-model-len -1`. `--dry-run` prints every command. |

How it was validated (all on the CPU):
- `bash -n` and shellcheck (from the shellcheck-py wheel, installed to /tmp and since deleted) on every script: clean.
- `py_compile` on every .py.
- `serve-*.sh --dry-run`. Every guard was triggered on purpose: ports 18080/18081, production's model dir as the HF config dir, production's tier root and a subdir of it, an unwritable tier root, no GPU gate, and baseline while production is live.
- `bootstrap.sh --dry-run` end to end, plus its argument errors.
- `g++ -fsyntax-only -Wall -Wextra` of `llama_logits.cpp` against b11211 headers: clean.
- `prompts.py --dry-run` twice gave the same fingerprint.
- The MTP `/metrics` parser was checked on synthetic counters.
- `soak_load.py report` on a synthetic 25 h run: passes, and fails on an injected IMA line.
- Reference math on real rows, VERIFIED: the Q4_K min-term split (zeroed quant bits → `dmin·m`) is block-constant and reproduces q81 exactly, and the q8_1 rounding is half-away-from-zero.
- All Python ran under `GSQ_LIGHT=1 tools/capped` with `no_gpu` imported. The one exception was a first bytecode-only `py_compile` pass, later redone capped.

Assumptions to confirm on the first GPU run (they are also TODOs in the code):
- Plugin API: `ops.ggml_dequantize(W_uint8[rows, bytes], type, rows, cols, dtype)` accepts float32 output. `ggml_mul_mat_vec_a8` / `ggml_mul_mat_a8(W, X, type, rows)` return `[n, rows]` in X's dtype. `_fused_mul_mat_gguf(x, W, type)` is importable from `quantization/linear.py`.
- Tolerances: TIGHT 4e-3 (bf16) / 1.5e-3 (fp16) against the best of q81/xsum/wround; LOOSE 5e-2 against full precision. On the CPU, the x-sum model already sits 1.5e-2 from full on Q4_K. Dequant fp16/bf16 is compared with atol 0 and may need 1 ulp.
- Guard tests are expected to fail at e2b8ad5: no checks exist. They are the acceptance test for the guards. Record which cases fault.
- Serving: `--hf-config-path` + `--tokenizer` on the hf-config dir is enough for the plugin's model_config.model/tokenizer. `LLM(max_logprobs=-1)` with `logprobs=-1` returns every vocab id in 0.27.1, and the probes hit the prefix cache in align mode.
- The KV capacity log lines match `kv_cache_utils.py:2389-2391`.
- llama-server b11211 reports `timings.draft_n`/`draft_n_accepted` with `--spec-type draft-mtp`.
- Cloud:
  - llama.cpp's CUDA build needs a system nvcc + cuBLAS; `build/cu130` has no cuBLAS dev files.
  - The deploy repo commit `2138d1ae` must be on GitHub (not checked).
  - `deploy-vllm.sh --init v0.27.1` must work on a fresh wheel, as it did here.

Defaults I chose, all visible in the argv diff:
- Host 127.0.0.1.
- A local test API key (`gsq-local-test`), never production's.
- fs tier at `/mnt/kvcache/gsq-tier`, capped at 100 GB. On this host it shares the drive with production's 300 GB tier; delete it after use.
- Parity runs vLLM with bf16 KV against llama.cpp f16 KV, so it measures the weights and kernels. fp8 is a separate informational run.
- Parity dumps are about 3 GB per engine and are deleted unless `KEEP=1`.
- The production environment variables come from this file's "Production's start script" notes, not from reading production's process environment.

(2026-09-27) STATUS.md itself was not committed then: it held other agents' uncommitted sections. It has been committed since phase 2.

## 2026-09-27 evening: CPU tests

Commit `27deeb5`. No GPU use: every run imported `no_gpu`, and the dry run also blocks torch CUDA init (it did catch one attempt, in `QwenGatedDeltaNetAttention` via `current_platform.current_device()`, now stubbed to meta). Run: `GSQ_LIGHT=1 tools/capped tools/pytest tests/cpu` → **416 passed, 54 xfailed**, 13 s, max RSS 1.2 GB.

- **Fixtures** (`tools/make_dequant_fixtures.py`, `tests/fixtures/dequant/`, 1.4 MB, committed): 84 real rows, 10 types × up to 3 tensors × 3 rows, schema as fixed above. Every gguf-py ref equals b11211 `libggml-base` `to_float` bit for bit (`tools/ggml_ref.py`, ctypes).
- **Plugin dequant vs gguf-py** (`test_dequant_fixtures.py`):
  - Triton kernels, run with `TRITON_INTERPRET=1` on CPU tensors: fp32 output bit-exact on all 84 rows, all 10 types. No bf16 check: the interpreter truncates fp32→bf16, compiled Triton rounds to nearest even.
  - CUDA `dequantize.cuh`, compiled for the host by `tools/cuda_dequant_host.py` (the unmodified source; `<<<>>>` launches rewritten to serial loops; `half` = `_Float16`, one RNE rounding per intrinsic). All 7 IQ types: bit-exact in fp32, and bf16 output = RNE(ref). Equality with the real sm86 build is INFERRED (fast-math, but there are no mul+add pairs or denormals on these paths).
  - **Finding: K-quant CUDA dequant is not bit-exact.** vLLM's b2899 copy does Q2_K/Q4_K/Q6_K in fp16 (`__hmul`/`__hsub` on `half`, and Q6_K's `__int2half_rn(sc*q)` rounds products above 2048), while ggml computes in fp32. In bf16 output, 16% (Q2_K), 28% (Q4_K) and 7% (Q6_K) of elements differ from RNE(ggml) by 1 ulp; max error ≤ 3.7e-3 of the row's absmax. Marked strict xfail, plus a bound test at 2⁻⁹·absmax. Impact on this model: small. linear.py sends K-quants to MMVQ/MMQ, not dequant, and the embedding is IQ2_S. `tests/gpu` dequant at atol 0 will fail for K-quants for this reason.
- **Kernel tables** (`test_kernel_tables.py`, 18 pass): the plugin's CUDA `iq2xxs/iq2xs/iq2s/iq3xxs` grids, `ksigns*`, `kmask`, `kvalues_iq4nl` equal b11211. `iq3xs_grid` = exactly 4 × b11211 `iq3s_grid` (all 512 entries; max 60 fits int8). `iq1s_grid_gpu` (uint64) fits 32 bits, equals b11211's uint32 table, and nibble-decodes to gguf-py's IQ1_S grid. The Triton tables equal gguf-py.
- **GDN round trip** (`test_gdn_roundtrip.py`, 4 pass): b11211 `Qwen3_5TextModel.modify_tensors` (via `__new__`, Swift's GDN dims) → plugin name map + `transform_weights` gives back every HF tensor exactly (A_log within 1e-6): qkv/z/a/b/conv1d/dt_bias/out_proj, the +1/−1 norms (not `linear_attn.norm`), q/k norms, and the MTP enorm/hnorm/shared_head norm/fc. Quantized out_proj stays GGML-tiled, and `input_to_gguf(x) @ W_gguf.T == x @ W_hf.T` exactly.
- **Meta dry run** (`tools/meta_dry_run.py`; 3 GB cap, peak RSS 2.0 GB, ~2 min): **PASS.** Real `GGUFModelLoader.load_model` + vLLM `Qwen3_5ForConditionalGeneration`, then `Qwen3_5MTP`. The platform is a NonNvmlCudaPlatform stub (sm86), with gloo world size 1, `load_general_plugins()`, `MTP_DRAFT_VOCAB=0` and fp8 KV + MTP k=3 config. The weight iterator yields meta tensors of the stored shape and dtype, so only the header is read (the plugin's own iterator does `torch.tensor(memmap)`, a full copy per tensor).
  - 866 = 851 main + 15 MTP; none unmapped, none overlapping.
  - Main: 917/917 params loaded. MTP: 17/19 (the missing `embed_tokens`/`lm_head` are shared from the target by design).
  - 258 GGUF layers, 498 shards: every stored logical (rows, cols) equals the partition size. GDN `in_proj_qkvz` loads as shards 0–3 (qkv split into q/k/v); 91 layers mix quant types across shards.
  - 48 `out_proj` layers have `GGUFHeadTilingLayout(3, 128)`.
  - One dry-run-only patch: `params._store_gguf_weight_type` calls `.item()` after moving to `param.device`, which fails on meta, so the script reads the value on the CPU first. That's harmless on a GPU (one sync per tensor).
- MemAvailable stayed ≥ 9.68 GB throughout. Scratch in /tmp removed. `tools/pytest` now passes `-p no:cacheprovider` (no `.pytest_cache` in the repo).
