# EXL3 at long context: where a decode step goes (DeepSWE soak, RTX 3090)

2026-10-11, track N2 of `orchestrator-handoff/12-exl3-next-steps-limited-compute.md`. Read-only: no GPU, no prod
contact beyond the existing 1 Hz recorder log and the unit's journal. Labels as in EXL3-OPT.md: VERIFIED (measured
or read in code), INFERRED (derived, not measured on this configuration).

Inputs: the prod 1 Hz log (`evidence/prod-exl3-deepswe-20261010/metrics-1hz.log`, 2026-10-10 02:24 to 2026-10-11
14:53 ET), job 23's torch traces (`~/vast-archive/workspace/logs/r3/23-attn-share/attn/`, GSQ-RCO, same model
architecture, fp8 KV, FlashInfer, k=5), the f4m14 GEMM micro (`cloud/results/exl3-opt-h16x/f4m14/11-mr-micro/`),
the 06-ladder long prefills, tiers.md, and vLLM main as installed in prod (`.venv-main`, v0.30.1rc1.dev285).
Section 1's numbers come from `long-context-1hz.py` (next to this file).

## Summary

- **Steps that carry a prefill chunk dominate.** Prod's mean is 88 ms per counted step. A decode-only step at
  2 x ~95k context costs 49 ms. A step that also carries one 128-token prefill chunk costs 220-250 ms. About 20 %
  of steps carry a chunk, and they take about 46 % of GPU-busy time (12.4 M freshly computed prompt tokens at
  ~1.6 ms each).
- **Decode-only model (3.50 bpw, k=3):** ms/step = ladder(c) + 0.097 ms per 1k tokens of total context + ~4 ms.
  ladder(c) is the short-context ladder (26.8 / 26.9 / ~29.3 / 31.8 at c=1/2/3/4). The slope comes from job 23's
  per-kernel times. The ~4 ms is a prod-only residual, fitted on the 1 Hz data and the same at c=1, 2 and 3. At
  c=2 x 85k this gives 47.4 ms.
- **Verify attention is bound by MMA work, not bytes.** At k=3 one request's verify query packs 4 x 6 (GQA) = 24
  rows. FlashInfer's planner runs those 24 rows in a 64-row CTA tile, at the same ~48 TFLOPS as a real long
  prefill. The same kernel at a 16-row tile (the drafter's q_len 1 passes) runs at 90 % of DRAM bandwidth. Two
  consequences:
  - fp8 vs bf16 KV does not change verify speed.
  - The verify cost does not depend on k for k+1 = 3..10.

  Both points are FlashInfer/vLLM territory (section 4).
- **The EXL3 GEMMs are still about half of GPU-busy time.** Half of that is not weight streaming: it is the
  compute-bound 132-136-row GEMMs of chunk steps, which run on fp32-accumulating 64-row tiles plus a separate
  launch for the 4-8 leftover decode rows that re-streams every weight. The handoff's "13 ms non-weight at short
  context" used the 13.4 ms byte floor; the 8-row target pass actually takes 18.9 ms.
- **Cheapest plugin lever:** run rows above 48 on the h16 48-row tile. Expected: -22 ms per chunk step, about
  -4.5 % of GPU-busy time, -11 % prefill time per new token. One micro run proves it (section 3).
- **Prod 4.50 bpw since 2026-10-11 03:29:** its KV pool is 194k tokens against 282k for 3.50 bpw. Decoding
  requests per step dropped from 1.64 to 1.21, and generation tok/s in busy hours from 52-75 to 44-58. At this
  context, capacity costs more than the ~1-2 ms/step the bigger weights add.

## 1. What the 1 Hz log says (VERIFIED, `long-context-1hz.py`)

**Method.** The log is cut into 10 s windows.

- Steps = d`iteration_tokens_total_count`. This counts only steps that produced output tokens
  (`async_llm.py:820-821`), so a step where every running request is mid-prefill is not counted.
- Decoding requests per step = d`spec_decode_num_drafts_total` / steps. At fixed k, each decoding request adds one
  draft per step.
- MTP acceptance = d`accepted` / d`drafts`.
- Prompt tokens are credited only when a prefill completes (`stats.py` PromptTokenStats, `loggers.py:1137-1140`).
  Chunk work therefore cannot be placed per window. A window is classed as decode-only when the running gauge is
  constant, drafts per step ≥ running - 0.02, and no prompt completed in the window.
- Context proxy = `kv_cache_usage_perc` x the GPU KV pool. This proxy is noisy: it attenuates fitted slopes and
  overstates c=1 context.

The log has two segments, split at the restarts:

| segment | span | busy windows | wall per counted step | generated tokens per step | decoding requests per step | MTP accepted per draft (tokens per request-step) | freshly computed prompt tokens | prompt tokens from cache (GPU + offload tiers) |
|---|---|---|---|---|---|---|---|---|
| 3.50 bpw (282k-token KV pool) | 23.8 h | 50 % | 88.0 ms | 5.15 | 1.64 | 2.138 (3.14) | 12.44 M (20 % of steps if 128-token chunks) | 361 M |
| 4.50 bpw, bf16 draft head (194k pool) | 11.5 h | 40 % | 69.4 ms | 3.73 | 1.21 | 2.097 (3.10) | 3.44 M (11 %) | 104 M |

Idle windows (no steps) almost all have running = waiting = 0. They are DeepSWE pauses: 14:00-23:00 on 10-10, and
03:30-09:30 on 10-11 after the switch. Inside active hours the GPU is busy in 94-100 % of windows, so the server is
the bottleneck whenever the workload runs. Queueing is common: waiting > 0 in 20 % of busy windows at 3.50 bpw and
52 % at 4.50.

**Step time vs running count, 3.50 bpw** (wall time / counted steps per 10 s window):

| running | windows | mean | p10 / p50 / p90 | gen tok/step | acc/draft | decode-only windows | decode-only mean, p10 / p50 / p90 | context proxy p50 |
|---|---|---|---|---|---|---|---|---|
| 1 | 298 | 48.1 | 39.6 / 45.2 / 67.7 | 3.04 | 2.04 | 81 % | 44.1, 39.4 / 44.4 / 50.5 | 137k |
| 2 | 785 | 76.0 | 46.9 / 74.1 / 243.9 | 5.76 | 2.10 | 27 % | 49.0, 42.3 / 49.0 / 56.8 | 195k total |
| 3 | 321 | 103.0 | 56.5 / 114.9 / 294.1 | 8.31 | 2.11 | 14 % | 54.5, 49.7 / 54.6 / 60.2 | 206k total |

4.50 bpw: running 1 has 466 windows, 47.7 mean, decode-only 45.8. Running 2 has 244 windows, 67.1 mean, decode-only
46.9 at a 147k proxy. Running 3 never occurs.

**Decode-only step vs context.** The 1 Hz data was fitted two ways:

| fit | 3.50 bpw | 4.50 bpw |
|---|---|---|
| slope free: ms/step - ladder(c) | 9.9 + 0.068 per 1k ctx (residual sd 4.2 ms) | 7.3 + 0.071 per 1k ctx |
| slope fixed at the trace value 0.097: residual constant | 4.4 / 4.0 / 4.3 ms at c=1/2/3 | 4.1 / 1.6 ms at c=1/2 |

The fixed-slope fit uses job 23's per-call kernel times, scaled to k=3 (section 2). The free slope comes out lower
because the noisy context proxy attenuates it. The residual is flat in c, i.e. a constant per-step cost that
production pays and the ladders do not. GSQ prod showed the same: 13.2 ms host idle per step at n=2, against 6.3 ms
on the box (r3 job 27). Jobs 21, 26 and 27 ruled out the KV connector, CPU contention and API parsing as causes.

**Steps that carry a prefill chunk.** In running-2 windows, f is the fraction of steps where one request did not
decode (it was prefilling a chunk):

| f | windows | wall per counted step | share of running-2 time |
|---|---|---|---|
| < 0.02 (decode-only) | 228 | 49.0 | 29 % |
| 0.02-0.10 | 116 | 63.1 | 15 % |
| 0.10-0.20 | 126 | 75.2 | 16 % |
| 0.20-0.30 | 65 | 92.1 | 8 % |
| 0.30-0.50 | 77 | 110.7 | 10 % |
| 0.50-0.80 | 40 | 156.1 | 5 % |
| ≥ 0.80 | 133 | 257.5 | 17 % |

The step-weighted linear fit is 47.3 + 202 f ms, so a step that carries a chunk costs ~250 ms at 3.50 bpw (~223 ms
if f ≥ 1 windows are excluded). At 4.50 bpw the fit is 45.7 + 161 f: ~207 ms, lower because that tier's smaller
pool holds shorter contexts.

Busy-time budget for 3.50 bpw: 42.8k s busy = 485k counted steps x ~47.5 ms (decode-equivalent, 23.0k s, 54 %)
plus 12.44 M freshly computed tokens x 1.57 ms (19.5k s, 46 %). The scheduler caps every prefill at 128 tokens per
step whenever two or more requests are eligible (`scheduler.py:635-647`).

**MTP acceptance over time.** Hourly accepted per draft at 3.50 bpw, in active hours, is 2.07-2.23 (3.1-3.2
tokens per request-step). It dips to 1.85-1.87 only in the single-request stretches on 10-10 17:00 and 23:00-00:30.
At 4.50 bpw it is 1.91-2.15. It does not drift with soak time and does not depend on concurrency (2.04 / 2.10 / 2.11
at c=1/2/3). These rates are healthy and match the ladders (per position 0.722 / 0.502 / 0.345, mean length 2.57 at
T=0).

## 2. Per-step cost model at 85k context, c=2 (3.50 bpw, MTP k=3, fp8 KV)

Model facts used below (VERIFIED, `hf-config/.../config.json` and the prod journal):

- 64 layers: 16 full-attention (24 q heads, 4 KV heads, head_dim 256) and 48 GDN (48 value heads, 128 x 128 state,
  fp16 SSM cache).
- fp8 KV = 2 x 4 x 256 B = 2 KiB per token per layer: 32 KiB per token for the target, plus 2 KiB for the MTP
  layer.
- KV block size 832 (set by `interface.py:989` to match the mamba page size).
- FlashInfer backend with **PIECEWISE** CUDA graphs. FULL is refused for spec decode on sm86 (journal
  `compilation.py:1462`; `flashinfer.py:1036-1071` returns UNIFORM_SINGLE_TOKEN_DECODE).
- Graph capture up to 32 tokens.
- Draft head: 40,960 rows, fp8.

**Decode-only step, 2 x 85k = 170k total context**

| term | work per step | ms | basis |
|---|---|---|---|
| target EXL3 GEMM, 8 rows (2 x (k+1)) | 11.28 GB trellis (byte floor 12.05 ms) | 18.9 | f4m14 micro, 8 rows (VERIFIED) |
| drafter weights, 3 passes | MTP layer 0.21 GB (4 bit) + fp8 head 0.21 GB per pass | 1.9 | micro: MTP layer 0.34 ms/pass; checkpoint 1: fp8 head saved 1.0 ms/step vs bf16 |
| target attention, 16 layers | 170k x 32 KiB = 5.6 GB fp8 (byte floor 5.9 ms) | 14.8 | job 23: tile-64 kernel 1.04 ms per call per 192k context (17.7 calls/step), scaled |
| drafter attention | pass 1: q_len 4, tile 64, 0.92 ms; passes 2-3: q_len 1, tile 16, 2 x 0.41 ms | 1.75 | job 23: tile-16 kernel 0.47 ms per call per 192k = 843 GB/s, 90 % of DRAM |
| GDN | per request 48 layers x (1 state read + k+1 = 4 state writes) x 1.5 MiB = 377 MB; conv1d | 1.3 | job 23: 0.77 ms/request at k=5 (7 x 1.5 MiB), scaled to 5/7; `fused_sigmoid_gating.py:140-188` writes every position's state |
| norms, activations, rope, Hadamards, casts, sampling, rejection; ladder host idle | | ~3.8 | ladder c=2 26.9 minus the four rows above at short context (INFERRED split) |
| prod-only residual | | ~4 | section 1 fit |
| **total** | | **46.5** | measured: decode-only running-2 windows at a 160-200k proxy, 48.4 mean |

Attention therefore accounts for 0.097 ms per 1k context tokens: 0.087 for the target and 0.010 for the drafter.
At 2 x 85k its 16.5 ms is 2.4 times the 8-row EXL3 GEMM's gap to its byte floor (6.9 ms).

**Why the target attention runs at 40 % of bandwidth** (VERIFIED in code, INFERRED as the cause):

1. On sm86 vLLM sends every spec-verify request through FlashInfer's *prefill* wrapper. `reorder_batch_threshold`
   stays 1 because `supports_spec_as_decode` holds only for the trtllm kernels (`flashinfer.py:928-931`), so q_len
   4 is not a "decode".
2. The planner picks the CTA tile from packed q rows = q_len x GQA group = 4 x 6 = 24. `FA2DetermineCtaTileQ`
   (`flashinfer/data/include/flashinfer/utils.cuh:408-440`) returns 64 for 17..64 packed rows on Ampere and 16 for
   ≤ 16. Its 32-row tile exists only for head_dim ≥ 512.
3. 62.5 % of the MMA rows are padding. The tile-64 kernel's throughput is ~48 TFLOPS (4.19 GFLOP per 1k context
   per step incl. padding, over 0.087 ms). That is the same rate the same kernel reaches on true long prefill: the
   06-ladder 65k/180k prefills give 0.0083 ms per token per 1k context = ~50 TFLOPS over 17 attention layers.
   Meanwhile the tile-16 variant streams the same fp8 KV at 90 % of DRAM bandwidth. The tile-64 kernel is
   therefore MMA-bound, not byte- or dequant-bound.
4. Consequences:
   - Doubling bytes with bf16 KV would barely change time and would halve capacity.
   - k+1 anywhere from 3 to 10 costs the same verify attention.
   - The levers sit in FlashInfer's tile choice or accumulation precision, not in the plugin.

**Step that carries a chunk.** Running 2: one request decodes at 85k, the other prefills a 128-token chunk at ~85k.

| term | ms | basis |
|---|---|---|
| target EXL3 GEMM, 132 rows: one launch of 2 x 64-row slices (fp32 accumulate) + one 4-row launch that re-streams every weight | 117.7 + 18.6 = 136.4 | micro 128 and 4 rows; `trellis_serve/exl3_marlin.cu:146-238` (`max_thread_m_blocks = 4`, remainder loop) |
| chunk attention, 16 layers: 128 tokens x 85k x 24 heads x 256 x 4 FLOP x 16 = 4.3 TFLOP | 90 | 06-ladder: 0.0083 ms per prefill token per 1k context |
| MTP layer on the chunk (drafter pass 1 runs all scheduled tokens, `llm_base_proposer.py:512-568`): GEMM 132 rows + 1-layer attention | 1.9 + 5.6 | micro (draft, 128 rows); 1/16 of the row above |
| the decoding request's attention (target + 3 draft passes) | 8.3 | decode table, half of 16.55 |
| its draft passes 2-3, GDN (chunked kernels on 128 tokens + one decode), glue | ~5 | INFERRED |
| host (mixed steps run eager: 132 tokens > capture size 32) | ~5-9 | r3 job 25: 9 ms idle per step, mixed c=4 |
| **total** | **~252** | measured 223-257 (section 1) |

Per freshly computed token at 85k context: ~0.93 ms GEMM plus ~0.75 ms attention, about 1.7 ms against the 1.57 ms
measured in aggregate. Prefill is compute-bound on both terms.

The 128-token threshold therefore barely changes total GPU time; it decides who waits. That matches r3 job 25,
where 128 was best for decoders.

**Where the busy time goes, 3.50 bpw soak** (INFERRED from the two tables and the busy-time split): EXL3 GEMMs about
50 % (decode weight streaming ~20 %, chunk-step GEMMs ~25 %, glue and drafter ~5 %), attention about 40 % (decode
~18 %, chunk ~22 %), GDN, host and the prod residual about 10 %.

**Decode-only ms/step vs per-request context** (3.50 bpw, k=3, ladder(c) + 0.097 x c x ctx + 4; INFERRED, checked
against section 1 at 44.1 / 49.0 / 54.5 measured vs 44.1 / 49.8 / 53.3 modelled at the observed proxies; c=3's
ladder is interpolated). Cells that exceed the 282k pool are omitted.

| c | 8k | 32k | 64k | 85k | 128k | 190k |
|---|---|---|---|---|---|---|
| 1 | 31.6 | 33.9 | 37.0 | 39.0 | 43.2 | 49.2 |
| 2 | 32.5 | 37.1 | 43.3 | 47.4 | 55.7 | |
| 3 | 35.6 | 42.6 | 51.9 | 58.0 | | |
| 4 | 38.9 | 48.2 | 60.6 | | | |

At 4.50 bpw add ~3.1 ms (tiers.md short-context delta), plus 1.0 ms while its draft head stays bf16.

## 3. The three cheapest levers the plugin controls

**L1. Rows above 48 on the 48-row fp16-accumulating tile** (chunk steps, prefill tails, c ≥ 13 verify).

Mechanism. The launcher slices rows into 64-row tiles (`max_thread_m_blocks = 4`, `exl3_marlin.cu:149`). The
leftover rows go to a separate launch that streams the whole weight again. The h16 patch already gives the 48-row
kernel fp16 partial accumulation folded every 4 stages (mask bit 3, parity-gated in checkpoint 1). That kernel does
48 rows in 37.98 ms (0.79 ms per row); the fp32 64-row kernel needs 62.91 ms for 64 rows (0.98 ms per row).

Change. One line in `exl3_marlin_h16.patch`: `max_thread_m_blocks = 3` when `TRELLIS_H16_MB_MASK` has bit 3.
c_tmp (sized for 64-row tiles per SM) and locks (per SM) need no change.

Expected target-pass times, from the f4m14 micro:

| rows | now | 48-row slices |
|---|---|---|
| 64 | 62.9 | ~60 |
| 96 | 93.6 | ~76 |
| 128 | 117.7 | ~103 |
| 132 | 136.4 | ~114 |
| 136 | 136.6 | ~114 |
| 144 | 140.6 | ~114 |

That is **-22 ms per chunk step** at c=2 (~9 %). Over the 3.50 bpw soak, ~97k chunk steps make it ~-2.0k s of
42.8k s busy (-4.5 %): about -4 ms per average counted step (88 -> ~84), and -11 % on each turn's prefill compute
(1.57 -> ~1.40 ms per new token).

No gain at 2048-row chunks: the dequant route there runs ~0.8 ms per row, about the same.

Test. Rerun `bench/micro/exl3_mr.py --rows 64,96,128,132,136,144,192,384` on the patched build. It passes if the
target pass at 132-144 rows is ≤ 116 ms on the 3090; on the 3070, compare ratios to its own f4m14 run. Prefill
numerics move to fp16 partials, so follow with 10-mr-parity at those rows and job 19's long-context KLD.

Follow-up (argv, after L1): `--long-prefill-token-threshold` 136-140 fills the third tile (4-8 decode rows + chunk
≤ 144) for ~6 % more prefill per chunk step at the same cost.

**L2. fp8 draft head on the 4.50 bpw tier** (prod since 2026-10-11 03:29 runs `EXL3_DRAFT_FP8=0`, because the
head's quantize/repack peak left 0.20 GiB free at load).

Expected: **-1.0 ms per decode step**. Checkpoint 1 measured -1.0 / -1.0 / -0.7 / -0.7 at c=1/2/4/8 on 3.50 bpw,
with acceptance within noise; the target's logits are unchanged. KV capacity rises by ~5.5k tokens (+2.8 % of 194k),
which matters on this tier: 2 x 85k = 170k of 194k.

Test. Start the 4.50 tier with `EXL3_DRAFT_FP8=1` and the exl3-load-fix sitecustomize, then run a c=2 decode A/B at
85k. It passes if the load completes with ≥ 0.5 GiB free, the pool is ≥ 199k tokens, and decode-only ms/step drops
about 1 ms.

If load headroom stays short: quantize the head entirely on the host before upload. `draft_head.py` already stages
the row chunks on the host, but it allocates the Marlin repack on the GPU next to the fp8 copy.

**L3. Concat on the 3.50 bpw tier** (`EXL3_MR_CONCAT=1`).

Expected: **-0.6 ms per decode step** at c=2 (tiers.md: 26.84 -> 26.20; -0.6 to -1.1 at c=1..8). There is no
attention or prefill effect: chunk steps are GPU-bound, so fewer launches do not show.

Cost: -1.7 % KV (281,648 -> 276,967 tokens at 196,608), which still fits 3 x 85k, and +4.8 GiB reserved during load
only. Do not use it on 4.50 bpw, which already needs the load fix.

Test. N6's fit check at prod's argv (max-len 190,000, gpu util 0.94): load succeeds and the pool is ≥ 270k tokens.
Then a c=2 decode A/B at 85k.

**Not levers in this regime:**

- MTP k. Verify attention is flat in k+1 up to 10, but GEMM rows are flat only up to 8. At c=2, k=4 costs ~+4.6 ms
  (10 rows plus one drafter pass of ~0.8 ms at 85k) for ~+5 % tokens. k=2 saves ~1 ms for -11 % tokens. A
  per-batch-size k schedule is what tripped #50021's conv1d bound. Keep fixed k=3.
- KV dtype: verify is MMA-bound and fp8 halves capacity cost, so keep fp8.
- Block size: forced to 832 by the GDN page size.
- The 2048-row prefill route: see L1.

Unmeasured, for a spare GPU hour: the 1-8-row decode kernel runs 18.9 ms against a 12.05 ms floor. That is the
largest plugin-owned decode term, but no data points to a cheap fix. The vendored launcher has `g_blocks_per_sm`
(default 1, `exl3_marlin.cu:85`) and `g_in_had_inlaunch` (default off), which the shim does not expose; one micro
sweep at 4/8 rows would size them. The gain is bounded by 6.9 ms.

## 4. vLLM / FlashInfer side: notes for the fork

| # | item | size at c=2 x 85k | code path |
|---|---|---|---|
| F1 | Spec-verify attention runs on FlashInfer's prefill kernel with a 64-row CTA tile for 24 real rows (MMA-bound, ~40 % of DRAM bandwidth). A 32-row tile for head_dim 256 (a 2x2 warp layout; the 1-warp-in-q 32-row tile would overflow o_frag registers at VO 256), or spec-as-decode on sm86, would move it toward the tile-16 kernel's 90 % of bandwidth. | -6 to -8 ms per decode step (14.8 -> ~7) | `flashinfer/.../utils.cuh:408-440`, `attention/scheduler.cuh:597-619`; vLLM `flashinfer.py:928-931` |
| F2 | fp16 QK accumulation (`allow_fp16_qk_reduction`) in the prefill kernel: GeForce Ampere runs fp16-accumulate MMA at 2x the fp32-accumulate rate. The plugin's own 48-row h16 kernel reaches ~63 TFLOPS against ~51 for its fp32 64-row tile. This would cut chunk attention and also F1's padded verify. Needs a long-context accuracy check. | up to -25 % of attention: ~-20 ms per chunk step, ~-4 ms per decode step | vLLM's FlashInfer wrapper plan calls (`flashinfer.py:1777-1795`) do not pass it |
| F3 | Full CUDA graphs for spec decode on sm86: FlashInfer reports UNIFORM_SINGLE_TOKEN_DECODE, so prod runs PIECEWISE with attention and GDN eager between pieces. TRITON_ATTN supports FULL but is no alternative at long context: for q_len > 1 it uses the 2D kernel without split-KV (2 x 4 x ceil(4 / BLOCK_Q=2) = 16 CTAs on 82 SMs) and reads the KV twice. | host share of the ~4 ms residual plus ladder idle | `flashinfer.py:1036-1071`; `triton_unified_attention.py:937-940, 1046-1055` |
| F4 | Prod-only per-step residual ~4 ms (EXL3, this fit) and ~7 ms idle excess (GSQ prod, r3 job 27). Not the connector (job 21), CPU contention (job 26) or API parsing (job 27). Open; a py-spy or instr capture on prod during a decode-only stretch would attribute it. | ~4 ms per step | r3 PLAN H1 |
| F5 | The GDN spec kernel writes a full 1.5 MiB state per layer for each of the k+1 positions, although only the accepted one is kept. Writing only the final-accepted state would need the acceptance before the kernel, so this is a design note, not a quick win. | ~0.8 of 1.1 ms GDN | `fused_sigmoid_gating.py:174-188`; `qwen_gdn_linear_attn.py:1448-1484` |
| F6 | The drafter rebuilds attention metadata and replans FlashInfer for every pass. This is host time (part of F3/F4), not context-dependent. | host time, unmeasured on EXL3 | `llm_base_proposer.py:689-716` |
| F7 | Metrics: `iteration_tokens_total` counts only steps with outputs and credits prompt tokens at prefill completion, so chunk steps cannot be placed per window. A per-step "prefill tokens scheduled" counter would make the next soak's model direct. | | `async_llm.py:820-821`, `loggers.py:1137-1140` |
| F8 | Capacity binds concurrency on this workload: 4.50 bpw's 194k pool cut decoding requests per step from 1.64 to 1.21 and busy-hour tok/s by ~20 %. GSQ-RCO (11.7 GB) or 3.00 bpw (326k) would raise c. This is a tier decision for Garrett, not a kernel item. | | REPORT.md diagnosis 1 |
