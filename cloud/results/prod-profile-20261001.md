# Production profile, 2026-10-01 03:42-04:02 EDT (ms4, RTX 3090, GSQ-RCO GGUF on vLLM main)

Source: `~/Downloads/vllm-1hz-20261001-0342.json` (1,201 vLLM `/metrics` samples and 1,200 nvidia-smi samples at 1 Hz,
20 min, `service_events` and `requests` empty), plus one read of production's `/metrics` for counter semantics and
lifetime histograms. Read-only analysis on CPU. The box numbers come from `REPORT.md` (vLLM 0.27.1, k=3, short
prompts). Production on main has not been benchmarked on the box yet (STATUS "Current state"), so every box
comparison below also includes the 0.27.1 -> main change. Labels: MEASURED (from the export), INFERRED (model or
reasoning), SOURCE (read in vLLM code).

## 0. What the counters mean (and one trap)

- Argv (`env/prod-main-serve-argv.txt`): `num_speculative_tokens_per_batch_size [[1,4,5],[5,8,3],[9,16,2]]`.
  k is picked from the number of scheduled requests (SOURCE: `scheduler.py`, `dynamic_sd_lookup[len(num_scheduled_tokens)]`).
  So k=5 (6 rows per sequence) only at 1-4 running; 5-8 running use k=3 (4 rows); 9-16 use k=2. MEASURED:
  `draft_tokens/drafts` is exactly 5 in 457 s and 3 in 669 s; running never exceeded 8, so k=2 never ran.
- `prompt_tokens`, `prompt_compute` (local_compute), `prompt_cache` and `prompt_external` grow by the **whole
  prompt** in the step that emits the first token (SOURCE: `v1/metrics/stats.py` `update_from_output`,
  `is_prefilling` -> `PrefillStats`). A 100k prompt's prefill runs for seconds, but its counters jump in a single
  second. "Seconds where prompt tokens increased" are completion seconds, not prefill seconds. Section 2 uses a
  duration model and checks it against the measured decode stalls.
- The `running` gauge reads 0 for 1-3 s while the GPU runs at 98-100 % / ~335 W right before each of 20
  long-prompt completions (MEASURED). Every one of these stretches ends in a long prefill. INFERRED: the gauge only
  refreshes when some request emits tokens, so it is stale during prefill-only stretches. I count those seconds as
  prefill.
- KV pool (`cache_config_info`): 241,245 tokens, 310 blocks of 848 tokens, fp8, `kv_cache_max_concurrency` 1.21 at
  200k. `kv_usage x 241,245` tracks the context of the running sequences (one sequence at 0.34 -> 82k tokens,
  matching its next prompt of 75-79k), so I use it as "context in the batch".
- Model: 64 layers, 16 full-attention layers (interval 4), 4 KV heads x 256. fp8 KV is 16x4x256x2 B = 32 KiB per
  token for the target, plus 2 KiB per token per MTP draft pass. GDN state is 48x48x128x128x2 B = 75.5 MB per
  sequence and does not grow with context.

## Workload in the window

Two regimes (MEASURED, minute table below):
- **0-7 min:** 1-2 long-context agent streams. 67 long turns in the whole window: prompt 72k-137k (median 88k,
  mean 97k), only ~2k new tokens per turn (median compute 1,723). Mean running 0.7-1.9, no waiting.
- **7-20 min:** the same long streams plus ~5-6 concurrent short requests (prompts 200-1,400 tokens, 18 % cache
  hit). Running 6-8, waiting 1-7, KV 0.95-1.00.

945 requests finished; 182k tokens were generated (151.7 tok/s mean). The lifetime histogram (since the server
started) shows the same split: 1,049 of 1,313 prompts are under 500 tokens and 182 are over 50k (mean 13.6k).
The "57k mean" is the mean of the long agent turns only, not of all requests.

| min | run mean | wait mean/max | kv mean/max | prompt k | compute k | ext k | gen tok/s | done | preempt | util % | W | MHz |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1.9 | 0.0/0 | 0.79/0.84 | 405 | 9.1 | 0 | 109 | 3 | 0 | 87 | 345 | 1845 |
| 1 | 1.9 | 0.0/0 | 0.78/0.85 | 598 | 12.7 | 0 | 80 | 5 | 0 | 81 | 338 | 1830 |
| 2 | 1.5 | 0.0/0 | 0.65/0.86 | 697 | 15.1 | 0 | 67 | 11 | 0 | 76 | 336 | 1845 |
| 3 | 0.7 | 0.0/0 | 0.33/0.86 | 774 | 15.1 | 0 | 30 | 7 | 0 | 62 | 327 | 1875 |
| 4 | 1.3 | 0.0/0 | 0.61/0.88 | 161 | 3.3 | 0 | 55 | 2 | 0 | 69 | 328 | 1860 |
| 5 | 1.9 | 0.0/0 | 0.84/0.89 | 444 | 11.5 | 110 | 79 | 3 | 0 | 80 | 338 | 1830 |
| 6 | 1.8 | 0.0/0 | 0.80/0.94 | 599 | 23.3 | 0 | 39 | 6 | 0 | 89 | 342 | 1845 |
| 7 | 3.8 | 2.1/7 | 0.71/1.00 | 708 | 20.7 | 303 | 98 | 34 | 1 | 84 | 337 | 1838 |
| 8 | 7.7 | 0.5/2 | 0.77/1.00 | 241 | 23.9 | 130 | 207 | 77 | 3 | 74 | 331 | 1868 |
| 9 | 6.8 | 1.9/7 | 0.94/1.00 | 236 | 18.3 | 218 | 212 | 56 | 2 | 79 | 341 | 1815 |
| 10 | 6.2 | 2.2/7 | 0.79/1.00 | 190 | 15.8 | 175 | 217 | 44 | 0 | 84 | 341 | 1815 |
| 11 | 5.1 | 3.4/7 | 0.79/1.00 | 554 | 24.3 | 442 | 130 | 61 | 3 | 84 | 341 | 1830 |
| 12 | 5.5 | 2.7/7 | 0.82/0.99 | 562 | 26.4 | 535 | 157 | 76 | 0 | 85 | 341 | 1845 |
| 13 | 7.5 | 0.0/0 | 0.49/0.84 | 27 | 26.5 | 0 | 255 | 101 | 0 | 70 | 337 | 1792 |
| 14 | 7.7 | 0.0/0 | 0.48/0.50 | 23 | 22.9 | 0 | 279 | 81 | 0 | 72 | 335 | 1792 |
| 15 | 7.2 | 0.8/2 | 0.66/0.95 | 153 | 18.8 | 134 | 216 | 62 | 0 | 68 | 332 | 1860 |
| 16 | 7.3 | 0.5/2 | 0.57/0.96 | 165 | 28.9 | 136 | 214 | 100 | 0 | 72 | 336 | 1815 |
| 17 | 6.8 | 1.7/2 | 0.94/0.97 | 40 | 30.0 | 0 | 185 | 82 | 0 | 80 | 338 | 1770 |
| 18 | 6.7 | 1.8/2 | 0.93/0.99 | 48 | 28.0 | 12 | 186 | 81 | 0 | 85 | 340 | 1800 |
| 19 | 6.8 | 1.9/3 | 0.96/1.00 | 179 | 21.4 | 158 | 220 | 53 | 2 | 85 | 342 | 1800 |

## 1. Steps/s and ms/step by running count

Method. In a second where all n running sequences decode, each engine step adds one draft per sequence, so
steps/s = Δdrafts / n and ms/step = 1000 n / Δdrafts. Tokens per step = n x (1 + Δaccepted/Δdrafts).
Clean decode seconds: same running count at both ends, k consistent with the schedule
(Δdraft_tokens = k x Δdrafts), no request finished in this second or the one before, no preemption, and outside the
prefill mask (section 2). Medians are quantized: at n=1, 17 vs 16 drafts/s is 58.8 vs 62.5 ms. The pooled value
(Σ n / Σ Δdrafts) is the precise one.

Cross-checks. (i) Δgen = 182,062 vs Δdrafts + Δaccepted = 182,663: 3.136 vs 3.146 tokens per sequence-step. The
0.3 % gap is accepted tokens dropped at stop strings, net of first tokens. So the gen counter gives the same step
rate. (ii) Window mean interval between one sequence's steps = Σ running-seconds / Σ drafts =
5,761 / 58,055 = 99.2 ms (prefill stalls included). Lifetime `inter_token_latency` = 9,640 s / 121,575 = 79.3 ms;
its count matches lifetime drafts (121,548), so the ITL histogram measures the same per-sequence step.

| running | k (rows/seq) | rows/step | seconds used | ms/step median | p90 | pooled | tok/step (per seq) | accepted/draft (rate) | context in batch (median) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 5 (6) | 6 | 59 clean | 58.8 | 71.4 | **58.8** | 3.19 (3.19) | 2.19/5 (0.44) | 122k |
| 2 | 5 (6) | 12 | 155 clean | 62.5 | 71.4 | **63.6** | 7.78 (3.89) | 2.89/5 (0.58) | 204k |
| 3 | 5 (6) | 18 | 4 | 62.5 | 65.4 | 63.5 | 10.9 | 2.64/5 | 168k |
| 4 | 5 (6) | 24 | 1 | 66.7 | - | - | - | 2.93/5 | 182k |
| 5-6 | 3 (4) | 20-24 | 3 decode-only; 17 with short prefills | 58.8; 87.0 | -; 140 | 58.8; 87.0 | 15.4 | 2.07-2.16/3 | 196-214k |
| 7-8 | 3 (4) | 28-32 | 72 decode-only; 334 with short prefills | 81.0; 94.6 | 167; 162 | **81.7; 92.5** | 20.9 (2.99) | 2.01/3 (0.67) | 229k |

Bins 3, 4 and 5-6 are too thin to use: running sits at 2 or at 7-8 almost all the time. In 7-20 min nearly every
second has a short-prompt completion, so 7-8 is shown both decode-only and with short prefills interleaved.
Window acceptance by k (MEASURED): k=5 2.79/5 (0.558), k=3 2.02/3 (0.674). Lifetime per-position acceptance (scrape;
solved assuming only k=5 and k=3 ran, so INFERRED): pos0-2 0.84 / 0.67 / 0.53 per draft, pos3 0.45 and pos4 0.37
per k=5 draft. k=5 therefore gives 3.85 tokens per sequence-step vs 3.04 at k=3 (+27 %).

## 2. Wall time: pure decode vs prefill-bearing

Prefill duration model (INFERRED from the box prefill ladder 1248/954/644 tok/s at 8k/64k/180k): time(N) = aN + bN²/2
with a = 0.77 ms/token and b = 8.7 ns/token², so the marginal cost at context C is 0.77 ms + 8.7 ns x C per token
(1.61 ms at 97k: about half GEMM, half attention). Each long completion masks the ceil(compute x marginal(C))
seconds before it. Check: measured decode-stall stretches before 63 long completions add up to 199.5 s for
131k compute tokens (658 tok/s). The model predicts 214.7 s, so production prefills 7 % faster than the box
model.

| class (1,200 s) | seconds | share | notes |
|---|---|---|---|
| long-turn prefill (masked) | 309 | 25.8 % | measured stall 199.5 s = 16.6 %; the co-running decode drops to ~4-5 steps/s (~200 ms/step) |
| short-prompt completions in the second | 490 | 40.8 % | decode continues. Short prefill GPU time: 490 s minus modeled decode time = 153 s, vs 192 s from 0.77 ms x 249,856 tokens |
| pure decode | 391 | 32.6 % | |
| idle (running 0, util < 50 %) | 7 | 0.6 % | plus 3 s busy with running=0 (stale gauge) |

GPU time split (INFERRED): prefill ≈ 200 s long + 150-190 s short ≈ **30-33 % of wall**, decode ≈ 67-70 %,
idle < 1 %. Under the naive counter definition, 574 s (47.8 %) are "prefill-bearing". Prompt tokens per second in
those seconds = 6.80M / 574 = 11.9k tok/s, which is meaningless because 94 % of those tokens are cached. Compute
tokens per second = 396k / 574 = 690 tok/s. Effective prefill rates: **long turns 658 tok/s at a median context of
88k** (alone or with one co-runner); short prompts ~1,300-1,600 tok/s (GEMM-bound, 128-token chunks).

## 3. Prefix cache

Window prompt tokens 6,804,540 = 396,204 computed (5.8 %) + 4,056,832 local GPU hits (59.6 %) + 2,351,504 from the
CPU/fs tier (34.6 %). **Hit ratio 94.2 %.** Long turns: 98.6 % (138k computed of ~6.49M). Short requests: 18 %.
21 of the 67 long turns reloaded their whole prefix (~110k tokens each, 2.32M in total) from the offload tier,
because short requests had evicted it from the GPU. Those turns stalled 2.9 s vs 3.3 s for GPU hits, so with
`sync_load` the reload cost does not show at 1 Hz. Without the cache, a 97k turn costs a x 97k + b x 97k²/2 ≈ 116 s,
so the 67 turns would need ~7,700 s of prefill against the ~200 s measured: 38x more, impossible in a 1,200 s
window. The cache, including the CPU tier, is what makes the long agent viable on this card.

## 4. Capacity: KV, waiting, preemptions

- `max_num_seqs` 16 was never reached (running max 8). **KV is the binding limit.** Waiting > 0 in 450 s
  (37.5 %), all after minute 7. Median kv_usage while requests waited was 0.951 (p10 0.754: a long prompt needs
  ~40-55 % of the pool at once). kv ≥ 0.95 for 262 s.
- Two long streams hold 90k + 135k ≈ 225k tokens, 93 % of the 241k pool. For 63 s, 2 sequences ran (12 rows,
  ~63 ms/step) while 5-7 short requests waited.
- 11 preemptions in the window (17 lifetime), all at kv ≥ 0.88 with 3-8 running.
- This costs throughput, not ms/step. The levers are config (KV pool size, admission policy, CPU-tier sizing):
  propose only.

## 5. GPU: power, utilization, clock

| class | util mean | power median / p90 / max | SM clock median / p10 | temp |
|---|---|---|---|---|
| long prefill | 87 % | 341 / 347 / 349 W | 1845 / 1785 MHz | 63 °C |
| short-prefill seconds | 76 % | 338 / 345 / 349 W | 1800 / 1710 MHz | 62 °C |
| pure decode | 76 % | 336 / 345 / 348 W | 1845 / 1755 MHz | 62 °C |
| clean decode n=1 / n=2 / 7-8 | 64 / 79 / 76 % | 324 / 338 / ~340 W | 1890 / 1845 / 1860 MHz | |

The power limit is not in the export. The box's stock limit was 350 W and the window maximum is 349.4 W. Prefill
runs at the cap (clock dips to 1710-1785 at p10). Decode does not: util is 64-79 %, so the GPU has no kernel
running for 21-36 % of each decode second, and the clock (1845-1890) is higher than the box's decode clock
(1725-1740 at 342-344 W). Decode is **under-utilized, not power-capped**: the losses are host gaps and
memory-bound kernels, not frequency. Only 29 % of seconds reach util ≥ 90 %, 40 % reach ≥ 340 W, and fan 68 %
at 62-64 °C is not thermal.

## 6. Prod ms/step vs the box ladder

Box model at short context (INFERRED). Target cost by rows T(r) comes from the no-MTP ladder (1/2/4/8 rows:
19.6/20.4/22.7/26.9 ms). One draft pass costs D = (MTP ms − T(4c)) / 3 = 1.73 ms at c=1 and 1.57 ms at c=2;
I use 1.65. T(16) and T(32) come from the k=3 MTP ladder: 35.7 − 3D = 30.7 and 44.2 − 3D = 39.2. The model then
reproduces the k=3 ladder (27.6/31.8/35.6/44.1 vs 27.9/31.6/35.7/44.2). For production's schedule,
box(n) = T(n(k+1)) + kD gives 33.0/37.0/40.0/43.1 at n=1-4 (k=5) and 37.7/39.8/42.0/44.1 at n=5-8 (k=3).

Split of the excess (INFERRED). nvidia-smi util gives the busy fraction u. GPU-busy ms = u x ms/step and idle
ms = (1−u) x ms/step, compared with the box's 88 % decode util.

| ms/step | box ladder k=3 | (a) rows + draft passes (k=5) | (b) long-context GPU work | (c) prefill interleave | (d) host idle above box | prod |
|---|---|---|---|---|---|---|
| n=1, 122k ctx | 27.9 | +5.1 | +8.6 (0.070 ms per 1k ctx) | 0 (clean) | +17.1 (21.1 idle vs 4.0) | **58.8** |
| n=2, 204k ctx | 31.6 | +5.4 | +17.7 (0.087 ms per 1k) | 0 (clean) | +9.0 (13.4 vs 4.4) | **63.6** |
| n=7-8, 229k ctx | 44.2 | −2.2 (mostly 7 running) | +25.1 (0.11 ms per 1k) | +10.8 (92.5 vs 81.7 decode-only) | +14.6 (19.6 vs 5.0) | **81.7 / 92.5** |

- (a) More rows explain only 5 ms (+16 % on the box, +8-9 % of the production step).
- (b) Context-proportional GPU work: the n=2 halves split by context give 61.7 ms at 154-204k and 65.5 ms at
  204-237k, about 0.09 ms per 1k tokens, consistent with the util split. HBM floor: (32 + 2k) KiB per context
  token per step / 936 GB/s = 0.046 ms per 1k at k=5 (0.042 at k=3). Attention therefore runs at ~40-65 % of the
  bandwidth floor. This is vLLM's attention backend (FlashInfer on this box), not plugin code. GDN state does not
  depend on context and is already in the box numbers (k=5 writes more per-position states, part of (a)).
- (c) Long turns stall co-running decode to ~200 ms steps for 16.6 % of wall time (section 2). Short prefills add
  10.8 ms to the average step at 7-8 running.
- (d) Host idle is 9-17 ms per step above the box. It does not scale with context (n=1 has more idle than n=2), so
  it is not attention. Candidates: vLLM main vs 0.27.1 per-step host path, the KV-offload connector working in the
  engine-core process (write-back/read threads, `sync_load`, 2.35M tokens loaded in the window), and per-step
  attention metadata at 848-token blocks. None of these can be told apart from 1 Hz data. nvidia-smi util is a
  coarse sampler, so treat (b)/(d) as ±3 ms.

## 7. Round-3 targets, ranked by expected ms/step at production's operating points

1. **Host idle in the decode step: −9 to −17 ms/step** (n=1: 58.8 -> ~42; n=2: 63.6 -> ~55; 7-8: −15). The owner
   is unknown: vLLM main, the KV-offload connector (overlay), or attention metadata. Measurement: on the box, run
   production's main argv with two 100k-context replay streams at c=1/2, with nsys + py-spy on the engine core.
   A/B: OffloadingConnector on/off, k=5 vs k=3, async scheduling off as in production. First step is the planned
   box re-benchmark on main (STATUS "Next").
2. **Long-context attention: 8.6 / 17.7 / ~25 ms/step at 122k / 204k / 229k; ~2 / 7 / 14 ms recoverable** if the
   kernel reached ~85 % of HBM bandwidth. Not ours (vLLM/FlashInfer decode with q_len 4-6 and fp8 KV, plus one MTP
   attention read per draft pass). Measurement: torch profiler at c=2 with 2 x 100k contexts. Attention kernel time
   vs bytes per layer; whether split-KV is used; whether each draft pass re-reads the full MTP-layer KV.
3. **Prefill, ~30 % of GPU time.** Long turns: 0.77 ms/token of GEMM (ours: Route L at 128-row chunks, about
   70 TFLOP/s effective) plus 0.84 ms/token of attention at 97k (vLLM). Short prompts: GEMM-bound, +10.8 ms/step
   averaged over 7-8 running. Halving the 128-row GEMM cost saves ~0.4 ms per long-turn token (~55 s/window) and
   ~5 ms/step at 7-8. Measurement: box microbench of the R2/MMQ products at 128-160 rows (a chunk plus decode rows),
   and a test-only A/B of threshold 128 / 256 / 512 on an agent replay. Bigger chunks cut the 13 ms weight stream
   paid per chunk step (~13 % of a 128-token chunk) but lengthen each decode stall.
4. **MTP k schedule (test only).** Extra drafts pay more on production's long steps than on the box. k=3 -> 5 costs
   ~5 ms (+8-9 %) and gives +27 % tokens/step. At 5-8 running, k=3 -> 4 would cost ~6 ms (rows 28 -> 35 plus one
   pass, +7 %) for ~+15 % tokens (pos3 ≈ 0.45), net ~+8 % tok/s. A 6th draft at 1-4 running: ~+2 ms (+3.6 %) for
   ~+8 %. Measurement: same-session box A/B on replayed production traffic, acceptance by position.
5. **9-32-row kernels (ours, K3/R2 at 12/18/24/28/32 rows): ~3-8 ms/step.** On the box, GEMM above the 13 ms floor
   is ~11 ms at 16 rows and ~17 ms at 32 rows. In production this is ~10 % of the step, behind items 1-3.
   Measurement: per-shape microbench at production row counts, ms vs floor.
6. **Draft head size: ~1 ms/step.** A pass costs ~1.65 ms, of which the 61,440-row lm_head is a part. Measurement:
   offline acceptance at 32k/48k/61k draft vocab plus per-pass timing.
7. **KV capacity (throughput, not ms/step).** 37.5 % of the window was KV-bound at 2-8 running. Config only:
   propose, do not change.

Not levers here: idle seconds (0.6 %), power cap during decode, thermals.
