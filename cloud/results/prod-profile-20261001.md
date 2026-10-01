# Production profile, 2026-10-01 03:42-04:33 EDT (ms4, RTX 3090, GSQ-RCO GGUF on vLLM main)

Sources: two contiguous 1 Hz exports, `~/Downloads/vllm-1hz-20261001-0342.json` (03:42:42-04:02:42) and
`~/Downloads/vllm-1hz-20261001-040243.json` (04:02:43-04:33:21). Together they hold 3,040 vLLM `/metrics` samples
(3,039 one-second deltas, all dt = 1 s, no counter resets) and 3,038 nvidia-smi samples (one missing second at
04:17:52), 50.6 min in total. One read of production's `/metrics` was used for counter semantics and lifetime
histograms. The analysis is read-only and ran on CPU. The box numbers come from `REPORT.md` (vLLM 0.27.1, k=3,
short prompts). Production on main has not been benchmarked on the box yet (STATUS "Current state"), so every box
comparison below also includes the 0.27.1 -> main change. Labels: MEASURED (from the exports), INFERRED (model or
reasoning), SOURCE (read in vLLM code).

## 0. What the counters mean (and one trap)

- Argv (`env/prod-main-serve-argv.txt`): `num_speculative_tokens_per_batch_size [[1,4,5],[5,8,3],[9,16,2]]`.
  k is chosen from the number of scheduled requests (SOURCE: `scheduler.py`, `dynamic_sd_lookup[len(num_scheduled_tokens)]`):
  k=5 (6 rows per sequence) at 1-4 running, k=3 (4 rows) at 5-8, k=2 (3 rows) at 9-16. MEASURED: `draft_tokens/drafts`
  is exactly 5 in 612 s, 3 in 1,756 s and 2 in 132 s. Running peaked at 9. At 9 running the ratio is often
  2.0-2.2, because the scheduled count crosses 8/9 within the second.
- `prompt_tokens`, `prompt_compute` (local_compute), `prompt_cache` and `prompt_external` grow by the **whole
  prompt** in the step that emits the first token (SOURCE: `v1/metrics/stats.py` `update_from_output`,
  `is_prefilling` -> `PrefillStats`). A long prefill runs for seconds, but its counters jump in a single second.
  "Seconds where prompt tokens increased" are completion seconds, not prefill seconds. Section 2 uses a duration
  model and checks it against the measured decode stalls.
- The `running` gauge reads 0 for 1-3 s while the GPU runs at 98-100 % / ~335 W right before each of 20
  long-prompt completions (MEASURED, first hour slice). Every one of these stretches ends in a long prefill.
  INFERRED: the gauge only refreshes when some request emits tokens, so it is stale during prefill-only stretches.
  I count those seconds as prefill.
- KV pool (`cache_config_info`): 241,245 tokens, 310 blocks of 848 tokens, fp8, `kv_cache_max_concurrency` 1.21 at
  200k. `kv_usage x 241,245` tracks the context of the running sequences (one sequence at 0.34 -> 82k tokens,
  matching its next prompt of 75-79k), so I use it as "context in the batch".
- Model: 64 layers, 16 full-attention layers (interval 4), 4 KV heads x 256. fp8 KV is 16x4x256x2 B = 32 KiB per
  token for the target, plus 2 KiB per token per MTP draft pass. GDN state is 48x48x128x128x2 B = 75.5 MB per
  sequence and does not grow with context.

## Workload

- **0-7 min:** 1-2 long-context agent streams only, never more than 2 running.
- **7-50 min:** the long streams plus short requests (prompts 200-1,400 tokens, 24 % cache hit), 4-9 running.

Totals: 154 long turns (prompt ≥ 20k): median 75k, mean 77k, range 30k-148k, median 1,825 new (computed) tokens
per turn. 2,839 requests finished and 547k tokens were generated (180 tok/s mean). The lifetime histogram from the
scrape: 1,049 of 1,313 prompts are under 500 tokens and 182 are over 50k. The "57k mean" in the memory note is the
long turns' mean, not the mean over all requests.

| min | run mean | wait mean/max | kv mean/max | prompt k | compute k | ext k | gen tok/s | done | preempt | util % | W med | MHz med |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0-5 | 1.5 | 0.0/0 | 0.63/0.88 | 2635 | 55 | 0 | 68 | 28 | 0 | 75 | 336 | 1845 |
| 5-10 | 4.4 | 0.9/7 | 0.81/1.00 | 2228 | 98 | 761 | 127 | 176 | 6 | 81 | 338 | 1845 |
| 10-15 | 6.4 | 1.7/7 | 0.67/1.00 | 1356 | 116 | 1152 | 208 | 363 | 3 | 79 | 339 | 1830 |
| 15-20 | 7.0 | 1.3/3 | 0.81/1.00 | 586 | 127 | 439 | 204 | 378 | 2 | 78 | 339 | 1815 |
| 20-25 | 6.3 | 2.4/4 | 0.94/1.00 | 530 | 94 | 435 | 213 | 243 | 2 | 84 | 341 | 1800 |
| 25-30 | 6.4 | 1.8/4 | 0.78/1.00 | 554 | 101 | 453 | 220 | 302 | 12 | 81 | 338 | 1815 |
| 30-35 | 5.8 | 2.7/5 | 0.87/1.00 | 1352 | 109 | 1174 | 179 | 312 | 9 | 81 | 338 | 1815 |
| 35-40 | 5.5 | 3.1/5 | 0.91/1.00 | 989 | 95 | 856 | 178 | 264 | 5 | 82 | 338 | 1815 |
| 40-45 | 8.3 | 0.3/5 | 0.76/0.99 | 804 | 163 | 43 | 200 | 316 | 0 | 84 | 341 | 1800 |
| 45-50.6 | 8.5 | 0.0/0 | 0.79/0.96 | 1820 | 164 | 90 | 202 | 457 | 0 | 82 | 339 | 1815 |

## 1. Steps/s and ms/step by running count

Method. In a second where all n running sequences decode, each engine step adds one draft per sequence, so
steps/s = Δdrafts / n and ms/step = 1000 n / Δdrafts. Tokens per step = n x (1 + Δaccepted/Δdrafts). Seconds used
must have the same running count at both ends, no preemption, and Δdraft_tokens = k x Δdrafts for the scheduled
k. At n=9 I accept a ratio of 2-3, because k flips with the scheduled count. Three classes:
- **clean:** no prompt completion, outside the long-prefill mask (section 2), no request finished in this second or
  the one before.
- **decode-only:** clean, but finished requests are allowed.
- **with short prefills:** also allows seconds with short-prompt completions; this is what decode sees in the mixed
  regime.

Medians are quantized: at n=1, 17 vs 16 drafts/s is 58.8 vs 62.5 ms. The pooled value (Σ n / Σ Δdrafts) is the
precise one.

Cross-checks. (i) Δgen = 547,496 vs Δdrafts + Δaccepted = 548,929: 2.947 vs 2.955 tokens per sequence-step. The
0.3 % gap is accepted tokens dropped at stop strings, net of first tokens. So the gen counter gives the same step
rate. (ii) Mean interval between one sequence's steps = Σ running-seconds / Σ drafts = 98.8 ms over the 50.6 min
(prefill stalls included). Lifetime `inter_token_latency` = 79.3 ms, and its count equals lifetime drafts, so the ITL
histogram measures the same per-sequence step.

| running | k (rows/seq) | rows | clean: s, median / p90 / **pooled** | decode-only: s, pooled | with short prefills: s, median / p90 / pooled | tok/step (per seq) | accepted/draft (rate) | ctx median | box model |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 5 (6) | 6 | 59, 58.8 / 71.4 / **58.8** | 62, 58.7 | 62, same | 3.19 (3.19) | 2.19/5 (0.44) | 122k | 33.0 |
| 2 | 5 (6) | 12 | 156, 62.5 / 71.4 / **63.6** | 159, 63.7 | 159, same | 7.79 (3.89) | 2.90/5 (0.58) | 204k | 37.0 |
| 3 | 5 (6) | 18 | 0 | 4, 63.5 | 5, 62.5 / 75.3 / 66.4 | 11.0 (3.68) | 2.68/5 (0.54) | 168k | 40.0 |
| 4 | 5 (6) | 24 | 18, 71.4 / 76.9 / **71.4** | 47, 75.0 | 80, 80.0 / 97.6 / 81.6 | 15.0 (3.75) | 2.75/5 (0.55) | 234k | 43.1 |
| 5 | 3 (4) | 20 | 27, 62.5 / 66.7 / **64.4** | 55, 69.0 | 114, 78.1 / 108 / 77.1 | 15.5 (3.10) | 2.10/3 (0.70) | 229k | 37.7 |
| 6 | 3 (4) | 24 | 27, 62.5 / 162 / **70.2** | 84, 79.0 | 181, 82.2 / 171 / 88.2 | 17.6 (2.93) | 1.93/3 (0.64) | 233k | 39.8 |
| 7 | 3 (4) | 28 | 22, 66.7 / 76.4 / **69.0** | 90, 80.9 | 369, 93.3 / 156 / 92.3 | 20.6 (2.94) | 1.94/3 (0.65) | 237k | 42.0 |
| 8 | 3 (4) | 32 | 12, 66.7 / 179 / **72.8** | 51, 73.5 | 248, 86.0 / 150 / 85.7 | 24.0 (3.00) | 2.00/3 (0.67) | 195k | 44.1 |
| 9 | 2 (3) | 27 | 11, 141 / 188 / 91.2 (noisy) | 28, **80.5** (median 73.2) | 169, 87.4 / 161 / 85.9 | 22.7 (2.53) | 1.53/2 (0.76) | 191k | 39.8 |

Notes:
- 3 running stays thin (37 s in total, almost all of them transitions).
- 5-8 running cluster at 62-73 ms clean, and every bin pays ~10-20 ms more once short prefills interleave.
- At 9 running, k=2 gives **fewer tokens per step than k=3 at 8 running** (22.7 vs 24.0) at a similar step time.
  Per sequence that is 2.53 tokens vs 3.00.
- Acceptance by k over the whole sample (MEASURED): k=5 2.80/5 (0.561), k=3 2.03/3 (0.676), k=2 1.48/2 (0.741).
- Lifetime per-position acceptance (scrape; solved assuming only k=5 and k=3 had run by then, so INFERRED): pos0-2
  0.84 / 0.67 / 0.53 per draft, pos3 0.45 and pos4 0.37 per k=5 draft. That gives 3.85 tokens per sequence-step at
  k=5 vs 3.04 at k=3 (+27 %).

## 2. Wall time: pure decode vs prefill-bearing

Prefill duration model (INFERRED from the box prefill ladder 1248/954/644 tok/s at 8k/64k/180k): time(N) = aN + bN²/2
with a = 0.77 ms/token and b = 8.7 ns/token², so the marginal cost at context C is 0.77 ms + 8.7 ns x C per token
(1.43 ms at 77k: about half GEMM, half attention). Each long completion masks the ceil(compute x marginal(C))
seconds before it.

Check: decode stalls before 109 of the 154 long completions add up to 310.5 s for their computed tokens
(726 tok/s), against 344 s from the model, so production runs ~10 % faster than the box model. The other 45 stalls
cannot be seen because they happen at 7-9 running, where one stalled stream hides among the others.

| class (3,039 s) | seconds | share | GPU time inside it |
|---|---|---|---|
| long-turn prefill (masked) | 712 | 23.4 % | model 504 s for all 154 turns, ~455 s at production's measured speed = **~15 %** of wall. The co-running decode drops to ~200 ms/step |
| short-prompt completions in the second | 1,515 | 49.9 % | short prefill: 365 s (residual after modeled decode) to 553 s (0.77 ms x 718k tokens) = **12-18 %** |
| pure decode | 802 | 26.4 % | |
| idle (running 0, util < 50 %) | 7 | 0.2 % | plus 3 s busy with running=0 (stale gauge) |

GPU time split (INFERRED): prefill ≈ **27-33 %**, decode ≈ 67-72 %, idle < 0.5 %. Under the naive counter
definition, 1,808 s (59.5 %) are "prefill-bearing". Prompt tokens per second in those seconds = 7.1k tok/s,
meaningless because 91 % of those tokens are cached. Compute tokens per second = 621 tok/s. Effective prefill rates:
**long turns 726 tok/s** at a median context of ~75k; short prompts ~1,300-2,000 tok/s (GEMM-bound, 128-token
chunks).

## 3. Prefix cache

Prompt tokens 12,853,009 = 1,123,473 computed (8.7 %) + 6,326,080 local GPU hits (49.2 %) + 5,403,456 from the
CPU/fs tier (42.0 %). **Hit ratio 91.3 %.** Long turns: 97.0 %. Short requests: 23.7 %. 60 of the 154 long turns
reloaded their prefix from the offload tier (5.19M tokens), because short requests had evicted it from the GPU.
Those turns stalled 2.9 s vs 2.8 s for GPU hits, so with `sync_load` the reload cost does not show at 1 Hz. Without
the cache, a 77k turn costs aN + bN²/2 ≈ 85 s, so the 154 turns would need ~13,000 s of prefill in a 3,039 s window.
The cache, including the CPU tier, is what makes the long agent viable on this card.

## 4. Capacity: KV, waiting, preemptions

- `max_num_seqs` 16 was never reached (running max 9). **KV is the binding limit.** Waiting > 0 in 1,452 s
  (47.8 %), all after minute 5. Median kv_usage while requests waited was 0.964 (p10 0.883). Waiting happened only
  at 1-8 running, and kv ≥ 0.95 held for 970 s.
- 9 running occurs only with KV free (median 0.825) and nobody waiting: at 9 the limit is demand.
- Two long streams can hold ~225k tokens, 93 % of the 241k pool. For 63 s, 1-2 sequences ran while 5-7 short
  requests waited.
- 39 preemptions (kv ≥ 0.68, 3-8 running; 21 in minutes 25-40).
- This costs throughput, not ms/step. The levers are config (KV pool, admission policy, CPU-tier sizing):
  propose only.

## 5. GPU: power, utilization, clock

| class | util mean | power median / p90 / max | SM clock median / p10 | temp |
|---|---|---|---|---|
| long prefill | 87 % | 342 / 346 / 352 W | 1845 / 1755 MHz | 63 °C (max 64) |
| short-prefill seconds | 79 % | 338 / 345 / 349 W | 1800 / 1725 MHz | 62 °C |
| pure decode | 79 % | 338 / 345 / 348 W | 1830 / 1740 MHz | 62 °C |
| clean decode n=1 / 2 / 4 / 5-7 / 8 / 9 | 64 / 79 / 78 / 82 / 79 / 80 % | 324-341 W | 1785-1890 MHz | |

The power limit is not in the exports. The box's stock limit was 350 W and the sample maximum is 352.2 W.
Prefill runs at the cap: clock p10 1725-1755. Decode does not: util is 64-83 %, so the GPU has no kernel running
for 17-36 % of each decode second, and its clock (1785-1890) is above the box's decode clock (1725-1740 at
342-344 W). Decode is **under-utilized, not power-capped**: the losses are host gaps and memory-bound kernels, not
frequency. Over the whole sample, util averages 80.7 %, only 30 % of seconds reach ≥ 90 % util, 41 % reach
≥ 340 W, and fan 68 % at 62-64 °C is not thermal.

## 6. Prod ms/step vs the box ladder

Box model at short context (INFERRED). Target cost by rows T(r) comes from the no-MTP ladder (1/2/4/8 rows:
19.6/20.4/22.7/26.9 ms). One draft pass costs D = (MTP ms − T(4c)) / 3 = 1.73 ms at c=1 and 1.57 ms at c=2; I use
1.65. T(16) and T(32) come from the k=3 MTP ladder: 35.7 − 3D = 30.7 and 44.2 − 3D = 39.2. The model then
reproduces the k=3 ladder (27.6/31.8/35.6/44.1 vs 27.9/31.6/35.7/44.2). For production's schedule,
box(n) = T(n(k+1)) + kD (last column of the section 1 table).

Split (INFERRED). nvidia-smi util gives the busy fraction u. GPU-busy ms = u x pooled ms/step and idle ms =
(1−u) x pooled ms/step. Box decode util is 88 %, so box idle = 0.12 x box(n). The idle 90 % intervals come from a
bootstrap over seconds.

| ms/step (clean) | box ladder k=3 | (a) rows + draft passes | (b) long-context GPU work | (d) host idle above box | prod clean | (c) short prefills interleaved | prod with prefills |
|---|---|---|---|---|---|---|---|
| n=2, 204k ctx | 31.6 | +5.4 (k=5) | +17.8 (0.088 ms per 1k ctx) | +8.7 (idle 13.2 [12.3-14.2]) | **63.6** | +0.1 | 63.7 |
| n=4, 234k ctx | 35.7 | +7.4 (k=5) | +17.9 (0.076 ms per 1k) | +10.4 (idle 15.6 [13.1-18.1]) | **71.4** | +10.2 | 81.6 |
| n=8, 195k ctx | 44.2 | −0.1 (k=3) | +18.6 (0.095 ms per 1k) | +10.1 (idle 15.4 [10.0-20.7]) | **72.8** | +12.9 | 85.7 |

**Idle per step does not scale with running count.**

| n | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| idle ms/step | 21.2 | 13.2 | 18.6 | 15.6 | 11.5 | 12.2 | 12.3 | 15.4 | 16.2 |

The linear fit is 17.6 − 0.49 n ms (flat to falling). Idle also does not depend on offload-tier activity: median
idle is 12.5 ms in decode seconds within ±10 s of a CPU/fs-tier load and 13.7 ms elsewhere. So the gap is a
**fixed per-step host cost of ~12-16 ms (7-10 ms above the box at n ≥ 2)**. It is not per-request scheduler work,
which would grow with n, and not connector transfer work. There is also no clear k trend: k=5 bins 13-16 ms
(21 at n=1), k=3 bins 11-15, k=2 16. Remaining suspects: a fixed per-step cost in vLLM main's engine loop or model
runner (the box ran 0.27.1 with the same connector and `--no-async-scheduling` and had ~4 ms), or the
connector's per-step polling. n=1 sits in minutes 2-4, while the other stream's 75-110k prompts are being prepared.
nvidia-smi util is a coarse sampler, so a per-bin idle value is good to about ±2-5 ms (see the intervals).

- (a) More rows and draft passes explain ≤ 7 ms; at n=8 (k=3) they explain nothing.
- (b) Context-proportional GPU work is the largest single term at every n ≥ 2: 18-19 ms at 195-234k context,
  0.076-0.095 ms per 1k tokens. HBM floor: (32 + 2k) KiB per context token per step / 936 GB/s = 0.046 ms per 1k at
  k=5 and 0.042 at k=3. Attention therefore runs at ~45-60 % of the bandwidth floor. This is vLLM's attention
  backend (FlashInfer on this box), not plugin code. GDN state does not depend on context and is already in the box
  numbers.
- (c) Short prefills add 10-13 ms to the average step at 4-8 running. Long turns stall co-running decode to
  ~200 ms steps for ~15 % of wall time.
- At n=9 (dec 80.5, 191k): box 39.8 + (b) ~16-18 + idle 16.2. The rest (~5-10 ms) is noise in a 28-s bin.

## 7. Round-3 targets, ranked by expected ms/step at production's operating points

1. **Host idle in the decode step: −7 to −10 ms/step at n ≥ 2** (−17 at n=1); every step pays it, at every running
   count. Section 6 shows it is a fixed per-step cost, not per-request and not driven by connector transfers. The
   owner is unknown: vLLM main's engine loop/runner, or connector polling. Measurement: on the box, run
   production's main argv with two 100k-context replay streams at c=2 and c=8, with nsys + py-spy on the engine
   core. A/B: OffloadingConnector on/off (polling cost), k=5 vs k=3, 0.27.1 vs main with the same argv. First step
   is the planned box re-benchmark on main (STATUS "Next").
2. **Long-context attention: 18-19 ms/step at 195-234k context, of which ~5-9 ms is recoverable** if the kernel
   reached ~85 % of HBM bandwidth. This is the largest single term, but not ours (vLLM/FlashInfer decode with
   q_len 3-6 and fp8 KV, plus one MTP attention read per draft pass). Measurement: torch profiler at c=2 and c=8
   with ~200k total context. Attention kernel time vs bytes per layer; whether split-KV is used; whether each
   draft pass re-reads the full MTP-layer KV.
3. **Prefill, 27-33 % of GPU time; +10-13 ms/step on decode at 4-8 running.** Long turns: 0.77 ms/token of GEMM
   (ours: Route L at 128-row chunks, about 70 TFLOP/s effective) plus attention at ~77k (vLLM). Short prompts are
   GEMM-bound. Halving the 128-row GEMM cost saves ~5-6 ms/step at 4-8 running and ~0.4 ms per long-turn token.
   Measurement: box microbench of the R2/MMQ products at 128-160 rows (a chunk plus decode rows), and a test-only
   A/B of threshold 128 / 256 / 512 on an agent replay.
4. **MTP k schedule (test only); the larger sample adds the 9-running case.**
   - At 9 running, k=2 gives 2.53 tokens per sequence-step vs 3.00 at k=3 (n=8). k=3 at 9 running would be 36 rows:
     ~+6 ms on ~80 (+8 %) for ~+20 % tokens.
   - At 5-8 running, k=3 -> 4 would cost ~6 ms (+7 %) for ~+15 % tokens (pos3 ≈ 0.45).
   - A 6th draft at 1-4 running: ~+2 ms for ~+8 %.
   - Measurement: same-session box A/B on replayed production traffic, acceptance by position.
5. **9-36-row kernels (ours, K3/R2 at production's 12-32 rows): ~3-8 ms/step.** On the box, GEMM above the 13 ms
   floor is ~11 ms at 16 rows and ~17 ms at 32 rows. In production this is ≤ 10 % of the step. Measurement:
   per-shape microbench at 12/18/24/27/28/32/36 rows, ms vs floor.
6. **Draft head size: ~1 ms/step.** Measurement: offline acceptance at 32k/48k/61k draft vocab plus per-pass
   timing.
7. **KV capacity (throughput, not ms/step).** 47.8 % of the sample was KV-bound at 1-8 running, with 39
   preemptions. Config only: propose, do not change.

Changes from the 20-min version:
- The host-idle range narrows to 7-10 ms at n ≥ 2 (per-bin ±2-5 ms), and it is shown to be flat in n. It stays #1
  because every step pays it and it is recoverable in full.
- Attention is the largest single term but only about half recoverable, and not ours.
- New: k=2 at 9 running loses tokens per step, so it joins the k-schedule test list.

Not levers here: idle seconds (0.2 %), power cap during decode, thermals.
