# Async scheduling vs production DeepSWE: measured effect

Read-only analysis of the passive sampler files. Analysis scripts: /tmp/an/{lib,an,final}.py (final.py reproduces everything below).

## Method
- Off: `prod-deepswe-20261011` up to 2026-10-12 01:27:57 (`before-all`: ~11 h; `before-last3h`: the 3 h closest to the switch).
- On: `async-prod-20261012`, from the DeepSWE resume at 01:53:43 EDT (running went 0 to 2) through 02:34:27. That is about 41 min, with running>0 for 94.8% of samples. Sampler copy started 01:52, so the probe/torture traffic (01:31-01:48) is not in it.
- 10 s windows from the 1 s /metrics. Steps = delta of `iteration_tokens_total_count` (one per engine step). Step time = dt/steps (ms). The ITL histogram agrees (ITL sum/count ~ step time, count = num_drafts).
- Primary metric: step time over decode-only windows (constant running count, KV bin by mean `kv_cache_usage_perc`, no prefill: all steps <=16 tokens and no `local_compute` prompt tokens). Step time is unaffected by MTP acceptance. Acceptance is reported separately.
- With running=2 about 30-45% of decode-only windows are in a "slow mode" (step 100-200 ms; one request is presumably stalled on prefill or KV offload). Those windows are excluded from the table ("fast mode", step <100 ms) and reported as the slow fraction.
- GPU busy = nvidia-smi `utilization.gpu` averaged over each window (a coarse kernel-active fraction).

## Matched bins: step time, ms, median [IQR], n (fast-mode decode-only windows)
| running | KV | off all-day | off last 3h | on (async) | on vs all-day | on vs last 3h |
|---|---|---|---|---|---|---|
| 1 | <30% | 41.7 [40.2-42.7] n=142 | 42.6 n=13 | no data (n=0) | - | - |
| 1 | 30-60% | 44.8 [43.5-46.1] n=489 | 45.5 [44.4-46.5] n=84 | 44.3 [41.5-45.1] n=33 | -1.3% | -2.7% |
| 1 | >60% | 49.8 [48.1-51.0] n=324 | 47.9 [46.0-49.2] n=51 | 51.2 [47.3-57.4] n=8 | +2.9% | +7.0% |
| 2 | 30-60% | 44.8 [42.9-45.3] n=25 | 44.7 n=6 | 43.3 [42.4-44.2] n=6 | -3.4% | -3.0% |
| 2 | >60% | 50.5 [48.3-52.9] n=121 | 50.8 [48.3-53.2] n=61 | 47.0 [44.8-52.1] n=15 | -7.0% | -7.5% |

Looser binning (round(mean running), same decode-only filter, fast mode): run~1 30-60: 45.5 (n=647) vs 44.3 (n=41), -2.7%; run~1 >60: 50.0 (n=383) vs 54.1 (n=14), +8%; run~2 30-60: 45.3 (n=35) vs 44.1 (n=9), -2.6%; run~2 >60: 52.1 (n=174) vs 47.1 (n=16), -9.7%.

Off-period spread: hour-to-hour medians in the same bin move by roughly +-2-5% (last-3h vs all-day differs by -4% to +2%). Most "on" deltas are inside that spread. Only running=2 KV>60% (-7%) is larger than it, and it rests on n=15.

Slow-mode fraction at running=2 KV>60%: off 0.44 (0.31 last 3h) vs on 0.35 (n=23). Not distinguishable.

## Acceptance (separate)
| | pooled accept rate | tokens per draft | median per-window accept |
|---|---|---|---|
| off all-day | 0.727 | 3.181 | 0.738 |
| off last 3h | 0.731 | 3.192 | 0.744 |
| on | 0.698 | 3.095 | 0.725 |

Acceptance is about 3 points lower with async on (tokens/draft about -3%). Possible causes: different prompt mix after the restart or a real async/MTP effect. This cannot be separated with 41 min of data. It cancels the small per-step gain in tok/s/req: 71.7 vs 73.8 at run=1 30-60 (+3%), 65.9 vs 59.1 at run=1 >60, 60.8 vs 62.0 at run=2 >60.

## GPU busy (nvidia-smi util %, median)
| scope | off all-day | off last 3h | on |
|---|---|---|---|
| decode-only windows | 82.7 | 83.0 | 86.0 |
| all running>0 windows | 85.3 | 86.6 | 89.6 |
| run=1 30-60% / >60% | 81.2 / 83.3 | 81.2 / 83.4 | 81.5 / 86.0 (n=33/8) |
| run=2 30-60% / >60% | 79.7 / 83.8 | 80.7 / 83.4 | 87.0 / 87.1 (n=6/15) |

Util is up about 3-4 points overall and about 3-7 points at running=2, which is consistent with fewer host gaps. Power is about +3-5 W (335 to 340 W). Prefill vs decode: 54% of off windows and 61% of on windows contain prefill steps (steps with >16 tokens or prompt compute). So more than half of wall time is prefill-contaminated in both. Prefill-containing windows are noisy and were not used for step time.

## Caveats
- The after sample is small. About 40% of the 244 on-windows with running>0 are decode-only, and the cells at running=1 >60% (n=8), running=2 (n=6, 15) and KV<30% (n=0) are too thin to be conclusive.
- Traffic mix differs. The on period starts from a cold prefix cache right after a restart and a pause, with a shorter, ramping context range (KV 0.13 rising to about 0.7). The off period is 11 h of steady-state agents. The KV proxy only partly controls for this.
- Acceptance dropped about 3 points. It is not controlled.
- smi util is a coarse kernel-active fraction.
- The sampler gzip is still being written, so more data will accrue by 10:00. Rerun /tmp/an/final.py for a tighter estimate.
