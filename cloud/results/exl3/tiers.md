# EXL3 tiers (3.0 / 3.5 / 4.0 / 4.5 bpw), concat probe, switch test

Source: `cloud/results/exl3-opt-dod/21-tier-*-k3`, `concat-probe/`, `12-mr-ladder-k3/`, and the switch-test stdout in `~/vast-archive/workspace/gpuq/out/1791183442573746-exl3dod-switch.log`.
All tiers: RTX 3090, MTP k=3, `EXL3_MR=2 EXL3_EMBED_HOST=1 EXL3_DRAFT_FP8=1 EXL3_MR_CONCAT=0`, vLLM 0.30.1rc1.dev285, 0.94 GPU memory utilization, fs KV tier bounded to 10 GB.

## Table

ms/step and tok/s are from production `run_benchmarks.sh single` pass 2 (pass 1 is warm-up), the T=0 rows. tok/s is decode (C/meanTPOT), aggregate over c requests. ms/step = c x tok/step / tok/s.

| tier | weights on GPU | load time | KV tokens (ctx) | c=1 | c=2 | c=4 | c=8 | MTP acceptance (mean len) | parity |
|---|---|---|---|---|---|---|---|---|---|
| SC_3.00bpw_H4_V4 | 9.89 GiB | 4.7 s load, 88.5 s engine init | 326,119 (196,608) | 98.6 tok/s, 26.27 ms | 196.7, 26.54 ms | 318.7, 32.51 ms | 500.3, 41.58 ms | 0.514 (2.543) | FAIL (top-1) |
| SC_3.50bpw_H4_V6 | 11.4 GiB | 5.0 s | 281,648 (196,608) | 96.2, 26.82 ms | 194.7, 26.91 ms | 330.0, 31.76 ms | 532.3, 38.47 ms | 0.523 (2.568) | pass flag false |
| SC_4.00bpw_H5_V6 | 13.06 GiB (16.35 GB on disk) | 5.3 s | 234,057 (196,608) | 95.2, 28.26 ms | 181.0, 28.07 ms | 325.2, 32.35 ms | 534.0, 38.80 ms | 0.510 (2.529) | FAIL (KLD) |
| SC_4.50bpw_H5_V6 | 14.42 GiB (17.87 GB on disk) | 5.4 s (second launch) | 193,291 (192,512) | 86.3, 29.90 ms | 173.2, 30.02 ms | 295.9, 34.34 ms | 495.0, 40.73 ms | 0.517 (2.550) | pass flag false |

Per-position acceptance (pos 1/2/3): 3.00 0.715/0.493/0.335; 3.50 0.722/0.502/0.345; 4.00 0.712/0.486/0.330; 4.50 0.718/0.493/0.339.
Prefill 8192 tokens, c=1: 1152 / 1143 / 1133 / 1141 tok/s (3.00 / 3.50 / 4.00 / 4.50).
Load time is the vLLM "Model loading took" figure; the harness only logged "healthy after 120 s" for every tier (poll cap, not a measurement).
On-disk size is known only for 4.00 and 4.50 (dl.log has byte counts); the 3.00 and 3.50 dl.log lines are "ok (cached)" with no sizes.

## Parity vs exllamav3

Gates (from parity.json): kld_mean <= 0.001 and top1 >= 0.99. All four result files report `"pass": false`.

| tier | KLD mean | top-1 | positions | gate result |
|---|---|---|---|---|
| 3.00 | 0.000400 | 0.9873 | 864 | KLD ok, top-1 below 0.99, fail |
| 3.50 | 0.000483 | 0.9965 | 864 | both gates met numerically, flag still false |
| 4.00 | 0.001318 | 0.9931 | 864 | KLD above 0.001 (seq_005 max 0.7466), fail |
| 4.50 | 0.000449 | 0.9907 | 864 | both gates met numerically, flag still false |

Per sequence (KLD mean / p99 / max, top-1):

| tier | seq_002 code 1536 tok | seq_003 code 4096 tok | seq_005 prose 8192 tok |
|---|---|---|---|
| 3.00 | 0.00031 / 0.00704 / 0.0131, 0.9896 | 0.00049 / 0.00593 / 0.0343, 1.0000 | 0.00039 / 0.00273 / 0.0087, 0.9722 |
| 3.50 | 0.00030 / 0.00402 / 0.0073, 1.0000 | 0.00079 / 0.01117 / 0.0740, 1.0000 | 0.00036 / 0.00307 / 0.0042, 0.9896 |
| 4.00 | 0.00034 / 0.00450 / 0.0152, 0.9965 | 0.00057 / 0.00677 / 0.0446, 0.9896 | 0.00305 / 0.00528 / 0.7466, 0.9931 |
| 4.50 | 0.00024 / 0.00208 / 0.0027, 1.0000 | 0.00067 / 0.01138 / 0.0313, 0.9931 | 0.00044 / 0.00593 / 0.0210, 0.9792 |

## Incomplete files

All four tiers have the same gap: of 11 parity sequences, only seq_002, seq_003 and seq_005 produced output. seq_000, 001, 004, 006 to 010 are logged "missing output, skipped", and `long_kld_mean` / `long_top1` are null. The parity verdicts above therefore rest on 864 positions from 3 sequences, and no tier ran the long-context parity. I did not verify why the 3.50 and 4.50 flags are false despite meeting both stated gates; the missing long metrics are the likely cause but that is an inference.
Other gaps: 3.00 and 3.50 have no on-disk byte sizes. 4.50 load.txt contains two launches: the first (pid 408192) stops after the CUDA-graph-memory lines with no KV size line, the second (pid 409123) completed; 4.50 also ran at a smaller context (192,512 vs 196,608) to fit. The 4.00 sequence 005 outlier (max KLD 0.7466, p99 only 0.005) is one position, not a tier-wide shift. argv.txt in each tier dir is a dry-run diff against production, not the launch record.

## Concat probe (EXL3_MR_CONCAT)

2hfc loaded. It was run on the 3.50 tier in `12-mr-ladder-k3/mr2hfc` (healthy after 125 s, model loading 11.41 GiB in 5.3 s, KV 276,967 tokens at 196,608 ctx) and also snapshotted in `concat-probe/`.

| mode | c=1 | c=2 | c=4 | c=8 | acceptance |
|---|---|---|---|---|---|
| 2hf | 97.1 tok/s, 26.78 ms | 196.7, 26.84 ms | 322.3, 32.02 ms | 540.2, 38.80 ms | 0.5071 |
| 2hfc | 100.9 tok/s, 26.16 ms | 200.0, 26.20 ms | 338.7, 31.06 ms | 532.6, 37.70 ms | 0.5138 |
| delta ms/step | -0.61 | -0.64 | -0.96 | -1.10 | |

2hfc is 0.6 to 1.1 ms/step faster (about 2 to 3 percent) at every concurrency. tok/s is higher at c=1/2/4 and lower at c=8 (532.6 vs 540.2); the summary's parenthesised second values (c=8: 526.7 vs 550.6) point the other way, so the c=8 tok/s difference is within noise. The cost is memory at load: the CUDA snapshot at the fp8 draft head's entry shows allocated 14.62 GiB in both modes but reserved 22.47 GiB for 2hfc vs 17.68 GiB for 2hf, with `_mr_concat` allocating 5.336 GiB during `process_weights_after_loading`. KV capacity was similar in the ladder (276,967 vs 281,648 tokens, -1.7 percent).

## Switch test

The test exited 1. Its log is 10 lines and its only verdict is:

```
# Torture switch: FAIL
| criterion | pass | value | limit |
| legs | **NO** | 3 | 0 |
| namespaces_disjoint | yes | [] | [] |
```

Reason: criterion `legs` failed, value 3 against a limit of 0. The log does not say what a "leg" is or which three legs failed; the output dir `/workspace/runs/exl3-opt-dod/20261005-091752-torture-switch` was not archived, so the cause beyond that is not recoverable from this record. `namespaces_disjoint` passed.
