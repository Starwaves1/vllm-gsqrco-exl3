# Benchmark report: Swift GSQ-RCO IQ3_S-mtp GGUF on vLLM 0.27.1 (Route L)

Final build: main (Integration 2 plus the bounded IQ3 repack, `b9cdfa5`), `VLLM_GGUF_LCPP=1`.
Speed, fit, parity and profile numbers come from Integration 2 (`4cbd091`), measured before the repack
fix. The fix changes only load-time code and gives the same bytes (section 10), so the numbers
carry over. Raw data: `cloud/results/<phase>/`; the dated write-ups are in `STATUS.md`.

## 1. Setup

| | |
|---|---|
| GPU | RTX 3090 24 GB (Vast.ai), power limit 350 W (stock), PCIe 4.0 x16, driver 595.84 (CUDA 13.2) |
| Host | Threadripper 3970X, 125 GB RAM; box-only: CPU KV tier 13 GiB (/dev/shm), fs tier 30 GB |
| Software | vLLM 0.27.1 + production's overlay `ba05ffab`, torch 2.13.0+cu130, isolated venv equal to production's freeze |
| Plugin | vllm-gguf-plugin `e2b8ad5` + our commits; vendored llama.cpp `b11211` MMVQ/MMQ (unmodified, sha256 in `VENDORED.md`); built sm_86 with CUDA 13.0 |
| Model | Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf, 12.1 GB; loads as 12.45 GiB |
| Baseline | production W4A16 (`Qwen3.8-27B-W4A16-AutoRound-fast`), same box, same argv (phase 1b) |
| Argv | production's (`env/prod-serve-argv.txt`): fp8 KV, max-model-len 200000, MTP k=3 with probabilistic draft sampling, cudagraph sizes <= 32, 2048 batched tokens, long-prefill threshold 128, prefix caching, priority scheduling. Only the model path, HF config dir, port and tier roots differ |
| Decode method | production's `run_benchmarks.sh single` (8 real prompts x 1024 tokens), pass 2, T=0; tok/s = C x 1000 / mean TPOT |
| Prefill method | salted ladder, c=1, 1 output token (`bench/speed/run.sh gsq`) |
| ms/step | C x 1000 / tok/s x tok/step (tok/step = tokens per engine step, from the MTP counters) |

Clocks during the Integration 2 ladder: decode median SM 1725-1740 MHz at 342-344 W, util 88%;
prefill 1680-1800 MHz at 346-348 W. Decode run-to-run noise is about 1% at c=1 and up to ~4% at
c=2 (the same code measured 154.2 and 160.4 tok/s in two sessions).

## 2. Decode ladder

tok/s (pass 2, T=0):

| build | c=1 | c=2 | c=4 | c=8 |
|---|---|---|---|---|
| **final (Integration 2)** | **110.3** | **192.7** | **348.1** | **541.3** |
| production W4A16 | 94.1 | 194.4 | 345.1 | 505.4 |
| ratio vs W4A16 | 1.17 | 0.99 | 1.01 | 1.07 |
| pre-campaign `f6b96bf` (phase 3 end) | 88.1 | 154.3 | 258.4 | 432.9 |
| ratio vs pre-campaign | 1.25 | 1.25 | 1.35 | 1.25 |
| phase 2 (Route L as first merged) | 77.8 | 117.0 | - | - |
| stock plugin kernels (phase 1b, greedy) | 32.4 | 38.0 | - | - |

ms per engine step (lower is better):

| build | c=1 | c=2 | c=4 | c=8 |
|---|---|---|---|---|
| **final** | **27.9** | **31.6** | **35.7** | **44.2** |
| production W4A16 | 27.6 | 27.3 | 30.0 | 41.3 |
| ratio vs W4A16 | 1.01 | 1.16 | 1.19 | 1.07 |
| pre-campaign `f6b96bf` | 33.7 | 38.5 | 47.2 | 56.4 |
| tok/step final / W4A16 | 3.08 / 2.60 | 3.04 / 2.65 | 3.11 / 2.59 | 2.99 / 2.61 |

The final build is slower than W4A16 per engine step at every concurrency. Its tok/s lead at c=1
and c=8 comes from higher MTP acceptance (the two models have different MTP heads, section 6). At
c=2 it is 0.9% below W4A16 in tok/s: inside the c=2 noise, but below the target.

## 3. Prefill (c=1, tok/s)

| build | 8k | 64k | 180k |
|---|---|---|---|
| **final** | **1248** | **954** | **644** |
| production W4A16 | 1108 | 868 | 603 |
| ratio | 1.13 | 1.10 | 1.07 |
| Integration 1 | 1152 | 900 | 622 |
| phase 2 | 1036 | - | 589 |
| stock plugin kernels (phase 1b) | 357 | 328 | 281 |

## 4. Fit

| | final | W4A16 | stock plugin (1b) |
|---|---|---|---|
| weights in VRAM | 12.45 GiB | 14.26 GiB | 12.29 GiB |
| GPU KV cache (fp8, MTP k=3, util 0.94) | 253,906 tokens | 207,812 | 245,312 |
| x of the 200k context | 1.27 | 1.04 | 1.23 |
| VRAM after load | 22,551 / 24,576 MiB | | 22,445 MiB |

The draft head is row-pruned to 61,440 vocabulary rows: production's 40,960 ids, the 3,218 ids
this model emits outside them, and filler. A 195k-token request completed with no fault in phase 1.

## 5. Logit parity vs llama.cpp b11211 (CUDA)

KLD = KL(llama.cpp CUDA || vLLM), in nats, over the full vocabulary, at the last positions of 11 fixed prompts
(vLLM bf16 KV, no spec decode). "floor" = llama.cpp's own CPU backend vs its CUDA backend on the same
ids (measured for seq_000-005 only).

| seq | tokens | stock (1b) | final | floor | top-1 final |
|---|---|---|---|---|---|
| seq_000 chat | 1024 | 0.11645 | 0.06807 | 0.22509 | 0.9531 |
| seq_001 chat | 2048 | 0.25106 | 0.14089 | 0.39918 | 0.9479 |
| seq_002 code | 1536 | 0.00505 | 0.00651 | 0.01449 | 0.9826 |
| seq_003 code | 4096 | 0.00186 | 0.00157 | 0.00352 | 0.9861 |
| seq_004 code | 8192 | 0.00238 | 0.00171 | 0.00825 | 1.0000 |
| seq_005 prose | 8192 | 0.06073 | 0.04188 | 0.07194 | 0.9826 |
| seq_006 code | 32768 | 0.00127 | 0.00089 | n/a | 0.9931 |
| seq_007 mixed | 32768 | 0.00145 | 0.00146 | n/a | 0.9896 |
| seq_008 mixed | 65536 | 0.00271 | 0.00334 | n/a | 0.9965 |
| seq_009 code | 102400 | 0.00195 | 0.00196 | n/a | 0.9826 |
| seq_010 mixed | 120000 | 0.00820 | 0.01075 | n/a | 0.9826 |
| overall | | 0.0404 | 0.0249 | | 0.9818 |
| >= 100k | | 0.0051 | 0.0064 | | 0.9826 |

- Absolute gate (mean KLD <= 0.001, top-1 >= 99%): **FAIL** for every build, stock kernels included.
- Relative gate (at or below llama.cpp's own CUDA-vs-CPU spread): **PASS** on seq_000-005, the
  only prompts where the floor exists. The >= 32k floor was never measured.
- A few high-entropy positions carry the means. For seq_005 it is one position (4910, KLD 8.89, a top-1
  miss); without it the mean is 0.0111. seq_000/001 have single-position spikes of 10-12 in every build.
- Ordering sensitivity. The pipeline is deterministic: repeat runs are bit-identical per position. In
  Integration 1, changing only the fp32 summation order of one product type (item 4b off, so MMQ
  stream-k splits differently) moved seq_010's worst spike from position 119966 (0.487 -> 0.065) to
  119784 (0.072 -> 0.583) and shifted seq_010's mean by ~20%. Build-to-build movement on the long
  prompts (seq_010 0.00740 -> 0.00902 -> 0.01075 over phase 2, Integration 1, final) is within that
  sensitivity, so it cannot be pinned on one kernel. INFERRED: the owned IQ3 and mma_k kernels
  change summation order the same way; a packing-off rerun would separate their share (not run).

## 6. MTP acceptance (k=3, production's 8 real prompts)

| | final (vLLM) | llama.cpp b11211 | delta | gate (within 2 pt) |
|---|---|---|---|---|
| greedy | 0.6684 | 0.6732 | -0.5 pt | PASS |
| T=1 (default sampling) | 0.6613 | 0.6276 | +3.4 pt | FAIL (vLLM samples drafts; every vLLM build shows this) |
| whole decode ladder | 0.650 | | | |
| W4A16 baseline (its own head, whole run, 1b) | 0.522 | | | |

The 61,440-row draft head costs ~2.8 pt of acceptance against the unpruned head (0.684) and is faster
overall (203 vs 809 us per head call).

## 7. Where the step goes: profile and gap to the floor

Torch profiler, 5 complete decode steps, per step (ms). The profiler inflates idle time (last row).

| | c=1 | c=4 | c=8 |
|---|---|---|---|
| span / busy | 34.52 / 22.81 | 37.16 / 29.94 | 46.79 / 39.48 |
| Route L GEMM | 18.40 | 23.87 | 30.37 |
| plugin plumbing (quantize, casts, runs' cat, in_proj_ba, memsets) | 1.41 | 1.44 | 1.81 |
| vLLM GPU work (GDN, attention, norms), of which GDN gating | 2.99 (0.5) | 4.63 (2.0) | 7.30 (3.9) |
| idle (profiled) | 11.71 | 7.22 | 7.31 |
| GPU launches | 2196 | 2325 | 2338 |
| bench ms/step (unprofiled) | 27.9 | 35.7 | 44.2 |

Largest GEMM terms: at c=1, MMVQ 7.92 (IQ4_XS 4.03; there is no owned IQ4_XS decode kernel), packed IQ3 7.78,
owned Q4_K/IQ2_S 2.70. At c=4, tiled IQ3 10.17, mma_k 8.75, MMQ 3.07. At c=8, tiled IQ3 13.11, mma_k 9.25,
MMQ 6.74 (Q6_K 1.86, IQ4_XS 1.49, IQ2_XS 1.20, Q2_K 1.12).

Gap to the floor. The floor is ~17.7 ms/step: ~13 ms to stream the weights once plus ~4.7 ms of vLLM
host idle, measured unprofiled at c=1 on both models. Above c=1 it is a lower bound, because GEMMs
become partly compute-bound.

| | c=1 | c=2 | c=4 | c=8 |
|---|---|---|---|---|
| ms/step | 27.9 | 31.6 | 35.7 | 44.2 |
| gap to 17.7 | 10.2 | 13.9 | 18.0 | 26.5 |
| GEMM above 13 ms | ~5.4 | (no profile) | ~10.9 | ~17.4 |
| vLLM GPU work | ~3.0 | | ~4.6 | ~7.3 |
| plugin plumbing | ~1.4 | | ~1.4 | ~1.8 |

Where the remaining gap sits: MTP steps, not single-row decode. The model matrix
(`cloud/results/models/summary.txt`, same box, same build) served every model with and without MTP.
Without MTP (1 row per sequence per step) the GGUF per-step cost vs production's W4A16 is:

| ms/step, no MTP | c=1 | c=2 | c=4 | c=8 |
|---|---|---|---|---|
| Swift GGUF | 19.6 | 20.4 | 22.7 | 26.9 |
| W4A16 | 20.6 | 22.3 | 22.8 | 22.8 |
| ratio | 0.95 | 0.92 | 1.00 | 1.18 |
| GGUF eff. GB/s (11.35 GB/step) / W4A16 (13.25 GB/step) | 580 / 644 | 556 / 595 | 500 / 581 | 422 / 581 |

The GGUF reads 14% fewer bytes per step but streams them 10-27% slower per byte. It is faster per
step at 1-2 rows, even at 4 rows, and loses at 8. An MTP step (4 rows per sequence: the target
verifies 4 tokens, plus 3 draft passes) costs 1.46 / 1.58 / 1.59 / 1.70x a plain step on the GGUF
(Base GGUF with vs without MTP, same session and build). On W4A16 it costs 1.34 / 1.22 / 1.32 / 1.81x
(cross-session). So the per-step deficit with MTP (1.01-1.19x) comes from the 4..32-row products
the MTP verify pass creates, which is where the owned kernels' efficiency (the "GEMM above 13 ms" row)
is lowest. The GGUF's tok/s lead with MTP comes from acceptance (0.63-0.65 vs 0.52). ISTA's base GGUF
has the same bytes and speed as Swift (no-MTP per-step ratio 1.00).

## 8. Owned kernels and changes, with measured contribution

Each contribution comes from the stage where the change landed (same-session A/B or the stage's
ladder). Decode in tok/s (pass 2, T=0); "ms" is profiled GEMM time per decode step.

| change | where | measured contribution |
|---|---|---|
| Route L: vendored b11211 MMVQ/MMQ behind `lcpp_shim.cu` (phase 2) | shim + vendored | decode 32.4 / 38.0 -> 77.8 / 117.0 at c=1 / 2; prefill 357 -> 1036 at 8k |
| MMQ from 8 rows instead of above 8 (phase 3 item 1) | `linear.py` | c=2 117.0 -> 135.5 |
| draft lm_head row-pruned, no bf16 placeholder (item 3; 61,440 rows since opt-p) | loader | c=1 77.8 -> 79.4; load-peak VRAM -2.37 GiB |
| owned X -> q8_1 quantizer, no input cast (item 4) | shim | c=1 79.4 -> 81.2, c=2 139.5 -> 142.9 |
| one product per same-type shard run, no dequant zero fill (item 4b) | `linear.py` | 433 -> 356 GEMMs per pass; ms/step -0.8% / -1.2% |
| owned IQ3 dp4a kernel at 1..8 rows (item 5) | `lcpp_shim.cu` | c=1 83.5 -> 90.9, c=2 146.7 -> 154.2 |
| owned Q4_K/IQ2_S dp4a kernel at 1..8 rows (K1) | `lcpp_owned_k4.cu` | c=1 88.1 -> 93.0, c=2 154.3 -> 158.9; lm_head 1229 -> 857 us |
| owned IQ3 int8 mma kernel at 6..8 rows (K2) | `lcpp_owned_iq3_mma.cu` | c=2 160.4 -> 173.2 (same session) |
| plumbing: 16-bit IQ3 output, one q8_1 per layer, BF16 gemv for in_proj_ba (opt-p) | shim, `linear.py` | c=1 88.1 -> 96.4; launches 2296 -> 2005 per step |
| MMQ tail zeroed in the quantize kernel; IQ1_M on vendored MMVQ (opt-p2) | shim, `linear.py` | -0.37 and -0.27 ms per step at c=4; KV +4.7k tokens |
| IQ3 repacked at load + packed mma kernel at 1..8 rows (R1) | `iq3_pack.py`, `lcpp_owned_iq3_mma.cu` | IQ3 10.27 -> 7.78 ms at c=1 (+0.26 ms of casts) |
| tiled packed IQ3 kernel above 8 rows (R2) | `lcpp_owned_iq3_mma.cu` | IQ3 19.8 -> 10.2 ms at c=4, ~23.3 -> 13.1 at c=8; 8k prefill +7.9% |
| Q4_K/IQ4_XS/IQ2_S int8 mma kernel at 9..32 rows (K3) | `lcpp_owned_mma_k.cu` | 10.3 -> 9.0 ms at c=4, ~-0.9 ms at c=8 |
| Integration 2 total vs Integration 1 | | ms/step -8.5 / -9.5 / -23.6 / -20.6% at c=1/2/4/8 |
| bounded IQ3 repack (this phase) | `iq3_pack.py` | load pack 6.78 -> 0.59 s, model loading 129.6 -> 124.7 s; GPU scratch 9.2x -> 0.99x the tensor |

Every owned kernel is tested against the vendored kernels on real GGUF blocks, with CUDA-graph
replay, guards, and compute-sanitizer memcheck + initcheck. Numerics: exact int32 per slice with the
vendored scales. The tiled IQ3 kernel is bit-identical to MMQ on whole tiles and within 1e-5
relative otherwise (fp32 summation order).

## 9. Tried and dropped

| what | result | why dropped |
|---|---|---|
| owned IQ4_XS 1..8-row kernel (K1) | all IQ4_XS calls 25.46 -> 25.24 ms per trace; within +-2 us of MMVQ/MMQ per call | no measurable gain; code kept in `8f78f3b` |
| opt-p2 item 6: fp32 product with the cast inside the traced graph | -0.12 / -0.19 ms per step at c=1 / c=4 | KV -3.1k tokens; not bit-exact (inductor keeps an fp32 intermediate) |
| "decode once" as the IQ3 lever (item 5 framing) | SASS shows vendored MMVQ already reuses the per-column decode across rows | the real gain was q8_1 reuse over 4 rows per warp and fewer instructions; ninfer-all's decode-once kernel was never built |
| MTP k sweep (opt-p item 4) | k=2 / 3 / 4: c=1 84.3 / 96.4 / 90.5, c=2 130.3 / 163.5 / 147.8 tok/s | k=3 best at c=1/2, ties k=4 at c=4/8; production stays at k=3 |
| lm_head via MMQ from 4 rows (item 1b) | 1119 vs 1077 us in situ | slower |
| cp.async double buffering or register prefetch in the IQ3 kernel (item 5) | slower than staging in shared memory | |
| R1's unpack + MMQ above 32 rows | 8k prefill -13% (unpacks 5.7 GB per step) | replaced by R2's tiled kernel |
| draft head 81,920 rows / unpruned | acceptance 0.658 / 0.684; head 270 / 809 us per call | 61,440 rows is faster overall |
| cached bf16 IQ1_M weight | -0.03 / -0.22 ms at c=4 / c=8 | costs 178 MB VRAM (~5.6k KV tokens) |
| Route A (batched MMVQ + closed vLLM PR #36226) | not benchmarked | no tensor cores at prefill; kept compile-ready on branch `route-a` |
| lossless IQ3 -> IQ4_XS transcode | not built | +1.77 GB, and IQ4_XS MMVQ is slower per token (llama.cpp PR #8215) |

## 10. IQ3 repack memory (this phase)

The repack (`quantization/iq3_pack.py`, called by `GGUFLinearMethod._pack_iq3`) runs on the
GPU-resident weight after load. The old code widened every byte to int32/int64 and built the output
with `torch.cat` of permuted copies, using ~35x its input in scratch. At load it went 256 rows at a
time, so the peak was ~9x on the smallest tensors; the tests' whole-tensor `pack()` peaked at
1.31 GiB. The new code copies from strided views straight into one uint8 output, in tile groups
sized so the scratch stays under min(tensor, 64 MiB). Measured on the box over all 222 IQ3 tensors
of the GGUF (5,453 MiB); data in `cloud/results/final/pack/`:

| | old | new |
|---|---|---|
| load path `pack_`, max GPU scratch / tensor | 9.16x (blk.3.attn_k, 1024 rows) | 0.99x (limit 1.0x, enforced by `test_iq3_pack_inplace_peak`) |
| load path `pack_`, total time | 6.78 s | 0.59 s |
| whole-tensor `pack()` on blk.1.ffn_down (36.5 MiB) | +1,339 MiB (36.7x) | +70 MiB (1.9x, including the output copy) |
| bytes | | identical to the old pack on every tensor |
| server "Model loading took" | 129.6 s | 124.7 s |
| host RSS / RssAnon peak during model loading (server session) | 17,494 / 5,215 MiB | 15,840 / 3,995 MiB |

The pack does not touch host memory. In both runs the load-window peak is one ~2 s transient about
68 s into the load: 8-13 GB of GGUF file-backed pages plus up to 1.3 GB of anonymous memory, sampled
every 0.5 s. Outside it, RSS stays at 4-5 GB (anon 3-4 GB), and the before/after anon difference is
within the sampling of that transient. After load the server session's host RSS is ~31 GiB, mostly
the 13 GiB CPU KV tier in /dev/shm (production's argv asks for 24 GiB) and GGUF page cache. On a host
with 62 GB shared with production, the CPU tier size is the number to watch.

## 11. Tests

| suite | result | build |
|---|---|---|
| kernel parity, Route L on | 3920 pass / 183 skip / 0 fail | Integration 2 |
| kernel parity, Route L off | 3822 / 281 / 0; stock-kernel tests per-test identical to Integration 1 (317 / 16) | Integration 2 |
| GPU guards (-k "lcpp or first_call") | 240 pass / 60 skip / 0 fail | Integration 2 + review fixes |
| compute-sanitizer memcheck + initcheck | 168 cases, 0 access or uninitialised-read errors | Integration 2 |
| CPU guards + routing table + iq3_pack (box) | 259 pass | final |
| kernel parity -k "pack or packed or iq3" | 2664 pass / 104 skip / 0 fail (incl. the new scratch-bound test) | final |
| CPU suite (local, no Route L build) | 564 pass / 54 xfail (K-quant fp16 dequant, see 14b); the 112 Route L guard tests need the Route L `.so` and run on the box | final |
| vendored files | sha256 match `VENDORED.md` | final |

## 12. Soak

The 24 h soak at c=2 has not run yet: it waits for the model/MTP test matrix on the same GPU.
A first start on the final build ran 1.19 h (2026-09-30 04:12-05:24 UTC) and was stopped on request
so the GPU could go to the test matrix. Nothing went wrong in that window
(`cloud/results/soak/partial-20260930/`):

| | 1.19 h partial |
|---|---|
| server alive / health 200 | every 60 s row (72 rows) |
| restarts / fault lines | 0 / 0 |
| requests | 621: ok 410, aborted streams 70, reasoning-only 139, empty completion 2 |
| tool calls parsed | 94 |
| GPU MiB (server) | 22,500 at the first row, then 23,472-23,496 |
| host RSS (server session) | 31.15-31.40 GiB |
| completion tokens/s (hour 0) | 43.9 |

"reasoning-only" = max_tokens ended inside the thinking, so vLLM returned content None with the
tokens in `message.reasoning`. The load generator counted these as bad output, and `soak_load.py`
now counts them as output (`cf8fbde`). The 2 empty completions are long_hit requests (prefix-cache
hits on 32k and 120k raw-completion prompts at T=1) whose first token was EOS; 2 of ~120 long_hit
requests. Not a fault, but worth watching in the full soak.

## 13. Definition of done (HANDOFF section 2)

| item | status |
|---|---|
| 1 drop-in | met: separate venv equal to production's, plugin loaded via `VLLM_PLUGINS`; argv differs only in model path, HF config dir, port and tier roots; Route L behind `VLLM_GGUF_LCPP=1` |
| 2 correct | partly: 866/866 tensors map (meta dry run); IQ dequant bit-exact; K-quant CUDA dequant 1 ulp off in fp16 (not on this model's linear path); tool calls and reasoning parse (smoke, soak); **logit gate KLD <= 0.001 / top-1 >= 99% FAIL** (0.0249 / 98.18%), relative gate PASS where measured |
| 3 fast | decode c=1 PASS (1.17x); **c=2 FAIL by 0.9%** (0.99x), **slower per engine step at every c**; prefill PASS (1.07-1.13x); MTP greedy PASS (-0.5 pt), T=1 FAIL (+3.4 pt, vLLM draft sampling) |
| 4 fits | PASS: 253,906 KV tokens (1.27x at 200k) |
| 5 stable | section 12 |
| 6 scientific (DeepSWE Pi run) | not run |
| 7 reproducible | pinned plugin fork (`e2b8ad5` + commits, subtree split `swift-gsq-rco`), vendored `b11211` with sha256, `tools/build-plugin.sh` / `VLLM_GGUF_BUILD_LCPP=1`, CPU + GPU suites, this report; no upstream PRs opened |

## 14. Upstream notes

### (a) llama.cpp b11211: MMQ read tail below 8 activation columns

`ggml_cuda_mul_mat_q` sizes the q8_1 buffer as the quantized data plus
`ggml_cuda_mmq_get_J_max(type, fallback, cc, ne11)` blocks of read tail. `get_J_max` starts from
`min(ne11, 512)` rounded down to a multiple of 8, so for `ne11 < 8` it returns 0: no tail. The kernel
still loads a full J-column tile of q8_1, so it reads past the allocation. In Maxwell-Lyu's bridge
(`f1d38ffdd0`) the uninitialised pool bytes there gave IMAs and NaNs. llama.cpp's own dispatch
sends <= 8 columns to MMVQ (`MMVQ_MAX_BATCH_SIZE 8`), so upstream is only exposed when MMQ is called
with fewer than 8 columns. The shim adds 128 `block_q8_1_mmq` (18 KiB) after upstream's tail and
zeroes both tails in its own quantize kernel (`lcpp_shim.cu`, `mul_mat_q`). With that, MMQ at 1..7
rows is memcheck- and initcheck-clean with one cudaMalloc per tensor (phase 2), initcheck-clean at
1/5/8/9 rows (opt-p2), and the 1..9-row parity tests pass on a poisoned allocator. Upstream's sizing
alone was not run under the sanitizer here. Proposed fix: size the tail from the largest J the kernel can select
for this `ne11` (at least one 8-column tile) instead of the rounded-down `ne11`.

### (b) vllm-gguf-plugin `e2b8ad5`: issues found and what was done

| issue at `e2b8ad5` | effect | status |
|---|---|---|
| unsharded and vocab weights staged through a GPU copy | in vLLM's cumem "weights" pool the freed segments could not hold the MTP draft's 2 x 2.37 GiB placeholders: OOM at the draft load | fixed (`e751e64`): copied from host |
| Qwen3.5 adapter with a multimodal config but no mmproj | raised; the text prefix `model.` would not map through `hf_to_vllm_mapper` | fixed (`7794689` on `swift-gsq-rco`): prefix follows `vision_config`; image/video limits 0 or `--language-model-only` required; no regression test committed |
| b2899 CUDA dequant of Q2_K/Q4_K/Q6_K in fp16 (`__hmul`, `__int2half_rn(sc*q)`) | 1 ulp off ggml's fp32 on 7-28% of bf16 elements, max 3.7e-3 of the row absmax | documented and bounded (strict xfail + a 2^-9 absmax bound test), not changed; this model's K-quant linears run on MMVQ/MMQ, not the dequant |
| no input checks on the b2899 ops | non-contiguous X silently wrong, narrow W view NaN, misaligned W device fault, too many rows / K mismatch accepted | Route L ops check everything before launch (dtype, 2-D, type, rows, K, alignment, strides, x_q8); b2899 ops unchanged |
| IQ types above the MMVQ cutoff dequantize the whole weight every forward | prefill 357 tok/s at 8k | replaced by Route L under `VLLM_GGUF_LCPP=1` |
| per-forward `.contiguous()` of each narrower shard in mixed-type fused layers | a copy per shard per forward | runs stored contiguously, read as views (Route L) |
| weight iterator does `torch.tensor(memmap)` per tensor | a full host copy per tensor at load | not changed |

### (c) What could go upstream, in order

1. llama.cpp: the MMQ tail sizing in (a). One function, self-contained.
2. vllm-gguf-plugin fixes that stand alone: host staging (`e751e64`), the adapter gate (with a test),
   an fp32 K-quant dequant (not written).
3. vllm-gguf-plugin: Route L as an opt-in backend. The vendored b11211 MMVQ/MMQ stay byte-identical;
   the shim (`lcpp_shim.cu`) with its guards, the routing, the owned q8_1 quantizer and the same-type
   shard runs in `linear.py` go with them, plus the build flag, `VENDORED.md` and the CPU/GPU tests.
4. Owned kernels, one at a time with microbench and parity tests, ordered by contribution and
   generality: the tiled + packed IQ3 kernels with the load-time repack (R1/R2; largest effect, but
   IQ3 only, and they change the in-memory layout); K3 mma_k (Q4_K/IQ4_XS/IQ2_S at 9..32 rows); K1
   (Q4_K/IQ2_S at 1..8 rows). The dp4a IQ3 kernel (item 5) and K2 only matter for unpacked layers.
5. Keep local: draft-head vocab pruning (production's mechanism) and routing thresholds tuned on one 3090.
