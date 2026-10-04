# EXL3 on production vLLM: benchmark report

Model: erlidev `Swift-1.5-Qwen3.8-27B-EXL3` @ `SC_3.50bpw_H4_V6` (14.69 GB), served by `plugin-exl3/`
(`vllm_exl3_plugin`) on vLLM main 0.30.1rc1.dev285 + overlay `2a0fe5e1e1`, in the same venv as the GSQ
plugin, on production's main argv (fp8 KV, prefix caching, KV offload tiers, PIECEWISE graphs up to 48,
16 seqs) with MTP at a **fixed k=3**. Rented RTX 3090 at 350 W. Design: `EXL3.md`, ADR
`docs/adr/0002-exl3-via-plugin.md`; phase 1: `cloud/results/exl3/PHASE1.md`; optimization log with every
measurement: `EXL3-OPT.md`. Labels: VERIFIED = measured on the box.

## 1. What runs

- Vendored, byte-identical: exllamav3 `d3739fd` dense-linear kernels (94 files, MIT) behind
  `exl3_shim.cu`; trellis-serve `1ace59c4b43c` Marlin-EXL3 (19 files, MIT / Apache-2.0) behind
  `exl3_mr_shim.cu`. `VENDORED.md` in each directory has the sha256s.
- Owned: `exl3_gemm_mr` on every routed row count 1..384 (K3/K4/K5; K4 repacked at load), with an owned
  patch (`csrc/exl3_marlin_h16.patch`) that accumulates in fp16 at 9..48 rows and folds into fp32 every
  4 k-stages; one opaque op per layer with the bf16 glue on fused layers; token embedding page-locked in
  host memory (2.37 GiB of VRAM freed); MTP draft head as per-row fp8 on vLLM's fp8 Marlin.
- No vLLM patches. Switches and defaults: `EXL3-OPT.md`, "Switches".

## 2. Decode ladder (VERIFIED, k=3, production's real-prompt cohorts, pass 2, T=0)

ms/step = tok/step / (decode tok/s / c). `cloud/results/exl3-opt-h16x/f4m14/12-mr-ladder-k3/`.

| c | EXL3 tok/s | EXL3 ms/step | GGUF GSQ-RCO ms/step | prod W4A16 ms/step |
|---|---|---|---|---|
| 1 | 98.1 | 27.0 | 27.9 | 27.6 |
| 2 | 188.3 | 27.1 | 31.6 | 27.3 |
| 4 | 323.1 | 32.2 | 35.7 | 30.0 |
| 8 | 539.1 | 38.9 | 44.2 | 41.3 |

The GGUF and W4A16 columns are `cloud/results/REPORT.md` (vLLM 0.27.1, k=3): same GPU and k, older
vLLM. MTP acceptance over the run 0.507 (per position 0.707 / 0.483 / 0.330), 2.5-2.65 tokens per step.
Phase 1 ran 39.9 / 43.0 / 69.9 / 72.1 ms/step at c=1/2/4/8 (k=5 schedule); the gains since are the
owned multi-row kernel and its fp16 accumulation, the per-layer op and glue, the host embedding and the
fp8 head (`EXL3-OPT.md` has each step's measured contribution).

## 3. Prefill (c=1, tok/s)

8k: 1133 (this build). 64k / 180k: 855 / 610 (phase 1; the route is unchanged: 2048-row chunks take the
dequant + fp16 GEMM route). GSQ: 1248 / 954 / 644; W4A16: 1108 / 868 / 603.

## 4. Fit (VERIFIED)

At max-model-len 196,608: 281,648 KV tokens (1.43x). 200,000 fits with the host embedding (job 14,
`cloud/results/exl3-opt/14-fit-2gah/`); without it production's 200,000 missed by 0.01 GiB (phase 1).

## 5. Correctness

- **Kernels** (VERIFIED, `cloud/results/exl3/01-kernel-parity`, `cloud/results/exl3-opt-h16x/f4m14/10-mr-parity`):
  dequant bit-exact with exllamav3 for every bit width in the checkpoint; `exl3_gemm_mr` against fp64
  never above exllamav3's own `exl3_gemm` error (rms ratio <= 1.00 on 150 shape/rows/dtype cases);
  compute-sanitizer memcheck and initcheck 0 errors; CUDA graph capture and replay tested.
- **Logits vs exllamav3** (VERIFIED, job 05, no MTP): KLD mean 0.0034 at bf16 KV (0.00067 without 4
  positions where the reference itself is wrong by inspection), 0.0094 at fp8 KV; top-1 0.993 / 0.991.
  Code and prose sequences 1.7e-4 to 7.1e-4 up to 120k tokens; the two chat prompts carry the excess,
  as they do between llama.cpp's own CUDA and CPU backends. exllamav3's own fp16 vs fp32 spread: 7.3e-5.
- **Generation corruption** (VERIFIED, `bench/corruption_check.py`, k=3, 608 requests at c=8/9/12/16, T=0
  and T=1.0, streamed and not): 0 early EOS, 0 bad UTF-8, 0 errors; 2 foreign-script flags, both at
  T=1.0 and both coherent text with one sampled foreign token. The T=1.0 baseline without MTP and the
  c=1/2/4/8 run are queued. The production k schedule's 3 -> 2 drop at 9 seqs corrupts EXL3 as it does
  GSQ (#50021's conv1d bound in the overlay, not the plugin); a fixed k avoids it.

## 6. Not finished

| item | state |
|---|---|
| other erlidev tiers (3.00 / 4.00 / 4.50 bpw) | queued on the box (`21-tier.sh`); table to come in `tiers.md` |
| single-stack switch test (GSQ <-> EXL3, one venv) | queued (`bench/torture.sh switch`, 4 legs of 10 min) |
| 12 h torture soak | queued behind GSQ round 3's soak (plog/echo skipped, as for GSQ) |
| `EXL3_MR_CONCAT` | off: with the fp8 head the drafter's load OOMs (probe queued); unmeasured |
| prompt_logprobs >= 4096 tokens | dies in vLLM's own `compute_logprobs` (stock W4A16 too); vLLM-side fix |

## 7. Definition of done (HANDOFF section 2, applied to EXL3)

| item | status |
|---|---|
| 1 drop-in | met: same venv as production's, plugin loaded via `VLLM_PLUGINS`; argv differs only in model path, host/port, tier roots, the chat template's repo copy (same sha256) and the fixed k=3; no vLLM patches |
| 2 correct | partly: kernels within exllamav3's error, dequant bit-exact; **logit gate KLD <= 0.001 FAIL on the raw numbers** (0.0034, the two chat prompts), PASS on code/prose; corruption check clean at T=0, 2 coherent foreign tokens at T=1.0 pending a baseline |
| 3 fast | decode faster per step than GGUF at every c and than W4A16 at c=1/2/8 (slower at c=4: 32.2 vs 30.0 ms); prefill 5-10 % behind GGUF, ahead of W4A16 at 8k and 180k (1.5 % behind at 64k) |
| 4 fits | PASS: 200,000 fits; 281,648 KV tokens at 196,608 |
| 5 stable | not yet: 12 h torture soak queued |
| 6 scientific (DeepSWE Pi run) | not run |
| 7 reproducible | vendored sources byte-identical with sha256, `VLLM_EXL3_BUILD=1` build, CPU tests (306 EXL3) and GPU suites (`tests/gpu/test_exl3_*.py`), box job scripts in `cloud/results/exl3*/box-scripts/`, this report |
