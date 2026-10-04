# EXL3 GPU phase 1: results

Rented RTX 3090 (350 W), 2026-10-01 05:35-07:58 UTC, after the GGUF soak was stopped at 19.4 h.
Model: `erlidev/Swift-1.5-Qwen3.8-27B-EXL3` @ `SC_3.50bpw_H4_V6` (041dc382), 18 files, 14.69 GB,
every file checked against the Hub (`00-model.log`). The A/B checkpoint (turboderp 3.50bpw) was not
downloaded: not enough disk. Stack: vLLM main 0.30.1rc1.dev285 plus the `2a0fe5e1e1` overlay, both
plugins in one venv, production's main argv (`env/prod-main-serve-argv.txt`: MTP k=5 with the
per-batch schedule, fp8 KV, 16 seqs, PIECEWISE graphs up to 48). Scripts: `box-scripts/`, one gpuq
job each. Labels as in EXL3.md: VERIFIED = measured here.

| job | status |
|---|---|
| 01 kernel parity | pass (run 2; run 1 failed 6 tests, all on bounds I had set too tight) |
| 02 draft head | pass |
| 03 smoke | pass at max-model-len 196,608 (the first run at 200,000 did not start) |
| 04 parity reference | incomplete: killed by SIGTERM (exit 143) three times while the phase-2 queue took the GPU; 8 of 11 sequences dumped |
| 05 parity vLLM | not run (needs 04) |
| 06 speed ladder | pass, at max-model-len 196,608 |
| 07 200k fit | FAIL: 200,000 does not fit on production's argv |

## Kernel parity (01, `01-kernel-parity/`)

VERIFIED. 731 tests: 411 passed, 0 failed, 304 xpassed, 16 xfailed. Eight checkpoint tensors cover
every bit width the checkpoint uses: K2 (`layers.0.mlp.up_proj`), K3, K4 (`k_proj`, `o_proj`, MTP,
lm_head) and K5. Reference: exllamav3 d3739fd in its own venv on the same tensors and activations.

- Dequant: `exl3_dequant` with `had=True` and `had=False` is bit-exact with exllamav3's
  `reconstruct_had_slice` / `reconstruct_slice` for all 8 tensors, K2 included (16/16, sha256 over
  every 32768-column slice).
- GEMM vs fp64, routed op at 1..17, 48, 145 and 1024 rows, fp16 and fp32 output: 320/320 inside
  exllamav3's own error (worst rel. RMS 2.5e-3; lm_head 3.8e-3).
- Bit identity with exllamav3's own output: 304/320. The 16 that differ are fp32 output at 145 and
  1024 rows. Our reconstruct route rounds the hgemm result to fp16 and then upcasts; exllamav3 writes
  fp32 directly.
- Also passing: route agreement at 145/1024, fp16/fp32 agreement, CUDA graph capture and replay
  (1..48 rows, 4 tensors), and fresh-process cases. Capture without warmup is refused cleanly, the
  first call inside a capture after warmup works, and the guards reject bad CUDA inputs.
- compute-sanitizer memcheck and initcheck: 0 errors on 84 cases.
- Run 1 (`01-kernel-parity-run1/`): 6 failures, all on bounds I had set too tight (two
  fp16-accumulate errors compared against each other), plus 4 memcheck errors that the sanitizer
  setup caused itself (capture cases with the caching allocator off). Both fixed in ef8411c.

## Draft head (02)

VERIFIED. `mtp_draft_head.safetensors` is 40,960 x 5,120 bf16 (0.39 GiB), built from production's
id list. Against the EXL3 lm_head on those ids: rel. RMS 4.1e-3, argmax agreement 100 %.

## Smoke (03, `03-smoke/`)

VERIFIED at `--max-model-len 196608`; every other argument is production's.

- Chat, reasoning split and the qwen3_coder tool call: SMOKE_OK. The draft head loads with 40,960
  rows.
- Weights: 13.94 GiB, loaded in 5.0 s. Healthy after 246 s (torch.compile 69 s, profiling run 66 s).
- KV memory 7.09 GiB, which holds 199,716 tokens (1.02x at 196,608). VRAM after load: 22,457 of
  24,576 MiB.
- MTP during the smoke: acceptance 0.542, 3.71 tokens/step at k=5. Per position 0.85, 0.62, 0.49,
  0.41, 0.33.

## Fit at 200k (07, `07-fit/`)

FAIL. On production's argv (gpu-memory-utilization 0.94, fp8 KV, 200,000) vLLM refuses to start:
7.07 GiB of KV is needed and 7.06 GiB is available, so the largest length that fits is 199,280.
The miss is 0.01 GiB. The bf16 input embedding on the GPU (2.37 GiB) is the obvious place to get it
back (a host-pinned or int8 embedding is the open question in ADR 0002). The GGUF fits 246k at
12.45 GiB.

## Speed ladder (06, `06-ladder/`)

VERIFIED. Production's `run_benchmarks.sh single`, real-prompt cohorts, pass 2 at T=0.
ms/step = C x 1000 / decode tok/s x tok/step (REPORT.md definition). Clocks during decode: median
SM 1755 MHz, 340 W.

The k differs between rows. EXL3 ran production's main argv: k=5 up to 4 sequences (6 rows per
sequence per step), k=3 at 5-8 sequences. The GSQ and W4A16 references ran k=3 (4 rows per
sequence). So an EXL3 step at c=1/2/4 verifies 1.5x the rows. tok/s also mixes in MTP acceptance;
ms/step is the kernel comparison.

| c | EXL3 decode tok/s | EXL3 e2e tok/s | EXL3 tok/step | EXL3 ms/step | GSQ Integration 2 ms/step (k=3) | prod W4A16 ms/step (k=3) |
|---|---|---|---|---|---|---|
| 1 | 74.7 | 72.4 | 2.98 | 39.9 | 27.9 | 27.6 |
| 2 | 136.6 | 117.6 | 2.94 | 43.0 | 31.6 | 27.3 |
| 4 | 171.2 | 143.5 | 2.99 | 69.9 | 35.7 | 30.0 |
| 8 | 308.6 | 255.4 | 2.78 | 72.1 | 44.2 | 41.3 |

| prefill, c=1 | EXL3 tok/s | GSQ Integration 2 | prod W4A16 |
|---|---|---|---|
| 8k | 1078 | 1248 | 1108 |
| 64k | 855 | 954 | 868 |
| 180k | 610 | 644 | 603 |

At c=2: 8k 724 tok/s, 64k 572 tok/s. MTP over the whole ladder: acceptance 0.406, mean accepted
length 2.77, per position 0.70 0.48 0.32 0.16 0.11.

What the numbers say (INFERRED from the routing table, not profiled):

- At c=4 there are 24 rows per step. The vendored GEMM streams the weights once per 16 rows, so it
  makes 2 passes, which explains 69.9 ms against 43.0 at c=2 (12 rows, 1 pass). c=8 (32 rows at
  k=3) also makes 2 passes.
- c=1 is 39.9 ms against GSQ's 27.9 even at 6 rows: a single-pass cost well above the ~13 ms weight
  stream. The 401 separate EXL3 products per pass (unfused qkv, gate_up and GDN in_proj parts) are
  the first suspect.
- Prefill is within 3-14 % of the references and beats W4A16 at 180k.

## Logit parity (04, partial; `04-parity-ref/exl3_logits-partial.json`)

The bug in 04's first run (an exllamav3 recurrent state created outside inference_mode) is fixed in
1adf032. The re-runs were then stopped by SIGTERM (exit 143) during the phase-2 queue's jobs. One run
completed seq_000..007 (1k-32k tokens) and the last completed seq_008 (65k). The 120k-token
exllamav3 cache fits: 18.8 GiB after load, peak 19.5 GiB. 05 (the vLLM side and the score) has not
run, so there is no logit-parity number yet. The 8 reference dumps are on the box at
`/workspace/runs/exl3/parity/exl3/` (2.2 GB).
