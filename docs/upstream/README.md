# Upstream PR series for vllm-project/vllm-gguf-plugin

Branches `upstream/NN-*` in this repo, cut from `plugin-upstream/main` at **e2b8ad5** (fetched
2026-09-30; that is also this repo's fork point). They use the upstream repo's own layout (package at the
repo root, tests in `tests/`), so a maintainer can fetch a branch or `git format-patch` / `git am` it
directly. No PR has been opened and nothing was pushed to any upstream remote.

## Order

Smallest and least controversial first. Independent PRs (based on e2b8ad5) can go in any order; the
lcpp stack (05 ... 11e) is stacked, each branch on the previous one.

| # | branch | base | commits | lines (+/-) | CPU tests (vLLM 0.27.1 and main) | why here |
| --- | --- | --- | --- | --- | --- | --- |
| 01 | `upstream/01-host-staging-copy` | e2b8ad5 | 2 | +72/-2 | 105 passed, 6 skipped | OOM at MTP draft load (cumem pool); 1 file |
| 02 | `upstream/02-qwen35-no-mmproj` | e2b8ad5 | 2 | +92/-8 | 110 passed, 6 skipped | text-only Qwen3.5 GGUF with the official config |
| 03 | `upstream/03-kernel-input-guards` | e2b8ad5 | 3 | +482/-0 | 149 passed, 7 skipped | bad inputs: exceptions instead of wrong results / faults |
| 04 | `upstream/04-kernel-parity-harness` | 03 | 2 | +534/-0 | 175 passed, 8 skipped, 16 xfailed | tight parity tests; fp16 dequant documented and bounded |
| 05 | `upstream/05-lcpp-vendored-mmvq-mmq` | 04 | 4 | +1248/-46 | 265 passed, 9 skipped, 16 xfailed | RFC: vendored llama.cpp MMVQ/MMQ, opt-in |
| 06 | `upstream/06-lcpp-q8-quantizer` | 05 | 2 | +209/-23 | 265 passed, 9 skipped, 16 xfailed | no activation cast |
| 07 | `upstream/07-lcpp-owned-iq3` | 06 | 3 | +897/-35 | 295 passed, 9 skipped, 16 xfailed | IQ3 decode 1..8 rows |
| 08 | `upstream/08-lcpp-owned-q4k-iq2s` | 07 | 3 | +442/-23 | 318 passed, 9 skipped, 16 xfailed | Q4_K / IQ2_S decode 1..8 rows |
| 09 | `upstream/09-lcpp-mma-k` | 08 | 3 | +1076/-21 | 338 passed, 9 skipped, 16 xfailed | Q4_K / IQ4_XS / IQ2_S at 9..32 rows |
| 10 | `upstream/10-lcpp-iq3-packed` | 09 | 4 | +1805/-25 | 411 passed, 9 skipped, 16 xfailed | IQ3 repack + packed kernels, all rows (largest gain) |
| 11a | `upstream/11a-small-unquantized-gemv` | e2b8ad5 | 2 | +117/-6 | 110 passed, 11 skipped | small unquantized GEMMs (independent) |
| 11b | `upstream/11b-dequant-no-zero-fill` | e2b8ad5 | 2 | +14/-1 | 104 passed, 6 skipped | no zero fill in dequantize (independent) |
| 11c | `upstream/11c-lcpp-shared-q8-input` | 10 | 3 | +275/-46 | 420 passed, 9 skipped, 16 xfailed | one q8_1 quantize per layer input |
| 11d | `upstream/11d-lcpp-mmq-tail-fold` | 11c | 1 | +37/-24 | 420 passed, 9 skipped, 16 xfailed | no memset per MMQ call |
| 11e | `upstream/11e-lcpp-iq1m-mmvq` | 11d | 3 | +89/-11 | 435 passed, 9 skipped, 16 xfailed | IQ1_M on MMVQ |
| 11f | `upstream/11f-mtp-draft-vocab-pruning` | e2b8ad5 | 3 | +82/-6 | 108 passed, 6 skipped | discussion only: needs a vLLM draft head |
| 12 | `upstream/12-mtp-draft-config` | e2b8ad5 | 2 | +45/-8 | 105 passed, 6 skipped | MTP draft config on newer vLLM (or fix in core) |

"lines" exclude the 21k vendored llama.cpp lines of 05 (and VENDORED.md wording updates in 07 / 11e).
CPU tests: `pytest tests --ignore=tests/test_kernels.py --ignore=tests/test_gguf_generation.py` with the
extension built in place where the branch changes C++ or CUDA (stock build for 03, 04, 11b;
`VLLM_GGUF_BUILD_LCPP=1` from 05 on),
torch 2.13.0+cu130, on vLLM 0.27.1 and on vLLM main (0.30.1rc1.dev285); both give the same counts. The
upstream baseline is 104 passed, 6 skipped. Every branch passes the upstream pre-commit hooks (ruff
0.14.0, ruff-format, typos 1.43.5, clang-format 21.1.2, markdownlint-cli2 0.21.0, run on all files), and
every branch that touches CUDA compiles for sm_86 with CUDA 13.0.

One-line rationale per PR:

- **01** fixes an OOM at MTP draft load under the cumem allocator: one small, obvious change.
- **02** fixes serving text-only Qwen3.5 GGUFs with the official multimodal config; small, with a test.
- **03** turns silently wrong results, NaNs and device faults on bad inputs into exceptions.
- **04** adds tight kernel parity tests (and a CPU run of the dequantize kernels) that the later
  kernel PRs rely on; documents and bounds the fp16 legacy / K-quant dequantization.
- **05** is the RFC-sized one: vendored llama.cpp b11211 MMVQ / MMQ behind a shim and a default-off
  flag; everything after it is optional and builds on it. Open the RFC issue (text in 05) first.
- **06** removes the activation cast (owned q8_1 quantizer, byte-identical to the vendored one).
- **07 / 08 / 09** add owned kernels where they measured faster: IQ3 at 1..8 rows, Q4_K / IQ2_S at
  1..8 rows, Q4_K / IQ4_XS / IQ2_S at 9..32 rows.
- **10** is the largest owned piece (IQ3 repacked at load, tensor-core kernels for every row count);
  the biggest single gain, and the one that changes the in-memory weight layout.
- **11a / 11b** are independent plumbing fixes (small unquantized GEMMs, dequantize zero fill).
- **11c / 11d / 11e** are lcpp plumbing (one quantize per layer input, no memset, IQ1_M on MMVQ).
- **11f** is for discussion only: its main part needs a vLLM-side draft head.
- **12** fixes MTP draft config creation on newer vLLM; a vLLM-core alternative is described in it.

## What was verified here, and what was not

- CPU only in this preparation: builds (compile and link, sm_86), CPU tests on two vLLM versions,
  lint. No GPU was used.
- The GPU tests in these branches are ported from the development suite, which ran them on an RTX 3090
  against the development model's own tensors (counts per PR in each description). Ported to the
  sample-GGUF weights upstream's tests use, they were collected and their harness exercised on the CPU
  with the CUDA ops replaced by the reference models (plumbing check only), but they have **not yet run
  on a GPU in this form**. Run them on a GPU box before opening each PR:
  `VLLM_GGUF_BUILD_LCPP=1 pip install -e . --no-build-isolation` then
  `VLLM_GGUF_LCPP=1 pytest tests/test_lcpp_kernels.py tests/test_kernel_parity.py tests/test_kernel_guards.py`
  (and once without `VLLM_GGUF_LCPP` for the stock path).
- The stack's end state (11e) was compared with the GPU-tested development build (`main` here): the
  owned CUDA files are token-identical apart from clang-format, a semicolon after each of 16 macro
  calls (two sites) and a split string literal; `iq3_pack.py` differs only in formatting; the shim differs only in the order of op definitions; `linear.py` differs only in the
  argument order of `_fused_mul_mat_gguf` (`packed` before `x_q8`, so 11c appends rather than inserts)
  and in the gemv of 11a, which is its own branch. The intermediate states (05 ... 11d) were never run
  on a GPU as such; 05's quantize path (vendored quantizers after a cast) is the development branch's
  phase-2 path with the MMVQ entry point of phase 3.
- 03's checks on the stock ops and 07's `TORCH_CUDA_ARCH_LIST` check for the lcpp build are new code
  written for upstream (not in the development build). The review caught a bug in the first version of
  03 (the `ggml_dequantize` check assumed n is a row width, but the embedding path passes the token
  count); fixed, with a CPU test for that call shape.

## Before opening each PR

- **DCO:** vLLM projects require `Signed-off-by`. These commits carry none, because signing off is the
  author's certification: run `git rebase --signoff <base>` on each branch (or `git commit -s --amend`).
- **Co-Authored-By:** no trailer was added to these commits; this repo's history uses one, and no
  policy for the upstream series was found. Add it if wanted.
- Numbers in the descriptions come from `cloud/results/` of this repo (RTX 3090, 350 W, vLLM 0.27.1,
  a 27B Qwen3.5-architecture IQ3_S-based GGUF with MTP k=3); the descriptions describe the model
  generically.

## Also here

- `llamacpp-mmq-tail-issue.md`: a ready-to-file llama.cpp issue on MMQ's q8_1 read tail below 8
  columns (file:line at b11211 / d7fb90e8, proposed fix).
- The RFC issue text for the lcpp backend is at the end of `05-lcpp-vendored-mmvq-mmq.md`.

## vLLM issues (not plugin PRs)

Ready-to-file issue texts for bugs found on vLLM main while running the plugin, each with its patch
in `cloud/results/r3/patches/` (overlay-style, against the installed tree):

- `vllm-issue-prompt-logprobs-oom.md`: prompt_logprobs allocates full-vocab logits for a whole chunk.
- `vllm-issue-prompt-logprobs-after-drafter.md`: prompt logprobs read after the MTP drafter, NaN under CUDA graphs.
- `vllm-issue-dynamic-spec-schedule-corruption.md` (the "#50021 review text"): PR #50021's conv1d
  bound (carried by our overlay) zeroes GDN output when the speculative query length shrinks between
  steps. Part 1 is a review comment for #50021, the deliverable (the bug is in the PR as proposed;
  merged main does not have it). Part 2 is an optional short issue against main asking for a test of
  the dynamic-K invariant (main is correct today). Fix: `cloud/results/r3/patches/conv1d-accepted-bound.patch`.
  Neither posted.

## HOLD (2026-10-01; reason gone 2026-10-03, HOLD lifted pending Garrett's go)

Do not open PRs 09 (9-32-row int8-mma kernel), 10 (IQ3 repack + packed kernels) or 11c-11e (shared q8 input, MMQ tail fold, IQ1_M chunking) until the production incident is resolved: GSQ-RCO corrupts generations only when ≥9 requests run at once (the MTP drafter's 9-16-row matmuls through these kernel paths inside CUDA graphs), 26% of answers at 9+ vs 0.1% at ≤8; see cloud/results/prod-garbled-tokens-20261001.md. PRs 01-08, 11a, 11b and 12 are unaffected but should wait for the root cause in case it touches the quantizer or routing they share.

**Update 2026-10-03: the reason for this HOLD is gone. HOLD lifted pending Garrett's go** (until he
says go, treat 09/10/11c-11e as held). The incident is not in these kernels. Stock W4A16 (Marlin, no plugin) corrupts the same way on the same
vLLM, turning the plugin's routing off changes nothing, and the 9-32-row kernels pass their unit tests
at the production row counts; these three facts are PROVEN by box runs and are all the HOLD decision
needs. The cause (PROVEN at kernel level on CPU, INFERRED end to end until the GPU rerun) is PR
#50021's conv1d bound in our vLLM overlay, which fires
when the per-batch-size MTP schedule lowers K between steps (`cloud/results/r3/incident-root-cause.md`).
The GPU rerun with the one-line fix is the last confirmation; nothing in the root cause touches the
quantizer or routing that PRs 01-08, 11a, 11b and 12 share.
