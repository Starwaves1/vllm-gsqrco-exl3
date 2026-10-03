# vLLM report (ready to post, not posted): PR #50021's conv1d bound corrupts output when the spec-decode query length shrinks between steps

Not filed or posted. Found on vLLM main `d28795f1a7` plus the qwen38/main overlay `2a0fe5e1e1`, which
carries vllm-project/vllm#50021 (open; head `71d7c782ca`, last updated 2026-09-27) verbatim for
`causal_conv1d.py`. Upstream main without that PR has no such check, and its kernel is correct for the
same inputs (CPU test below). So the right place is a review comment on #50021, not a new issue
against main. If a GPU run on unpatched main ever reproduces the corruption, use the issue title at the
end instead. Line numbers are from `d28795f1a7` (upstream) and the PR head.

Evidence and the full incident write-up: `cloud/results/r3/incident-root-cause.md`. Fix:
`cloud/results/r3/patches/conv1d-accepted-bound.patch`. Kernel test:
`cloud/results/r3/box-scripts/r3conv_kchange.py`.

---

**Comment on #50021 (title for an issue, if one is needed):** [Bug][Spec Decode][Mamba] causal_conv1d_update: `num_accepted > seqlen` check zeroes GDN output after the speculative query length shrinks (dynamic `num_speculative_tokens_per_batch_size`, truncated drafts)

## Summary

The `causal_conv1d.py` hunk in this PR rejects `num_accepted_tokens > seqlen`:

```python
num_accepted = tl.load(num_accepted_tokens_ptr + idx_seq).to(tl.int64)
if (num_accepted < 1) | (num_accepted > seqlen):
    # zero the output for this request, return without updating conv_state
```

In the varlen path `seqlen` has just been replaced by this request's own query length
(`causal_conv1d.py:846` upstream). But `num_accepted_tokens` is the previous step's count, and the
previous step can have verified more tokens than this one:

- with `num_speculative_tokens_per_batch_size`, K is picked per step from the batch size
  (`v1/core/sched/scheduler.py:1446-1451`), so a request that verified 6 tokens and accepted all 6
  can verify 4 in the next step when the batch grows from 4 to 5 under `[[1,4,5],[5,8,3],...]`;
- with a fixed K, the scheduler shortens drafts for structured-output requests
  (`update_draft_token_ids` keeps the grammar-valid prefix) and under a tight token budget.

In both cases the check fires on a valid count. The request's conv output is zero for every token of
the step and its conv state is not advanced, so all GDN layers see zero input for that step and a
stale window afterwards. On Qwen3.5/3.6-architecture models (48 of 64 layers GDN) this shows up as
wrong tokens mid-word, duplicated fragments, and early EOS.

The other three kernels in the PR bound the same index by the state row width
(`i_t < stride_indices_seq` in `fused_recurrent.py` and `fused_sigmoid_gating.py`,
`init_token_idx < stride_state_indices_batch` in `mamba_ssm.py`), as does the CUDA
`fused_gdn_decode_post_conv_mtp` (`accepted <= state_indices_width`). Only the conv1d hunk uses the
current query length.

## Environment

- vLLM main `d28795f1a7` (0.30.1rc1.dev285) + #50021 (`causal_conv1d.py` identical to the PR head),
  V1 model runner (`VLLM_USE_V2_MODEL_RUNNER=0`), `--no-async-scheduling`, FlashInfer attention,
  PIECEWISE CUDA graphs (also with `--enforce-eager`).
- RTX 3090 24 GB (350 W), CUDA 12.8 image.
- Models: a Qwen3.5-architecture 27B hybrid (GDN + full attention) with its MTP head, in two
  quantizations: W4A16 AutoRound (Marlin) and a GGUF IQ3_S via an out-of-tree plugin. Both corrupt.

## Reproduction

Server (the exact argv we ran, minus host paths and our KV-offload connector, which did not matter):

```
vllm serve <Qwen3.5-arch model with MTP> --max-model-len 200000 --max-num-seqs 16 \
  --gpu-memory-utilization 0.94 --kv-cache-dtype fp8 --mamba-ssm-cache-dtype float16 \
  --no-async-scheduling --max-num-batched-tokens 2048 --long-prefill-token-threshold 128 \
  --enable-prefix-caching --mamba-cache-mode align \
  --compilation-config '{"max_cudagraph_capture_size":48,"custom_ops":["+rms_norm","+silu_and_mul"]}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":5,"draft_sample_method":"probabilistic",
                         "num_speculative_tokens_per_batch_size":[[1,4,5],[5,8,3],[9,16,2]]}' \
  --reasoning-parser qwen3 --scheduling-policy priority
```

Load (two clients at once):

1. Main: 30 chat requests, 4 at a time, non-streamed, `temperature: 0`, `max_tokens: 600`, thinking
   on, ordinary English questions (history, units, dates).
2. Neighbour: one thread sending `POST /v1/completions` with
   `{"prompt": "The capital of Denmark is", "max_tokens": 1, "temperature": 0}` back to back (one in
   flight at a time). Each one is a prefill-only request that lifts the batch from 4 to 5 for a step,
   so K goes 5 -> 3 -> 5.

Count answers with U+FFFD, EOS inside the reasoning, characters outside the prompt's scripts, or a
fragment repeated 4+ times.

## Results (T=0, c=4, 30 answers each)

| server | neighbour | corrupted |
| --- | --- | --- |
| schedule `[[1,4,5],[5,8,3],[9,16,2]]` (GGUF) | none | 0 |
| same | 1-token completions | 23-27 |
| same | 4-token completions / logprobs / prompt_logprobs / echo | 19-26 |
| same, W4A16 (Marlin) | none | 0 |
| same, W4A16 (Marlin) | 1-token completions | 23 |
| schedule + `--enforce-eager` | 1-token completions | 25 |
| schedule, no KV connector | 1-token completions | 25 |
| schedule, `--mamba-cache-mode none` | 1-token completions | 24 |
| fixed `num_speculative_tokens: 3`, no schedule | 1-token completions | 0 |
| no speculative decoding | 1-token completions | 0 |
| `[[1,16,2]]` (K=2 at every batch size), c=3..8, no neighbour | none | 0-1 |

Without a neighbour, the same schedule corrupts when the running count hovers at a tier boundary:
c=8: 0-1/32, c=9: 8-16/36 (K flips 3 <-> 2), c=12: 3/48, c=16: 1-3/64.

Examples (T=0): `Serbia accepted 7 of the 10 10 demands;`, `1377: JikI377: Jikji printed in Korea`,
`The old town's main streets:'s.\n- The old town's.\n- The old town's.`, `11111111111...`.

## Kernel-level reproduction (CPU, Triton interpreter)

`r3conv_kchange.py` calls `causal_conv1d_update` exactly as `qwen_gdn_linear_attn.py` does
(`query_start_loc` per request, `max_query_len = num_spec + 1`, conv state `width - 1 + num_spec`
wide), feeds a sequence of verify steps with given query lengths and accepted counts, and compares
every output with a plain causal conv over the accepted tokens (width 4, num_spec 5):

| steps (L = query length, a = accepted) | main | main + #50021 | #50021 + fix |
| --- | --- | --- | --- |
| fixed K=3: L=4 each step | exact | exact | exact |
| L=6 a=6, then L=4 (K 5 -> 3) | exact | step 2 err 9.9, step 3 err 7.6 | exact |
| L=6 a=4, then L=4 | exact | exact | exact |
| L=4 a=4, then L=3 (K 3 -> 2) | exact | step 2 err 5.1, step 3 err 1.9 | exact |
| L=4 a=4, then L=6 (K 3 -> 5) | exact | exact | exact |
| num_accepted 0 or num_spec + 2 | reads out of row | rejected | rejected |

## Suggested fix

Bound the count by the launch's `max_query_len` (the width of the spec-decode state row, which the
caller passes as `spec_state_indices_tensor.size(-1)`), not by the request's own query length. Every
count up to that width selects a window inside the `width - 1 + num_spec` conv state:

```diff
@@ def _causal_conv1d_update_kernel(
+    max_seqlen = seqlen  # launch-wide max_query_len: the spec-decode state row width
     if IS_VARLEN:
         query_start_index = tl.load(query_start_loc_ptr + idx_seq).to(tl.int64)
@@
+        # num_accepted comes from the previous step, which may have verified more
+        # tokens than this one (dynamic K, trimmed drafts): bound it by the state
+        # row (max_seqlen), not by this request's query length.
         num_accepted = tl.load(num_accepted_tokens_ptr + idx_seq).to(tl.int64)
-        if (num_accepted < 1) | (num_accepted > seqlen):
+        if (num_accepted < 1) | (num_accepted > max_seqlen):
```

A test for the PR: one request, two `causal_conv1d_update` calls with `max_query_len = num_spec + 1`,
the first with a query of `num_spec + 1` tokens, the second with fewer tokens and
`num_accepted_tokens = num_spec + 1`; compare with the reference conv.

The non-varlen (3D) path keeps `seqlen` as its bound. GDN layers use the varlen path. If a 3D caller
can see a shrinking query, it needs the conv state width passed in.

## Not yet verified

- On the GPU, the end-to-end rerun with only this change applied (in progress on our side; this text
  gets the result before it is posted).
- Whether the same symptom appears on main without #50021 (the CPU test says the kernel is correct
  there; no GPU run on unpatched main yet).
- A production deployment on vLLM 0.27.1 with the same schedule and the same check showed no
  corruption signature for a W4A16 model (running count mostly 9-16). We do not have an explanation
  for that run yet.
