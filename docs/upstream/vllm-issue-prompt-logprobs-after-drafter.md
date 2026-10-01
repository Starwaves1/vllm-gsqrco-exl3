# vLLM issue (ready to file): prompt logprobs read the target's hidden states after the EAGLE/MTP drafter ran, which under CUDA graphs can overwrite them

Not filed. Measured on vLLM main `d28795f1a7` with the qwen38/main overlay `2a0fe5e1e1` on an RTX 3090.
Line numbers are from that tree. Patch: `cloud/results/r3/patches/prompt-logprobs-after-drafter.patch`.

**Title:** [Bug][Spec Decode] echo / prompt_logprobs return NaN (HTTP 400 "Out of range float values are not JSON compliant: nan") for short prompts with MTP + CUDA graphs

## What happens

The setup is MTP speculative decoding (`num_speculative_tokens` 5, a dynamic schedule, PIECEWISE CUDA
graphs, max capture 48) on a Qwen3.5-architecture model. Two requests fail with HTTP 400 "nan": a
`/v1/completions` request with `echo: true, logprobs: 1` on a 5-token prompt, and `prompt_logprobs=1`
on the same prompt. A 40-token prompt fails the same way; 200 tokens works. Generated-token logprobs
(`logprobs` without `echo`) are fine. A probe on `LogitsProcessor.forward` shows NaN already in its
input hidden states, at some prompt positions. The NaN count varies between runs.

| configuration | 5 / 40-token echo + logprobs |
| --- | --- |
| CUDA graphs on (default) | 400 nan |
| `--enforce-eager` | 200 |
| `cudagraph_mode: NONE` (torch.compile kept) | 200 |
| `max_cudagraph_capture_size: 4` | 200 |
| without the KV-offload connector | 400 nan |
| graphs on + the patch below | 200 |

The same server with a different model (stock W4A16, same argv) did not show it. Which model is hit
depends on the memory layout.

## Why (SOURCE; the aliasing mechanism is INFERRED, the fix is verified)

With the padded drafter (`use_gpu_toks`), `sample_tokens` (`v1/worker/gpu_model_runner.py:4580`) runs
the drafter, `propose_draft_token_ids(...)` (:4685), before `_bookkeeping_sync` (:4743). Only after
that, `_bookkeeping_sync` computes prompt logprobs from the target forward's `hidden_states`
(:3838 `_get_prompt_logprobs_dict(hidden_states[:num_scheduled_tokens], ...)`). For a prompt short
enough to run in a captured graph, those hidden states are a graph output. They live in the global
CUDA graph memory pool, which the drafter's graphs share (`compilation/cuda_graph.py:200`,
`get_global_graph_pool()`). A draft pass can therefore reuse that memory before the prompt logprobs are
read. Without graphs, or for prompts above the capture size, the tensor is ordinary memory and stays
intact.

## Fix

At the top of `sample_tokens`, after unpacking `execute_model_state`, copy the hidden states when any
request wants prompt logprobs and a drafter will run. Batches without prompt logprobs pay nothing.

```diff
         # Clear ephemeral state.
         self.execute_model_state = None

+        if self.num_prompt_logprobs and self.speculative_config is not None:
+            hidden_states = hidden_states.clone()
+
```

An alternative is to compute prompt logprobs before the drafter runs. That needs the bookkeeping
order changed, and this copy is the smaller change.

## Evidence (rental RTX 3090, production argv, `cloud/results/r3/box-scripts/54-nan-matrix.sh`, 55)

Job 54 is the configuration table above. Job 55: graphs on, with this patch and
`prompt-logprobs-chunked.patch`. echo + logprobs on 5 / 40 / 200-token prompts and `prompt_logprobs=1`
on 5 to 3,936 tokens all return 200 with finite values, on both models.
