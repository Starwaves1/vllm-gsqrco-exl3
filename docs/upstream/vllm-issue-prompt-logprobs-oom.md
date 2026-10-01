# vLLM issue (ready to file): prompt_logprobs allocates full-vocab logits for a whole prefill chunk and kills the EngineCore

Not filed. Measured on vLLM main `d28795f1a7` with the qwen38/main overlay `2a0fe5e1e1` on an RTX 3090.
Line numbers are from that tree. Patch: `cloud/results/r3/patches/prompt-logprobs-chunked.patch`.

**Title:** [Bug] prompt_logprobs on a few-hundred-token prompt OOMs the engine at high gpu_memory_utilization (logits for the whole chunk are outside the memory profile)

## What happens

The server runs `--gpu-memory-utilization 0.94`, a 248k vocab (Qwen3.5), and `--max-num-batched-tokens 2048`. A
single `/v1/completions` request with `prompt_logprobs=1` and a 512-token prompt raises
`torch.OutOfMemoryError` inside `EngineCore.step`, and the server dies (`EngineDeadError`). The same
happens with a stock compressed-tensors W4A16 model, so it is not the GGUF plugin.

## Why

`GPUModelRunner._get_prompt_logprobs_dict` (`v1/worker/gpu_model_runner.py:5632`) handles each request in
the batch with prompt logprobs. For the chunk's prompt rows it runs, at once:

- `logits = self.model.compute_logits(prompt_hidden_states)` (:5701): `[num_logits, vocab]` in the model
  dtype, 0.95 GiB at 2,048 rows x 248,320;
- `scores = self.sampler.compute_logprobs(logits)` (:5714): the same shape in fp32, 1.9 GiB, plus
  temporaries in `gather_logprobs`.

The startup memory profile only sizes logits for the sampled rows (one per request). At high
`gpu_memory_utilization` the free memory left after the KV cache (0.2-0.8 GiB on a 24 GB card here)
cannot hold these tensors.

## Fix

Compute the prompt logprobs of a request in row passes with bounded scratch: 64 MiB of fp32 per pass,
which is 67 rows at a 248k vocab. Each pass copies its token ids, logprobs, and ranks into the
existing CPU `LogprobsTensors` slices with `non_blocking=True`, as before. The final `_sync_device()`
covers all of them. Results are identical; the per-request cost is a few more small kernel launches.

```diff
-            prompt_hidden_states = hidden_states[offset : offset + num_logits]
-            logits = self.model.compute_logits(prompt_hidden_states)
-            ...
+            rows_per_pass = max(1, (64 << 20) // (4 * self.model_config.get_vocab_size()))
+            for s in range(0, num_logits, rows_per_pass):
+                e = min(num_logits, s + rows_per_pass)
+                logits = self.model.compute_logits(hidden_states[offset + s : offset + e])
+                tgt_token_ids = prompt_token_ids[start_tok + s : start_tok + e]
+                ... compute_logprobs / gather_logprobs as before ...
+                chunk_slice = slice(start_idx + s, start_idx + e)
+                ... the three non_blocking copies as before ...
```

A better upstream fix may be to make the memory profiler reserve logits for `max_num_batched_tokens`
rows whenever prompt logprobs are allowed. The chunking costs nothing and needs no profile change.

## Evidence (rental RTX 3090, production argv, `cloud/results/r3/box-scripts/50-plp-repro.sh`, 55)

| prompt_logprobs=1, prompt tokens | 512 | 1,024 | 2,048 | 2,600 | 3,936 |
| --- | --- | --- | --- | --- | --- |
| stock W4A16, unpatched | engine dies | - | - | - | - |
| stock W4A16, patched (64 MiB passes) | 200 | 200 | 200 | 200 | 200 |

With 256 MiB passes the W4A16 server, which had 237 MiB free, still died at 512 tokens.
