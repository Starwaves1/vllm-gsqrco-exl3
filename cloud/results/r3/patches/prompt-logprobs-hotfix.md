# Hotfix: prompt_logprobs OOM and echo+logprobs NaN on vLLM main (torture smoke 2026-10-01)

Two separate bugs. Both reproduce on the rental 3090 with production's main argv (box jobs 50-55,
logs under `/workspace/logs/r3/`). Labels: MEASURED (box), SOURCE (vLLM main d28795f1a7 + overlay
2a0fe5e1e1).

## 1. prompt_logprobs kills the EngineCore (CUDA OOM)

- MEASURED (50): `prompt_logprobs=1` on a 512-token prompt already kills the EngineCore on GSQ and on
  the stock W4A16 model alike. This is vLLM's own buffer, not only ours.
- SOURCE: `GPUModelRunner._get_prompt_logprobs_dict` computes full-vocab logits and fp32
  log-softmax scores for every prompt row of the prefill chunk at once. At 2,048 rows x 248,320 vocab
  that is 0.95 GiB bf16 + 1.9 GiB fp32. The startup memory profile only sizes logits for sampled rows,
  and at `--gpu-memory-utilization 0.94` the server keeps only 0.2-0.8 GiB free.
- Ours made it worse: Route L's MMQ writes an fp32 [rows, 248,320] dst and then casts it (the 1.57 GiB
  allocation in the smoke). MEASURED (51, `test_lm_head_prompt_rows`): peak 2,910 MiB at 2,048 rows and
  5,593 MiB at 3,936 on 32ae6ec.

Fixes:
- plugin (ours): `quantization/linear.py` runs any Route L product whose fp32 dst would exceed
  256 MiB in 128-row-multiple chunks into one X-dtype output. Only the lm_head at prompt-logprob row
  counts hits it. Peak drops to 1,334 / 2,228 MiB. Outputs are finite and match the dequantized
  product at 1..3,936 rows. Kernel parity: 3,932 passed / 0 failed (51).
- vLLM (proposal for the qwen38/main overlay, `prompt-logprobs-chunked.patch`): prompt logprobs in
  passes of at most 64 MiB of fp32. With the plugin fix alone GSQ still dies at 512 tokens (vLLM's
  scores buffer). With 256 MiB passes GSQ survives 512..3,936 tokens, but W4A16 (only 237 MiB free)
  still died, hence 64 MiB.

## 2. echo + logprobs returns HTTP 400 "nan"

- MEASURED (50/52/54): on GSQ the final hidden states hold NaN at some prompt positions when the prompt
  is short enough to run in a CUDA graph (5 and 40 tokens fail, 200 is fine, max capture is 48). The
  NaN count varies between runs. The lm_head is finite on the same rows (51). There is no NaN with
  `--enforce-eager` (52), with `cudagraph_mode NONE` (54), or with max capture 4 (54). The KV
  connector is irrelevant (54 `noconn` fails the same way). W4A16 does not show it. `logprobs` without
  `echo` is fine.
- SOURCE: with the padded EAGLE/MTP drafter, `sample_tokens` runs the drafter before
  `_bookkeeping_sync`, and `_bookkeeping_sync` reads the target's `hidden_states` for prompt logprobs.
  Under CUDA graphs the target's output lives in the global graph pool, which the drafter's graphs
  share (`compilation/cuda_graph.py:200`), so a draft pass can overwrite it before it is read. Whether
  a given model's pool layout overlaps is luck: GSQ's does and W4A16's does not.
- Fix (vLLM, proposal, `prompt-logprobs-after-drafter.patch`): clone `hidden_states` at the top of
  `sample_tokens` when any request in the batch wants prompt logprobs. There is no cost otherwise.
  Not ours: no plugin code is involved.

Side finding (not production, which runs Route L): `VLLM_GGUF_LCPP=0` (the plugin's stock kernels)
under graphs on vLLM main gives NaN logits on every request (54 `lcpp0`).

## Deploy

1. Plugin: branch `hotfix-prompt-logprobs` (main + the linear.py chunking + its GPU test).
2. vLLM overlay: apply both patches to qwen38/main
   (`patch -p1 -d <site-packages> < prompt-logprobs-*.patch`, or commit them there).
   Box evidence, 55 (both patches + the fixed plugin, CUDA graphs on, production's main argv),
   on GSQ and on W4A16: echo + logprobs on 5 / 40 / 200-token prompts, prompt_logprobs=1 on 5 and
   512 / 1,024 / 2,048 / 2,600 / 3,936 tokens, and logprobs without echo all return HTTP 200 with
   finite values; both servers stay alive. Without the patches (50): 400 "nan" on echo (GSQ), and
   the EngineCore dies at 512 prompt tokens (both models).
