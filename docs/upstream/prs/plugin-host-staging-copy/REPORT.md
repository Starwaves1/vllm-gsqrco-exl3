# [BugFix] Copy unsharded and vocab GGUF weights to the device from host memory

Target: vllm-project/vllm-gguf-plugin. Branch `upstream/01-host-staging-copy` (2 commits on e2b8ad5, which is
upstream `main` as of 2026-10-03). Description: `docs/upstream/01-host-staging-copy.md`.
Related upstream: #101 (open issue, load uses more memory than expected; the commenter attributes the fix to
#109, merged and already in e2b8ad5, which this does not overlap), #141 (open, large kernel PR whose unsharded path
becomes `param.data = loaded_weight`, which fixes the linear half of this OOM independently but leaves
`_gguf_embedding_weight_loader` untouched: if #141 lands first, this PR reduces to the vocab path plus the test;
if this lands first, #141 rebases one hunk). No open PR or issue covers this OOM.

Status 2026-10-05: approved by Garrett; held until `pytest tests -m "not slow"` completes on a GPU for this branch
(signed branch ready, not pushed).

## Initial problem and concise proof

With vLLM's cumem allocator on (`--enable-sleep-mode`, or `enable_cumem_allocator`), a 27B Qwen3.5-architecture
GGUF with MTP k=3 on a 24 GB RTX 3090 (vLLM 0.27.1) dies while the MTP draft loads
(`cloud/results/phase1/memdiag.txt`, `cloud/results/phase1/runs/p1-smoke-oom-before-fix/server.log`):

```
[memdiag] load_model end arch=['Qwen3_5ForConditionalGeneration'] alloc=12.04GiB reserved=18.42GiB
[memdiag] after initialize_model alloc=16.78GiB reserved=23.16GiB        <- Qwen3_5MTP
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 70.00 MiB. GPU 0 has a total capacity of 23.56 GiB of which 14.00 MiB is free.
```

With the fix, same box and argv: the target leaves 17.36 GiB reserved (12.02 allocated), and the draft load
completes at 22.54 GiB reserved. The new CPU test fails on e2b8ad5 with `assert [(4, 4), (10, 6)] == []` (the
unsharded linear and the vocab weight each made a device staging copy) and passes on the branch; the CPU
suite passes, 105 passed, 6 skipped (`pytest tests --ignore=tests/test_kernels.py
--ignore=tests/test_gguf_generation.py`, re-run 2026-10-03 on vLLM 0.30.1rc1.dev285).

## Problem mechanism

- vLLM loads the target and its drafter in one `CuMemAllocator` "weights" pool: `gpu_worker.py:538-545`
  wraps the whole `model_runner.load_model` in it (vLLM main d28795f1a7). Inside the pool, expandable segments
  are forced off (`device_allocator/cumem.py:378-391`). Freed segments are only released when the context
  exits (`cumem.py:405-420`, which says `empty_cache()` cannot be used with the pluggable allocator). vLLM also
  loads with `max_split_size_mb=20` (`gpu_worker.py:542-543`): a freed block of 20 MiB or more is never split
  and only serves a request within 20 MiB of its own size. That is why the final 70 MiB request fails with
  about 6 GiB reserved but free.
- The plugin moved every unsharded weight to the device before copying it into its materialized parameter:
  `_clone_loaded_weight(loaded_weight).to(device=param.device)` in `_store_gguf_loaded_weight`
  (`quantization/params.py:80`) and in `_gguf_embedding_weight_loader` (`params.py:134`). The staging
  tensor is freed after `param.data.copy_`. The next weight of the same shape reuses that block, so the pool
  ends with one dead block per distinct unsharded shape. Most of it is the embedding and lm_head staging
  copies, and on this model it comes to 1.06 GiB (18.42 versus 17.36 GiB reserved for the same allocations).
- The MTP draft then allocates its bf16 vocab-sized placeholders (248,320 x 5,120, 2.37 GiB each). They fit
  in none of the freed blocks, so the pool maps new memory and runs out.

Verified: the line numbers above against e2b8ad5 and d28795f1a7; the measured reserved and allocated numbers
before and after; the red/green test.

## Fix mechanism

For the unsharded path (`shard_id is None`) and the vocab path, the weight is copied from host memory into the
materialized parameter with one `copy_`. No device temporary is created, so no block is freed. Sharded weights
still move to the device first, because a single-shard parameter keeps the stored tensor itself as its data
(`params.py:88-95`). The fix is 3 changed lines (+72/-2 with comments and the test).

Nothing more is needed for this OOM. The sharded path (`_create_padded_weight_param`) still leaves about 5.3 GiB
reserved-but-free on this model, and a larger GGUF could hit that; it is a separate change. Downside: none
found. Each such weight now takes one host-to-device copy instead of a host-to-device plus a device-to-device
copy. Load time was not measured separately. Without the cumem allocator the change only drops a copy.

## Final check agent's concise opinion

Fable 5.1 /check, 2026-10-03: **ready with edits (text only; code and test correct and minimal)**. Verified every
cited line in e2b8ad5 and d28795f1a7. The draft loads inside the same pool: `gpu_model_runner.py:5286-5288` runs
under `load_model`'s pool context, and the OOM fires in `cumem_allocator.cpp:163` (server.log line 52). 18.42 vs
17.36 GiB reserved at the same allocation (memdiag pids 35152 vs 36822), so the 1.06 GiB is the staging copies.
The test is red on e2b8ad5 and green on the branch. Pageable `copy_` into the materialized parameter uses no
allocator memory, dtype and shape are unchanged, and the host `narrow` on dim 0 keeps TP > 1 embeddings correct.
The sharded path keeps its device move, as it must. No open issue or PR covers this OOM. Edits applied: the
`max_split_size_mb=20` rule added to the mechanism (it is why freed blocks cannot serve other sizes); "one block
per weight" corrected to one per distinct shape; #141's overlap stated precisely; the test command qualified.

## Fable's comment

**Recommendation:** approve. **Confidence:** high.

The mechanism is real and now complete: inside vLLM's cumem weights pool, segments are never returned until the pool closes and `max_split_size_mb=20` stops a freed block from serving any request outside its own size band, so one dead block per distinct unsharded shape is exactly what the 1.06 GiB gap between 18.42 and 17.36 GiB reserved measures. The draft's two 2.37 GiB vocab placeholders then have nowhere to go. The fix removes the only thing that created those blocks, a device staging copy that was never needed because the materialized parameter can take a host-to-device `copy_` directly. Three changed lines, the sharded path untouched for the stated reason, red/green test, and the plugin's own CPU suite green.

Residual points, none blocking: the test patches `torch.Tensor.to` globally for its duration, which is fine in a unit test but should stay the only test doing so; load time was not measured, and I expect it unchanged or better since one copy replaces two. Open PR #141 rewrites the linear half of this path, so whichever lands second rebases one hunk. I would open this one first.
