# [Spec Decode] Follow seq_lens on the host across draft passes (one sync per step instead of one per pass)

Status: PROPOSAL. Measured on the rental box only (`box-scripts/42-drafter-sync-ab.sh`). Not applied
to production or to any shared venv. Patch: `drafter-host-seq-lens.patch`, against vLLM main
`d28795f1a7` plus the qwen38/main overlay `2a0fe5e1e1` (site-packages paths).

## Problem

With an EAGLE/MTP drafter and `num_speculative_tokens > 1`, the drafter rebuilds attention metadata
once per draft position (`SpecDecodeBaseProposer.propose`, the loop over `token_index`). On every
build, `FlashInferMetadataBuilder.build` needs host sequence lengths for `plan()`. Under spec decode it
takes them with a blocking `common_attn_metadata.seq_lens.cpu()`: the exact-host-lengths shortcut is
gated on `num_speculative_tokens == 0`. On GPUs without the trtllm-gen path (sm86 and other non-
Blackwell parts) `needs_seq_lens_cpu` is always true. A decode step at k=5 therefore stops the host
about 6 times: the target build, the first draft pass, and 4 loop passes. After each stop the GPU
idles while the host runs `plan()`, builds the next pass's inputs and launches it, because nothing
else is queued.

The host already knows these lengths:
- Outside async scheduling, `seq_lens_cpu_upper_bound` (`optimistic_seq_lens_cpu`) is exact. The
  dataclass docstring says so: "precise ... for all rows outside async spec decode".
- The padded drafter's first pass uses the target's lengths.
- Each later pass is `seq_lens - num_rejected + i`. The drafter already adds the `+1` per pass to
  `seq_lens_cpu_upper_bound` (`_update_positions_dependent_metadata`). Only `num_rejected` is
  GPU-only.

## Change

1. `CommonAttentionMetadata.seq_lens_cpu_exact: bool = False`, set to `not use_async_scheduling` by
   the model runner.
2. `FlashInferMetadataBuilder.build` uses `seq_lens_cpu_upper_bound` when
   `num_speculative_tokens == 0 or seq_lens_cpu_exact`.
3. `SpecDecodeBaseProposer.propose` copies `num_rejected_tokens_gpu` to pinned memory at the top of
   `propose`, before the first draft pass is enqueued, and records an event. Before the loop it waits
   on that event only, which completes as soon as rejection sampling has, while the first draft pass
   still runs. It then sets the host lengths to `upper_bound - rejected`. It keeps the exact flag only
   when no row can reach `max_model_len` within the remaining passes, the only case where the GPU
   update (`eagle_step_update_slot_mapping_and_metadata`) is not `+1`. In every other case the flag
   is cleared and the builder falls back to `seq_lens.cpu()`, the old behaviour.

The values passed to `plan()` are identical to the synced ones, so attention inputs and outputs do
not change (the A/B job checks that greedy outputs are identical). This is safe with FlashInfer's
plan staging only because the plan buffers are pageable. The overlay makes them pageable by default:
`VLLM_FLASHINFER_UNPINNED_PLAN_BUFFERS`, see the race note at the top of `flashinfer.py`. Upstream
should land this together with that, or with an explicit sync between plans of the same wrapper.

## Results

(filled in from 42-drafter-sync-ab: ms/step at c=2 x 96k k=5 and c=8 x 20k k=3, stock vs patched,
same box and session; greedy probe identity; acceptance)
