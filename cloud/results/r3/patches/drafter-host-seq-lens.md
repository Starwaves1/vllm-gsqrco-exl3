# [Spec Decode] Follow seq_lens on the host across draft passes (one sync per step instead of one per pass)

Status: MEASURED, NOT KEPT (no speed gain on the box; see Results). Measured on the rental box only (`box-scripts/42-drafter-sync-ab.sh`). Not applied
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

## Results (42-drafter-sync-ab, same box and session, production's main argv, c=2 x 96k, k=5, 90 s windows)

| | ms/step | GPU busy | GPU idle | accepted/draft |
|---|---|---|---|---|
| stock | 57.5 | 51.1 | 6.4 | 3.28 |
| patched | 57.2 | 51.1 | 6.1 | 3.28 |

-0.3 ms/step (0.5 %), inside run-to-run noise. The per-pass `seq_lens.cpu()` syncs are not where
the step's idle goes on this host: they wait on GPU work that is already queued, and the host work
after each one (plan + launch of one MTP pass) is short next to the pass. The greedy probe
(4 concurrent requests) differed between the two servers from token 6..161. That is inconclusive,
because T=0 under MTP is not batch-invariant across server runs (the torture harness records the
same). Not pursued: there is no gain to justify the change. The c=8 windows failed in both variants
(sizing bug, since fixed).
