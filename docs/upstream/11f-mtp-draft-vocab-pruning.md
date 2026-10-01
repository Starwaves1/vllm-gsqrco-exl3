# 11f: MTP draft head: GGUF placeholder lm_head, optional row-pruned head (for discussion)

Branch: `upstream/11f-mtp-draft-vocab-pruning` (3 commits, on e2b8ad5). No plugin dependency, but the
second commit needs a vLLM change that does not exist upstream. **Not proposed for merge as is**; the
first commit can go on its own.

**Title:** [Perf] qwen3_5 MTP: no bf16 lm_head placeholder for the draft; optional pruned draft head

## Motivation

1. The Qwen3.5 MTP draft's `lm_head` is replaced by the target's after loading, but as an unquantized
   module it first materializes a full-vocab bf16 placeholder: 2.37 GiB for 248320 x 5120, at the
   draft-load memory peak.
2. The draft head scores the whole 248k vocabulary on every draft step. Scoring a fixed subset (the
   tokens the model actually drafts) makes the head 6x cheaper for a small acceptance cost.

## What changed

- Commit 1 (`weights_adapter/qwen3_5.py`): the MTP adapter no longer lists `lm_head` in
  `extra_unquantized_modules`, so it stays an empty GGUF placeholder until vLLM shares the target's.
  `embed_tokens` stays unquantized: vLLM probes the draft's `embed_input_ids` before sharing.
- Commit 2: when the model (HF config) directory holds `mtp_draft_vocab_ids.pt`, `output.weight` is also
  mapped to `mtp.draft_lm_head.weight` with only those rows kept (a lossless slice of the quantized
  blocks); `MTP_DRAFT_VOCAB=0` turns it off. This needs a Qwen3.5 MTP model that defines
  `draft_lm_head` and scores only those ids (-inf elsewhere): vLLM does not have that.
- Commit 3: CPU tests of both.

## How tested

- CPU tests: 108 passed, 6 skipped on vLLM 0.27.1 and vLLM main.
- GPU, development branch (RTX 3090, a vLLM with such a `draft_lm_head`): load peak -2.37 GiB
  (23,427 -> 21,001 MiB); with 40,960 of 248,320 rows kept, the draft lm_head went 809 -> 136 us per
  call and decode 77.8 -> 79.4 tok/s at c=1, 135.5 -> 139.5 at c=2, at an MTP acceptance of 0.630
  instead of 0.688. A 61,440-row head later gave the best trade-off (0.650 acceptance, 203 us).
- Commit 1 with the pruned head off (`MTP_DRAFT_VOCAB=0`): a CPU meta-device load of the 27B model
  passed on vLLM 0.27.1 and main, with the out-of-tree vLLM change installed. Commit 1 has not been
  checked on stock vLLM nor served on a GPU without commit 2.

## Discussion

The pruned head is a vLLM-side feature first (the draft model must restrict its logits); the plugin
part only maps and slices the rows. If vLLM grows such an option, commit 2 is the GGUF side of it.
