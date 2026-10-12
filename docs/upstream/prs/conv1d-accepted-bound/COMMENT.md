# Draft comment for vllm-project/vllm#50021 (posted by Garrett 2026-10-12)

Goal: get the PR labeled and merged. It is blocked only on `pre-run-check` (needs `ready`/`verified`); amittell
already pinged @ZJY0516 and @tdoublep. This gives the maintainers independent end-to-end evidence on a second
architecture and a direct ask.

---

@ZJY0516 @tdoublep could one of you add `ready` so CI runs? Independent verification on SM86 below, in case it helps.

**End to end** (RTX 3090, Qwen3.5-arch 27B + MTP, vLLM main `d28795f1a7` + this PR as of `71d7c782ca`, then with the bound swapped, `num_speculative_tokens_per_batch_size` `[[1,4,5],[5,8,3],[9,16,2]]`, 30 T=0 chats at c=4 with a neighbour that keeps pushing the batch from 4 to 5 so K goes 5 -> 3 -> 5). Answers with garbled text:

| conv1d hunk | GGUF IQ3_S | W4A16 (Marlin) |
| --- | --- | --- |
| old bound (`num_accepted > seqlen`) | 23-27 / 30 | 16-23 / 30 |
| current bound (`> max_seqlen`) | 0 / 30 | 0 / 30 |
| conv1d hunk reverted (main's kernel) | 0 / 30 | not run |

**Kernel** (CPU, Triton interpreter, `causal_conv1d_update` called like the GDN spec path, vs a plain causal conv over the accepted tokens), head `ace8eb7d64` on main `03091e28b8`: exact on fixed K, K 5->3, K 3->2 and K 3->5. The old bound gives max error 9.9 and 5.1 on the two shrinking cases. Out-of-range counts (0, K+2) still zero the output and leave the state alone.

@amittell go ahead with your own test; I won't send a separate PR.
