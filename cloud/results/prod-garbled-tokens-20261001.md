# Prod garbled tokens 2026-10-01: U+FFFD, dropped characters, early EOS, drift

Read-only diagnosis of ms4 production (`qwen-vllm.service`, GSQ-RCO IQ3_S GGUF via the plugin on vLLM main
d28795f1a7 + overlay 2a0fe5e1e1, MTP k-schedule `[[1,4,5],[5,8,3],[9,16,2]]`, MAX_SEQS 16, fp8 KV, prefix
caching, CPU + fs offload tiers, PIECEWISE graphs, `--no-async-scheduling`). Sources: unit journal, `/metrics`,
llama-dashboard's 1 Hz `vllm_samples` (running, KV usage, preemptions), a copy of knowledge-bench `runs.db`
(every bench request is non-streamed, no sampling params, i.e. generation_config T=1.0 top_k 20 top_p 0.95,
thinking on, priority 100000, concurrency 16, via ai.starlab.one -> :18080), and 9 small requests of my own.

## Verdict (ranked)

1. **GSQ x running >= 9 (the `[9,16,2]` tier) injects wrong tokens. Confirmed by data, mechanism open.**
   Answer-side corruption (U+FFFD or `finish=stop` with empty content) by the highest `running` seen during
   the request, GSQ runs 17-19 (8,994 answers):

   | max running during request | 0-7 | 8 | 9 | 10-12 |
   |---|---|---|---|---|
   | corrupted / answers | 2/764 | 7/5,880 (0.1%) | 587/2,265 (25.9%) | 11/79 |

   Judge calls: 0 failures in 6,673 calls whose window never reached 9 running; 132/2,076 (6.4%) when it
   did (dose-response: 3.7% if <50% of the window at >=9, 7.1% if >=50%). Per 10-min bin, every failing bin
   had max running >= 9; every bin capped at 8 was clean.
   Control, same vLLM, same schedule, same tiers, same bench: W4A16 runs 13-16 spent ~24,000 s at running
   >= 9 (11,984 answers, 5.4 M tokens): **0 U+FFFD**, 0.08% empty-stop, judge failures 0-0.07% (a stray
   letter, and ReadTimeouts). So the >=9 regime is necessary but not sufficient: it needs GSQ.
   Signature = isolated wrong token ids, then the model repairs or drifts: `ASC �AP ASC Topic 606`,
   `PCAOB Rule �B Rule 3520`, `capitalized� not be capitalized`, `TheThe`; judge replies `AK`, `D2`,
   `p\n\n\nB`; reasoning cut mid-word (`reasoning=26 chars, tail='The c'`, i.e. an EOS sampled mid-sentence);
   content that echoes the judge prompt (`Question: ...`, `Rules: 1. YEAR must ...`, consistent with an
   injected `</think>` or `<|im_start|>`). U+FFFD = a lone UTF-8 byte token, not a stream split.
   What changes at 9 running: k drops 3 -> 2 and the drafter's per-step batch goes from <= 8 rows to 9-16
   rows (padded 16). The verify GEMM is 27 rows -> padded 32, the same padded size as c=8/k=3, which is
   clean. In the plugin, <= 8 rows route to MMVQ / own vec kernels and >= 9 rows to MMQ / mma_k
   (`quantization/linear.py:79-92`), so **the MTP drafter (MTP block + vocab-truncated draft lm_head) only
   runs our MMQ/mma_k paths, inside graphs, at c >= 9**. Leading mechanism: non-finite or invalid draft
   probabilities from that path reaching the probabilistic rejection sampler (recovered token drawn from a
   NaN residual => arbitrary id, often specials at the draft vocab's tail: EOS, `</think>`), with the
   GSQ-only graph-pool overlap (hotfix notes) as the alternative. A correct sampler cannot emit these
   tokens at top_k 20 / top_p 0.95 from finite probabilities.
2. **H1 offload tiers / preemption: not the driver.** W4A16 runs 13-16 had 2,203 preemptions and the same
   tiers with no corruption; GSQ 05:00-08:55 had **zero** preemptions and 1.5% judge failures. The fs tier
   (4,184 blocks written 03:40-08:54 by the earlier GSQ run, same plugin build, same GGUF) served 69 chunk
   hits after the 11:04 restart; my 17.6k-token verbatim copy at T=0, cold then hit (+16,112 GPU and
   +2,544 external prefix tokens, 3 fs chunks), came back exact both times (436 tokens, em dashes, curly
   quotes, arrows). Not excluded as a minor contributor, but nothing points to it.
3. **H2 streaming detokenization: ruled out for the bench** (all non-streamed) and the proxy is a byte
   passthrough (`aiter_raw`, `vision-sidecar/proxy.py:501`). Stream vs non-stream, :18080 vs :18081, T=0,
   text full of multi-byte characters: all four identical and exact. Note vLLM's SSE carries raw UTF-8
   (no `\u` escapes), so a client that decodes each TCP read separately can still add U+FFFD in chats;
   that would never drop ASCII letters, and the bench shows the same garbling without streaming.
4. **H5 torture: did not run.** No process, no `torture-runs/` dir, `finished_reason=abort` 0. The load at
   04:47 and 12:12-12:24 was knowledge-bench (runs 17/18 and 19). py-spy ran `--nonblocking` (memory
   reads only) at 12:12-12:28; it cannot alter state.
5. **Async-scheduling D2H overwrite: void** (`async_scheduling: False` in the startup args).

Rates by temperature: every bench request ran at T=1.0 (no T=0 traffic exists in the data). My own probes at
running 9-13 (k=2) were clean: ~1.3k tokens at T=0 and 4,800 sampled tokens (n=4, logprobs top-20: no
sampled token outside its own top-20, none below logprob -6). At the bench's rate (~8e-4 events/token) the
sampled probe expected ~4 events, P(0) ~ 2%. Differences from the bench: my probes asked for logprobs,
had short or no thinking, and ran after run 19 ended at 12:24 (lighter mix). Worth one box cell.

Time: not a post-restart window. Run 17 (03:50-05:32) was clean throughout despite 58 preemptions;
failures track running >= 9 in run 18 (05:50, 07:20, 07:50-08:30) and run 19 (12:12-12:24, 13.3% of judge
calls; that hour was mostly at 9 running).

## Mitigation

Already in place: at 12:28:31 Garrett installed `deploy/override.conf.c8` (MAX_SEQS 8, schedule
`[[1,4,5],[5,8,3]]`, CG 32) and restarted; healthy at 12:33:11. That removes the regime. Keep it until the
box run below names the path. No need to stop the bench or disable the fs tier.

## Settling measurements (round-3 box, prod argv, plugin 32ae6ec)

Detector, no text heuristics: per request, return `token_ids`; flag a byte token not completing a valid
UTF-8 sequence, an EOS/`<|im_end|>`/`</think>`/`<|im_start|>` at a position where the target's own top-1 is
something else, or any sampled id outside the target's top-k=20. Workload: knowledge-bench AA prompts,
exactly C concurrent, no sampling params (T=1 defaults), thinking on, no logprobs unless stated, 600
requests per cell (~25 min at c=9).

| cell | change from prod | separates |
|---|---|---|
| A | prod c16 argv, C=9 | must reproduce (expect ~25% of answers) |
| B | A + `[[1,4,5],[5,16,3]]` | k=2 vs drafter batch >= 9 |
| C | A + `[[1,4,5],[5,8,2]]`, C=8 | k=2 at drafter batch <= 8 |
| D | A with `VLLM_GGUF_LCPP` MMVQ for the MTP module only (force `n<=16` to vec) | our drafter kernels |
| E | A + `--enforce-eager` | graph-pool overlap |
| F | A + temperature 0 | T>0 only? |
| G | A + `draft_sample_method greedy` | probabilistic acceptance path |
| H | A + logprobs top-20 | why my probe was clean |

Unit side, parallel: the MTP block's tensors and the draft lm_head (61,440 ids) through `lcpp_mul_mat_q` /
`lcpp_mul_mat_mma_k` at 9..16 rows, eager and graph replay, vs dequantized reference; assert finite.
In A, add a one-line `torch.isfinite` check on `draft_probs` and `target_logits` in
`rejection_sampler.py` (box build only) and log the batch shape when it fires.

Needs Garrett: exact timestamps of his garbled chat messages, and the client (to confirm whether it
streams and how it decodes SSE).

## Shutdown at 12:28:31

Not a crash. `sudo tee .../override.conf`, `daemon-reload`, `systemctl restart qwen-vllm.service` from
garrett on pts/6 at 12:28:31 (journal sudo lines); SIGTERM, "aborting in-flight requests count=1", clean
teardown, unit deactivated 12:28:36, restarted 12:28:37 with the c8 override, serving 12:33:11. 18 chat
requests in the last minute before it, no Traceback/EngineDeadError/OOM.
