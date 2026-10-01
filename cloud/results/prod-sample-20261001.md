# Production samples under agent traffic, 2026-10-01 12:13-12:19 EDT (read-only)

Inputs: `/tmp/prod-sample-20261001-121351/`, `/tmp/prod-sample-20261001-121525/` (S1, S2: py-spy dump of
EngineCore pid 1671826, 30 s py-spy flame graph of the API server pid 1671480 at 250 Hz `--nonblocking`,
`ps`, `/metrics` before and after) and `/tmp/prod-sample-20261001-121925/` (S3: engine dump and a 30 s
EngineCore flame graph at 250 Hz `--nonblocking`). Production: ms4, RTX 3090, vLLM main d28795f1a7 + qwen38/main overlay 2a0fe5e1e1,
GSQ-RCO GGUF, k schedule `[[1,4,5],[5,8,3],[9,16,2]]`, `--no-async-scheduling`. Frames were mapped to
`.venv-main` (same tree). Labels: MEASURED (these files), SOURCE (vLLM code), INFERRED.

**S3 (the engine flame graph) ends with production shutting down.**
- `engine-30s.svg` (5,274 samples, again `--nonblocking`, no native frames) was written at 12:28:32, not
  ~12:19:55. py-spy stops after duration x rate sampling ticks or when the target exits, so its window
  lies somewhere in 12:19:25-12:28:32 and is not pinned.
- S3's API record (12:28:32-36) holds 8 samples, all in interpreter shutdown: `_exitfunc` ->
  `weakref` -> `torch.library._del_library`.
- The closing `ps` has no EngineCore line and shows the API process at RSS 0, and `metrics-after.txt` is
  empty. **Production's vLLM exited at ~12:28:32-36.**
- These files cannot tell a restart from a crash. The atexit path in the API server means it ran Python
  shutdown, not a hard kill.
- Sample proportions in the engine graph are valid. Absolute per-second rates are not, because the
  window is uncertain and part of it may have been light load.

## 1. Load during the windows

The `/metrics` snapshots are 30.5 s (S1) and 30.4 s (S2) apart, not ~65 s. Steps come from
`iteration_tokens_total_count`. SOURCE: `async_llm.py` output_handler creates one `IterationStats` per
engine-core output message, and messages with no outputs are skipped. MEASURED: its sum equals
generated + computed prompt tokens exactly (S1: 7,066 + 13,407 = 20,473). So it counts engine steps
that emitted tokens, which is every step while anything decodes. The prompt's drafts/running method
assumes all 9 running sequences draft every step. Here only 7.8 (S1) and 8.5 (S2) drafted per step, so
that method overstates ms/step by 13-15 %.

| | S1 12:13:52-12:14:22 | S2 12:15:25-12:15:55 | gap 12:15:55-12:19:25 (209.7 s) |
|---|---|---|---|
| running (gauge, start -> end) | 9 -> 9 | 9 -> 9 | -> 9 |
| waiting, KV usage | 0, 0.75 -> 0.81 | 1 -> 0, 0.95 -> 0.76 | 0, -> 0.57 |
| preemptions (total) | 0 -> 0 | 2 -> 4; queue time 16.7 s summed | 4 -> 8 |
| engine steps | 342 = **11.2 steps/s, 89.2 ms/step** | 304 = **10.0 steps/s, 100 ms/step** | 2,907 = 13.9 steps/s, 72.1 ms/step |
| drafts/9 method | 9.73 /s, 102.8 ms | 9.45 /s, 105.8 ms | |
| mean ITL (= per-sequence step) | 87.8 ms | 96.0 ms | |
| drafts per step; draft tokens / draft (k) | 7.8; 2.70 (k=3/k=2 mix) | 8.5; 2.05 (k=2) | 8.2 |
| accepted / draft | 1.65 | 1.39 | |
| generated tok/s; tok/step | 232; 20.7 | 204; 20.4 | 277; 20.0 |
| prompt tok/s: total / computed / external | 2,052 / 440 / 0 | 4,103 / 2,039 / 251 | |
| requests finished; /tokenize calls | 46; 45 | 46; 44 | |
| API CPU (`process_cpu_seconds_total`) | 1.40 s = 4.6 % of a core | 1.55 s = 5.1 % | |

`ps.txt` %CPU is the **lifetime average** (API 2.8 %, EngineCore 8.4-15.3 %), not instantaneous. The
API server started at 11:04:46, so most of its lifetime was light load. Using that start time, the
EngineCore's lifetime CPU seconds grew by 0.9-1.2 s per second in S1 and S2 (±0.05 % ps rounding) and by
1.01 s/s across the gap (INFERRED). **The engine burns about one full core while serving**, which is
the main thread.

Comparison with the 50.6-min profile:
- S1 matches the n=9 "with short prefills" bin: 89 vs 85.9 ms pooled (median 87.4), and 440 computed
  prompt tok/s vs ~545 in the 40-50.6 min bins. Tokens per step are 20.7 vs 22.7 because only 7.8
  sequences drafted per step, not 9. Per sequence-step S1 gave 2.65 tokens (k=3/k=2 mix) vs 2.53.
- S2 is prefill-heavy: 2,039 computed tok/s, about 4x any profile bin, plus a waiting request and two
  preemptions. The extra ~11 ms/step is the prefill rows.
- The gap (72 ms at ~8 drafting) sits at the profile's clean 5-8 band (64-73 ms).
- These windows have no nvidia-smi, so GPU idle per step cannot be measured here.

Garbled-output window (coordinator note): `finished_reason` error, abort and repetition stay at 0. The 8
`length` finishes already existed at 12:13:51 and did not grow. Preemptions rose 0 -> 2 -> 4 -> 8
between 12:13:51 and 12:19:25. None of the three dumps is in a preemption, recompute or exception path,
but a dump is one instant.

## 2. API server (S1 311 samples, S2 369 samples at 250 Hz)

py-spy without `--idle` drops idle samples. So the samples are on-CPU time: 1.24 s (S1) and 1.48 s (S2)
in 30 s, which matches the process's own CPU counter (1.40 / 1.55 s, the rest being native threads).
**The API server is ~95 % idle.** py-spy reported "Errors: 37/34". These are stack-read races from
`--nonblocking`, ~0.5 % of ~7,500 attempts. They also produced 26 / 34 truncated stacks rooted directly
under `all`. All of those are `prometheus_client` frames and are counted in that category.

### By category (self samples, stack-classified)

| category | S1 ms / 30 s (%) | S2 ms / 30 s (%) | thread |
|---|---|---|---|
| Prometheus `/metrics` exposition (`generate_latest`, `sample_line`, `collect`, escaping) | 592 (47.6) | 620 (42.0) | anyio worker thread |
| event loop, native (uvloop C: socket I/O, httptools, ZMQ, SSE writes; self in `runners.py:118`) | 288 (23.2) | 264 (17.9) | main loop |
| tokenization (`_tokenize_prompt` -> HF `_encode_plus`; /tokenize + chat prompts) | 144 (11.6) | 144 (9.8) | threadpool |
| tool parser, streaming (`extract_tool_calls_streaming` -> `_qwen3_arg_converter`, `_safe_arg_prefix`) | 4 (0.3) | 236 (16.0) | main loop |
| detokenization (`detokenizer.decode_next`) | 48 (3.9) | 56 (3.8) | main loop |
| reasoning parser (`count_reasoning_tokens`, non-streaming) | 40 (3.2) | 16 (1.1) | main loop |
| request setup (`_make_parser`, `add_request`) | 40 (3.2) | 32 (2.2) | main loop |
| stats logging (`loggers.record`) | 20 (1.6) | 8 (0.5) | main loop |
| HTTP/ASGI middleware, SSE framing (`stream_response` own) | 24 (1.9) | 36 (2.5) | main loop |
| ZMQ/IPC with the engine (`output_handler` outside processing) | 8 (0.6) | 4 (0.3) | main loop |
| chat template, threadpool queue, misc | 32 (2.6) | 64 (4.3) | |
| asyncio idle / GC | not sampled (idle) / no GC frames | same | |

GC: 22 / 20 gen-0 and 2 / 1 gen-1 collections per window, no gen-2 (counters), so it is negligible. The
main event-loop thread was on-CPU 0.48 s (S1) and 0.63 s (S2) per 30 s, 1.6-2.1 %. Notes:
- The scrape costs ~600 ms per 30 s, which would be ~20 ms per scrape at a 1 Hz scraper (INFERRED). It
  holds the GIL in a worker thread, so it can delay SSE delivery by up to that much once a second. It
  does not touch the engine.
- In S2 the tool parser spent ~0.75 ms per streaming tool-call invocation (236 ms / 315).

### Top-20 inclusive (pure wrappers removed: thread bootstrap, CLI entry, asyncio runner, threadpool run)

| # | frame | S1 samples (%) | S2 samples (%) |
|---|---|---|---|
| 1 | `metrics (prometheus_fastapi_instrumentator/instrumentation.py:301)` | 123 (39.5) | 131 (35.5) |
| 2 | `run (asyncio/runners.py:118)` (whole event loop) | 119 (38.3) | 158 (42.8) |
| 3 | `generate_latest (prometheus_client/exposition.py:348)` | 60 (19.3) | 69 (18.7) |
| 4 | `collect (prometheus_client/registry.py:107)` | 37 (11.9) | 50 (13.6) |
| 5 | `sample_line (prometheus_client/exposition.py:299)` | 35 (11.3) | 46 (12.5) |
| 6 | `generate_latest (prometheus_client/exposition.py:317)` | 36 (11.6) | 43 (11.7) |
| 7 | `_tokenize_prompt (vllm/renderers/base.py:576)` | 36 (11.6) | 35 (9.5) |
| 8 | `__call__ (vllm/tokenizers/hf.py:93)` | 36 (11.6) | 35 (9.5) |
| 9 | `__call__ (transformers/tokenization_utils_base.py:2512)` | 36 (11.6) | 35 (9.5) |
| 10 | `_encode_plus (transformers/tokenization_utils_tokenizers.py:1027)` | 36 (11.6) | 35 (9.5) |
| 11 | `stream_response (starlette/responses.py:252)` | 3 (1.0) | 62 (16.8) |
| 12 | `chat_completion_stream_generator (chat_completion/serving.py:661)` | 1 (0.3) | 56 (15.2) |
| 13 | `parse_delta (vllm/parser/abstract_parser.py:884)` | 1 (0.3) | 56 (15.2) |
| 14 | `extract_tool_calls_streaming (vllm/parser/engine/parser_engine.py:610)` | 1 (0.3) | 58 (15.7) |
| 15 | `_events_to_delta (vllm/parser/engine/parser_engine.py:812)` | 1 (0.3) | 56 (15.2) |
| 16 | `_handle_arg_chunk (vllm/parser/engine/parser_engine.py:924)` | 1 (0.3) | 54 (14.6) |
| 17 | `collect (prometheus_client/metrics.py:91)` | 27 (8.7) | 28 (7.6) |
| 18 | `create_chat_completion (chat_completion/serving.py:269)` | 19 (6.1) | 10 (2.7) |
| 19 | `output_handler (vllm/v1/engine/async_llm.py:832)` (detokenize path) | 12 (3.9) | 15 (4.1) |
| 20 | `count_reasoning_tokens (vllm/parser/abstract_parser.py:944)` | 10 (3.2) | 4 (1.1) |

### Top-20 self

| # | frame | S1 samples (%) | S2 samples (%) |
|---|---|---|---|
| 1 | `run (asyncio/runners.py:118)` (uvloop native) | 71 (22.8) | 64 (17.3) |
| 2 | `_encode_plus (transformers/tokenization_utils_tokenizers.py:1027)` | 36 (11.6) | 35 (9.5) |
| 3 | `sample_line (prometheus_client/exposition.py:299)` | 13 (4.2) | 23 (6.2) |
| 4 | `sample_line (prometheus_client/exposition.py:310)` | 15 (4.8) | 9 (2.4) |
| 5 | `_escape (prometheus_client/openmetrics/exposition.py:224)` | 10 (3.2) | 11 (3.0) |
| 6 | `_protected_step (vllm/v1/engine/detokenizer.py:224)` | 9 (2.9) | 10 (2.7) |
| 7 | `<lambda> (<string>:1)` (prometheus sample namedtuple) | 7 (2.3) | 11 (3.0) |
| 8 | `_qwen3_arg_converter (vllm/parser/qwen3.py:72)` | 0 (0.0) | 17 (4.6) |
| 9 | `floatToGoString (prometheus_client/utils.py:25)` | 9 (2.9) | 6 (1.6) |
| 10 | `_qwen3_arg_converter (vllm/parser/qwen3.py:78)` | 1 (0.3) | 15 (4.1) |
| 11 | `escape_label_name (prometheus_client/openmetrics/exposition.py:207)` | 6 (1.9) | 4 (1.1) |
| 12 | `_multi_samples (prometheus_client/metrics.py:278)` | 5 (1.6) | 5 (1.4) |
| 13 | `_is_valid_legacy_labelname (prometheus_client/validation.py:98)` | 6 (1.9) | 3 (0.8) |
| 14 | `floatToGoString (prometheus_client/utils.py:24)` | 3 (1.0) | 6 (1.6) |
| 15 | `_multi_samples (prometheus_client/metrics.py:277)` | 3 (1.0) | 6 (1.6) |
| 16 | `_decode (transformers/tokenization_utils_tokenizers.py:1100)` | 5 (1.6) | 3 (0.8) |
| 17 | `_worker (concurrent/futures/thread.py:90)` | 4 (1.3) | 4 (1.1) |
| 18 | `escape_metric_name (prometheus_client/openmetrics/exposition.py:184)` | 4 (1.3) | 3 (0.8) |
| 19 | `add_sample (prometheus_client/metrics_core.py:39)` | 4 (1.3) | 3 (0.8) |
| 20 | `collect (prometheus_client/process_collector.py:92)` | 3 (1.0) | 4 (1.1) |

## 3. EngineCore

### 3a. Threads (dumps S1 12:13:52, S2 12:15:25, S3 12:19:25; flame graph S3)

| thread | S1 dump | S2 and S3 dumps | flame graph samples | class |
|---|---|---|---|---|
| MainThread | **active**: `execution_fn (<string>:683)` in the target forward (`_model_forward` <- `execute_model` :4477 <- `core.py:641`) | **active**: `build (flashinfer.py:1527)` <- `build_for_drafting` <- `propose` :568 (first draft pass) <- `sample_tokens` :4685 <- `core.py:649` | 5,168 (98.0 %) | step loop |
| Thread-1 `_report_usage_worker` | idle | idle | 0 | telemetry |
| `tqdm_monitor`, `signal-callback` | idle | idle | 0 | |
| `vllm_offloading_lookup_fs` + 12 `vllm_kv_py_fs_*` | idle (`Condition.wait_for`, thread_pool :166) | idle | 0 | connector fs tier |
| Thread-2 `process_input_sockets` | idle in `zmq poll` (core.py:1811) | idle | 0 | IPC in (from API) |
| Thread-3 `process_output_sockets` | idle in `queue.get` (core.py:1891) | idle | 0 | IPC out (to API) |
| truncated stacks (non-blocking read races) | | | 106 (2.0 %) | mostly fragments of main-thread ops |

All non-main threads have zero samples, so the engine does no work outside the main thread: no
connector I/O, no IPC encode/decode and no GC shows up. With `--nonblocking`, leaf Python frames under
eager ops are often missing. For example, 890 samples stop at `qwen_gdn_attention_core :1927` (the
`self._forward_core(...)` call) without a `_forward_core` child, and the plugin does not patch it. Time is
therefore attributed at the level of the calling op, which is enough for the split below.

### 3b. Main-thread step loop by segment (S3 flame graph)

ms/step assumes the main thread is on-CPU for the whole step at 72 ms/step (the gap just before S3) or
89 ms/step (S1). INFERRED: it is not exactly on-CPU the whole step, see the last row.

| segment | samples | % of step loop | ms/step at 72 / 89 | GPU state during it (INFERRED) |
|---|---|---|---|---|
| **drain wait**: `seq_lens.cpu()` at `flashinfer.py:1527`, first draft pass (`propose` :568) | 2,393 | 46.3 | 33 / 41 | busy: finishing target forward + sampler |
| draft-loop `seq_lens.cpu()` waits (`propose` :714, passes 2..k) | 71 | 1.4 | 1.0 / 1.2 | short waits, GPU nearly drained |
| target forward launch (`_model_forward`): | 1,998 | 38.7 | 28 / 34 | busy (host ahead, see below) |
| - eager GDN op `qwen_gdn_attention_core` (48 layers) | 1,180 | 22.8 | 16 / 20 | |
| - CUDA-graph piece replays (`cuda_graph.py:256` -> inductor `output_code.py:763`) | 455 | 8.8 | 6.3 / 7.8 | |
| - stitching `execution_fn` + other eager ops (`torch/_ops.py`, FlashInfer attention) | 363 | 7.0 | 5.1 / 6.3 | |
| drafter host work outside the waits (`propose` lines 651/714/762, build/plan rest, `_copy_draft_token_ids_to_cpu`, `propose_draft_token_ids` own) | 275 | 5.3 | 3.8 / 4.7 | **mostly idle** (after the drain) |
| runner prep: `_build_attention_metadata` (104), `execute_model` own :4281/:4411 (90), wrapper (34) | 228 | 4.4 | 3.2 / 3.9 | **idle** (before the first target kernel) |
| sampler + rejection sampler (`apply_top_k_top_p_small_k`) | 155 | 3.0 | 2.2 / 2.7 | busy (enqueued behind the target) |
| `schedule` (core.py:640), `update_from_output` (:654), `post_step`, busy loop | 30 | 0.6 | 0.4 / 0.5 | idle |
| KV connector / offload hooks | 0 | 0 | 0 | |
| of which GDN chunked prefill (`chunk_gated_delta_rule`, inside the eager GDN op above) | 123 | 2.4 | | prefill steps occurred |
| main thread off-CPU (not sampled: 7,500 ticks - 5,168) | ~2,330 | (31 % of ticks) | | unknown: idle engine, blocking sync, or D-state |

What the main-thread frames are (SOURCE):
- **`flashinfer.py:1527`** is `seq_lens_cpu = common_attn_metadata.seq_lens.cpu()` under
  `gpu_sync_allowed()`. The gate at :1520 skips it only when `_num_speculative_tokens == 0`, and sm86
  has no TRTLLM path. The first MTP draft pass reaches it inside `sample_tokens`, after the target
  forward, sampler and rejection sampler are enqueued. The blocking D2H copy waits for all of that, and
  "active" fits the CUDA driver spin-waiting.
  - **The main thread is blocked in a CUDA sync in 2/3 dumps and ~47 % of samples.** During the wait the
    GPU is busy. When it returns, the GPU queue is empty, so everything up to the first MTP kernel
    (`plan()`, input prep, launch) and each later host segment runs with the GPU idle.
  - The sync is on the critical path as the drain point. It does not itself add idle.
  - This also explains why removing the per-pass syncs did nothing on the box: the host is ahead at
    that point, so the wait just moves to the next sync.
- **`execution_fn <string>:683`** (S1) is vLLM's generated stitching function for the PIECEWISE graph
  (`compilation/codegen.py`): one line per graph-piece replay or eager splitting op. The flame graph
  shows that 59 % of target-forward host time is the eager GDN op (48 calls per forward). The host is
  ~33-41 ms ahead when it reaches the drain point, so this launch work overlaps GPU work. It is not idle
  today, but at 16-20 ms/step it is the largest piece of host work.
- **Exposed host work per step** (INFERRED: every segment that starts after a drain with an empty GPU
  queue): drafter host work (5.3 %) + runner prep and target metadata (4.4 %) + scheduler/update
  (0.6 %) ≈ 10.3 % of the step, **~7.5-9 ms/step at 72-89 ms**. The profile's nvidia-smi estimate was
  12-16 ms. The remaining ~3-7 ms must come from the 31 % of ticks where the main thread was off-CPU,
  or from GPU-side bubbles. Neither can be seen without `--idle` and a CUDA trace.
- Nothing unusual (garbled-output window): no exception, abort, retry, `_preempt_request`,
  recompute, connector-failure or fault-sentinel frames. `run_with_fault_tolerance` has 3 own samples,
  which are line-level fragments. The step loop looks like normal decode with some chunked prefill
  (`chunk_gated_delta_rule`, 2.4 %). The engine's exit at ~12:28:32 is not visible in the samples.

### 3c. EngineCore top-20 (S3 flame graph, 5,274 samples)

Inclusive. Pure pass-through wrappers are removed: process/loop entry, executor/worker RPC,
torch.compile and AOT-autograd wrappers, and the drafter metadata pass-throughs. Each has one child
with the same count. The full chain is in 3b.

| # | frame | samples | % |
|---|---|---|---|
| 1 | `step (vllm/v1/engine/core.py:649)` | 2,909 | 55.2 |
| 2 | `sample_tokens (vllm/v1/worker/gpu_model_runner.py:4685)` | 2,739 | 51.9 |
| 3 | `build (vllm/v1/attention/backends/flashinfer.py:1527)` | 2,464 | 46.7 |
| 4 | `propose (vllm/v1/spec_decode/llm_base_proposer.py:568)` | 2,424 | 46.0 |
| 5 | `step (vllm/v1/engine/core.py:641)` | 2,226 | 42.2 |
| 6 | `execute_model (vllm/v1/worker/gpu_model_runner.py:4477)` | 1,998 | 37.9 |
| 7 | `__call__ (vllm/compilation/caching.py:224)` | 1,986 | 37.7 |
| 8 | `__call__ (torch/_ops.py:1279)` | 1,250 | 23.7 |
| 9 | `qwen_gdn_attention_core (vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:1927)` | 1,180 | 22.4 |
| 10 | `__call__ (vllm/compilation/cuda_graph.py:256)` | 455 | 8.6 |
| 11 | `__call__ (torch/_inductor/output_code.py:763)` | 433 | 8.2 |
| 12 | `sample_tokens (vllm/v1/worker/gpu_model_runner.py:4616)` | 157 | 3.0 |
| 13 | `_sample (vllm/v1/worker/gpu_model_runner.py:3713)` | 155 | 2.9 |
| 14 | `apply_top_k_top_p (vllm/v1/sample/ops/topk_topp_sampler.py:509)` | 153 | 2.9 |
| 15 | `_forward_core (vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:1509)` | 140 | 2.7 |
| 16 | `forward (vllm/model_executor/custom_op.py:134)` | 136 | 2.6 |
| 17 | `forward_native (vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:312)` | 136 | 2.6 |
| 18 | `_fn (torch/_dynamo/eval_frame.py:1446)` | 130 | 2.5 |
| 19 | `chunk_gated_delta_rule (vllm/third_party/flash_linear_attention/ops/chunk.py:230)` | 123 | 2.3 |
| 20 | `_build_attention_metadata (vllm/v1/worker/gpu_model_runner.py:2652)` | 104 | 2.0 |

Self:

| # | frame | samples | % |
|---|---|---|---|
| 1 | `build (vllm/v1/attention/backends/flashinfer.py:1527)` | 2,464 | 46.7 |
| 2 | `qwen_gdn_attention_core (vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:1927)` | 890 | 16.9 |
| 3 | `__call__ (torch/_inductor/output_code.py:763)` | 402 | 7.6 |
| 4 | `__call__ (vllm/compilation/caching.py:224)` | 162 | 3.1 |
| 5 | `_forward_core (vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:1383)` | 86 | 1.6 |
| 6 | `__call__ (torch/_ops.py:1279)` | 76 | 1.4 |
| 7 | `_build_attn_group_metadata (vllm/v1/worker/gpu_model_runner.py:2586)` | 66 | 1.3 |
| 8 | `apply_top_k_top_p_small_k (vllm/v1/sample/ops/topk_topp_sampler.py:477)` | 66 | 1.3 |
| 9 | `apply (torch/autograd/function.py:625)` | 55 | 1.0 |
| 10 | `forward (vllm/third_party/flash_linear_attention/ops/chunk.py:113)` | 55 | 1.0 |
| 11 | `apply_top_k_top_p (vllm/v1/sample/ops/topk_topp_sampler.py:509)` | 53 | 1.0 |
| 12 | `_copy_draft_token_ids_to_cpu (vllm/v1/worker/gpu_model_runner.py:4921)` | 50 | 0.9 |
| 13 | `all` | 44 | 0.8 |
| 14 | `decorate_context (torch/utils/_contextlib.py:124)` | 38 | 0.7 |
| 15 | `forward (vllm/model_executor/models/qwen3_next.py:689)` | 37 | 0.7 |
| 16 | `execute_model (vllm/v1/worker/gpu_model_runner.py:4281)` | 35 | 0.7 |
| 17 | `build_for_drafting (vllm/v1/attention/backend.py:754)` | 33 | 0.6 |
| 18 | `propose_draft_token_ids (vllm/v1/worker/gpu_model_runner.py:5265)` | 30 | 0.6 |
| 19 | `_forward_core (vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:1335)` | 20 | 0.4 |
| 20 | `apply_top_k_top_p_small_k (vllm/v1/sample/ops/topk_topp_sampler.py:478)` | 19 | 0.4 |

## 4. Verdict

**The API server is not the cause (refuted).**
- Its process uses 4.6-5.1 % of one core. The event loop is on-CPU 1.6-2.1 %, and 40-48 % of samples are
  `/metrics` exposition in a worker thread.
- Even fully serialized it would be at most 3.6-4.9 ms/step. It is not serialized:
  - SOURCE: `run_busy_loop` blocks on input only when `has_work()` is false, and outputs go
    `put_nowait` to an unbounded queue drained by its own thread.
  - MEASURED: both IPC threads were idle in every dump and have 0 samples in the engine graph.
- Its work overlaps the engine's steps. The engine never waits for it.

**Also refuted for this window (engine graph):** the KV-offload connector hooks (0 samples), scheduler +
`update_from_output` + `post_step` (0.6 %, ~0.5 ms/step), GIL contention from the engine's own threads
(0 samples on any other thread), and GC (no frames).

**Supported:** the engine main thread is the bottleneck thread at ~1 core. About 47 % of its time is the
drain wait at the first draft pass's `seq_lens.cpu()`, 39 % is the target-forward launch (overlapped),
and ~10 % is host work that runs with the GPU empty.

Remaining hypotheses, ranked:
1. **H1, exposed serial host work after the drain points: ~7.5-9 ms/step.** This is the drafter's
   first-pass `build` + `plan()` and inputs, the k-1 loop rebuilds (each pass waits only ~0.1 ms, so the
   draft loop is host-bound), then runner prep (`_build_attention_metadata` 1.3 % self in
   `_build_attn_group_metadata`, `execute_model` own lines). It accounts for most of the measured idle.
   Levers:
   - build the draft-pass metadata from host-known lengths, so it does not need the drain;
   - overlap via async scheduling (test-only config);
   - prepare the next step's inputs before the drafter's final copy.
2. **H2, slower or contended host thread on prod.** The same exposed segments are ~2x the box's 6.5 ms
   in total. Compare per-frame ms/step with the box's `22-idle-profile` py-spy. A uniform ~2x means
   host speed (3.9 GHz cap, EPP `power`, VM on an SMT sibling).
3. **H4, off-CPU main-thread time: 31 % of ticks unsampled.** This is engine idle (no requests), a
   blocking CUDA sync, or page-fault/compaction D-state. It would explain the ~3-7 ms not covered by H1.
   It is only decidable with `--idle` plus `/proc/vmstat` deltas.
4. **H3, traffic-driven scheduler/connector work: refuted** in this window (< 1 %). Recheck only during a
   preemption burst.

## 5. What the next engine capture must show

The S3 graph answered most of the earlier questions. What is still open:
- **Exact window and rate:** record with `--idle` at ≤ 100 Hz for 30 s and scrape `/metrics` at both
  ends. With that, samples per step and ms per segment are exact instead of the 72-89 ms bracket.
- **H1 vs H2:** ms/step of the exposed segments (`propose` minus the :1527 waits,
  `_build_attention_metadata`, `execute_model` prep, `schedule`, `update_from_output`) next to the box's
  values for the same frames. If prod is ~2x on every frame: H2. If one frame is out of line: that frame.
- **H4:** with `--idle`, where the off-CPU samples land:
  - `input_queue.get` means no work, so the gap is not real;
  - `:1527` or `_copy_draft_token_ids_to_cpu` means a blocking sync (check the CUDA sync mode);
  - any other leaf means faults or contention. Pair with `/proc/vmstat` `compact_stall`/`allocstall`
    deltas.
- **Native frames** (`--native` without `--nonblocking`, so it pauses the engine; keep the rate low)
  would name the leaves under the eager GDN op and `plan()`, and show `cuStreamSynchronize` vs spin.
- **Garbled output / exit:** this capture shows no error path. The exit at ~12:28:32 needs the service
  logs (out of scope here).
