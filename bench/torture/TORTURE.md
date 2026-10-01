# Torture soak

Garrett's bar (Q3): "Abuse it in every way possible for at least ~12 hours. If that is good, then it's
probably good." `torture` does that to any OpenAI-compatible vLLM endpoint.

It is standard-library Python (3.12+), with no torch, vllm or tokenizer on the client. It can run
from anywhere, against anything: production on ms4, a rented box, or a server it starts itself.

Canonical source: this directory (`bench/torture/` on branch `torture`). The installed copy on ms4 is
`~/tools/torture-harness/`; refresh it with `./sync-from-repo.sh`.

| file | role |
|---|---|
| `torture` | the CLI |
| `load.py` | schedule, request mix, sending, per-response classifier |
| `monitor.py` | optional server-side monitor, server lifecycle, leftover checks |
| `report.py` | `report.json`, `REPORT.md`, verdict |

`bench/torture.sh` (repo only) wraps it for this repo's own models. That wrapper:
- serves them from `.venv-main` under production's vLLM-main argv (`env/prod-main-serve-argv.txt`:
  MTP k=5 with the per-batch schedule, 16 seqs, fp8 KV, prefix caching, CPU + fs KV offload tiers,
  CUDA graphs up to 48, `VLLM_USE_V2_MODEL_RUNNER=0`);
- applies the repo's guards;
- writes results under `$GSQ_RUNS`.

## Usage

```bash
./torture --plan [--hours 12 | --minutes 20] [--seed N]              # the schedule; contacts nothing
./torture --base-url http://HOST:PORT/v1 [--hours 12 | --minutes 20] [options]
./torture serve --cmd SCRIPT --port N [options]                       # start, torture, stop
./torture switch --a SCRIPT --b SCRIPT --port N [--rounds 3] [--minutes 10] [--tier-root DIR] [options]
./torture report RUN_DIR                                              # rewrite the report
```

The model comes from `/v1/models` (or `--model`). The API key comes from `--api-key` or
`VLLM_API_KEY` / `OPENAI_API_KEY`. Only the given host:port is ever contacted.

Ctrl-C, or the server dying or restarting, stops the run at once: requests in flight are
cut, and the report is still written.

### Politeness (default)

Torture traffic shares the server with real users. vLLM gives them only limited protection, so the
harness throttles itself:
- Priority. Every request carries vLLM's `priority` field: `--priority` P (default 100000) plus 0-20.
  The spread exercises priority ordering among the torture's own requests. Under
  `--scheduling-policy priority` (production's argv), vLLM admits *waiting* requests in priority
  order, so a user's request (priority 0) goes ahead of queued torture requests. It never preempts a
  *running* request to admit one: a user still waits for a seat or for KV blocks. Under FCFS, vLLM
  main ignores the field. Older vLLM rejects non-zero priority with HTTP 400; pass `--priority 0`
  there.
- Seats. At most `--max-conc` torture requests are in flight (default 12). Production's 16 seqs
  then leave 4 seats for users.
- KV. Torture requests in flight hold at most `--token-budget` tokens of prompt + n × max_tokens.
  The default is half the server's KV cache, read from `vllm:cache_config_info` (num_gpu_blocks ×
  block_size), or 65,536 when the metric is missing. A request bigger than the budget runs alone.
  A 196k prompt still runs, alone, and holds most of the KV while it does.
- `--brutal` drops the seat and KV throttles. Against ports 18080/18081 it also needs `--i-know`.

The prefix and long-context phases write tens of GB through the CPU and fs KV tiers. On production
that wears the KV drive (the used 660p, where wear is the limit) and evicts users' cached prefixes
from the GPU and CPU tiers. Expect slower first tokens for real users during and after the run.

### On ms4, against production

```bash
cd ~/tools/torture-harness
./torture --base-url http://127.0.0.1:18081/v1 --minutes 20 --priority 100000    # smoke first
journalctl -u qwen-vllm -f -n 0 -o cat > /tmp/qwen-vllm.log &                     # optional: fault grep, new lines only
./torture --base-url http://127.0.0.1:18081/v1 --hours 12 --priority 100000 \
  --server-pid "$(systemctl show -p MainPID --value qwen-vllm)" --server-log /tmp/qwen-vllm.log
```

Without `--server-pid` and `--server-log`, the server side is only `/health` and `/metrics` (gauges,
plus restart detection through `process_start_time_seconds`). Note that systemd restarts production
by itself: a restart shows up as a fault and stops the run.

### This repo's models (GSQ-RCO GGUF, EXL3)

```bash
bench/torture.sh --plan
GSQ_ALLOW_GPU=1 bench/torture.sh run --serve scripts/serve-gsq.sh --minutes 20            # smoke
GSQ_ALLOW_GPU=1 bench/torture.sh run --serve scripts/serve-gsq.sh --hours 12
GSQ_ALLOW_GPU=1 bench/torture.sh run --serve ../wt-exl3/scripts/serve-exl3.sh --hours 12
GSQ_ALLOW_GPU=1 bench/torture.sh switch --a scripts/serve-gsq.sh --b ../wt-exl3/scripts/serve-exl3.sh --rounds 3
```

The wrapper refuses ports 18080/18081 and refuses to run while production is live
(`GSQ_ALLOW_BESIDE_PROD=1` overrides, and only with Garrett's go). It refuses an argv with
`--enforce-eager`. `--port N` moves the server (default 18090). Other options pass through to `torture`.

The EXL3 serve script lives on branch `exl3`. Point at a checkout of it; it reads its own
`scripts/env.sh`. That checkout needs:
- the exl3 plugin installed in the venv;
- `tools/exl3_draft_head.py` run once on the model dir (see the script's header).

### As one gpuq job (12 h)

gpuq stores a job's command as `"$*"`, so quoting is lost. Use a wrapper like the soak's
(`cloud/results/final/box-scripts/soak.sh`). Without PYTHONPATH, the box's shared venv imports an old
editable plugin build. Check that `GSQ_VENV` points at the vLLM-main venv.

```bash
cat > /workspace/torture-gsq.sh <<'EOF'
#!/bin/bash
source /workspace/box-env.sh
export PYTHONPATH=/workspace/wt-torture/plugin:/workspace/wt-torture/tools GSQ_ALLOW_GPU=1 VLLM_GGUF_LCPP=1 GSQ_RUNS=/workspace/runs
cd /workspace/wt-torture && exec bench/torture.sh run --serve scripts/serve-gsq.sh --hours 12
EOF
gpuq submit torture-gsq --cwd /workspace/wt-torture -- bash /workspace/torture-gsq.sh
```

For EXL3, the same with `--serve /workspace/wt-exl3/scripts/serve-exl3.sh`. The job runs about
12 h 20 min: startup, the plan, then draining the last phase.

## What it sends

A seeded schedule of phases. Each cycle shuffles all eleven types, gives each 10-20 min, and adds a
2-5 min idle gap after every third. Twelve hours is about four cycles. Shorter runs scale the
durations down (floor 20 s, idle 5 s), so a 20-minute smoke still reaches every type.

| phase | traffic |
|---|---|
| ramp | concurrency 1 → 16 → 1 across the phase, mixed requests |
| burst | waves of 16 requests sent at the same instant |
| long_ctx | one lane of 32k/96k/196k raw prompts (30% prefix hits) beside two chat lanes |
| prefix | six lanes of 1k/8k/32k prompts, 70% prefix hits: half reuse one of the last four prompts of that length, half any earlier one (older ones are likelier to come back from the CPU/fs tiers) |
| cancel | eight streaming lanes: 30% cancelled mid-stream (after 1-64 chunks or 0.05-5 s), 10% client timeouts |
| tools | qwen3_coder tool calls; tool_choice absent / auto / required / named / none |
| structured | `response_format` json_schema and json_object (vLLM serves both by default; production's `bench/api_smoke.py` uses json_schema) |
| plogprobs | `prompt_logprobs=1` on 2k-4k prompts. For the first half of the phase it is the harness's only lane. When it is also the server's only request, vLLM drops the 128-token prefill cap and the lm_head sees 2048-row chunks (the EXL3 >144-row reconstruct path); real traffic on a shared server prevents that. Then beside two chat lanes. |
| repeat | eight lanes resending identical requests (exact prefix hits) |
| api | n=2, stop strings, seed, logprobs + top_logprobs, min_tokens + penalties, completions echo |
| full | sixteen lanes of everything up to 32k prompts (KV pressure, preemptions) |
| idle | no traffic: do graphs and memory survive a quiet gap? |

These apply in every phase:
- 5% of streams are cancelled at a random point and 1% of requests get a 0.5-10 s client timeout.
- max_tokens runs from 1 to 4096.
- Temperature is 0 for 30% of requests; the rest sample (1.0/0.95/20 with thinking on, 0.7/0.8/20 with it off).
- Thinking is on or off via `chat_template_kwargs.enable_thinking`, as production's clients send it.
- The greedy probe: fixed prompt, T=0, thinking off, 256 tokens, streamed. It opens every phase on a
  drained server and measures latency drift, idle recovery and T=0 repeatability.
- `/tokenize` → `/detokenize` round trips and `/metrics` are polled every 15 s.

Long prompts are token ids, so the server-side prompt count can be checked exactly. The corpus is
Python's standard-library sources, shuffled by the seed and tokenized once at start by the server's
own `/tokenize` (per file, each under max_model_len). Each prompt is either a fresh 64-bit salt in
front of that corpus or a known earlier salt plus a new tail; that is how the hit ratios are known.
The salt digits are token ids of the hex characters. The corpus fingerprint and the Python version
that read it go to `plan.json`.

### Per-response checks (`classify` in `load.py`)

Each response gets one of these classes:
- `ok`
- `reasoning_only`: no content but some reasoning, usually because max_tokens ended inside the thinking.
- `eos_first`: the first token was EOS. The soak traced this to the model's own distribution at T>0.
- `short_empty`: under 8 tokens with no text.
- `truncated_json`: a structured output cut off by max_tokens.
- `cancelled`, `client_timeout`: induced by the client on purpose.

All of those are expected. Every other class fails the run:

| class | meaning |
|---|---|
| `stalled` | an unplanned timeout, over 1 h |
| `conn_error` | the connection failed |
| `http_4xx`, `http_5xx` | non-200 status |
| `stream_error` | an error event mid-stream |
| `bad_response` | missing choices, or a choice count different from n |
| `bad_count` | prompt tokens ≠ ids sent; completion > max_tokens; `length` with fewer than max_tokens; under min_tokens |
| `bad_finish` | a finish reason other than stop, length or tool_calls |
| `bad_logprobs` | non-finite or positive logprobs, or prompt_logprobs not one entry per prompt token |
| `bad_tool_json` | tool-call arguments that are not a JSON object |
| `no_tool_call` | required or named tool_choice but no call |
| `bad_tool_choice` | a tool call under tool_choice none |
| `bad_json` | structured output that is not JSON, or misses the schema |
| `empty_output` | 8+ tokens with no content, reasoning or tool call. This is the shape of the soak's two unexplained empties. |
| `tokenize_mismatch` | the `/tokenize` → `/detokenize` round trip changed the text |
| `metrics_error` | `/metrics` went wrong |
| `harness_error` | a bug in the harness itself |

Every response is logged to `load.jsonl` with its text: 300 characters when it is fine, 8 KB when it is not.

## Verdict (`report.py`)

The run passes only if every criterion holds:

| criterion | rule |
|---|---|
| faults | `faults.log` is empty: server-log lines matching illegal memory access, misaligned address, launch failure, CUDA error, EngineDeadError, engine core died, Traceback, segfault, out of memory; or server exited / restarted / unhealthy for 3 rows / never came up |
| alive | every 60 s row shows the server alive (with a pid); `/health` 200 except isolated single misses (never two rows in a row; three end the run); 0 restarts; monitored ≥ 98% of the plan (less 2 min) |
| errors | every response is in the OK classes above |
| memory | with `--server-pid`, after warm-up (first hour, or first quarter of a shorter run), measured as trend (least-squares slope × span): server GPU MiB grows ≤ 256 and host **anon** RSS grows ≤ 1 GiB. Shmem is the CPU KV tier filling up, bounded by its size, and is reported but not judged. |
| drift | greedy-probe TPOT, 25th percentile, last hour vs first hour: ≤ +15%. The probes run at c=1 on a drained harness, so the load is equal. The 25th percentile keeps real traffic on a shared server from deciding it. Applies to runs ≥ 3 h. |
| idle | probe TPOT p25 right after idle gaps ≤ +15% over the other probes (lost CUDA graphs would show); needs ≥ 3 such probes |
| lifecycle | serve mode: came up healthy; stopped without SIGKILL; no process left in its session after 60 s; port free after 30 s |
| coverage | every phase type ran (runs ≥ 6 h) |
| switch | per leg: lifecycle as above; GPU back within 256 MiB of its idle level after 60 s (all GPUs summed); no new vLLM `/dev/shm` files (`vllm_*`, `sem.mp-*`, `torch_*`, `psm_*`); the leg's own run passes faults/alive/errors/memory. Across legs: the KV fs-tier namespace dirs (`--tier-root`) changed by A and by B are disjoint. Each side keeps its seed across rounds, so round 2 re-sends round 1's prompts and reads back what its tier stored. The tier check sees writes only, not reads. |

T=0 repeatability is recorded but never judged: distinct probe outputs and the share of the most
common one, plus, for every T=0 request body sent more than once in a phase (mostly the repeat
phase), whether the outputs differed.
Under concurrent MTP, T=0 is not batch-invariant (soak finding).

`REPORT.md` has one row per phase:
- requests and failures by class;
- cancels and timeouts;
- TTFT and TPOT p50/p95/p99 (streams only);
- measured prefix-cache hit ratio: local, and external (the KV tiers), from `/metrics` deltas;
- the planned hit ratio and the share of long prompts actually sent as hits;
- preemptions and MTP mean acceptance length.

It also has min/max/growth for every monitor column and the fault lines. `report.json` has everything,
including up to 20 failed responses with their text.

Run dir contents:

| file | contents |
|---|---|
| `plan.json` | the schedule |
| `load.jsonl` | one line per response |
| `phases.jsonl` | phase times plus `/metrics` counters at each phase's start and end |
| `monitor.csv`, `faults.log` | the 60 s rows, the fault lines |
| `server.log` | serve/switch modes only |
| `serve.json` / `switch.jsonl` | start-to-healthy, stop and leftover facts |
| `report.json`, `REPORT.md` | the report |

## Assumptions to confirm on the first real run (vLLM main)

- Completions return `choices[0].prompt_logprobs` as `[None, {token_id: {"logprob": ...}}, ...]`.
  If the shape differs, every plogprobs request shows as `bad_logprobs`.
- Chat streams carry thinking in `delta.reasoning` (`reasoning_content` is also read).
  `stream_options.include_usage` gives the final usage chunk, which TPOT and the count checks need.
- `/tokenize` takes `{"model", "prompt", "add_special_tokens"}` and returns `tokens`; `/detokenize`
  takes `{"model", "tokens"}` and returns `prompt`. The round trip is exact for Qwen's byte-level BPE.
- The metric names: `vllm:prefix_cache_{hits,queries}`, `vllm:external_prefix_cache_*`,
  `vllm:kv_offload_cpu_cache_usage_perc`, `vllm:kv_offload_fs_cache_bytes` (read from the
  `.venv-main` source), and `process_start_time_seconds` (prometheus_client's process collector).
- `tool_choice` `required`/named with the qwen3 reasoning parser and thinking on yields
  `tool_calls`, or ends on `length`.
- The 20-minute smoke is the place to check all of these. Every class it reports should be one
  listed above.

## On Garrett's production 3090

Garrett re-tests on his production 3090, either way:
- against the live server, politely: the ms4 commands above;
- with production stopped, the repo wrapper with this repo's serve scripts. The harness tier root
  `/mnt/kvcache/gsq-tier` is not production's, and `scripts/env.sh` refuses production's.

A pass on a rented box is necessary. The pass that counts is the one on his card.
