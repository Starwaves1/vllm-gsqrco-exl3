# Round 3: diagnosis plan for production's per-step host idle

Input: `cloud/results/prod-profile-20261001.md`. Production on vLLM main shows 12-16 ms of GPU idle in
every decode step at 1-9 running (nvidia-smi util x pooled ms/step). The box on 0.27.1 showed ~4.7 ms at
c=1 with short context. The box scripts in `box-scripts/` run on the rental 3090 (gpuq, one job each, logs
under `/workspace/logs/r3/<script>/`, `summary.txt` per job). Each script has `--plan`. Every server runs
production's main argv (`env/prod-main-serve-argv.txt`) with the GGUF. The only differences are host/port,
model path, tier roots and sizes, and the variable under test, which is printed in `argv-diff.txt`.

Labels: SOURCE (read in vLLM main 0.30.1rc1.dev285 + overlay 2a0fe5e1e1, i.e. `.venv-main`), HOST (read
on the production host, read-only), INFERRED.

## Facts that shape the hypotheses

- SOURCE: the step loop is synchronous under production's `--no-async-scheduling`. `EngineCore.step`
  (`v1/engine/core.py:630-659`) runs schedule, then `execute_model`, then `sample_tokens`, then
  `update_from_output`. After it, `post_step` (`core.py:661-668`) calls `take_draft_token_ids`
  (`v1/worker/gpu_model_runner.py:4886-4942`, `draft_token_ids_event.synchronize()` at :4937) and
  `scheduler.update_draft_token_ids`. None of this overlaps GPU work. The runner has further host syncs:
  `_update_states` :1275, `_prepare_inputs` :2153, :3863 `prepare_inputs_event`, and `_bookkeeping_sync`
  -> `_to_list` :7506-7519.
- SOURCE: graph mode is PIECEWISE. `config/vllm.py:1064-1080` forces it for any dynamic k schedule on the
  V1 runner. FlashInfer with spec decode also forced it on 0.27.1 (`phase3/item2/cg.txt`), so this is not
  the 0.27.1 -> main difference. The splitting ops (`config/compilation.py:764-776`) include FlashInfer
  attention and `qwen_gdn_attention_core`. Each target forward therefore runs ~65 graph pieces with 16
  attention + 48 GDN ops launched eagerly between them, and the MTP passes add more. The box measured
  5.6 ms of target-forward idle at c=1 on 0.27.1 (`phase3/item4/profile-phase.txt`).
- SOURCE: the KV-offload connector runs these on every step:
  - worker side: `kv_connector_model_runner_mixin.py:67-107` (bind, `start_load_kv`, `get_transfer_results`,
    `build_connector_worker_meta`) and `offloading/worker.py:223-262` (`handle_preemptions` submits the
    previous step's deferred stores; sync loads block);
  - scheduler side: `offloading/scheduler.py:1821-1887` `build_connector_meta` (`_update_req_states` :1246,
    partial-tail stores :1370, `_build_store_jobs` :1535);
  - tiering manager: `v1/kv_offload/tiering/manager.py:1205-1236` `on_schedule_end`. It polls finished
    jobs, serves tier requests, flushes promotions and cascades, and runs `_writeback_step` (:1037-1070).
    While the CPU tier is at least 85 % full, `_writeback_step` walks up to `num_chunks - low_chunks`
    (half the tier, ~435 chunks at production's 24 GiB / 29.5 MB) on every step. `fs/manager.py:1032`
    also flushes the lookup manager. Production's tier is full: 375 dirty and 1152 written blocks in
    the one /metrics read.
- HOST: production's host is a Ryzen 5 9600X (6C/12T). It is held at 3.9 GHz: `cpuinfo_max_freq` 3.9 GHz
  vs `amd_pstate_max_freq` 5.49 GHz, with EPP `power`. A qemu VM (`pm2`, up since 2026-09-30 18:24, so
  through the whole profile window) averages ~395 % CPU. The EngineCore main thread sits at ~97 % CPU.
  The box is a 32C/64T Threadripper 3970X with nothing else on it.
- INFERRED: "flat in n" does not mean "fixed per step". Production's total context sat at ~200k at every
  running count (kv usage 0.8-0.95), so a cost that grows with total context would also look flat. Scripts
  20 and 22 add a short-context c=2 window to separate the two.

## Hypotheses, ranked, with the evidence that decides each

| # | hypothesis | owner | decided by |
|---|---|---|---|
| H1 | Synchronous engine-loop host work (schedule, update, draft-token D2H sync, input prep, attention/GDN metadata, eager launches between graph pieces) is ~5 ms on a fast idle host and grew on main or with long context | vLLM main | 20 (box c2 vs 0.27.1's ~4.7), 22 timers + py-spy + trace, 21 `async` |
| H2 | Production's host CPU is slower or contended (3.9 GHz cap, EPP power, VM on 4 of 12 threads), so the same host work takes 2-3x longer | prod host config | 20 (box does not reproduce) then 26 (pin to 6C/12T + 4 busy threads) |
| H3 | The OffloadingConnector's per-step hooks, mainly the tiering write-back scan with a full CPU tier | connector (overlay fork + upstream) | 21 `noconn` / `nofs` vs 20, 22 timer block "KV connector" |
| H4 | Host cost that grows with total context (block tables, metadata for ~200k tokens) | vLLM main | 20 c2 vs c2s, 22 timers c2 vs c2s, 23 idle by context |
| H5 | Long-context attention: GPU time at 45-60 % of HBM bandwidth (not idle, but the largest term) | vLLM / FlashInfer | 23 attention ms vs floor at 8k / 100k / 195k |

H1 ranks first because the 0.27.1 box already showed ~5 ms from the same structure. H2 is second
because it fits a host-bound gap of 2-3x and nothing on the box resembles it. H3 has a concrete per-step
loop that only runs once the tier is full, which the box runs never reached; the scripts fill it first.

## Scripts, decision rules, GPU time

| script | measures | decision rule | GPU min |
|---|---|---|---|
| 20-idle-baseline | prod argv, CPU tier filled past the watermark. c2s (2 x 4k), c2 (2 x 96k, k=5), c8 (8 x 20k, k=3). ms/step and idle as in the prod profile | box c2 idle within ~3 ms of prod's 13.2: the gap reproduces, go on with 21/22. Box idle <= ~6 ms: it does not reproduce, run 26 and ask Garrett for a prod-side check. c2 - c2s idle >= 3 ms: H4 | 22 |
| 21-idle-connector-off | c2 with `noconn` (no `--kv-transfer-config`), `nofs` (CPU tier only), `async` (`--async-scheduling`, test only, + c8) | idle drops >= 8 ms with noconn: the connector is the cost (nofs then splits fs tier vs core), next is its hooks in 22. Drops < 3 ms: not the connector, go to 22's loop breakdown. `async`: the saving is the overlappable host time | 40 |
| 22-idle-profile | same c2 workload. sitecustomize timers per engine cycle (scheduler, runner, connector, tiering, CUDA syncs), py-spy record --native 30 s of EngineCore + API, py-spy dumps, torch profiler 25 steps (CUPTI; nsys is not in the container) | the largest host segment that runs while the GPU idles (trace "GPU idle by host scope") is the target. Connector hooks > 3 ms/cycle: H3. Event.synchronize >> GPU busy: D2H wait chain. Main-thread CPU << cycle-minus-sync: GIL contention from the tier's I/O threads | 18 |
| 23-attn-share | torch traces at c1 8k / 100k / 195k (prefix hit) and c2 x 96k: attention, GDN, GEMM ms per step, split target vs draft, vs HBM floor | attention at < 60 % of the floor at >= 100k: a FlashInfer decode-kernel item (split-KV / q_len 6), not ours. GPU idle rising with context: H4 | 20 |
| 24-k-schedule-test | TEST ONLY. c=9 x 20k at T=default and T=0, schedule `[9,16,2]` (prod) vs `[5,9,3],[10,16,2]` | propose only if >= +5 % gen tok/s at T=default without an ms/step regression beyond the added rows | 25 |
| 25-prefill-chunk-test | TEST ONLY. `--long-prefill-token-threshold` 128 / 256 / 512, c=4 mixed (3 decoders + prefill lane with 600 / 1400 / 60k+1.8k turn / 4000). TTFT, decoder ms/step, tok/s | propose if turn TTFT drops >= 20 % for <= 3 ms of decoder ms/step | 30 |
| 26-idle-cpu-contention | conditional on 20. c2 pinned to 6C/12T, without and with 4 busy threads | pin6-load idle approaching prod's: H2, a host-side fix (Garrett) | 25 |

Total about 2.6 GPU-hours, or 3 h with 26.

## Ours vs vLLM, and the upstream path

- Ours (gsq-vllm plugin): Route L GEMMs. They are captured inside the graph pieces, so their host cost is
  near zero. The eager lm_head and draft-head calls and the plugin plumbing are ours. If 22 shows plugin
  frames on the host path, the fix goes in `plugin/` on branch `r3`, measured with the ladder and parity.
- vLLM main (engine loop, runner, FlashInfer metadata, graph splitting): a minimal patch on a branch in
  this repo (`cloud/results/r3/patches/`). It is measured on the box with an overlaid copy of venv-main
  (never the shared venv, never prod) and comes with an upstream-ready description for
  vllm-project/vllm.
- OffloadingConnector / tiering: the tiering write-back is in the overlay fork (Starwaves1/vllm
  `qwen38/main`, e.g. #7). A fix there is a patch against `qwen38/main`, proposed to Garrett. The core
  connector hooks go upstream like the vLLM item.
- Production host (H2): proposal only (VM pinning off the EngineCore's cores, CPU boost/EPP). Garrett's
  call. Any prod-side capture (py-spy on prod's EngineCore, a VM pause test) also needs his approval.
- Config items (async scheduling, k schedule, prefill threshold): test-only numbers and a written
  proposal. Nothing is left changed.
