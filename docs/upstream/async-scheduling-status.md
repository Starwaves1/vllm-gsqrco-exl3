# Async scheduling: upstream status and our fixes (2026-10-11)

Question: has upstream vLLM fixed the async-scheduling deadlock, and do we have a fix now?

Trees used:
- **Latest main**: `/home/garrett/vllm-latest` at `03091e28b8ba` (2026-10-12 09:25 +0800, #59026). Read-only. Tests ran on `git archive` copies in `/tmp/async-sched-check/{stock,fixed}`.
- **Production**: `qwen-vllm.service` runs `/home/garrett/gsq-vllm/.venv-main/bin/python` (`qwen38-27b-rtx3090/venv` is a symlink to it). Deployed commit: Starwaves1/vllm `qwen38/main` `05efb4d1fc2c` (`deploy/vllm-deployed-commit`), based on upstream `d28795f1a7` (2026-09-29). Main is 758 commits ahead of that base.
- **Running process** (PID 3270955): `--no-async-scheduling`, `"sync_load":true`, MTP `num_speculative_tokens: 3`.

## Bug 1: deferred-frees head-of-line deadlock

### Upstream still has it
At `03091e28b8ba`, `vllm/v1/core/sched/scheduler.py`:
- `:2798-2810` `_drain_deferred_frees` still stops at the first pending fence (`if fence > self.processed_step_seq: break`). Commit `481e481be7` (#46384, merged 2026-07-12) changed only the docstring, to say a satisfied entry behind a pending fence "is merely freed later". That is the assumption our fix refutes.
- `:1484-1492` A CoW retention fence is still `sched_step_seq + 1`, recorded even when the schedule ends with 0 tokens.
- `:1560-1563` and `:2054-2058` `sched_step_seq` and `processed_step_seq` still advance only on non-empty steps, and the drain runs only there.
- `:2776-2785` Request frees still queue behind with fence `sched_step_seq`, which is at most the CoW fence.
- `:186-192` with `vllm/config/vllm.py:577-585` The mechanism is on whenever `max_concurrent_batches > 1` (async scheduling or PP) and the connector is a KV consumer. OffloadingConnector `kv_both` counts as a consumer.

Reachability on main: one path that produces a 0-token schedule carrying CoW copies is async-load admission. `allocate_slots` runs with external tokens (`:1256-1272`), the request parks in `WAITING_FOR_REMOTE_KVS` (`:1320-1323`), and nothing is scheduled. The CoW comes from a partial prefix hit, `single_type_kv_cache_manager.py:454-474` and `:1900-1944`. If no productive step follows, because the pool is exhausted and the pinned blocks are exactly what admission needs, the head fence never clears. I did not reproduce this end to end on main. The in-tree regression test injects the CoW copy directly (below).

### Upstream PRs and issues
I searched `deferred_frees`, `_drain_deferred_frees`, `defer_block_free`, `processed_step_seq`, "deferred free", "CoW retention fence", "0-token step" and "async scheduling deadlock". **No PR or issue reports or fixes the stop-at-head wedge.** Related items:
- #49675 (merged 2026-09-11, `84030bbe3d`; fixes issue #49674): stops zero-progress preemption cascades caused by deferred frees. It touches the fence predicate (`_request_blocks_can_be_freed`, main `:2769-2774`), not the drain. Its body says "It does not change the deferred-free safety fence". Both main and our prod base contain it.
- #48532 (open, 2026-07-13; issue #48489 open): per-group eviction order in the deferred free. It keeps the `break`.
- #56525 (open, 2026-09-11): Mamba commit fence. It moves the drain under `if self.defer_block_free:` and keeps stop-at-head.
- #55227 (open): skip cascade prefixes during deferred frees. Not this bug.
- We have never upstreamed it. `gh search prs --author Starwaves1` on vllm-project/vllm returns nothing, and `patches-local/UPSTREAM_CONTRIBUTIONS.md` has no entry for it.

### Our fix
- **Source**: Starwaves1/vllm `2089dc2b3c` on `qwen38/main` ("Full-scan deferred-frees drain", 2026-09-23). It is a port of `1d184a46f6` (qwen38/v0.27.1) and adds `test_pending_cow_fence_does_not_pin_satisfied_frees_behind_it` to `tests/v1/core/test_deferred_block_free.py`.
- **In production: yes.** The installed `scheduler.py:2831-2855` has the full scan ("Scan the whole queue", `self.deferred_frees = pending`). `2089dc2b3c` is an ancestor of the deployed `05efb4d1fc2c`. Today the code is dormant: with `--no-async-scheduling` and PP=1, `defer_block_free` stays False.
- **Applies to latest main**:
  - `patches/scheduler-deferred-frees-liveness.patch`: `git apply --check` **fails**. The context moved: the docstring changed, and `fence, blocks =` became `fence, _ =`.
  - The commit `2089dc2b3c`: `git apply --check` **passes**, hunk at `:2798`, offset +75.
- **Tests on latest main** (`GSQ_LIGHT=1 GSQ_MIN_AVAIL_GB=3 tools/capped`, `.venv-main` python, `PYTHONPATH` set to the archive copy, compiled `.so` files symlinked from `.venv-main`):
  - `tests/v1/core/test_deferred_block_free.py`, stock main plus the new test: **1 failed, 14 passed**. The wedge test fails with `assert [2, 1] == [2]` at `:597`, so the satisfied fence 1 stays pinned behind CoW fence 2.
  - Same file, main plus `2089dc2b3c`: **15 passed**.
  - `patches/test_drain_deferred_frees.py` (simulation harness) as committed: **cannot run** on latest main or on the prod venv. Its fidelity check (`:85`) looks for `last_sched_seq <= self.processed_step_seq` inside `_free_request_blocks`, and #49675 moved that predicate into `_request_blocks_can_be_freed`.
  - With a 2-change adapted copy (`/tmp/async-sched-check/test_drain_adapted.py`: append the predicate source in the fidelity check, bind the method on the stub): **18/18 PASS** on main plus the fix, and **18/18 PASS** on the prod venv. This includes "(f') wedge sequence still WEDGES under stock" and 3000-seed no-premature/no-double-free fuzz. On stock main it stops at the fidelity assert, which requires the full-scan drain.

## Bug 2: async-KV-load admission livelock ("BUG #2")

This bug is in the async *KV load* path, not async *scheduling*. A probe on stock main spins with `async_scheduling` both False and True.
- **Upstream**: not fixed. On main the async-load admission still calls `allocate_slots(..., full_sequence_must_fit=self.scheduler_reserve_full_isl)` (`scheduler.py:1256-1272`; default True, `config/scheduler.py:190`). On `None` it does an unconditional `break` (`:1275-1282`), with no retry bound or degrade.
- **Probe on stock main**: idle engine, a 12-block pool, a 320-token candidate with a 160-token async hit (the same setup as our `test_idle_engine_degrades_after_n_failures`). After 50 steps it is still `WAITING` with 0 scheduled tokens. **2 passed**, `[False]` and `[True]`. The setup is artificial (the request is larger than the pool).
- **Upstream PRs, both open**: #54348 (2026-08-29, "Resolve admission deadlock for blocked-waiting statuses"; continues the scan when nothing is running) and #57959 (2026-09-21, `max_recompute_pending_tokens`). Neither is merged.
- **Our fix**: `qwen38/main` `e0945f32a2`, in prod at `scheduler.py:78-93`, `:413-419` and `:1045-1058`. `git apply --check` **fails** on latest main: the patch file at `:831`, and the commit with 3 of 8 hunks rejected (`:1034`, `:2696`, `:3086`). Our tests for it (`tests/v1/kv_connector/unit/test_async_load_admission.py` from `05efb4d1`) fail 9/9 on stock main because they read `ASYNC_LOAD_ADMIT_MAX_FAILS` and `_async_load_admit_fails`, which do not exist there. That is a missing-symbol failure, not a behavioural red.
- **Inert in prod**: `sync_load: true` means the connector never returns `load_kv_async=True`, as the header of `patches/test_admission_livelock.py` notes.

## Busy-wait refinement (stage2-2)
- This changes our own BUG #2 guard so a busy engine waits instead of dropping the hit. Upstream already waits (no degrade), so there is nothing upstream to fix here. It only matters on top of Bug 2's patch.
- **In prod: yes.** `qwen38/main` `766fac7c25` is an ancestor of `05efb4d1`, and `_should_degrade_async_load` at `scheduler.py:3183-3192` checks `_async_load_engine_busy()`.
- **Latest main**: `git apply --check` on `patches-local/stage2-2-async-admit-busy-wait.patch` **fails**, and the commit rejects 4 of 4 hunks, because it depends on Bug 2's patch.

## Gotchas 21-22 (draft count pinned under async)
Still true on main: `vllm/v1/engine/core.py:684-691` `post_step` feeds draft token ids back only when `not self.async_scheduling`. This does not affect prod's fixed `num_speculative_tokens: 3`. It only matters for adaptive or dynamic verify blocks (LOOKUP/DFlash2, SPEC_SCHEDULE).

## Bottom line: can prod turn async scheduling back on?
In code, yes: the deadlock fix is already in the deployed scheduler, and the livelock patches are inert under `sync_load`. Before flipping `ASYNC_SCHED=1`:
1. **Soak it.** Turning async on makes `defer_block_free` True, which turns on the full-scan drain, the #49675 guard and the CoW fences. None of that has run in prod since 2026-09-04. Async scheduling with `sync_load` has also never been tested together: `README-sync-load-ordering-fix.md:56` only says part (c) costs nothing *with* `--no-async-scheduling`. The GPU headroom at 0.94 is 440-460 MiB under load (`override.conf`), and whatever async adds has not been measured.
2. **Keep `2089dc2b3c` on every redeploy.** Upstream does not have it, and its docstring argues the opposite.
3. **Update the harness.** `patches/test_drain_deferred_frees.py` is stale against main (fidelity check). The in-tree pytest in `tests/v1/core/test_deferred_block_free.py` is the guard that still works.
4. **Keep spec decode fixed-k.** No SPEC_SCHEDULE or adaptive k while async is on (gotcha 21).
5. **Optional: upstream the fix.** It is a clean cherry-pick with a red/green test on main, and nobody has filed it.
