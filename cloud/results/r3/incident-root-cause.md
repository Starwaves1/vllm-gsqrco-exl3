# Incident root cause: corrupted generations under the per-batch-size MTP schedule (2026-10-03)

Labels: PROVEN (shown by a box job or by a test run here), CODE (read in source, file:line given),
INFERRED (reasoned from the two). Source paths are in `.venv-main/lib/python3.12/site-packages/vllm/`
(vLLM main d28795f1a7 + overlay 2a0fe5e1e1, the bytes production and the box run) unless they say
"upstream", which means `git show d28795f1a7:<path>` in `~/vllm`.

## Verdict

The evidence points away from the GSQ-RCO kernels and from vLLM main's dynamic schedule code as
merged, and at the interaction of two things:

1. vLLM's dynamic speculative decoding (`num_speculative_tokens_per_batch_size`) changes the number
   of draft tokens K between steps whenever the batch size crosses a tier boundary. A smaller K in
   step N+1 means a shorter verify query (K+1 tokens) than step N had.
2. A bounds check from the open upstream PR vllm-project/vllm#50021, which our overlay backports
   (commit 46ba368c70, "[Backport #50021] Bound accepted-token state lookups in GDN/Mamba spec
   kernels (site 1)"). In `causal_conv1d_update` it rejects `num_accepted > seqlen`, where
   `num_accepted` is the previous step's accepted count and `seqlen` is this step's query length.
   After a K decrease that inequality is legitimate (accepted 6 of 6 last step, verifying 4 now).
   The kernel then writes zeros for the request's whole conv output and returns without updating
   its conv state. Every GDN layer (48 of 64) gets zero conv output for that request for that step,
   and a stale conv window for the next one or two steps. The recurrent update still runs on that
   zero q/k/v (its own bound is the row width, `qwen_gdn_linear_attn.py:1455-1464`), so each layer's
   recurrent state is perturbed from then on, not just for one step. The token sampled from that step
   is garbage, and the context it leaves behind makes the model repeat or drift. (Zeroed conv output
   and stale conv state: PROVEN on CPU. The recurrent-state effect and the token-level effect:
   INFERRED, the CPU test covers only the conv, and no step was traced on the GPU.)

PROVEN at kernel level here (CPU, Triton interpreter, `box-scripts/r3conv_kchange.py`): upstream
main's kernel is exact for every K change; the overlay's kernel is wrong for the two steps after a
decrease with full acceptance (max abs error 9.9 and 7.6 on outputs of magnitude ~1-10), and exact
otherwise. INFERRED end to end: every box result and the production pattern follow from this rule
(section "Why the pattern looks the way it does"). The GPU run that applies only the one-line fix
and reruns job 60 is Task B (below); until it is green, the end-to-end attribution is inference.

So the answer to "is it the overlay's bug?" is: at kernel level yes (PROVEN on CPU). The overlay, via
the still-open upstream PR, introduced the check; upstream main's kernel has no such check and is
correct for K changes. That this check is the whole of the box and production corruption is INFERRED
until Task B.

## Timeline

| when (EDT) | what | label |
|---|---|---|
| 08-21 | PR #50021's conv1d hunk is in prod's 0.27.1 venv (`venv-0271/.../causal_conv1d.py:875`, file mtime; deploy repo `patches/vllm-pr50021-gdn-spec-bounds.patch`) | CODE |
| 09-23 | overlay commit 46ba368c70 ports it to main (verbatim from the PR head 71d7c782ca) | CODE |
| 09-28 22:37 | overlay 2a0fe5e1e1 deployed into venv-main (deploy-vllm history, 2026-09-29T02:37Z) | PROVEN |
| 09-29 12:55 | prod starts using `SPEC_SCHEDULE=[[1,4,5],[5,8,3],[9,16,2]]` (journal: logged speculative_config) | PROVEN |
| 09-30 16:42 | prod cuts over to vLLM main (venv-main) | DOCUMENTED (vllm-main-compat.md) |
| 10-01 02:55-12:50 | GSQ-RCO live with the schedule; knowledge-bench runs 17-19 corrupted | PROVEN |
| 10-01 12:28 | cap-8 mitigation (`[[1,4,5],[5,8,3]]`, MAX_SEQS 8) | PROVEN |
| 10-01 12:51 | rollback to W4A16, MAX_SEQS 8, fixed k=3, no schedule | PROVEN (journal, override.conf.w4a16-tt709-nonfast-c8) |
| 10-01/02 | box jobs 28b/28c/28d/56/57/58/59/60 | PROVEN |
| 10-03 | mechanism found in code, kernel-level CPU proof, fix patch | this doc |

## Evidence (quoted from the raw logs in /tmp/gpuq-out, box `/workspace/logs/r3/*/summary.txt`)

Raw logs on Garrett's machine (`/tmp/gpuq-out/`): job 60 `1790847241760000-r3-60-churn.log`, job 58
`1790847241900000-` and `1790875263931040-r3-58-side-client.log`, job 28b
`1790847241000000-r3-28b-token-corruption.log`, 28c `1790847243000000-r3-28c-token-corruption.log`,
28d `1790847241800000-r3-28d-parity.log`. Each ends with r3tok's table and per-answer details.

All box servers ran production's main argv (`env/prod-main-serve-argv.txt`), GSQ-RCO IQ3_S-mtp GGUF
unless noted, plugin 32ae6ec, T=0 unless noted. "flagged" = r3tok.py's per-answer flags (HTTP/UTF-8
errors, U+FFFD, text != decode(ids), EOS inside reasoning, characters outside Latin/Greek/punctuation/
math/emoji, a 12-80 char fragment repeated 4+ times).

**Job 60 (r3-60-churn, 2026-10-01 18:20-19:03 UTC).** Main load: 30 non-streamed chat answers at
c=4, 600 tokens. Side client: one thread sending `{"prompt": "The capital of Denmark is",
"max_tokens": 1}` completions back to back (`plain1`, a prefill-only request; one in flight at a time).

```
tag          pass        T  c   n  flagged  http  utf8  fffd  text  eos_reason  odd_script  repeat
base         side-plain1 0.0  4  30       23     0     0    13     0          10          16       4
eager        side-plain1 0.0  4  30       25     0     0    10     0          15          13       1
k3fixed      side-plain1 0.0  4  30        0     0     0     0     0           0           0       0
mambanone    side-plain1 0.0  4  30       24     0     0    13     0          13          15       2
noconn       side-plain1 0.0  4  30       25     0     0    10     0          16          13       4
nospec       side-plain1 0.0  4  30        0     0     0     0     0           0           0       0
variant noprefix: FAILED   (server did not come up: "Engine core initialization failed")
```

`nospec` = no `--speculative-config` at all (not "fixed k=2", as the job summary has it). `k3fixed`
= `{"method":"mtp","num_speculative_tokens":3,"draft_sample_method":"probabilistic"}`, no schedule.
Sample (base, prompt 0): `Serbia accepted 7 of the 10 10 demands;`; prompt 20: `1377: JikI377: Jikji
printed in Korea`; prompt 8 degenerates into `1111...`.

**Job 58 (r3-58-side-client, latest run 2026-10-02 02:23-02:49 UTC).** Same load; side client kind
varies: `none` (no side client), `plain` (4 tokens), `plain1` (1 token), `lp` (logprobs), `plp`
(prompt_logprobs), `echo` (echo + logprobs).

```
tag          pass        T  c   n  flagged  fffd  eos_reason  odd_script  repeat
gsq          side-none   0.0  4  30        0     0           0           0       0
gsq          side-plain  0.0  4  30       19    13           4          14       3
gsq          side-plain1 0.0  4  30       27    12          12          17       3
gsq          side-lp     0.0  4  30       26     9          12          12       2
gsq          side-plp    0.0  4  30       20    10           9          11       2
gsq          side-echo   0.0  4  30       26     9          12          12       2
w4a16        side-none   0.0  4  30        0     0           0           0       0
w4a16        side-plain  0.0  4  30       16     4           9           4       4
w4a16        side-plain1 0.0  4  30       23    15           5          17       6
w4a16        side-lp     0.0  4  30       14     4           5           5       5
w4a16        side-plp    0.0  4  30       14     4           5           5       5
w4a16        side-echo   0.0  4  30       16     4           9           4       4
gsq-patched  side-none   0.0  4  30        0     0           0           0       0
gsq-patched  side-plain1 0.0  4  30       25    13          12          15       5
(gsq-patched plain/lp/plp/echo: 26/26/26/20)
```

The table is the second of two runs of the same job (wt 29d3761): raw logs
`/tmp/gpuq-out/1790875263931040-r3-58-side-client.log` (this table) and
`1790847241900000-r3-58-side-client.log` (first run, 2026-10-01 17:44 UTC: gsq plain1 24, w4a16
plain1 16, gsq plain/lp/plp/echo 26/22/26/25, w4a16 14/15/14/14, side-none 0 on all three servers).
`w4a16` = a dense W4A16 AutoRound checkpoint (Marlin), `Qwen3.8-27B-W4A16-AutoRound-fast`, production's
previous W4A16 (production now runs `...TT709-W4A16-AutoRound-fast`), same venv, same argv. Any side
client corrupts, including plain 4-token completions; the job summary's "only logprobs/echo requests
corrupt" is wrong.

**Job 28b (r3-28b-token-corruption, 2026-10-01), production schedule, no side client unless `plp`:**

```
prod main c=8  T=1.0: 0/32    c=8  T=0: 1/32
prod main c=9  T=1.0: 8/36    c=9  T=0: 16/36
prod main c=12 T=1.0: 3/48    c=12 T=0: 3/48
prod main c=16 T=1.0: 1/64    c=16 T=0: 3/64
(rows below: main c=9 / c=12 at T=1.0; plp c=4 = prompt_logprobs side client, T=0)
g-mmq-k2      (VLLM_GGUF_MMA_K=0)              main c=9: 13/36  c=12: 3/48  plp c=4: 26/30
e-ours-k2     (--enforce-eager)                main c=9:  5/36  c=12: 3/48  plp c=4: 21/30
g-ours-k3at9  (schedule [[1,4,5],[5,16,3]])    main c=9:  3/36  c=12: 3/48  plp c=4: 23/30
g-ours-cap8   (max-num-seqs 8, [[1,4,5],[5,8,3]]) main c=9: 2/36  c=12: 1/48  plp c=4: 23/30
```

**Job 28c (2026-10-01 19:25-19:48 UTC), production schedule, plugin routing toggles (main at T=1.0):** `alloff`
7/36 at c=9, 1/48 at c=12, plp 23/30; `noiq1m` 10/36, 2/48, 21/30; `notiled` 8/36, 1/48, 25/30.
The "-k2" in these tags names the tier under test; the argv kept production's schedule.

**Job 28d (2026-10-01 19:03-19:24 UTC).** `k2-small`: schedule `[[1,16,2]]` (K=2 at every batch
size): c=3..8 at T=0: 1, 0, 0, 1, 0, 0 flagged of 30-32; both flags are prompt 15 with U+0304 (combining
macron, math notation) as the only odd character, a false positive of the odd-script rule. `k2-parity`: production's schedule unchanged
(argv diff shows no `--speculative-config` change; the job summary calls it a flat k=2 schedule,
which it was not), c=10/11/13/14/15: 2/40, 2/44, 1/52, 2/56, 2/60 (6 with U+FFFD, a real corruption
sign; 3 are the same U+0304 false positive on prompt 15).

**Units.** Job 56: draft_head 61440x5120 at n=9..16 and lm_head 248320x5120 at n=20..32 through
`lcpp_mul_mat_mma_k`, rel err <= 6.2e-2, 29 passed. Job 59: embedding rows 1..64, 64 passed.

**Production (knowledge-bench runs.db copy, 1 Hz vllm_samples copy from llama-dashboard).**

| runs | model, vLLM, schedule | answers | U+FFFD | stop with empty content |
|---|---|---|---|---|
| 13-16 | W4A16, 0.27.1, `[[1,4,5],[5,8,3],[9,16,2]]` | 12,000 | 0.00% | 0.00-0.23% |
| 17-19 | GSQ-RCO, main, same schedule | 9,000 | 2.0-8.7% | 3.3-12.3% |
| 20-21 | W4A16, main, fixed k=3, MAX_SEQS 8 | 8,400 | 0.00% | 0.02% |

Running-count profile: GSQ runs 17-19 sat at the 8/9 boundary (11,675 s at 8, 3,064 s at 9, 729
upward 8->9 crossings visible at 1 Hz in 6.4 h). W4A16 runs 13-16 sat at 9-16 (mean drafted k 2.01;
223 visible 8->9 crossings in 4.1 h). The prod report (`cloud/results/prod-garbled-tokens-20261001.md`, on main at 204d6f5, not on this
branch) found GSQ
corruption concentrated in requests whose max running was exactly 9 (587/2,265) rather than 10-12
(11/79).

## Exclusions

| candidate (as a necessary cause) | excluded by | label |
|---|---|---|
| GSQ-RCO / GGUF plugin kernels | W4A16 (Marlin) corrupts the same way (58); routing toggles change nothing (28b mmq, 28c); units 56/59 pass | PROVEN |
| CUDA graphs / graph-pool overlap | `--enforce-eager` corrupts (60: 25/30; 28b) | PROVEN |
| KV offload connector | `noconn` 25/30 (60) | PROVEN |
| mamba prefix-cache align mode | `mambanone` 24/30 (60) | PROVEN |
| prompt-logprobs / echo hotfix paths | plain 1-token neighbour corrupts; `gsq-patched` corrupts (58) | PROVEN |
| request shape / prefill mixing as such | `k3fixed` with the same neighbour: 0/30 (60) | PROVEN |
| spec decode as such, or K < num_speculative_tokens | `k3fixed` 0/30; flat `[[1,16,2]]` with num_speculative_tokens 5: <= 1/30 at c=3..8, both flags false positives (28d) | PROVEN |
| async scheduling | argv has `--no-async-scheduling` | PROVEN |
| prefix caching | NOT excluded by a run (`noprefix` never booted); excluded only by inference (k3fixed keeps prefix caching on and is clean) | INFERRED |
| OOB writes elsewhere | NOT tooled: job 57's memcheck and initcheck both exited rc=255 before reporting | open |

## Mechanism (code path, file:line)

1. K for a step is chosen from that step's scheduled request count, after scheduling:
   `v1/core/sched/scheduler.py:1550-1555` (upstream :1446-1451):
   `num_spec_tokens_to_schedule = self.dynamic_sd_lookup[len(num_scheduled_tokens)]`. A prefill-only
   neighbour counts, so at c=4 its arrival makes the batch 5 and K drops from 5 to 3. The lookup is
   built by `v1/spec_decode/dynamic/utils.py:77-148` (dense batch-size -> K table). CODE
2. The runner hands that K to the drafter at the end of the same step:
   `v1/worker/gpu_model_runner.py:5022` (upstream :4972), `v1/spec_decode/llm_base_proposer.py:532`
   (`self.num_speculative_tokens = num_speculative_tokens`) and `:689` (draft loop). Next step, each
   running request verifies K+1 tokens. CODE
3. After verification, `num_accepted_tokens = (output_token_ids != -1).sum(dim=1)`
   (`gpu_model_runner.py:1632`, upstream :1582), range 1..K_prev+1, copied to the next step's GPU
   buffer at `:2158-2173`. CODE
4. The GDN metadata builder treats requests with `num_scheduled == drafts + 1` as spec rows
   (`gpu_model_runner.py:2291-2300`, `v1/attention/backends/gdn_attn.py:253`) and gives the layer a
   state-index row of `num_spec + 1` columns (`gdn_attn.py:357-358, 378-379`). CODE
5. The GDN layer calls `causal_conv1d_update(..., num_accepted_tokens=..., query_start_loc=spec_query_start_loc,
   max_query_len=spec_state_indices_tensor.size(-1))` (`model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:1332-1346`,
   fused CUDA path :1710-1719). The conv state is `conv_kernel_size - 1 + num_spec` wide
   (`model_executor/layers/mamba/mamba_utils.py:292`). CODE
6. In the kernel, the varlen branch replaces `seqlen` by this request's query length
   (`model_executor/layers/mamba/ops/causal_conv1d.py:847`), then the overlay's check runs:
   `:875-888`, `if (num_accepted < 1) | (num_accepted > seqlen):` zero the output, return. Upstream
   (:874-876) has no check and reads the window at offset `num_accepted - 1`, which stays inside the
   `width - 1 + num_spec` state for every `num_accepted <= num_spec + 1`. CODE
7. Result: after a step that accepted a tokens, if the next step verifies L < a tokens, that request's
   GDN conv output is zero for all L positions and its conv state is not advanced. PROVEN on CPU:

```
upstream  K 5->3, accept 6 then L=4   ok   max err per step 4.8e-07 9.5e-07 4.8e-07 7.2e-07
overlay   K 5->3, accept 6 then L=4   BAD  max err per step 4.8e-07 9.9     7.6     7.2e-07
overlay   K 5->3, accept 4 then L=4   ok
overlay   K 3->2, accept 4 then L=3   BAD  max err per step 3e-07   5.1     1.9
overlay   K 3->5 (increase)           ok
overlay   fixed K=3                   ok
fixed     all of the above            ok; num_accepted 0 and 7 still rejected (zero output, state untouched)
```

The recurrent (SSM) kernels in the same PR bound the index by the state row width
(`fused_recurrent.py` `i_t < stride_indices_seq`, `fused_sigmoid_gating.py` same, `mamba_ssm.py`
`init_token_idx < stride_state_indices_batch`), so only the conv1d hunk uses the wrong bound. Merged
main's CUDA `fused_gdn_decode_post_conv_mtp` (not part of the PR) uses the same row-width bound
(`accepted <= state_indices_width`, upstream `csrc/libtorch_stable/gdn/fused_gdn_decode_kernel.cu:176`). CODE

## Why the pattern looks the way it does (INFERRED from the rule "a K decrease after full acceptance")

- c=4 + neighbour: running 4 <-> 5 flips K 5 <-> 3 many times per answer; a 5->3 flip hurts when
  the step before accepted >= 5 tokens (per-position acceptance 0.82/0.63/0.49/0.25/0.20 at k=5, job
  40, so roughly a quarter of flips). 23-27/30 answers hit at least one.
- No neighbour at c=4: running never exceeds 4, K stays 5: 0/30.
- c=9 with the production schedule: running hovers 8 <-> 9 as answers finish and new ones arrive,
  K flips 3 <-> 2, and a 3->2 flip hurts when all 3 drafts were accepted: 8-16/36.
- c=12/16: running stays above 9 except at ramp-up/down, so few decreases: 1-3 of 48-64.
- c=8 with the production schedule: K is 3 throughout (decreases only at ramp-up from <= 4): 0-1/32.
- Flat `[[1,16,2]]`, fixed k=3, no spec: K never changes: clean.
- `[[1,4,5],[5,16,3]]` and cap 8 (no 3->2 tier) cut c=9 at T=1.0 from 8/36 to 3/36 and 2/36, not to
  zero, and leave the c=4 prompt_logprobs neighbour at 23/30 (5->3 remains). The residual 1-3 per 36-48
  at T=1.0 (also c=12 under every schedule in 28b/28c) has no T=1.0 fixed-K control to compare with;
  ramp-up at the start of each level crosses the tier boundaries too. Task B's `fix-c9` cell measures it.
- Production GSQ: the bench saturated KV at 8-9 running, so it crossed the 8/9 boundary constantly.

Not explained: W4A16 runs 13-16 on 0.27.1 used the same schedule and the same conv1d check and
crossed 8->9 at least 223 times, yet show no corruption signature. `venv-0271` has all three
ingredients: dynamic K from the scheduled count (`scheduler.py:1299-1303`), the GDN call passing
`max_query_len=spec_state_indices_tensor.size(-1)` (`qwen_gdn_linear_attn.py:1279`), and the check
(`causal_conv1d.py:875`; the v0.27.1 tag has no check, the deploy repo's
`patches/vllm-pr50021-gdn-spec-bounds.patch` put it there, file mtime 2026-08-21, before those runs).
On the box W4A16 corrupts about as often as GSQ (14-23 vs 19-27 of 30 with a neighbour), so the model
does not explain 0.00%. Their running count sat mostly at 10-16, which lowers the rate, but by this
rule it should not reach zero. This is the one open hole in the end-to-end attribution. A box cell on
a 0.27.1 venv with the c=4 neighbour would settle it; the current box has no 0.27.1 venv, so it was
not run.

## Production exposure today

Production runs W4A16 on venv-main with the same overlay, fixed k=3, no schedule. The schedule path
is closed. One path remains open by the same rule (INFERRED, not observed): any step where a
request's drafts are shortened after a step that accepted more. Structured output does this:
`scheduler.update_draft_token_ids` (`v1/core/sched/scheduler.py:2612-2614`) truncates drafts to the
grammar-valid prefix, so a JSON-schema request that accepted 4 tokens and then gets 1 valid draft
verifies 2 tokens with `num_accepted` 4. Token-budget trimming (`scheduler.py:861-864`) does the same
but rarely binds at 2048 tokens. The one-line fix closes it too. Flagged for Garrett, not acted on.

## Mitigation

- Now: keep fixed K (production's k=3). Do not enable `num_speculative_tokens_per_batch_size` on any
  venv that carries PR #50021's conv1d hunk (prod venv-main, venv-0271, the box venvs) until the fix
  is in.
- Fix: `patches/conv1d-accepted-bound.patch` (bound `num_accepted` by the launch's `max_query_len`,
  the state row width, instead of this request's query length; invalid counts are still rejected).
  For the overlay this is a one-line change to `causal_conv1d.py` on qwen38/main; for upstream it is a
  review comment on #50021 (`docs/upstream/vllm-issue-dynamic-spec-schedule-corruption.md`).
- Cost of fixed K, from what we have: production at n=9 ran k=2 at 80.5 ms/step, 2.53 tokens per
  sequence per step; at n=8, k=3 gave 3.00 per sequence (job 24's reference lines). Production
  acceptance by k on 2026-10-01: k=5 2.80/5, k=3 2.03/3, k=2 1.48/2 (`prod-profile-20261001.md`).
  On 0.27.1 the k sweep put k=3 best at c=1/2 and tied with k=4 at c=4/8 (REPORT.md). Job 24, the
  schedule-vs-schedule comparison at c=9, never reached steady state (a stream finished inside the
  window on both variants), so there is no box number for the schedule's benefit yet. Task B measures
  fixed k=3 against the schedule at c=1/2/4/8.

## Still open

- GPU confirmation (Task B): K changing every step must corrupt, the same traffic with fixed K must
  be clean, and job 60's matrix with only the conv1d patch applied must be clean.
- Job 57: compute-sanitizer memcheck and initcheck both exited rc=255 with no report, so OOB was
  never checked by tooling.
- Job 28e: the headcheck server never booted (engine-core init failure), so the live lm_head/draft
  head comparison never ran. Same boot failure for job 60 `noprefix` and job 40 `steady`.
- The 0.27.1 W4A16 discrepancy above.
