# conv1d accepted-count bound: CPU re-verification against latest vLLM (2026-10-11)

## Bottom line

- **The fix is already in #50021.** amittell pushed it as `f07a68f` on 2026-10-05 ("Bound conv1d
  spec-decode num_accepted by the state row", co-authored by Starwaves1), then rebased. The current
  head is `ace8eb7d64`, and its conv1d hunk is byte-identical to our `conv1d-accepted-bound.patch`:
  the patch reverse-applies cleanly to the PR head, and forward-applying it reports "previously applied".
- The drafted review comment (`docs/upstream/vllm-issue-dynamic-spec-schedule-corruption.md`, Part 1)
  is **obsolete as a review comment**. It targets head `71d7c782ca`, which was force-pushed away (GitHub
  returns "No commit found"). Nothing in it needs to be posted. At most, the two-call kernel test could go
  in as a PR against `amittell:bugfix/gdn-mtp-spec-decode-index-bounds`, which amittell invited.
- Latest main does not carry the bug. #50021 has not merged, and main has no accepted-count check in
  the conv1d update kernel.
- We could not find an issue from Starwaves1 or a comment by them on #50021. amittell's 10-05 reply
  ("@Starwaves1 thanks - you're right") answers something that is not in the issue comments or review
  comments (deleted, or sent another way). Garrett should know the origin.

## Upstream state (gh, read-only, 2026-10-11)

| item | state |
|---|---|
| #50021 "[Bugfix] Bound accepted-token state lookups in GDN/KDA spec decode" (amittell) | OPEN, not merged; head `ace8eb7d64635862e94e3fa004c9c6f3262f99a0` (force-pushed 2026-10-10T14:25Z, 11 commits on main `2bea22002d`) |
| conv1d hunk since `71d7c782ca` | changed: `seqlen` became `max_seqlen` (the launch-wide max_query_len), plus a docstring and a comment. This is commit `ace8eb7d64` and is the only commit that touches causal_conv1d.py's bound |
| blocker | `pre-run-check` needs a maintainer label (`ready`/`verified`); amittell has asked @ZJY0516 @tdoublep |
| #51508 (maxpla3, OPEN) | edits the same conv1d lines. It only clamps the offset to `max(num_accepted - 1, 0)`, with no upper bound and no seqlen check, so it does not cause this bug. It does conflict textually with #50021 in `causal_conv1d.py`, and the second to land must pick one zero-count policy (#50021's description says so) |
| other reports | none. Searched PRs and issues for "causal_conv1d num_accepted", "conv1d num_accepted_tokens seqlen", "dynamic speculative mamba conv state": 0 hits. "num_speculative_tokens_per_batch_size" turns up only DSD crash and perf items (#58692, #60981, #49548, #51510, ...), none about the GDN conv state |

## SHAs and files tested

- vLLM main: `03091e28b8ba956634c153beab0e52decafdef31` (2026-10-12 09:25 +0800, #59026), clone at
  `/home/garrett/vllm-latest` (`--filter=blob:none`). causal_conv1d.py blob `e8254e4f`, the same as the
  PR's merge base.
- #50021 head: `ace8eb7d64` (`git fetch origin pull/50021/head:pr50021`). causal_conv1d.py blob `331ef707`.
- `git log 2bea22002d..origin/main` shows no commits to any of the 12 files the PR touches, so the PR applies to main as-is.

| copy | how it was made | sha256 (first 8) |
|---|---|---|
| a_main.py | `git show origin/main:<conv1d>` | cb16cc92 |
| b_pr50021_head.py | `git show pr50021:<conv1d>` | 81a0ef48 |
| c = #50021 head + our fix | our patch is already in b (forward apply: "Reversed (or previously applied) patch detected", 2/2 hunks skipped), so c == b | 81a0ef48 |
| d = main + #50021 conv1d hunk + fix | b's file is main's blob plus exactly that hunk (`git diff origin/main pr50021 -- <conv1d>`), so d == b | 81a0ef48 |
| b0_pr50021_without_fix.py (control: the hunk as reviewed at `71d7c782ca`) | `patch -p1 -R < conv1d-accepted-bound.patch` on b (applies cleanly) | 893158b4 |

## Kernel test (CPU, Triton interpreter)

```
cd /tmp/conv1d-test   # copies above + cloud/results/r3/box-scripts/r3conv_kchange.py
CUDA_VISIBLE_DEVICES= TRITON_INTERPRET=1 GSQ_LIGHT=1 GSQ_MIN_AVAIL_GB=3 tools/capped \
  .venv-main/bin/python r3conv_kchange.py a_main.py b_pr50021_head.py b0_pr50021_without_fix.py
```

Output (verbatim, minus one torch "No CUDA runtime" warning), exit 1 because of the control b0:

```
a_main.py: fixed K=3                    ok  max err per step ['3e-07', '4.8e-07', '9.5e-07', '4.8e-07']
a_main.py: K 5->3, accept 6 then L=4    ok  max err per step ['4.8e-07', '9.5e-07', '4.8e-07', '7.2e-07']
a_main.py: K 5->3, accept 4 then L=4    ok  max err per step ['4.8e-07', '4.8e-07', '9.5e-07']
a_main.py: K 3->2, accept 4 then L=3    ok  max err per step ['3e-07', '4.8e-07', '9.5e-07']
a_main.py: K 3->5, accept 4 then L=6    ok  max err per step ['3e-07', '9.5e-07', '7.2e-07']
a_main.py: num_accepted=0: output zeroed False, state untouched False
a_main.py: num_accepted=7: output zeroed False, state untouched False
b_pr50021_head.py: fixed K=3                    ok  max err per step ['3e-07', '4.8e-07', '9.5e-07', '4.8e-07']
b_pr50021_head.py: K 5->3, accept 6 then L=4    ok  max err per step ['4.8e-07', '9.5e-07', '4.8e-07', '7.2e-07']
b_pr50021_head.py: K 5->3, accept 4 then L=4    ok  max err per step ['4.8e-07', '4.8e-07', '9.5e-07']
b_pr50021_head.py: K 3->2, accept 4 then L=3    ok  max err per step ['3e-07', '4.8e-07', '9.5e-07']
b_pr50021_head.py: K 3->5, accept 4 then L=6    ok  max err per step ['3e-07', '9.5e-07', '7.2e-07']
b_pr50021_head.py: num_accepted=0: output zeroed True, state untouched True
b_pr50021_head.py: num_accepted=7: output zeroed True, state untouched True
b0_pr50021_without_fix.py: fixed K=3                    ok  max err per step ['3e-07', '4.8e-07', '9.5e-07', '4.8e-07']
b0_pr50021_without_fix.py: K 5->3, accept 6 then L=4    BAD max err per step ['4.8e-07', '9.9', '7.6', '7.2e-07']
b0_pr50021_without_fix.py: K 5->3, accept 4 then L=4    ok  max err per step ['4.8e-07', '4.8e-07', '9.5e-07']
b0_pr50021_without_fix.py: K 3->2, accept 4 then L=3    BAD max err per step ['3e-07', '5.1', '1.9']
b0_pr50021_without_fix.py: K 3->5, accept 4 then L=6    ok  max err per step ['3e-07', '9.5e-07', '7.2e-07']
b0_pr50021_without_fix.py: num_accepted=0: output zeroed True, state untouched True
b0_pr50021_without_fix.py: num_accepted=7: output zeroed True, state untouched True
```

| plan | a main | b = c = d (#50021 head, fix included) | b0 (old #50021 hunk, control) |
|---|---|---|---|
| fixed K=3 | ok | ok | ok |
| K 5->3, accept 6 then L=4 | ok | ok | **BAD** (9.9, 7.6) |
| K 5->3, accept 4 then L=4 | ok | ok | ok |
| K 3->2, accept 4 then L=3 | ok | ok | **BAD** (5.1, 1.9) |
| K 3->5, accept 4 then L=6 | ok | ok | ok |
| num_accepted=0 / 7 fail closed | no (no check) | yes | yes |

This differs from the brief's expectation ("b wrong on the K-shrink plans") only because b now
contains the fix. The b0 control shows the test still catches the old hunk on the same plans as before.
The "accept 4 then L=4" plan passes for b0 as expected, since 4 <= L.

## Code reading (file:line at the tested shas)

- Main's kernel has no equivalent check. At `03091e28b8` `causal_conv1d.py:874-876`,
  `conv_state_token_offset = tl.load(num_accepted_tokens_ptr + idx_seq) - 1`, unchecked. Lines 846
  (`seqlen = query_end_index - query_start_index`) and 1182 (`seqlen = max_query_len`, the host side)
  are the two meanings of `seqlen` that the PR's bound confused.
- In the PR head, `causal_conv1d.py:842` saves `max_seqlen = seqlen` before the varlen reassignment at :848,
  and :880 checks `(num_accepted < 1) | (num_accepted > max_seqlen)`.
- The other kernels in the PR bound by the state row width (`stride_indices_seq` /
  `stride_state_indices_batch`), not the query length, at `ace8eb7d64`:
  `fla/ops/fused_recurrent.py:122` and `fla/ops/fused_sigmoid_gating.py:121`
  (`idx_in_row = (i_t >= 0) & (i_t < stride_indices_seq)`), KDA
  `kimi_k3/nvidia/.../fused_recurrent.py:203` and `kimi_k3/amd/.../fused_recurrent.py:199`
  (`initial_token < stride_indices_seq`), and `mamba_ssm.py:344`
  (`init_token_idx < stride_state_indices_batch`).
- Every GDN caller passes the row width as `max_query_len`: `qwen_gdn_linear_attn.py:1441`
  (`spec_state_indices_tensor.size(-1)`), :1815, `olmo_gdn_linear_attn.py:365`, and
  `kimi_gdn_linear_attn.py:480/498`.
- Main's scheduler still picks K by batch size. At `scheduler.py:309-314`,
  `dynamic_sd_lookup = build_dynamic_sd_schedule_lookup(num_speculative_tokens_per_batch_size, ...)`,
  and at :1497-1500, `num_spec_tokens_to_schedule = self.dynamic_sd_lookup[len(num_scheduled_tokens)]`.
  So K can shrink between steps on main, and the old hunk would have hit it.

## Build check (latest main, isolated venv)

```
cd /home/garrett/vllm-latest && GSQ_MIN_AVAIL_GB=3 /home/garrett/gsq-vllm/tools/capped bash -c \
  'uv venv --python 3.12 .venv-latest && VIRTUAL_ENV=.venv-latest VLLM_USE_PRECOMPILED=1 uv pip install -e .'
```

The first attempt failed on a transient DNS error fetching
`wheels.vllm.ai/03091e28.../cu130/vllm/metadata.json`. The retry succeeded (exit 0; log
`/tmp/vllm-latest-build2.log`). With `CUDA_VISIBLE_DEVICES=` and no GPU, `import vllm` gives
`0.31.1rc1.dev291+g03091e28`, and `vllm.model_executor.layers.mamba.ops.causal_conv1d` imports
(`causal_conv1d_update` is present; torch 2.13.0+cu130, `cuda.is_available()` False). Rerunning the
kernel test with this venv's python (triton 3.7.1) gives the same result lines as the `.venv-main` run above.
