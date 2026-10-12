# Draft comment for vllm-project/vllm#50021 (Garrett posts it; not posted)

Reply to amittell's 2026-10-05 comment. Closes our side: nothing else from us is needed on this PR.

---

@amittell thanks for picking it up, and for the SM120 A/B.

I re-checked the rebased head (`ace8eb7d64`) against current main (`03091e28b8`) with our CPU kernel test (Triton interpreter, `causal_conv1d_update` called the way the GDN spec path calls it, compared against a plain causal conv over the accepted tokens):

| verify steps | main | this PR (ace8eb7d64) | PR with the old `> seqlen` bound |
| --- | --- | --- | --- |
| fixed K=3 | exact | exact | exact |
| K 5->3: accept 6, then L=4 | exact | exact | wrong (max err 9.9) |
| K 3->2: accept 4, then L=3 | exact | exact | wrong (max err 5.1) |
| K 3->5: accept 4, then L=6 | exact | exact | exact |
| num_accepted 0 or 7 | not checked (main has no bound) | output zeroed, state unchanged | output zeroed, state unchanged |

So the current head is correct on every case we have, and main isn't affected because it has no accepted-count check in this kernel. Nothing more is needed from our side. Please go ahead and add your own test rather than waiting on a PR from me.
