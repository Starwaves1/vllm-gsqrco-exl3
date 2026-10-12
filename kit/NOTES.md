# Kit status (2026-10-04, paused for budget)

Where I stopped:
- Kit complete and pushed (run.sh bootstrap/tier1/tier2/tier3/all, synth.py, micro_gguf.py, ladder.py,
  report.py with collect/agree/selftest, README, RETURN). Per-generation reporting and the cross-card
  agreement check (report.py agree -> kit/results/generation-smXY.{md,json}) are in.
- CPU dry-run verified here: bash -n; bootstrap --no-gpu --skip-build into a throwaway venv (pinned cu130
  stack installs; vendored sha256 check passes for lcpp, exl3, trellis_serve); pip CUDA 13.0 toolchain +
  stock GGUF build for sm86; synthetic GGUF (100 tensors, every block dequantizes finite) and EXL3
  checkpoint (loads through exl3_cases.load); micro_gguf shape inventory, routes and variant lists;
  ladder.py against a stub SSE server (ms/step 5.9 for a 6 ms / 3-token step); report.py selftest incl. a
  two-card agreement; CUDA-12 path resolves with uv --dry-run (torch 2.13.0+cu129 via --torch-backend).
- box2 (torn-gpu): repo at /workspace/kit (branch validation-kit), env in /workspace/kit-env.sh, launcher
  /workspace/kit-launch.sh, work dir /workspace/kit-work. Bootstrap DONE on sm86: gguf_route_l ok, exl3 ok
  (nvcc 13.0 accepted GCC 15.2; glibc 2.43 rsqrt patch applied). Log /workspace/logs/kit/bootstrap.log.

What is running: nothing (no GPU job was launched on box2; the 2080 Ti was never used).

Open:
- The single /check review was started on commit da944c7 and had not reported when the pause came; its
  findings are unverified and not applied. Rerun the check (or read its report) before tier 1.
- The bootstrap on box2 predates 888e3c7 (generation/agreement + compute probe): `cd /workspace/kit &&
  git pull` first (Python only, no rebuild needed).

Next:
1. Apply the check findings, push, `git pull` on box2.
2. Tier 1 on the RTX 3070 only, when idle (<= ~630 MiB owner use):
   `nohup /workspace/kit-launch.sh tier1 --gpu 0 --name torn-gpu-rtx3070 --max-gpu-gb 6.5 > /workspace/logs/kit/tier1.log 2>&1 &`
   Watch the first parity minutes for the runtime (full parity is ~4100 GGUF + ~1000 EXL3 cases; use
   --parity-k if it is too long). Never CUDA_VISIBLE_DEVICES=1 / --gpu 1 (2080 Ti under hardware watch).
3. Copy kit/results/torn-gpu-rtx3070-<date>/ back, commit it on validation-kit, push; then tier 2 on the 3070.
4. The sm86 agreement test needs a vast-rtx3090 tier-1 run of the kit (queued job, not exploratory).
