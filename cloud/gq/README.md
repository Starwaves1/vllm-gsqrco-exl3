# gq: the cluster GPU job queue

`gq` runs scripted GPU jobs on every machine in `machines.json` from one place: ms4. A dispatcher on ms4
(user service `gq-dispatcher`) checks every GPU every 15 s. When a GPU is free and allowed, it starts the
highest-priority queued job that fits, over ssh. Logs stream back while the job runs. On exit the dispatcher
records the exit code and copies the results to ms4. If a box dies, everything already copied is safe.

Python 3 stdlib, ssh and tar only. All state lives in plain files, with no database and no daemon on the boxes.

## Submit

```
gq submit --name ladder-c4 --on vast --prio ladder --est 90 -- bash /workspace/wt-r3/scripts/ladder.sh c4
gq submit --name kit-t1 --on 'cc>=8.6' --prio kit --script ./kit-tier1.sh -- --quick
```

- `-- CMD...` runs exactly that argv on the target, with no shell. Pipelines and `&&` belong in a script file.
- `--script LOCAL_FILE [-- ARGS]` copies a file from ms4 to `$GQ_DIR/script.sh` on the target and runs `bash script.sh ARGS`.
- `--cwd DIR`: working directory on the target. Default: the job dir.
- `--results DIR`: directory on the target that is copied back on exit. Default: `$GQ_DIR/results`.
- `--est MIN`: an estimate, shown in `gq ls`. `--exclusive`: no other gq job runs on that machine meanwhile.
- The command prints the job id (`MMDD-HHMMSS[-N]`).

The job sees these variables: `CUDA_VISIBLE_DEVICES` (the assigned GPU, with `CUDA_DEVICE_ORDER=PCI_BUS_ID`, so
the index is nvidia-smi's), `GQ_ID`, `GQ_DIR=/workspace/gq/<id>`, `GQ_RESULTS`, and `GQ_MAX_MEM_GB` when the
card has a memory cap. Box environments still apply: source `/workspace/box-env.sh` on Vast and put
`/usr/local/cuda/bin` on PATH on torn-gpu, inside your script.

### Selectors (`--on`)

`any` (default) · `vast` · `torn-gpu:0` · `cc>=8.6` · `vram>=20G`. gq refuses a selector that matches no
schedulable GPU.

### Priorities (`--prio`)

`dev` (default) > `kit` > `ladder` > `soak` > `backlog`. Higher priority goes first, then the oldest
submission. A running job is never preempted.

## Other commands

| command | does |
|---|---|
| `gq ls [--json]` | GPUs (busy reason or idle minutes), running and queued jobs, backlog depth, recent results |
| `gq wait ID [--max SEC]` | blocks until done; exits with the job's rc (124 on timeout) |
| `gq log ID [-n LINES]` | live tail over ssh while running, the synced copy afterwards (`-n 0` = all) |
| `gq cancel ID` | queued: dropped; running: TERM to its process group, KILL 15 s later |
| `gq machines [--json]` | live nvidia-smi summary of every card, plus the rules |
| `gq backlog add [submit options] -- CMD...` / `gq backlog` | add to / list the Vast backlog |

## Machine rules (`machines.json`)

- **vast** (RTX 3090): `never_idle`. When nothing in the queue fits, the dispatcher takes the first matching line
  from the backlog. The box-local `gpuq` runs first (see below).
- **torn-gpu:0** (RTX 3070, a friend's card): `only_when_idle`. A job starts only while the card shows less than
  1 GB in use; the owner keeps about 600 MiB on it. Under `max_mem_gb: 6.5`, the dispatcher kills a job when the
  card's total use goes over 6.5 + 1 GB.
- **torn-gpu:1** (RTX 2080 Ti): `schedulable: false`, telemetry only. gq never starts anything on it.

## Backlog

`~/tools/gq-data/backlog.txt` holds one `gq submit` argument line per job, consumed in order. A line runs only
when a `never_idle` GPU (the Vast 3090) is idle and the queue holds nothing for it. Lines become `backlog`
priority unless they carry `--prio`. Agents refill it with `gq backlog add ...`. The file is state, not code,
so it lives in the data dir and not in this repo.

## The box-local gpuq on Vast

`/usr/local/bin/gpuq` keeps working, and gq stays out of its way. The Vast GPU counts as busy while gpuq has a
running job, queued jobs, or a job in preflight. Agents should submit from ms4 with gq from now on.

The other direction needs one line in gpuq's preflight (`cloud/box/gpuq`): it waits for
`/workspace/gq/gpu*.lock`. Every gq job holds that lock while it runs (`flock -o`, not inherited by children).
**Until that line is installed on the box, a job submitted to gpuq can start on top of a running gq job.**
To install it, copy the repo file over the old one with a write-then-rename, so running `gpuq wait` loops keep
the old inode. No gpuq restart is needed, because the daemon re-runs `gpuq __preflight` for every job.

## Where things land (the dashboard reads these directly)

```
~/tools/gq-data/
  jobs/<id>.json      one per job: id name on prio est_minutes exclusive cmd script cwd results
                      status(queued|running|done|failed|cancelled) machine gpu rc note submitted started ended (epoch s)
  logs/<id>.log       the job's stdout+stderr, appended every 15 s while it runs
  results/<id>/       contents of the job's results dir, copied on exit
  state.json          written every cycle: {updated, backlog, gpus: {"vast:0": {machines.json fields, machine,
                      live: {name, mem_used_mib, mem_total_mib, util, temp_c, power_w}, busy, idle_since}}}
  backlog.txt         pending backlog lines
```

On the boxes, each job runs in `/workspace/gq/<id>/` (wrap.sh, script.sh, pid, log, rc, results/), detached
with `setsid nohup`. A dispatcher restart re-attaches through the pid and rc files and never re-runs a job. A
job whose process is gone with no rc becomes `failed`, noted "process vanished". The usual cause is a
container restart. rc 97 means the cwd is missing; rc 98 means the GPU lock was held. Old job dirs on the
boxes can be deleted by hand.

## Install on ms4 (not done yet, see below)

```
git clone -b gq ~/gsq-vllm ~/tools/gq && git -C ~/tools/gq remote set-url origin https://github.com/Starwaves1/vllm-gsqrco-exl3.git
printf '#!/bin/sh\nexec /usr/bin/python3 ~/tools/gq/cloud/gq/gq.py "$@"\n' > ~/.local/bin/gq && chmod +x ~/.local/bin/gq
cp ~/tools/gq/cloud/gq/gq-dispatcher.service ~/.config/systemd/user/ && systemctl --user daemon-reload && systemctl --user enable --now gq-dispatcher
journalctl --user -u gq-dispatcher -f     # events: started / done / requeued / killed / backlog
```

For tests, `GQ_DATA` and `GQ_MACHINES` point gq at a scratch data dir and a scratch machines file.

## Where I stopped (2026-10-04, budget pause)

**Works (tested from a scratch instance against both boxes):**
- a 10 s job on torn-gpu:0, on vast:0 (CPU only, beside the soak, with the gpuq check off in the test
  machines file), and a `cc>=8.6` job. All finished with rc 0, and their logs and results were copied back.
- `CUDA_VISIBLE_DEVICES=0` maps to the RTX 3070 on torn-gpu (checked with cuDeviceGetName).
- a backlog line was consumed once the queue was empty.
- cancelling a queued job and a running one; selectors that match nothing are rejected.
- a second dispatcher refuses to start.
- the dispatcher was SIGKILLed mid-job. The job kept running, and the restarted dispatcher re-attached and
  finished it with the full log and no duplicate lines.

**Not done yet:**
1. Install on ms4: the clone, the wrapper and `gq-dispatcher` (commands above). The service is not installed,
   and nothing is running.
2. Install the gpuq preflight line on Vast (above). Until then, only the gq → gpuq direction is safe.
3. Live-test `gpuq` busy detection: submit a vast job while the soak runs and expect busy `gpuq` in `gq ls`.
   Also test the `max_mem_gb` kill path, e.g. with a tiny `max_mem_gb` on vast in a test machines file.
   Never test it by allocating memory on the 3070.
4. Write `gq ls --json` sample output for the dashboard agent. The fields are listed above.
5. Run one Fable /check over `cloud/gq` and fix what it finds.
