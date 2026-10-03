#!/usr/bin/env python3
"""Poll the Vast.ai box once a minute over one ssh call; append to DATA."""
import json, os, subprocess, time

DATA = os.environ.get("DASH_DATA") or os.path.expanduser("~/tools/dashboard-data")
Q = "/workspace/gpuq"
REMOTE = f"""
echo @@dash:gpu; nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total,power.draw,power.limit,clocks.sm,clocks.mem,temperature.gpu --format=csv,noheader,nounits
echo @@dash:mem; free -b | awk 'NR==2{{print $3,$7}}'
echo @@dash:disk; df -B1 / | awk 'NR==2{{print $3,$4}}'
echo @@dash:load; cat /proc/loadavg
echo @@dash:cgmax; cat /sys/fs/cgroup/memory.max
echo @@dash:gpuq; gpuq ls 2>&1 | head -n 60
echo @@dash:running; [ -f {Q}/running ] && echo "$(cat {Q}/running) $(stat -c %Y {Q}/running)"
echo @@dash:queued; for f in {Q}/jobs/*.job; do [ -e "$f" ] && printf '%s\\t%s\\n' "$(basename "$f" .job)" "$(head -c 400 "$f" | tr '\\n\\t' '  ')"; done
echo @@dash:status; grep -H '' {Q}/out/*.status 2>/dev/null
echo @@dash:logs; stat -c '%n %W %Y' {Q}/out/*.log 2>/dev/null
echo @@dash:tail; r=$(cut -d' ' -f1 {Q}/running 2>/dev/null); [ -n "$r" ] && tail -n 20 {Q}/out/$r.log | cut -c1-300
echo @@dash:end
"""
BANNER = ("Welcome to vast.ai", "Have fun!", "AI agents: READ")


def sections(text):
    out, cur = {}, None
    for line in text.splitlines():
        if line.startswith(BANNER):
            continue
        if line.startswith("@@dash:"):
            cur = out.setdefault(line[7:], [])
        elif cur is not None:
            cur.append(line)
    return out


def num(s):
    try:
        return float(s)
    except ValueError:
        return None


def job_id(path, ext):
    return os.path.basename(path)[: -len(ext)]


def poll():
    now = time.time()
    box = json.load(open(os.path.join(DATA, "box.json")))  # {"host":..., "port":...}; kept out of the public tree
    ssh = ["ssh", "-p", str(box["port"]), "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
           "-o", "ServerAliveInterval=10", "-o", "ServerAliveCountMax=2", "root@" + box["host"]]
    r = subprocess.run(ssh + [REMOTE], capture_output=True, text=True, timeout=60)
    s = sections(r.stdout)
    if "end" not in s or not s.get("gpu"):
        raise RuntimeError((r.stderr.strip() or "no output")[-300:])
    g = [x.strip() for x in s["gpu"][0].split(",")]
    keys = ["util", "vram_used", "vram_total", "power", "power_limit", "sm_clock", "mem_clock", "temp"]
    sample = {"ts": round(now), **{k: num(v) for k, v in zip(keys, g[1:])}}
    mem = (s.get("mem") or ["0 0"])[0].split()
    disk = (s.get("disk") or ["0 0"])[0].split()
    sample.update(ram_used=int(mem[0]), ram_avail=int(mem[1]), disk_used=int(disk[0]),
                  disk_free=int(disk[1]), load=[num(x) for x in (s.get("load") or ["0 0 0"])[0].split()[:3]])
    cg = (s.get("cgmax") or ["max"])[0].strip()
    sample["ram_limit"] = int(cg) if cg.isdigit() else None

    running = None
    if s.get("running") and s["running"][0].strip():
        rid, _pid, rstart = (s["running"][0].split() + ["", ""])[:3]  # "<id> <pid> <mtime of running file>"
        running = {"id": rid, "started": int(rstart) if rstart.isdigit() else None, "tail": s.get("tail", [])}
    queued = [dict(zip(("id", "cmd"), l.split("\t", 1))) for l in s.get("queued", []) if l]
    queued = [q for q in queued if not running or q["id"] != running["id"]]
    status = {}
    for l in s.get("status", []):
        path, _, code = l.partition(":")
        status[job_id(path, ".status")] = code.strip()
    logs = {}
    for l in s.get("logs", []):
        path, birth, mtime = l.rsplit(" ", 2)
        logs[job_id(path, ".log")] = (int(birth), int(mtime))
    gpuq = [l for l in s.get("gpuq", []) if l.strip()]
    return sample, running, queued, status, logs, gpuq


def main():
    os.makedirs(DATA, exist_ok=True)
    jobs_path = os.path.join(DATA, "jobs.json")
    while True:
        t0 = time.time()
        try:
            jobs = json.load(open(jobs_path)) if os.path.exists(jobs_path) else {}
        except ValueError:
            jobs = {}
        try:
            sample, running, queued, status, logs, gpuq = poll()
            with open(os.path.join(DATA, "samples.jsonl"), "a") as f:
                f.write(json.dumps(sample) + "\n")
            # Finished jobs: history is kept locally so it survives the box being destroyed.
            # A job is recorded once, when its status file first appears; logs copied from an older box have a
            # birth time after their mtime, so their start is unknown.
            for jid, code in status.items():
                if jid not in jobs:
                    birth, mtime = logs.get(jid, (0, 0))
                    jobs[jid] = {"status": code, "started": birth if 0 < birth <= mtime else None, "ended": mtime or None}
            now = {"online": True, "ts": sample["ts"], "sample": sample,
                   "running": running, "queued": queued, "gpuq": gpuq}
        except Exception as e:  # ssh down, timeout, parse failure: mark offline, keep last data
            old = {}
            try:
                old = json.load(open(os.path.join(DATA, "now.json")))
            except (OSError, ValueError):
                pass
            now = {**old, "online": False, "error": str(e)[-300:], "checked": round(t0)}
        for path, obj in ((jobs_path, jobs), (os.path.join(DATA, "now.json"), now)):
            with open(path + ".tmp", "w") as f:
                json.dump(obj, f)
            os.replace(path + ".tmp", path)
        time.sleep(max(1, 60 - (time.time() - t0)))


if __name__ == "__main__":
    main()
