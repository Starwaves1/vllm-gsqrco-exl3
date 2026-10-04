#!/usr/bin/env python3
"""gq: cluster GPU job queue. One dispatcher on ms4 launches jobs over ssh on the GPUs in machines.json.

  gq submit [--name N] [--on SEL] [--prio P] [--est MIN] [--cwd DIR] [--results DIR] [--exclusive] -- CMD...
            (or --script LOCAL_FILE [-- ARGS]: the file is copied to the box and run with bash)
  gq ls [--json] | gq wait ID [--max SEC] | gq log ID [-n LINES] | gq cancel ID | gq machines [--json]
  gq backlog [ls] | gq backlog add [submit options] -- CMD...      gq dispatch   (the dispatcher loop; run by gq-dispatcher)

SEL: any | MACHINE | MACHINE:GPU | cc>=X | vram>=Y.  P: dev > kit > ladder > soak > backlog.  See README.md.
"""
import argparse, fcntl, json, os, shlex, subprocess, sys, time, traceback
from contextlib import contextmanager

HERE = os.path.dirname(os.path.realpath(__file__))
DATA = os.environ.get("GQ_DATA", os.path.expanduser("~/tools/gq-data"))
MACHINES = os.environ.get("GQ_MACHINES", os.path.join(HERE, "machines.json"))
PRIOS = ["dev", "kit", "ladder", "soak", "backlog"]
FINAL = ("done", "failed", "cancelled")
REMOTE = "/workspace/gq"  # on every box: REMOTE/<id>/{wrap.sh,script.sh,pid,log,rc,results/}, REMOTE/gpu<N>.lock
IDLE_MIB = 1024  # only_when_idle: a card counts as idle below this much used memory (the 3070's owner keeps ~600 MiB)
POLL_S = 15
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=15", "-o", "LogLevel=ERROR"]


def now():
    return int(time.time())


def log(msg):
    print(time.strftime("%F %T"), msg, flush=True)


def path(*p):
    return os.path.join(DATA, *p)


def machines():
    with open(MACHINES) as f:
        return json.load(f)


def gpus(ms):
    for name, m in ms.items():
        for g in m["gpus"]:
            yield name, m, g


def matches(sel, name, g):
    for key, field in (("cc>=", "cc"), ("vram>=", "vram_gb")):
        if sel.startswith(key):
            return g[field] >= float(sel[len(key):].rstrip("GgBb"))
    return sel in ("any", name, f"{name}:{g['index']}")


def sshcmd(m):
    cmd = shlex.split(m["ssh"])
    return cmd[:1] + SSH_OPTS + cmd[1:]


def ssh(m, script, timeout=60):
    """Run a bash script on machine m; returns its stdout (bytes), or None if the box did not answer."""
    try:
        r = subprocess.run(sshcmd(m) + ["bash -s"], input=script.encode(), capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    return None if r.returncode == 255 else r.stdout


@contextmanager
def locked():
    with open(path("lock"), "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield


# ---- jobs: one JSON file per job in DATA/jobs ----

def load(i):
    with open(path("jobs", i + ".json")) as f:
        return json.load(f)


def save(j):
    p = path("jobs", j["id"] + ".json")
    with open(p + ".tmp", "w") as f:
        json.dump(j, f, indent=1)
    os.replace(p + ".tmp", p)


def all_jobs():
    js = []
    for fn in os.listdir(path("jobs")):
        if fn.endswith(".json"):
            try:
                js.append(load(fn[:-5]))
            except (OSError, ValueError):
                pass
    return sorted(js, key=lambda j: (j["submitted"], j["id"]))


def update(i, **kw):
    with locked():
        j = load(i)
        j.update(kw)
        save(j)
    return j


def parse_submit(argv):
    k = argv.index("--") if "--" in argv else len(argv)
    cmd = argv[k + 1:]
    p = argparse.ArgumentParser(prog="gq submit")
    p.add_argument("--name", default="job")
    p.add_argument("--on", default="any", help="any | MACHINE | MACHINE:GPU | cc>=X | vram>=Y")
    p.add_argument("--prio", choices=PRIOS)
    p.add_argument("--est", type=int, default=60, help="estimated minutes")
    p.add_argument("--cwd", help="working dir on the target (default: the job dir)")
    p.add_argument("--results", help="dir on the target synced back on exit (default: $GQ_DIR/results)")
    p.add_argument("--script", help="local file copied to $GQ_DIR/script.sh and run with bash; CMD = its arguments")
    p.add_argument("--exclusive", action="store_true", help="no other gq job on the machine meanwhile")
    a = p.parse_args(argv[:k])
    if not cmd and not a.script:
        p.error("no command after -- and no --script")
    try:
        ok = any(g.get("schedulable", True) and matches(a.on, n, g) for n, _, g in gpus(machines()))
    except ValueError:
        ok = False
    if not ok:
        p.error(f"--on {a.on} matches no schedulable GPU")
    return a, cmd


def create(argv, prio="dev"):
    a, cmd = parse_submit(argv)
    script = open(a.script).read() if a.script else None
    base = time.strftime("%m%d-%H%M%S")
    with locked():
        i, n = base, 1
        while os.path.exists(path("jobs", i + ".json")):
            n += 1
            i = f"{base}-{n}"
        d = f"{REMOTE}/{i}"
        j = dict(id=i, name=a.name, on=a.on, prio=a.prio or prio, est_minutes=a.est, exclusive=a.exclusive,
                 cmd=["bash", f"{d}/script.sh", *cmd] if script else cmd, script=script, cwd=a.cwd or d,
                 results=a.results or f"{d}/results", status="queued", machine=None, gpu=None, rc=None,
                 note="", submitted=now(), started=None, ended=None)
        save(j)
    return j


def backlog_lines():
    try:
        with open(path("backlog.txt")) as f:
            return [l.rstrip("\n") for l in f if l.strip() and not l.startswith("#")]
    except FileNotFoundError:
        return []


def from_backlog(n, g):
    """Pop the first backlog line that can run on this GPU and submit it (prio backlog unless the line says otherwise)."""
    with locked():
        lines = backlog_lines()
        for k, line in enumerate(lines):
            try:
                a, _ = parse_submit(shlex.split(line))
            except SystemExit:
                continue
            if matches(a.on, n, g):
                with open(path("backlog.txt.tmp"), "w") as f:
                    f.writelines(l + "\n" for l in lines[:k] + lines[k + 1:])
                os.replace(path("backlog.txt.tmp"), path("backlog.txt"))
                break
        else:
            return None
    try:
        j = create(shlex.split(line), prio="backlog")
    except (OSError, SystemExit) as e:
        log(f"backlog line dropped ({e}): {line}")
        return None
    log(f"backlog -> {j['id']} {j['name']}")
    return j


# ---- remote side ----

def gpuq_busy_sh(q):
    """Shell condition: the box-local gpuq has a running job, queued jobs, or a job in preflight."""
    return f'[ -e {q}/running ] || ls {q}/jobs/*.job >/dev/null 2>&1 || pgrep -f "gpuq __preflight" >/dev/null'


def num(x):
    try:
        return float(x) if "." in x else int(x)
    except ValueError:
        return None


def probe(m):
    """Live GPU state of a machine: {index: {...}, "gpuq_busy": bool}, or None if unreachable."""
    q = "index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw"
    s = f"nvidia-smi --query-gpu={q} --format=csv,noheader,nounits\n"
    if m.get("gpuq"):
        s += f"if {gpuq_busy_sh(m['gpuq'])}; then echo gpuq-busy; fi\n"
    out = ssh(m, s)
    if out is None:
        return None
    st = {"gpuq_busy": False}
    for line in out.decode(errors="replace").splitlines():
        f = [x.strip() for x in line.split(",")]
        if line == "gpuq-busy":
            st["gpuq_busy"] = True
        elif len(f) == 7 and f[0].isdigit():
            st[int(f[0])] = dict(name=f[1], mem_used_mib=num(f[2]), mem_total_mib=num(f[3]), util=num(f[4]),
                                 temp_c=num(f[5]), power_w=num(f[6]))
    return st


def launch(m, g, j):
    """Start job j on GPU g, detached (setsid nohup) so it survives ssh and dispatcher restarts.
    Returns "launched", "busy" (box-local gpuq took the GPU first) or None (no answer)."""
    d = f"{REMOTE}/{j['id']}"
    env = dict(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=g["index"], GQ_ID=j["id"], GQ_DIR=d,
               GQ_RESULTS=j["results"])
    if "max_mem_gb" in g:
        env["GQ_MAX_MEM_GB"] = g["max_mem_gb"]
    # flock -o: the lock is held while CMD runs but not inherited by it; gpuq's preflight waits on it (cloud/box/gpuq)
    wrap = (f"cd {shlex.quote(j['cwd'])} || {{ echo 97 > {d}/rc; exit; }}\n"
            "export " + " ".join(f"{k}={shlex.quote(str(v))}" for k, v in env.items()) + "\n"
            f"flock -n -E 98 -o {REMOTE}/gpu{g['index']}.lock {shlex.join(j['cmd'])}\n"
            f"echo $? > {d}/rc.tmp && mv {d}/rc.tmp {d}/rc\n")
    s = f"mkdir -p {d} {shlex.quote(j['results'])} && cd {d} || exit 1\n"
    if m.get("gpuq"):
        s += f"if {gpuq_busy_sh(m['gpuq'])}; then echo busy; exit 0; fi\n"
    s += "[ -e pid ] && { echo launched; exit 0; }\n"
    for fname, text in (("wrap.sh", wrap), ("script.sh", j["script"])):
        if text is not None:
            s += f"cat > {fname} <<'GQ_EOF_{j['id']}'\n{text.rstrip(chr(10))}\nGQ_EOF_{j['id']}\n"
    s += "setsid nohup bash wrap.sh > log 2>&1 < /dev/null & echo $! > pid; echo launched\n"
    out = ssh(m, s)
    return None if out is None else out.decode().strip()


def poll(m, j):
    """Append new log bytes to DATA/logs/<id>.log; return "rc N", "alive", "dead" (gone without rc) or "nopid"."""
    d = f"{REMOTE}/{j['id']}"
    lp = path("logs", j["id"] + ".log")
    off = os.path.getsize(lp) if os.path.exists(lp) else 0
    out = ssh(m, f"""cd {d} 2>/dev/null || {{ echo nopid; exit 0; }}
if [ -e rc ]; then echo "rc $(cat rc)"; elif [ ! -e pid ]; then echo nopid
elif ps -o args= -p "$(cat pid)" | grep -q wrap.sh; then echo alive; else echo dead; fi
tail -c +{off + 1} log 2>/dev/null; exit 0""")
    if out is None:
        return None
    head, _, tail = out.partition(b"\n")
    with open(lp, "ab") as f:
        f.write(tail)
    return head.decode()


def kill(m, j):
    """TERM the job's process group, KILL it 15 s later if anything is left."""
    d = f"{REMOTE}/{j['id']}"
    ssh(m, f"""p=$(cat {d}/pid 2>/dev/null) && ps -o args= -p "$p" | grep -q wrap.sh || exit 0
kill -TERM -- -"$p"; setsid sh -c "sleep 15; kill -KILL -- -$p" >/dev/null 2>&1 < /dev/null &""")


def fetch_results(m, j):
    dst = path("results", j["id"])
    os.makedirs(dst, exist_ok=True)
    src = subprocess.Popen(sshcmd(m) + [f"tar -C {shlex.quote(j['results'])} -cf - ."], stdout=subprocess.PIPE,
                           stderr=subprocess.DEVNULL)
    ok = subprocess.run(["tar", "-C", dst, "-xf", "-"], stdin=src.stdout, stderr=subprocess.DEVNULL).returncode == 0
    src.stdout.close()
    return src.wait() == 0 and ok


def finish(m, j, rc):
    synced = fetch_results(m, j)
    with locked():
        j = load(j["id"])
        status = "cancelled" if j.get("cancel") else "done" if rc == 0 else "failed"
        note = j["note"] if rc is not None else (j["note"] or "process vanished without an exit code")
        if not synced:
            note = (note + "; " if note else "") + "results sync failed"
        j.update(status=status, rc=rc, ended=now(), note=note)
        save(j)
    log(f"{status} {j['id']} {j['name']} rc={rc} {note}")


# ---- dispatcher ----

def cycle():
    ms = machines()
    live = {n: probe(m) for n, m in ms.items()}

    for j in all_jobs():  # 1. poll running jobs: log tail, exit, memory rule
        if j["status"] != "running" or not live.get(j["machine"]):
            continue
        m = ms[j["machine"]]
        g = next(x for x in m["gpus"] if x["index"] == j["gpu"])
        st = poll(m, j)
        if st is None:
            continue
        if st.startswith("rc "):
            finish(m, j, int(st[3:]))
        elif st == "dead":
            finish(m, j, None)
        elif st == "nopid":  # the dispatcher died between marking it running and launching it
            update(j["id"], status="queued", machine=None, gpu=None, started=None)
            log(f"requeued {j['id']} (never started)")
        elif "max_mem_gb" in g:
            used = live[j["machine"]].get(g["index"], {}).get("mem_used_mib") or 0
            if used > g["max_mem_gb"] * 1024 + IDLE_MIB:
                update(j["id"], note=f"killed: card memory {used} MiB over max_mem_gb {g['max_mem_gb']} + 1 GB")
                kill(m, j)
                log(f"killed {j['id']}: {used} MiB on {j['machine']}:{j['gpu']}")

    js = all_jobs()  # 2. fill free GPUs
    running = [j for j in js if j["status"] == "running"]
    prev = {}
    if os.path.exists(path("state.json")):
        with open(path("state.json")) as f:
            prev = json.load(f).get("gpus", {})
    state = {}
    for n, m, g in gpus(ms):
        key = f"{n}:{g['index']}"
        L = live[n]
        mine = next((j for j in running if j["machine"] == n and j["gpu"] == g["index"]), None)
        if not L or g["index"] not in L:
            busy = "unreachable"
        elif mine:
            busy = "gq:" + mine["id"]
        elif not g.get("schedulable", True):
            busy = "not schedulable"
        elif L["gpuq_busy"]:
            busy = "gpuq"
        elif g.get("only_when_idle") and L[g["index"]]["mem_used_mib"] >= IDLE_MIB:
            busy = f"in use ({L[g['index']]['mem_used_mib']} MiB)"
        elif any(j["exclusive"] for j in running if j["machine"] == n):
            busy = "exclusive job on machine"
        else:
            busy = None
            cands = sorted((j for j in js if j["status"] == "queued" and matches(j["on"], n, g)),
                           key=lambda j: (PRIOS.index(j["prio"]), j["submitted"], j["id"]))
            j = cands[0] if cands else (g.get("never_idle") and from_backlog(n, g)) or None
            if j and j["exclusive"] and any(r["machine"] == n for r in running):
                j = None  # wait for the machine to drain
            if j:
                busy = start(n, m, g, j)
                if busy:
                    running.append(load(j["id"]))
                    js = all_jobs()
        idle_since = None if busy else (prev.get(key, {}).get("idle_since") or now())
        state[key] = dict(g, machine=n, live=L.get(g["index"]) if L else None, busy=busy, idle_since=idle_since)

    with open(path("state.json.tmp"), "w") as f:
        json.dump(dict(updated=now(), gpus=state, backlog=len(backlog_lines())), f, indent=1)
    os.replace(path("state.json.tmp"), path("state.json"))


def start(n, m, g, j):
    with locked():
        j = load(j["id"])
        if j["status"] != "queued":
            return None
        j.update(status="running", machine=n, gpu=g["index"], started=now())
        save(j)
    r = launch(m, g, j)
    if r == "busy":
        update(j["id"], status="queued", machine=None, gpu=None, started=None)
        return "gpuq"
    log(f"started {j['id']} {j['name']} on {n}:{g['index']}" + ("" if r else " (unconfirmed; the next poll decides)"))
    return "gq:" + j["id"]


def dispatch():
    lk = open(path("dispatcher.lock"), "w")
    try:
        fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit("another gq dispatcher is running")
    log(f"dispatcher up: data {DATA}, machines {MACHINES}")
    while True:
        t = time.time()
        try:
            cycle()
        except Exception:
            log(traceback.format_exc())
        time.sleep(max(1, POLL_S - (time.time() - t)))


# ---- CLI ----

def mins(a, b=None):
    return f"{((b or now()) - a) // 60}m"


def cmd_ls(as_json):
    st = {}
    if os.path.exists(path("state.json")):
        with open(path("state.json")) as f:
            st = json.load(f)
    js = all_jobs()
    done = [j for j in js if j["status"] in FINAL][-50:]
    live = [j for j in js if j["status"] not in FINAL]
    if as_json:
        print(json.dumps(dict(state=st, jobs=live + done, backlog=backlog_lines()), indent=1))
        return
    age = now() - st["updated"] if st else None
    print("no dispatcher state yet" if age is None else f"dispatcher state {age}s old" + ("  STALE: is gq-dispatcher running?" if age > 60 else ""))
    for key, g in st.get("gpus", {}).items():
        L = g["live"] or {}
        mem = f"{L.get('mem_used_mib')}/{L.get('mem_total_mib')} MiB {L.get('util')}%" if L else "-"
        what = f"idle {mins(g['idle_since'])}" if g["idle_since"] else g["busy"]
        print(f"  {key:11} {g['name']:12} {mem:22} {what}")
    print("running:")
    for j in live:
        if j["status"] == "running":
            print(f"  {j['id']} {j['name']} on {j['machine']}:{j['gpu']} {j['prio']} {mins(j['started'])} of ~{j['est_minutes']}m")
    print("queued:")
    for j in sorted((j for j in live if j["status"] == "queued"), key=lambda j: (PRIOS.index(j["prio"]), j["submitted"])):
        print(f"  {j['id']} {j['name']} --on {j['on']} {j['prio']} ~{j['est_minutes']}m, waiting {mins(j['submitted'])}")
    print(f"backlog: {len(backlog_lines())} lines")
    print("recent:")
    for j in done[-10:]:
        print(f"  {j['id']} {j['name']} {j['status']} rc={j['rc']} on {j['machine']}:{j['gpu']} {j['note']}")


def cmd_wait(i, mx):
    t0 = time.time()
    while True:
        j = load(i)
        if j["status"] in FINAL:
            print(f"{j['status']} rc={j['rc']} {i} {j['name']}")
            sys.exit(0 if j["status"] == "done" else j["rc"] or 1)
        if time.time() - t0 >= mx:
            sys.exit(124)
        time.sleep(5)


def cmd_log(i, n):
    j = load(i)
    if j["status"] == "running":
        out = ssh(machines()[j["machine"]], f"tail -n {n or '+1'} {REMOTE}/{i}/log")
        if out is not None:
            sys.stdout.buffer.write(out)
            return
    try:
        with open(path("logs", i + ".log"), "rb") as f:
            lines = f.readlines()
    except FileNotFoundError:
        sys.exit(f"no log for {i} ({j['status']})")
    sys.stdout.buffer.write(b"".join(lines[-n:] if n else lines))


def cmd_cancel(i):
    with locked():
        j = load(i)
        if j["status"] == "queued":
            j.update(status="cancelled", ended=now())
            save(j)
            print(f"cancelled queued {i}")
            return
        if j["status"] != "running":
            sys.exit(f"{i} is already {j['status']}")
        j["cancel"] = True
        save(j)
    kill(machines()[j["machine"]], j)
    print(f"signalled running {i} on {j['machine']}:{j['gpu']}")


def cmd_machines(as_json):
    ms = machines()
    live = {n: probe(m) for n, m in ms.items()}
    if as_json:
        print(json.dumps(live, indent=1))
        return
    for n, m, g in gpus(ms):
        L = (live[n] or {}).get(g["index"])
        rules = " ".join(k if v is True else f"{k}={v}" for k, v in g.items()
                         if k in ("schedulable", "only_when_idle", "never_idle", "max_mem_gb", "note"))
        if L:
            print(f"{n}:{g['index']:<3} {L['name']:26} {L['mem_used_mib']:>6}/{L['mem_total_mib']} MiB {L['util']:>3}% "
                  f"{L['temp_c']}C {L['power_w']}W  {rules}")
        else:
            print(f"{n}:{g['index']:<3} UNREACHABLE  {rules}")
    for n, L in live.items():
        if L and L["gpuq_busy"]:
            print(f"{n}: box-local gpuq is busy (gq waits for it)")


def main(argv):
    for d in ("jobs", "logs", "results"):
        os.makedirs(path(d), exist_ok=True)
    cmd, args = (argv[0], argv[1:]) if argv else ("help", [])
    opt = lambda k, d: type(d)(args[args.index(k) + 1]) if k in args else d
    if cmd == "submit":
        print(create(args)["id"])
    elif cmd == "ls":
        cmd_ls("--json" in args)
    elif cmd == "wait":
        cmd_wait(args[0], opt("--max", 10**9))
    elif cmd == "log":
        cmd_log(args[0], opt("-n", 200))
    elif cmd == "cancel":
        cmd_cancel(args[0])
    elif cmd == "machines":
        cmd_machines("--json" in args)
    elif cmd == "backlog" and args[:1] == ["add"]:
        parse_submit(args[1:])
        with locked(), open(path("backlog.txt"), "a") as f:
            f.write(shlex.join(args[1:]) + "\n")
        print(f"backlog: {len(backlog_lines())} lines")
    elif cmd == "backlog":
        print("\n".join(backlog_lines()))
    elif cmd == "dispatch":
        dispatch()
    else:
        print(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
