"""Server side of the torture harness (standard library only): the 60 s monitor, and for the serve
and switch modes the server's lifecycle and the leftover checks.

Monitor rows (monitor.csv; bench/soak.sh's quantities, plus the restart detector):
  t, server_alive (pid given: 1/0, else empty), health (/health HTTP code), gpu_mib (nvidia-smi, the
  server pid's process tree), rss_kib / rss_anon_kib / shmem_kib (/proc of that tree; shmem = CPU KV
  tier), running, waiting, kv_usage, preemptions, cpu_tier_perc, fs_tier_bytes (/metrics),
  start_time (/metrics process_start_time_seconds), restarts.
Fault lines go to faults.log: new server-log lines matching FAULTS (--server-log), "server exited"
(pid gone), "server restarted" (start_time changed), "server unhealthy" (3 rows without /health
200). Any of the last three sets `dead`: the run stops and is reported. Nothing is ever restarted.
A row that raises (the monitor's own bug) goes to monitor-errors.log; the monitor keeps going.
"""

import hashlib
import os
import re
import shlex
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path

FAULTS = re.compile(r"illegal memory access|misaligned address|unspecified launch failure|CUDA error|EngineDeadError"
                    r"|Engine core .* died|Traceback|Segmentation fault|out of memory")
COLUMNS = ["t", "server_alive", "health", "gpu_mib", "rss_kib", "rss_anon_kib", "shmem_kib", "running", "waiting",
           "kv_usage", "preemptions", "cpu_tier_perc", "fs_tier_bytes", "start_time", "restarts"]
GAUGES = {"running": "vllm:num_requests_running", "waiting": "vllm:num_requests_waiting", "kv_usage": "vllm:kv_cache_usage_perc",
          "preemptions": "vllm:num_preemptions_total", "cpu_tier_perc": "vllm:kv_offload_cpu_cache_usage_perc",
          "fs_tier_bytes": "vllm:kv_offload_fs_cache_bytes", "start_time": "process_start_time_seconds"}


def proc_table() -> dict[int, tuple[int, int]]:
    """pid -> (ppid, session id) for every process."""
    out = {}
    for d in os.listdir("/proc"):
        if d.isdigit():
            try:
                s = Path(f"/proc/{d}/stat").read_text()
            except OSError:
                continue
            f = s[s.rindex(")") + 2:].split()
            out[int(d)] = (int(f[1]), int(f[3]))
    return out


def tree(pid: int) -> list[int]:
    t, kids = proc_table(), [pid]
    out = []
    while kids:
        p = kids.pop()
        if p in t:
            out.append(p)
            kids += [c for c, (pp, _) in t.items() if pp == p]
    return out


def session(sid: int) -> list[int]:
    return [p for p, (_, s) in proc_table().items() if s == sid]


def memory(pids: list[int]) -> tuple[int, int, int]:
    """kB summed over pids: VmRSS, RssAnon, RssShmem."""
    tot = {"VmRSS": 0, "RssAnon": 0, "RssShmem": 0}
    for p in pids:
        try:
            for line in Path(f"/proc/{p}/status").read_text().splitlines():
                k, _, v = line.partition(":")
                if k in tot:
                    tot[k] += int(v.split()[0])
        except OSError:
            pass
    return tot["VmRSS"], tot["RssAnon"], tot["RssShmem"]


def gpu_mib(pids: list[int] | None = None) -> int | None:
    """MiB used by pids (compute apps), or by all GPUs when pids is None; None without nvidia-smi."""
    if not shutil.which("nvidia-smi"):
        return None
    q = "--query-compute-apps=pid,used_memory" if pids is not None else "--query-gpu=index,memory.used"
    try:
        out = subprocess.run(["nvidia-smi", q, "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    rows = [line.split(", ") for line in out.splitlines() if ", " in line]
    return sum(int(m) for k, m in rows if m.strip().isdigit() and (pids is None or int(k) in pids))  # "[N/A]" cells skipped


def pid_alive(pid: int) -> bool:
    """Exists and is not a zombie (a dead child of ours stays in /proc until reaped)."""
    try:
        s = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return False
    return s[s.rindex(")") + 2] != "Z"


class Monitor(threading.Thread):
    def __init__(self, api, out: Path, server_pid: int | None = None, server_log: str | None = None, interval: float = 60):
        super().__init__(daemon=True)
        self.api, self.out, self.pid, self.log, self.interval = api, out, server_pid, server_log, interval
        self.dead, self.finished = threading.Event(), threading.Event()
        self.restarts, self.start0, self.bad_health = 0, None, 0
        self.logpos = os.path.getsize(server_log) if server_log and os.path.exists(server_log) else 0

    def fault(self, line: str, name: str = "faults.log") -> None:
        with open(self.out / name, "a") as f:
            f.write(line.rstrip("\n") + "\n")

    def row(self) -> None:
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        alive = int(pid_alive(self.pid)) if self.pid else ""
        health = self.api.call("GET", self.api.root + "/health", timeout=10)[0]
        met = self.api.call("GET", self.api.root + "/metrics", timeout=10)[1] if health else ""
        g = {}
        for line in met.splitlines():
            for col, name in GAUGES.items():
                if line.startswith(name) and line[len(name):len(name) + 1] in (" ", "{"):
                    try:
                        g[col] = g.get(col, 0.0) + float(line.split()[-1])
                    except ValueError:
                        pass
        r = {"t": int(time.time()), "server_alive": alive, "health": health, **{c: g.get(c, "") for c in GAUGES}}
        if self.pid:
            pids = tree(self.pid)
            r["rss_kib"], r["rss_anon_kib"], r["shmem_kib"] = memory(pids)
            r["gpu_mib"] = gpu_mib(pids)
        if self.start0 is None:
            self.start0 = r["start_time"] if r["start_time"] != "" else None
        elif r["start_time"] not in ("", self.start0):
            self.restarts += 1
            self.fault(f"{now} server restarted (process_start_time_seconds {self.start0} -> {r['start_time']})")
            self.start0 = r["start_time"]
            self.dead.set()
        r["restarts"] = self.restarts
        with open(self.out / "monitor.csv", "a") as f:
            f.write(",".join("" if r.get(c) is None else str(r.get(c, "")) for c in COLUMNS) + "\n")
        if self.log and os.path.exists(self.log):
            with open(self.log, "rb") as f:
                f.seek(self.logpos)
                new = f.read()
            self.logpos += len(new)
            for line in new.decode(errors="replace").splitlines():
                if FAULTS.search(line):
                    self.fault(line)
        if alive == 0:
            self.fault(f"{now} server exited")
            self.dead.set()
        self.bad_health = 0 if health == 200 else self.bad_health + 1
        if self.bad_health >= 3:
            self.fault(f"{now} server unhealthy: no /health 200 for 3 rows")
            self.dead.set()

    def run(self) -> None:
        (self.out / "monitor.csv").write_text(",".join(COLUMNS) + "\n")
        while not self.finished.is_set() and not self.dead.is_set():
            try:
                self.row()
            except Exception as e:  # noqa: BLE001
                self.fault(f"{time.strftime('%FT%TZ', time.gmtime())} {e!r}", "monitor-errors.log")
            self.finished.wait(self.interval)

    def stop(self) -> None:
        self.finished.set()
        self.join(timeout=120)


# ---------------------------------------------------------------- serve / switch

def port_listening(port: int) -> bool:
    for f in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            for line in Path(f).read_text().splitlines()[1:]:
                p = line.split()
                if p[3] == "0A" and int(p[1].rsplit(":", 1)[1], 16) == port:
                    return True
        except OSError:
            pass
    return False


def start_server(cmd: str, log: Path, api, timeout: float = 2400) -> tuple[subprocess.Popen, float | None]:
    """Start cmd in its own session, output appended to log; wait for /health 200.
    Returns (process, seconds to healthy or None if it exited or timed out)."""
    t0 = time.time()
    p = subprocess.Popen(shlex.split(cmd), stdout=open(log, "ab"), stderr=subprocess.STDOUT, start_new_session=True)
    while time.time() - t0 < timeout:
        if p.poll() is not None:
            return p, None
        if api.call("GET", api.root + "/health", timeout=5)[0] == 200:
            return p, round(time.time() - t0, 1)
        time.sleep(2)
    return p, None


def stop_server(p: subprocess.Popen) -> tuple[bool, int]:
    """SIGINT the session, wait up to 60 s for all of it to exit, then SIGKILL the group.
    Returns (the server itself needed SIGKILL, processes left in the session at that point)."""
    try:
        os.killpg(p.pid, signal.SIGINT)
    except ProcessLookupError:
        pass
    for _ in range(60):
        if p.poll() is not None and not session(p.pid):  # poll() first: it reaps the server's zombie
            break
        time.sleep(1)
    killed, left = p.poll() is None, len([x for x in session(p.pid) if x != p.pid])
    try:
        os.killpg(p.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    p.wait()
    return killed, left


def tier_snapshot(root: str | None) -> dict[str, str]:
    """namespace dir -> fingerprint of its files (relative path, size, mtime)."""
    out = {}
    if root and os.path.isdir(root):
        for d in sorted(os.listdir(root)):
            full = os.path.join(root, d)
            if os.path.isdir(full):
                items = []
                for r, _, fs in os.walk(full):
                    for f in fs:
                        try:
                            st = os.stat(os.path.join(r, f))
                        except OSError:
                            continue
                        items.append(f"{os.path.relpath(os.path.join(r, f), full)} {st.st_size} {st.st_mtime_ns}")
                out[d] = hashlib.sha1("\n".join(sorted(items)).encode()).hexdigest()[:16]
    return out
