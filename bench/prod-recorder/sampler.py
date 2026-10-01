"""1 Hz /metrics sampler + per-request condition windows. Stdlib only.

Runs inside the recorder process (so each second's line also says which
proxied requests were in flight, by type) or standalone:
    python3 sampler.py --upstream http://127.0.0.1:18081 --data data
"""
import argparse
import asyncio
import json
import os
import re
import time
from pathlib import Path

_PROM_LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(.*)\})?\s+(\S+)\s*$")
_PROM_LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="([^"]*)"')

# Same counters/gauges as the llama-dashboard export.
COUNTERS = {
    "vllm:prompt_tokens_total": "prompt",
    "vllm:generation_tokens_total": "gen",
    "vllm:prefix_cache_queries_total": "pfx_q",
    "vllm:prefix_cache_hits_total": "pfx_h",
    "vllm:external_prefix_cache_hits_total": "ext_h",
    "vllm:spec_decode_num_drafts_total": "drafts",
    "vllm:spec_decode_num_draft_tokens_total": "draft_toks",
    "vllm:spec_decode_num_accepted_tokens_total": "acc_toks",
    "vllm:num_preemptions_total": "preempt",
}
GAUGES = {
    "vllm:num_requests_running": "running",
    "vllm:num_requests_waiting": "waiting",
    "vllm:kv_cache_usage_perc": "kv_usage",
}

# Production MTP schedule: running 1-4 -> k=5, 5-8 -> 3, 9-16 -> 2.
DEFAULT_SCHEDULE = [[1, 4, 5], [5, 8, 3], [9, 16, 2]]


def parse_metrics(text):
    """Prometheus text -> {"g": gauges, "c": counters, "fin": {reason: n},
    "start_ts"} (only the series the recorder needs)."""
    g, c, fin, start = {}, {}, {}, None
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        m = _PROM_LINE.match(line)
        if not m:
            continue
        name, labels, sval = m.groups()
        if name in COUNTERS or name in GAUGES or name == "vllm:request_success_total" \
                or name == "process_start_time_seconds":
            try:
                val = float(sval)
            except ValueError:
                continue
            if val != val:
                continue
            if name in COUNTERS:
                c[COUNTERS[name]] = c.get(COUNTERS[name], 0.0) + val
            elif name in GAUGES:
                g[GAUGES[name]] = g.get(GAUGES[name], 0.0) + val
            elif name == "process_start_time_seconds":
                start = val
            else:
                reason = dict(_PROM_LABEL.findall(labels or "")).get("finished_reason", "?")
                fin[reason] = fin.get(reason, 0.0) + val
    return {"g": g, "c": c, "fin": fin, "start_ts": start}


def k_for(running, spec):
    """Speculative tokens per step for `running` requests under `spec`
    ({"k": fixed k or 0, "schedule": [[lo, hi, k], ...] or None})."""
    if not spec:
        return None
    sched = spec.get("schedule")
    if not sched:
        return spec.get("k")
    if running is None or running < 1:
        return None
    r = int(round(running))
    for lo, hi, k in sched:
        if lo <= r <= hi:
            return k
    return sched[-1][2] if r > sched[-1][1] else sched[0][2]


# ------------------------------------------------------------ upstream info

def _arg(argv, name):
    for i, a in enumerate(argv):
        if a == name and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return None


def spec_from_argv(argv):
    """Speculative-decoding config of a `vllm serve` argv."""
    raw = _arg(argv, "--speculative-config") or _arg(argv, "--speculative_config")
    if not raw:
        return {"method": "off", "k": 0, "schedule": None}
    try:
        cfg = json.loads(raw)
    except ValueError:
        return {"method": "unparsed", "k": None, "schedule": None}
    return {"method": cfg.get("method") or "?",
            "k": cfg.get("num_speculative_tokens"),
            "schedule": cfg.get("num_speculative_tokens_per_batch_size")}


def find_upstream(port, proc="/proc"):
    """argv-derived facts about the `vllm serve` listening on `port` (same
    user, read-only /proc scan). Returns {} when not found."""
    try:
        pids = [p for p in os.listdir(proc) if p.isdigit()]
    except OSError:
        return {}
    for pid in pids:
        try:
            with open(f"{proc}/{pid}/cmdline", "rb") as f:
                argv = [a.decode("utf-8", "replace") for a in f.read().split(b"\0") if a]
        except OSError:
            continue
        if not argv or not any("vllm" in a for a in argv[:3]) or "serve" not in argv[:4] and \
                not any(a.endswith("serve") for a in argv[:4]):
            continue
        if str(_arg(argv, "--port") or "8000") != str(port):
            continue
        if "--no-async-scheduling" in argv:
            asyn = "off"
        elif "--async-scheduling" in argv:
            asyn = "on"
        else:
            asyn = "auto"
        return {"pid": int(pid), "async": asyn, "spec": spec_from_argv(argv),
                "max_num_seqs": _arg(argv, "--max-num-seqs"),
                "argv": " ".join(argv)[:6000]}
    return {}


# ----------------------------------------------------- per-request window

class Window:
    """Conditions seen by one request while it was in flight."""

    def __init__(self, spec, sample=None):
        self.spec = spec
        self.n = 0
        self.rmin = self.rmax = None
        self.rsum = 0.0
        self.k = {}
        self.wmax = 0.0
        self.kvmax = 0.0
        self.first = self.last = None
        self.plp = 0
        self.echo = 0
        self.inflight_max = 0
        if sample:
            self.add(sample, 0, 0, 0)

    def add(self, sample, other_plp, other_echo, inflight):
        if not sample or "g" not in sample:
            return
        g = sample["g"]
        r = g.get("running")
        if r is not None:
            self.n += 1
            self.rmin = r if self.rmin is None else min(self.rmin, r)
            self.rmax = r if self.rmax is None else max(self.rmax, r)
            self.rsum += r
            k = k_for(r, self.spec)
            self.k[k] = self.k.get(k, 0) + 1
        self.wmax = max(self.wmax, g.get("waiting", 0.0))
        self.kvmax = max(self.kvmax, g.get("kv_usage", 0.0))
        if self.first is None:
            self.first = sample["c"]
        self.last = sample["c"]
        self.plp = max(self.plp, other_plp)
        self.echo = max(self.echo, other_echo)
        self.inflight_max = max(self.inflight_max, inflight)

    def summary(self):
        def d(key):
            if not self.first or not self.last or key not in self.last:
                return None
            v = self.last[key] - self.first.get(key, 0.0)
            return v if v >= 0 else None
        drafts, dtoks, acc = d("drafts"), d("draft_toks"), d("acc_toks")
        ks = {k: n for k, n in self.k.items() if k is not None}
        kmode = max(ks, key=ks.get) if ks else None
        return {
            "n_samples": self.n,
            "running_min": self.rmin, "running_max": self.rmax,
            "running_mean": round(self.rsum / self.n, 2) if self.n else None,
            "k_mode": kmode, "k_set": sorted(ks), "k_mixed": len(ks) > 1,
            "acc_rate": round(acc / dtoks, 4) if acc is not None and dtoks else None,
            "acc_len": round(1 + acc / drafts, 3) if acc is not None and drafts else None,
            "preempt_delta": d("preempt"),
            "gen_tokens_delta": d("gen"),
            "waiting_max": self.wmax, "kv_max": round(self.kvmax, 4),
            "other_prompt_logprobs_inflight": self.plp > 0,
            "other_echo_inflight": self.echo > 0,
            "inflight_max": self.inflight_max,
        }


# ------------------------------------------------------------ the sampler

async def http_get(host, port, path, timeout=2.0):
    """Minimal GET returning (status, body bytes)."""
    async def go():
        r, w = await asyncio.open_connection(host, port)
        try:
            w.write(f"GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\n"
                    f"User-Agent: prod-recorder\r\nConnection: close\r\n\r\n".encode())
            await w.drain()
            raw = await r.read()
        finally:
            w.close()
        head, _, body = raw.partition(b"\r\n\r\n")
        status = int(head.split(b" ", 2)[1])
        if b"transfer-encoding: chunked" in head.lower():
            body = dechunk(body)
        return status, body
    return await asyncio.wait_for(go(), timeout)


def dechunk(data):
    out, i = bytearray(), 0
    while True:
        j = data.index(b"\r\n", i)
        size = int(data[i:j].split(b";")[0], 16)
        if size == 0:
            return bytes(out)
        out += data[j + 2:j + 2 + size]
        i = j + 2 + size + 2


class HourlyWriter:
    """Append JSON lines to <dir>/<prefix>-YYYYmmdd-HH.jsonl (thread-safe)."""

    def __init__(self, data_dir, prefix):
        import threading
        self.dir = Path(data_dir)
        self.prefix = prefix
        self.lock = threading.Lock()

    def path(self, ts):
        return self.dir / f"{self.prefix}-{time.strftime('%Y%m%d-%H', time.localtime(ts))}.jsonl"

    def write(self, obj, ts=None):
        line = json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n"
        with self.lock, open(self.path(ts or time.time()), "a", encoding="utf-8") as f:
            f.write(line)


class Sampler:
    def __init__(self, host, port, data_dir, interval=1.0, inflight_fn=None,
                 on_sample=None, on_restart=None, log=print):
        self.host, self.port = host, port
        self.out = HourlyWriter(data_dir, "metrics")
        self.interval = interval
        self.inflight_fn = inflight_fn
        self.on_sample = on_sample
        self.on_restart = on_restart
        self.log = log
        self.latest = None
        self.start_ts = None
        self.errors = 0

    async def tick(self):
        ts = time.time()
        try:
            status, body = await http_get(self.host, self.port, "/metrics",
                                          timeout=max(0.5, self.interval * 0.9))
            sample = await asyncio.to_thread(parse_metrics, body.decode("utf-8", "replace"))
            sample["ts"] = round(ts, 3)
            if status != 200:
                sample = {"ts": round(ts, 3), "error": f"status {status}"}
        except Exception as e:  # upstream down, timeout, garbage
            self.errors += 1
            sample = {"ts": round(ts, 3), "error": f"{type(e).__name__}: {e}"[:200]}
        if sample.get("start_ts") and sample["start_ts"] != self.start_ts:
            old, self.start_ts = self.start_ts, sample["start_ts"]
            if self.on_restart:
                try:
                    self.on_restart(old, self.start_ts)
                except Exception as e:
                    self.log(f"sampler: on_restart failed: {e!r}")
        if self.inflight_fn:
            try:
                sample["inflight"] = self.inflight_fn()
            except Exception as e:
                self.log(f"sampler: inflight_fn failed: {e!r}")
        if "g" in sample:
            self.latest = sample
        try:
            await asyncio.to_thread(self.out.write, sample, ts)
        except Exception as e:
            self.errors += 1
            self.log(f"sampler: write failed: {e!r}")
        if self.on_sample and "g" in sample:
            try:
                self.on_sample(sample)
            except Exception as e:
                self.log(f"sampler: on_sample failed: {e!r}")
        return sample

    async def run(self, stop_event=None):
        nxt = time.monotonic()
        while not (stop_event and stop_event.is_set()):
            await self.tick()
            nxt += self.interval
            delay = nxt - time.monotonic()
            if delay < 0:
                nxt, delay = time.monotonic(), 0
            await asyncio.sleep(delay)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--upstream", default="http://127.0.0.1:18081")
    ap.add_argument("--data", default=str(Path(__file__).resolve().parent / "data"))
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--seconds", type=float, default=0, help="stop after N s (0 = forever)")
    a = ap.parse_args()
    hostport = a.upstream.split("://", 1)[-1].rstrip("/")
    host, port = hostport.rsplit(":", 1)
    Path(a.data).mkdir(parents=True, exist_ok=True)
    s = Sampler(host, int(port), a.data, a.interval)

    async def go():
        stop = asyncio.Event()
        if a.seconds:
            asyncio.get_running_loop().call_later(a.seconds, stop.set)
        await s.run(stop)
    asyncio.run(go())


if __name__ == "__main__":
    main()
