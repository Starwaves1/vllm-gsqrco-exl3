"""Round-3 load driver and /metrics-based step accounting (stdlib + `tokenizers`).

Prompts are token ids from bench/parity/prompts.py's build_text (salted, so no prefix-cache or
KV-tier entry matches unless a prompt is reused on purpose), cached as JSON under --cache.
Requests go to /v1/completions with the ids, streaming, ignore_eos, so the API server still
detokenizes every step. Per-step numbers come from /metrics exactly as the 2026-10-01 production
profile computed them (cloud/results/prod-profile-20261001.md section 1):
  steps/s = d(spec_decode_num_drafts) / n,  ms/step = 1000 n / (drafts/s)
plus the engine's own step counter (iteration_tokens_total_count) as a cross-check, and
nvidia-smi utilization at 1 Hz for the busy/idle split (idle ms = (1 - util) x ms/step).

  r3load.py warm    --url U
  r3load.py fill    --url U --n 2 --tokens 90000 [--conc 2]
  r3load.py steady  --url U --conc 2 --tokens 96000 --max-tokens 12000 --window 90 --tag c2 --out DIR
                    [--kind chat] [--temperature 0|default] [--settle 5] [--hook 'shell cmd']
                    [--reuse NAME:TOKENS]  (stream 0 reuses a named prompt, e.g. for prefix hits)
                    [--prefix NAME:KIND:TOKENS]  (stream 0 = that prompt + a fresh tail up to --tokens)
  r3load.py mixed   --url U --decoders 3 --dec-tokens 40000 --lane 600,1400,turn,4000
                    --turn-prefix 60000 --turn-new 1800 --window 120 --tag t128 --out DIR
  r3load.py probe   --url U --n 4 --tokens 2000 --max-tokens 256 --tag X --out DIR   (greedy outputs, saved)
  r3load.py plan    (prints what each subcommand would send; no network)
Every subcommand exits non-zero with a reason when its validity checks fail.
"""

import argparse
import http.client
import json
import os
import random
import re
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "bench" / "parity"))
MODEL = os.environ.get("R3_SERVED_MODEL", "qwen3.8-27b")
API_KEY = os.environ.get("GSQ_API_KEY", "gsq-local-test")
CACHE = Path(os.environ.get("R3_PROMPT_CACHE", "/workspace/logs/r3/_prompts"))


def die(msg: str) -> None:
    print(f"r3load: FAIL: {msg}", file=sys.stderr, flush=True)
    sys.exit(3)


# ---------------------------------------------------------------- prompts
_tok = None


def prompt_ids(name: str, kind: str, n: int) -> list[int]:
    """Salted token ids, cached. The same (name, kind, n) always gives the same ids."""
    global _tok
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{name}-{kind}-{n}.json"
    if f.exists():
        return json.loads(f.read_text())
    import prompts  # bench/parity/prompts.py (imports tokenizers + numpy only)
    if _tok is None:
        from tokenizers import Tokenizer
        _tok = Tokenizer.from_file(str(prompts.HF_CONFIG / "tokenizer.json"))
    ids = prompts.build_text(kind, n, random.Random(f"r3:{name}:{kind}:{n}"), _tok)
    if abs(len(ids) - n) > 64:
        die(f"prompt {name} has {len(ids)} tokens, wanted {n}")
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps(ids))
    tmp.rename(f)
    return ids


def prompt_text(name: str, n: int) -> str:
    """A ~n-token code text (the "code" kind's ids decoded), for chat messages."""
    global _tok
    ids = prompt_ids(name, "code", n)
    if _tok is None:
        import prompts
        from tokenizers import Tokenizer
        _tok = Tokenizer.from_file(str(prompts.HF_CONFIG / "tokenizer.json"))
    return _tok.decode(ids)


# Tool definitions as bench/torture/load.py sends them (an agent's read_file / run_shell).
TOOLS = [{"type": "function", "function": {
    "name": "read_file", "description": "Read a file from the repository",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "start_line": {"type": "integer"}},
                   "required": ["path"]}}},
         {"type": "function", "function": {
             "name": "run_shell", "description": "Run a shell command in the repository and return its output",
             "parameters": {"type": "object", "properties": {"command": {"type": "string"}, "timeout_s": {"type": "integer"}},
                            "required": ["command"]}}}]


# ---------------------------------------------------------------- HTTP
class Server:
    def __init__(self, url: str):
        u = urlparse(url)
        self.host, self.port = u.hostname, u.port or 80
        self.hdr = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    def conn(self, timeout=3600):
        return http.client.HTTPConnection(self.host, self.port, timeout=timeout)

    def get(self, path: str, timeout=10) -> str:
        c = self.conn(timeout)
        c.request("GET", path, headers=self.hdr)
        r = c.getresponse()
        body = r.read().decode()
        c.close()
        if r.status != 200:
            raise RuntimeError(f"GET {path}: HTTP {r.status} {body[:200]}")
        return body

    def post(self, path: str, obj=None, timeout=600) -> str:
        c = self.conn(timeout)
        c.request("POST", path, body=json.dumps(obj or {}), headers=self.hdr)
        r = c.getresponse()
        body = r.read().decode()
        c.close()
        if r.status != 200:
            raise RuntimeError(f"POST {path}: HTTP {r.status} {body[:300]}")
        return body


def completion_body(ids, max_tokens, temperature, seed, stream=True):
    b = {"model": MODEL, "prompt": ids, "max_tokens": max_tokens, "ignore_eos": True,
         "stream": stream, "seed": seed}
    if stream:
        b["stream_options"] = {"include_usage": True}
    if temperature != "default":
        b["temperature"] = float(temperature)
    return b


class ChatStream(threading.Thread):
    """Streaming chat completions (reasoning on by the template default), optionally with tools,
    for `turns` turns: each turn appends the assistant's text and a short user follow-up, so every
    turn after the first is a prefix-cache hit plus a few hundred new tokens."""

    def __init__(self, srv, text, max_tokens, temperature, seed, label, tools, turns, ignore_eos):
        super().__init__(daemon=True)
        self.srv, self.label, self.turns = srv, label, turns
        self.msgs = [{"role": "user", "content": "Review this code. Explain what it does, then list concrete bugs "
                      "and fixes, using the tools to inspect files where useful.\n\n" + text}]
        self.base = {"model": MODEL, "max_tokens": max_tokens, "stream": True, "seed": seed,
                     "stream_options": {"include_usage": True}, "ignore_eos": ignore_eos}
        if temperature != "default":
            self.base["temperature"] = float(temperature)
        if tools:
            self.base.update(tools=TOOLS, tool_choice="auto")
        self.ids = text  # len() for the summary: characters, not tokens
        self.t_send = self.t_first = self.t_end = None
        self.chunks, self.turn_times, self.error = 0, [], None
        self._stop = threading.Event()
        self._conn = None

    def run(self):
        try:
            for t in range(self.turns):
                if self._stop.is_set():
                    break
                self._conn = c = self.srv.conn(timeout=7200)
                ts = time.time()
                if self.t_send is None:
                    self.t_send = ts
                c.request("POST", "/v1/chat/completions", body=json.dumps({**self.base, "messages": self.msgs}),
                          headers=self.srv.hdr)
                r = c.getresponse()
                if r.status != 200:
                    self.error = f"HTTP {r.status} {r.read()[:300]!r}"
                    return
                content = []
                while not self._stop.is_set():
                    line = r.readline()
                    if not line:
                        break
                    if not line.startswith(b"data: "):
                        continue
                    data = line[6:].strip()
                    if data == b"[DONE]":
                        break
                    obj = json.loads(data)
                    ch = obj.get("choices") or []
                    if ch and ch[0].get("delta"):
                        d = ch[0]["delta"]
                        if self.t_first is None:
                            self.t_first = time.time()
                        self.chunks += 1
                        if d.get("content"):
                            content.append(d["content"])
                c.close()
                self.turn_times.append(time.time() - ts)
                self.msgs += [{"role": "assistant", "content": "".join(content) or "(tool call)"},
                              {"role": "user", "content": "Continue: go deeper on the next part of the file."}]
            self.t_end = time.time()
        except Exception as e:  # noqa: BLE001
            if not self._stop.is_set():
                self.error = repr(e)
        finally:
            if self.t_end is None and not self._stop.is_set():
                self.t_end = time.time()

    def stop(self):
        self._stop.set()
        try:
            if self._conn is not None and self._conn.sock is not None:
                self._conn.sock.shutdown(2)
        except OSError:
            pass


def proc_cpu_ticks(pid: int) -> int:
    """utime + stime of a process and all its threads (clock ticks)."""
    f = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()
    return int(f[11]) + int(f[12])


class Stream(threading.Thread):
    """One streaming completion. Records first-token time and chunk times; stop() closes the
    socket, which makes vLLM abort the request."""

    def __init__(self, srv: Server, ids, max_tokens, temperature, seed, label):
        super().__init__(daemon=True)
        self.srv, self.ids, self.label = srv, ids, label
        self.body = completion_body(ids, max_tokens, temperature, seed)
        self.t_send = self.t_first = self.t_end = None
        self.chunks = 0
        self.usage = None
        self.error = None
        self._stop = threading.Event()
        self._conn = None

    def run(self):
        try:
            self._conn = c = self.srv.conn(timeout=7200)
            self.t_send = time.time()
            c.request("POST", "/v1/completions", body=json.dumps(self.body), headers=self.srv.hdr)
            r = c.getresponse()
            if r.status != 200:
                self.error = f"HTTP {r.status} {r.read()[:300]!r}"
                return
            while not self._stop.is_set():
                line = r.readline()
                if not line:
                    break
                if not line.startswith(b"data: "):
                    continue
                data = line[6:].strip()
                if data == b"[DONE]":
                    break
                obj = json.loads(data)
                if obj.get("usage"):
                    self.usage = obj["usage"]
                if obj.get("choices") and obj["choices"][0].get("text"):
                    if self.t_first is None:
                        self.t_first = time.time()
                    self.chunks += 1
        except Exception as e:  # noqa: BLE001
            if not self._stop.is_set():
                self.error = repr(e)
        finally:
            self.t_end = time.time()

    def stop(self):
        self._stop.set()
        try:
            if self._conn is not None and self._conn.sock is not None:
                self._conn.sock.shutdown(2)
        except OSError:
            pass


# ---------------------------------------------------------------- metrics + GPU sampling
METRIC_RE = re.compile(r'^(vllm:[a-z_]+)(\{[^}]*\})?\s+([-+0-9.eE]+)$')
KEEP = ("vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:spec_decode_num_drafts_total",
        "vllm:spec_decode_num_draft_tokens_total", "vllm:spec_decode_num_accepted_tokens_total",
        "vllm:spec_decode_num_accepted_tokens_per_pos_total", "vllm:generation_tokens_total",
        "vllm:prompt_tokens_total", "vllm:kv_cache_usage_perc", "vllm:num_preemptions_total",
        "vllm:iteration_tokens_total_count", "vllm:iteration_tokens_total_sum",
        "vllm:prompt_tokens_by_source_total", "vllm:kv_offload_total_bytes_total",
        "vllm:kv_offload_tiering_writeback_flushed_blocks_total",
        "vllm:kv_offload_tiering_writeback_dirty_blocks", "vllm:kv_offload_fs_cache_blocks")


def parse_metrics(text: str) -> dict:
    out = {}
    for line in text.splitlines():
        m = METRIC_RE.match(line)
        if not m or m.group(1) not in KEEP:
            continue
        name, labels, val = m.groups()
        key = name
        if labels:
            extra = [kv for kv in re.findall(r'(\w+)="([^"]*)"', labels)
                     if kv[0] not in ("engine", "model_name")]
            if extra:
                key += "{" + ",".join(f"{k}={v}" for k, v in extra) + "}"
        out[key] = float(val)
    return out


class Sampler:
    """/metrics at 1 Hz (aligned to the wall-clock second) + nvidia-smi -lms 1000."""

    def __init__(self, srv: Server, out_dir: Path, tag: str):
        self.srv, self.samples = srv, []
        self.gpu_file = out_dir / f"{tag}-nvsmi.csv"
        self.metrics_file = out_dir / f"{tag}-metrics.jsonl"
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._smi = None

    def start(self):
        self._smi = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=timestamp,utilization.gpu,power.draw,clocks.sm,clocks.mem,temperature.gpu",
             "--format=csv,noheader,nounits", "-lms", "1000"],
            stdout=open(self.gpu_file, "w"), stderr=subprocess.DEVNULL)
        self._t.start()

    def _loop(self):
        with open(self.metrics_file, "w") as f:
            while not self._stop.is_set():
                time.sleep(1.0 - (time.time() % 1.0))
                t = time.time()
                try:
                    m = parse_metrics(self.srv.get("/metrics", timeout=5))
                except Exception as e:  # noqa: BLE001
                    m = {"error": repr(e)}
                m["t"] = t
                self.samples.append(m)
                f.write(json.dumps(m) + "\n")
                f.flush()

    def stop(self):
        self._stop.set()
        self._t.join(timeout=5)
        if self._smi:
            self._smi.terminate()
            self._smi.wait(timeout=5)

    def gpu_rows(self):
        rows = []
        for line in self.gpu_file.read_text().splitlines():
            p = [x.strip() for x in line.split(",")]
            if len(p) < 6:
                continue
            try:
                ts = time.mktime(time.strptime(p[0].split(".")[0], "%Y/%m/%d %H:%M:%S"))
                rows.append({"t": ts, "util": float(p[1]), "power": float(p[2]), "sm": float(p[3]),
                             "mem": float(p[4]), "temp": float(p[5])})
            except ValueError:
                continue
        return rows


def window_stats(samples, gpu, t0, t1, n_expected, k_expected=None, n_decoders=None):
    """Production-profile accounting over [t0, t1]. n_decoders: streams that draft (default
    = n_expected); the running gauge must equal n_expected at every sample."""
    w = [s for s in samples if t0 <= s["t"] <= t1 and "error" not in s]
    if len(w) < 5:
        die(f"only {len(w)} metric samples in the window")
    a, b = w[0], w[-1]
    T = b["t"] - a["t"]
    d = {k: b.get(k, 0.0) - a.get(k, 0.0) for k in b if k != "t"}
    nd = n_decoders or n_expected
    drafts = d.get("vllm:spec_decode_num_drafts_total", 0.0)
    if drafts <= 0:
        die("no drafts in the window (MTP off or nothing decoding)")
    running = [s.get("vllm:num_requests_running", -1) for s in w]
    per_sec = []
    for x, y in zip(w, w[1:]):
        dd = y.get("vllm:spec_decode_num_drafts_total", 0) - x.get("vllm:spec_decode_num_drafts_total", 0)
        dt = y["t"] - x["t"]
        if dd > 0:
            per_sec.append(1000.0 * nd * dt / dd)
    # production-profile "clean" seconds: running == n at both ends, no prompt tokens counted,
    # drafts advanced (turn boundaries / prefills excluded)
    clean = []
    for x, y in zip(w, w[1:]):
        dd = y.get("vllm:spec_decode_num_drafts_total", 0) - x.get("vllm:spec_decode_num_drafts_total", 0)
        dp = y.get("vllm:prompt_tokens_total", 0) - x.get("vllm:prompt_tokens_total", 0)
        if dd > 0 and dp == 0 and x.get("vllm:num_requests_running") == y.get("vllm:num_requests_running") == n_expected:
            clean.append((x["t"], y["t"], dd))
    iters = d.get("vllm:iteration_tokens_total_count", 0.0)
    g = [r for r in gpu if t0 <= r["t"] <= t1]
    util = statistics.mean(r["util"] for r in g) / 100.0 if g else float("nan")
    ms = 1000.0 * nd * T / drafts
    k = d.get("vllm:spec_decode_num_draft_tokens_total", 0.0) / drafts
    acc = d.get("vllm:spec_decode_num_accepted_tokens_total", 0.0)
    pos = {key.split("=")[1].rstrip("}"): v / drafts for key, v in d.items()
           if key.startswith("vllm:spec_decode_num_accepted_tokens_per_pos_total")}
    st = {
        "t0": a["t"], "t1": b["t"], "seconds": T, "n_running_expected": n_expected, "n_decoders": nd,
        "running_min": min(running), "running_max": max(running),
        "waiting_max": max(s.get("vllm:num_requests_waiting", 0) for s in w),
        "preemptions": d.get("vllm:num_preemptions_total", 0.0),
        "prompt_tokens_in_window": d.get("vllm:prompt_tokens_total", 0.0),
        "drafts": drafts, "k_mean": k, "accepted_per_draft": acc / drafts,
        "accepted_per_pos": pos,
        "ms_per_step_pooled": ms, "steps_per_s": 1000.0 / ms,
        "ms_per_step_median": statistics.median(per_sec) if per_sec else None,
        "ms_per_step_p90": (sorted(per_sec)[int(0.9 * (len(per_sec) - 1))] if per_sec else None),
        "engine_steps_per_s": iters / T if T else None,
        "ms_per_engine_step": (1000.0 * T / iters) if iters else None,
        "gen_tok_s": d.get("vllm:generation_tokens_total", 0.0) / T,
        "tok_per_step_per_seq": 1.0 + acc / drafts,
        "gpu_util": util, "gpu_busy_ms": util * ms, "gpu_idle_ms": (1.0 - util) * ms,
        "gpu_power_med": statistics.median(r["power"] for r in g) if g else None,
        "gpu_sm_mhz_med": statistics.median(r["sm"] for r in g) if g else None,
        "gpu_temp_max": max(r["temp"] for r in g) if g else None,
        "kv_usage_mean": statistics.mean(s.get("vllm:kv_cache_usage_perc", 0) for s in w),
        "offload_bytes_in_window": {k2: v for k2, v in d.items() if "kv_offload" in k2},
    }
    if clean:
        cs = sum(b_ - a_ for a_, b_, _ in clean)
        cms = 1000.0 * nd * cs / sum(dd for _, _, dd in clean)
        cg = [r for r in g if any(a_ <= r["t"] <= b_ for a_, b_, _ in clean)]
        cu = statistics.mean(r["util"] for r in cg) / 100.0 if cg else float("nan")
        st.update({"clean_seconds": cs, "clean_ms_per_step": cms, "clean_gpu_util": cu,
                   "clean_gpu_idle_ms": (1.0 - cu) * cms})
    problems = []
    if st["running_min"] != n_expected or st["running_max"] != n_expected:
        problems.append(f"running gauge {st['running_min']}-{st['running_max']} != {n_expected}")
    if st["preemptions"] > 0:
        problems.append(f"{st['preemptions']:.0f} preemptions")
    if k_expected is not None and abs(k - k_expected) > 0.05:
        problems.append(f"k {k:.2f} != {k_expected}")
    st["problems"] = problems
    return st


def fmt_stats(tag: str, st: dict) -> str:
    def f(x, p=1):
        return "-" if x is None else f"{x:.{p}f}"
    return (f"{tag}: n={st['n_running_expected']} {st['seconds']:.0f}s  ms/step pooled {f(st['ms_per_step_pooled'])} "
            f"(median {f(st['ms_per_step_median'])}, p90 {f(st['ms_per_step_p90'])}; engine-iter {f(st['ms_per_engine_step'])})  "
            f"steps/s {f(st['steps_per_s'], 2)}  util {f(100 * st['gpu_util'], 0)}%  busy {f(st['gpu_busy_ms'])} idle {f(st['gpu_idle_ms'])} ms  "
            f"k {f(st['k_mean'], 2)} acc/draft {f(st['accepted_per_draft'], 2)} tok/step/seq {f(st['tok_per_step_per_seq'], 2)}  "
            f"gen {f(st['gen_tok_s'])} tok/s  SM {f(st['gpu_sm_mhz_med'], 0)} MHz {f(st['gpu_power_med'], 0)} W  "
            f"kv {f(st['kv_usage_mean'], 2)}"
            + (f"  | clean {st['clean_seconds']:.0f}s: ms/step {st['clean_ms_per_step']:.1f} idle {st['clean_gpu_idle_ms']:.1f}"
               if st.get("clean_seconds") else "")
            + "".join(f"  | {k} CPU {v:.0f}%" for k, v in (st.get("proc_cpu_pct") or {}).items())
            + f"  problems: {'; '.join(st['problems']) or 'none'}")


# ---------------------------------------------------------------- subcommands
def cmd_warm(a):
    srv = Server(a.url)
    srv.get("/v1/models")
    for i in range(2):
        out = json.loads(srv.post("/v1/completions", completion_body(
            prompt_ids(f"warm{i}", "chat", 1024), 64, 0, i, stream=False)))
        n = out["usage"]["completion_tokens"]
        if n != 64:
            die(f"warm-up request {i} returned {n} tokens")
    print("warm: ok")


def run_fill(srv, n, tokens, conc, name="fill"):
    todo = list(range(n))
    errs = []

    def worker():
        while todo:
            try:
                i = todo.pop(0)
            except IndexError:
                return
            try:
                srv.post("/v1/completions", completion_body(prompt_ids(f"{name}{i}", "chat", tokens), 1, 0, i,
                                                            stream=False), timeout=3600)
            except Exception as e:  # noqa: BLE001
                errs.append(repr(e))
    th = [threading.Thread(target=worker) for _ in range(conc)]
    t = time.time()
    [x.start() for x in th]
    [x.join() for x in th]
    if errs:
        die(f"fill errors: {errs[:2]}")
    return time.time() - t


def cmd_fill(a):
    dt = run_fill(Server(a.url), a.n, a.tokens, a.conc)
    print(f"fill: {a.n} x {a.tokens} tokens prefilled in {dt:.0f} s")


def start_streams(srv, specs, max_tokens, temperature, prefix0=None):
    """specs: list of (name, kind, tokens). All streams start together. prefix0: ids put in
    front of stream 0's prompt."""
    ids = [prompt_ids(*s) for s in specs]
    if prefix0:
        ids[0] = prefix0 + ids[0]
    st = [Stream(srv, x, max_tokens, temperature, 1000 + i, specs[i][0]) for i, x in enumerate(ids)]
    for s in st:
        s.start()
    return st


def wait_first_tokens(streams, timeout):
    t_end = time.time() + timeout
    while time.time() < t_end:
        bad = [s for s in streams if s.error]
        if bad:
            die(f"stream {bad[0].label}: {bad[0].error}")
        if all(s.t_first for s in streams):
            return max(s.t_first for s in streams)
        if any(s.t_end and not s.t_first for s in streams):
            die("a stream ended before its first token")
        time.sleep(0.5)
    die(f"no first token on every stream within {timeout} s")


def cmd_steady(a):
    srv, out = Server(a.url), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    specs = [(f"{a.pname or a.tag}-s{i}", a.kind, a.tokens) for i in range(a.conc)]
    if a.reuse:
        nm, nt = a.reuse.split(":")
        specs[0] = (nm, a.kind, int(nt))
    prefix = None
    if a.prefix:  # stream 0 = a named prompt (prefix-cache hit) + a fresh salted tail
        nm, kd, nt = a.prefix.split(":")
        prefix = prompt_ids(nm, kd, int(nt))
        specs[0] = (f"{a.tag}-tail", "code", a.tokens - len(prefix))
    samp = Sampler(srv, out, a.tag)
    samp.start()
    t_send = time.time()
    if a.chat:
        streams = [ChatStream(srv, prompt_text(f"{a.pname or a.tag}-txt{i}", a.tokens), a.max_tokens, a.temperature,
                              1000 + i, f"chat{i}", a.tools, a.turns, not a.natural) for i in range(a.conc)]
        for s_ in streams:
            s_.start()
    else:
        streams = start_streams(srv, specs, a.max_tokens, a.temperature, prefix)
    t_dec = wait_first_tokens(streams, a.prefill_timeout)
    print(f"steady {a.tag}: all {a.conc} streams decoding {t_dec - t_send:.0f} s after send", flush=True)
    t0 = t_dec + a.settle
    t1 = t0 + a.window
    pids = {k: int(v) for k, v in (x.split(":") for x in os.environ.get("R3_PIDS", "").split(",") if x)}
    c0 = {}
    while time.time() < t0:
        time.sleep(0.2)
    for k, p in pids.items():
        c0[k] = (proc_cpu_ticks(p), time.time())
    while time.time() < t1 + 1.5:
        if any(s.error for s in streams):
            die(f"stream error during window: {[s.error for s in streams if s.error][0]}")
        if any(s.t_end for s in streams):
            die("a stream finished inside the window (raise --max-tokens / --turns)")
        time.sleep(0.5)
    cpu = {k: 100.0 * (proc_cpu_ticks(p) - c0[k][0]) / os.sysconf("SC_CLK_TCK") / (time.time() - c0[k][1])
           for k, p in pids.items()}
    hook = None
    if a.hook:
        th0 = time.time()
        p = subprocess.run(["bash", "-c", a.hook], env={**os.environ, "R3_WINDOW_T0": str(t0), "R3_WINDOW_T1": str(t1)})
        hook = {"rc": p.returncode, "seconds": time.time() - th0}
        if any(s.t_end for s in streams):
            hook["warning"] = "a stream ended during the hook"
    for s in streams:
        s.stop()
    samp.stop()
    st = window_stats(samp.samples, samp.gpu_rows(), t0, t1, a.conc, a.k)
    st["proc_cpu_pct"] = cpu
    if a.chat:
        st["turn_seconds"] = [s.turn_times for s in streams]
    st.update({"tag": a.tag, "prompt_tokens_each": [len(s.ids) for s in streams], "prefill_seconds": t_dec - t_send,
               "window_t0": t0, "window_t1": t1, "hook": hook, "temperature": a.temperature,
               "max_tokens": a.max_tokens, "stream_chunks": [s.chunks for s in streams]})
    (out / f"{a.tag}-steady.json").write_text(json.dumps(st, indent=1))
    line = fmt_stats(a.tag, st)
    print(line, flush=True)
    with open(out / "lines.txt", "a") as f:
        f.write(line + "\n")
    if hook and hook["rc"] != 0:
        die(f"hook failed rc={hook['rc']}")
    if st["problems"] and not a.allow_problems:
        die(f"window invalid: {st['problems']}")


def cmd_mixed(a):
    """D long-running decoders + one lane of back-to-back max_tokens=1 requests (prefill load)."""
    srv, out = Server(a.url), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lane = a.lane.split(",")
    if "turn" in lane:  # put the shared long-turn prefix in the prefix cache first
        srv.post("/v1/completions", completion_body(prompt_ids("turnprefix", "chat", a.turn_prefix), 1, 0, 0,
                                                    stream=False), timeout=3600)
    samp = Sampler(srv, out, a.tag)
    samp.start()
    specs = [(f"{a.tag}-d{i}", "chat", a.dec_tokens) for i in range(a.decoders)]
    streams = start_streams(srv, specs, a.max_tokens, a.temperature)
    t_dec = wait_first_tokens(streams, a.prefill_timeout)
    t0 = t_dec + a.settle
    t1 = t0 + a.window
    while time.time() < t0:
        time.sleep(0.2)
    rec, i = [], 0
    base = prompt_ids("turnprefix", "chat", a.turn_prefix) if "turn" in lane else None
    while time.time() < t1:
        kind = lane[i % len(lane)]
        if kind == "turn":
            new = prompt_ids(f"{a.tag}-turn{i}", "prose", a.turn_new)
            ids, fresh = base + new, len(new)
        else:
            ids = prompt_ids(f"{a.tag}-lane{i}", "chat", int(kind))
            fresh = len(ids)
        ts = time.time()
        srv.post("/v1/completions", completion_body(ids, 1, 0, 5000 + i, stream=False), timeout=1800)
        rec.append({"kind": kind, "fresh_tokens": fresh, "ctx": len(ids), "t": ts, "ttft": time.time() - ts})
        i += 1
        if any(s.t_end for s in streams):
            die("a decoder finished inside the window (raise --max-tokens)")
    t1 = time.time()
    for s in streams:
        s.stop()
    samp.stop()
    st = window_stats(samp.samples, samp.gpu_rows(), t0, t1, a.decoders, None, a.decoders)
    # the lane request is counted as running while it prefills: the gauge check does not apply
    st["problems"] = [p for p in st["problems"] if not p.startswith("running gauge")]
    by = {}
    for r in rec:
        by.setdefault(r["kind"], []).append(r["ttft"])
    st.update({"tag": a.tag, "lane": rec, "ttft": {k: {"n": len(v), "median": statistics.median(v), "max": max(v)}
                                                   for k, v in by.items()},
               "lane_fresh_tok_s": sum(r["fresh_tokens"] for r in rec) / (t1 - t0)})
    (out / f"{a.tag}-mixed.json").write_text(json.dumps(st, indent=1))
    tt = "  ".join(f"ttft[{k}] {v['median']:.2f}s (n={v['n']})" for k, v in st["ttft"].items())
    line = fmt_stats(a.tag, st) + f"  | lane fresh prefill {st['lane_fresh_tok_s']:.0f} tok/s  {tt}"
    print(line, flush=True)
    with open(out / "lines.txt", "a") as f:
        f.write(line + "\n")
    if st["problems"]:
        die(f"window invalid: {st['problems']}")


def cmd_probe(a):
    """Greedy outputs for a fixed prompt set, sent together (one batch, k from the schedule):
    the correctness check for a change that must not alter numerics."""
    srv, out = Server(a.url), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = [None] * a.n

    def one(i):
        body = completion_body(prompt_ids(f"probe{i}", "chat", a.tokens), a.max_tokens, 0, 77 + i, stream=False)
        body["logprobs"] = 1
        r = json.loads(srv.post("/v1/completions", body, timeout=1800))
        ch = r["choices"][0]
        res[i] = {"text": ch["text"], "tokens": (ch.get("logprobs") or {}).get("tokens"),
                  "completion_tokens": r["usage"]["completion_tokens"]}
    th = [threading.Thread(target=one, args=(i,)) for i in range(a.n)]
    [t.start() for t in th]
    [t.join() for t in th]
    if any(r is None for r in res):
        die("probe: a request failed")
    (out / f"{a.tag}-probe.json").write_text(json.dumps(res, indent=1))
    print(f"probe {a.tag}: {a.n} x {a.max_tokens} greedy tokens saved")


def cmd_profile(a):
    """POST /start_profile, wait, POST /stop_profile (the server's --profiler-config bounds it)."""
    srv = Server(a.url)
    srv.post("/start_profile")
    time.sleep(a.seconds)
    srv.post("/stop_profile", timeout=1800)
    print("profile: done")


def cmd_plan(a):
    print(__doc__)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("warm", "fill", "steady", "mixed", "profile", "probe", "plan"):
        p = sub.add_parser(name)
        p.add_argument("--url", default=os.environ.get("GSQ_URL", "http://127.0.0.1:18090"))
        p.add_argument("--out", default=".")
        p.add_argument("--tag", default="run")
        p.add_argument("--temperature", default="0")
        p.add_argument("--max-tokens", type=int, default=12000)
        p.add_argument("--window", type=float, default=90)
        p.add_argument("--settle", type=float, default=5)
        p.add_argument("--prefill-timeout", type=float, default=1500)
    sp = sub.choices
    sp["fill"].add_argument("--n", type=int, default=2)
    sp["fill"].add_argument("--tokens", type=int, default=90000)
    sp["fill"].add_argument("--conc", type=int, default=2)
    sp["steady"].add_argument("--conc", type=int, required=True)
    sp["steady"].add_argument("--tokens", type=int, required=True)
    sp["steady"].add_argument("--kind", default="chat")
    sp["steady"].add_argument("--k", type=float, default=None, help="expected draft length (checked)")
    sp["steady"].add_argument("--hook", default="")
    sp["steady"].add_argument("--reuse", default="")
    sp["steady"].add_argument("--pname", default="", help="prompt name prefix (default: --tag); same name = same prompts")
    sp["steady"].add_argument("--prefix", default="", help="NAME:KIND:TOKENS put in front of stream 0")
    sp["steady"].add_argument("--allow-problems", action="store_true")
    sp["steady"].add_argument("--chat", action="store_true", help="streaming chat completions (parsers on)")
    sp["steady"].add_argument("--tools", action="store_true", help="with --chat: tools in the request")
    sp["steady"].add_argument("--turns", type=int, default=1, help="with --chat: turns per stream")
    sp["steady"].add_argument("--natural", action="store_true", help="with --chat: no ignore_eos")
    sp["mixed"].add_argument("--decoders", type=int, default=3)
    sp["mixed"].add_argument("--dec-tokens", type=int, default=40000)
    sp["mixed"].add_argument("--lane", default="600,1400,turn,4000")
    sp["mixed"].add_argument("--turn-prefix", type=int, default=60000)
    sp["mixed"].add_argument("--turn-new", type=int, default=1800)
    sp["profile"].add_argument("--seconds", type=float, default=5)
    sp["probe"].add_argument("--n", type=int, default=4)
    sp["probe"].add_argument("--tokens", type=int, default=2000)
    a = ap.parse_args()
    {"warm": cmd_warm, "fill": cmd_fill, "steady": cmd_steady, "mixed": cmd_mixed,
     "profile": cmd_profile, "probe": cmd_probe, "plan": cmd_plan}[a.cmd](a)


if __name__ == "__main__":
    main()
