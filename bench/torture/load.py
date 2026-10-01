"""Schedule, request mix, sending and per-response classification for the torture harness.

Standard library only (Python >= 3.12): it runs against any OpenAI-compatible vLLM endpoint,
production included, from any Python. Long prompts are token ids built through the server's own
/tokenize (so their exact prompt-token counts are known): Python's standard-library sources,
shuffled by the seed, behind a per-request 64-bit salt.

The schedule cycles through every phase type in a seeded order, 10-20 min each, with an idle gap
of 2-5 min after every third. Shorter runs scale the durations down (floor 20 s, idle 5 s), so a
20-minute smoke still visits every type.
  ramp        concurrency 1 -> 16 -> 1 across the phase, mixed requests
  burst       waves of 16 requests sent at the same instant
  long_ctx    one lane of 32k/96k/196k raw prompts (30% prefix hits) beside two chat lanes
  prefix      six lanes of 1k/8k/32k prompts, 70% prefix hits (recent ones and old ones, which
              come back from the CPU/fs KV tiers)
  cancel      eight streaming lanes: 30% cancelled mid-stream, 10% client timeouts
  tools       tool calls (qwen3_coder parser): tool_choice absent/auto/required/named/none
  structured  response_format json_schema and json_object
  plogprobs   prompt_logprobs=1 on 2k-4k prompts: alone (2048-row lm_head chunks), then beside chats
  repeat      eight lanes resending identical requests (exact prefix-cache hits)
  api         n=2, stop strings, seeds, logprobs, min_tokens + penalties, echo
  full        sixteen lanes of everything up to 32k prompts (KV pressure, preemption)
  idle        no traffic
Everywhere: 5% of streams cancelled at a random point, 1% client timeouts, max_tokens 1-4096,
T=0 and sampled, thinking on/off (chat_template_kwargs.enable_thinking), priority P..P+20 (P =
--priority, default 100000: behind any real traffic under --scheduling-policy priority). A greedy
probe opens every phase (latency drift, idle recovery, repeatability); /tokenize + /detokenize
(round trip) and /metrics are polled every 15 s.
"""

import copy
import hashlib
import http.client
import json
import math
import platform
import random
import re
import socket
import sysconfig
import threading
import time
import urllib.parse
from pathlib import Path

TYPES = ["ramp", "burst", "long_ctx", "prefix", "cancel", "tools", "structured", "plogprobs", "repeat", "api", "full"]
DEFAULTS = dict(lengths=[1000, 8000], hit=0.2, cancel=0.05, timeout=0.01, stream=0.7)
# lanes: (request mix, concurrency) with concurrency an int, "ramp", "burst" or "half" (0, then 2)
PHASES = {
    "ramp": dict(lanes=[("mixed", "ramp")]),
    "burst": dict(lanes=[("mixed", "burst")]),
    "long_ctx": dict(lanes=[("long", 1), ("chat", 2)], lengths=[32000, 96000, 196000], hit=0.3),
    "prefix": dict(lanes=[("long", 6)], lengths=[1000, 8000, 32000], hit=0.7),
    "cancel": dict(lanes=[("mixed", 8)], cancel=0.3, timeout=0.1, stream=1.0),
    "tools": dict(lanes=[("tool", 4)]),
    "structured": dict(lanes=[("json", 4)]),
    "plogprobs": dict(lanes=[("plog", 1), ("chat", "half")]),
    "repeat": dict(lanes=[("repeat", 8)]),
    "api": dict(lanes=[("api", 4)]),
    "full": dict(lanes=[("mixed", 16)], lengths=[1000, 8000, 32000]),
    "idle": dict(lanes=[]),
}
MIXED = {"chat": 40, "tool": 12, "json": 8, "long": 15, "api": 8, "repeat": 7, "plog": 2}
RAMP = [1, 2, 4, 8, 12, 16, 16, 12, 8, 4, 2, 1]
BURST = 16
MAX_TOKENS = ([1, 2, 8, 32, 128, 256, 512, 1024, 2048, 4096], [3, 2, 5, 10, 15, 15, 15, 15, 10, 10])
SHORT = [
    "Write a Python function that merges overlapping intervals, with tests.",
    "Explain the difference between a mutex and a semaphore with a C example.",
    "Summarize how TCP congestion control works, in five bullet points.",
    "Refactor this into idiomatic Rust: for i in 0..v.len() { if v[i] > 3 { out.push(v[i] * 2) } }",
    "What does `git rebase --onto A B C` do? Show a before/after graph.",
    "Draft a SQL schema for a library lending system with constraints.",
]
TOOLS = [{"type": "function", "function": {
    "name": "read_file", "description": "Read a file from the repository",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "start_line": {"type": "integer"}},
                   "required": ["path"]}}},
         {"type": "function", "function": {
             "name": "run_shell", "description": "Run a shell command in the repository and return its output",
             "parameters": {"type": "object", "properties": {"command": {"type": "string"}, "timeout_s": {"type": "integer"}},
                            "required": ["command"]}}}]
TOOL_PROMPTS = ["Open src/main.py and tell me what the entry point does.",
                "Run the test suite with pytest -x and tell me which test fails first.",
                "Read README.md, then list the build steps it describes."]
CITY = {"type": "object", "properties": {"city": {"type": "string"}, "population": {"type": "integer"}},
        "required": ["city", "population"]}
PROBE = {"messages": [{"role": "user", "content": SHORT[1]}], "max_tokens": 256, "temperature": 0.0,
         "chat_template_kwargs": {"enable_thinking": False}}
LONG_TIMEOUT = 3600.0  # a request that takes longer is "stalled" (196k prefill under load is ~10 min)

# Response classes. OK ones are expected outcomes; anything else fails the run.
OK_CLASSES = {"ok", "reasoning_only", "eos_first", "short_empty", "truncated_json", "cancelled", "client_timeout"}


def make_plan(total_s: float, seed: int) -> list[dict]:
    """Phases back to back filling total_s: shuffled cycles of TYPES, idle after every third."""
    rng = random.Random(f"plan:{seed}")
    scale = min(1.0, total_s / (12 * 3600))
    plan, t = [], 0.0
    while t < total_s:
        order = TYPES[:]
        rng.shuffle(order)
        for n, ty in enumerate(order):
            items = [(ty, max(rng.uniform(600, 1200) * scale, 20))]
            if n % 3 == 2:
                items.append(("idle", max(rng.uniform(120, 300) * scale, 5)))
            for ty, d in items:
                d = min(d, total_s - t)
                if d > 0:
                    plan.append({"i": len(plan), "type": ty, "start": round(t, 1), "dur": round(d, 1)})
                    t += d
    return plan


# ---------------------------------------------------------------- classification

def classify(req: dict, res: dict) -> str:
    """One class per response. req: what was asked (see Gen); res: what came back (see send)."""
    if res.get("cancelled"):
        return "cancelled"
    if res.get("timed_out"):
        return "client_timeout" if req.get("planned_timeout") else "stalled"
    if res.get("error"):
        return "conn_error"
    if res.get("http") != 200:
        return f"http_{res.get('http', 0) // 100}xx"
    if res.get("stream_error"):
        return "stream_error"
    ch, usage = res.get("choices") or [], res.get("usage") or {}
    n = req["body"].get("n", 1)
    if len(ch) != n or "completion_tokens" not in usage:
        return "bad_response"
    pt, ct, mt = usage.get("prompt_tokens"), usage["completion_tokens"], req["body"]["max_tokens"]
    if req.get("expect_prompt_tokens") is not None and pt != req["expect_prompt_tokens"]:
        return "bad_count"
    if ct > n * mt or (n == 1 and ch[0]["finish"] == "length" and ct != mt):
        return "bad_count"
    if req["body"].get("min_tokens") and ct < min(req["body"]["min_tokens"], mt):
        return "bad_count"
    if any(c["finish"] not in ("stop", "length", "tool_calls") for c in ch):
        return "bad_finish"
    if res.get("lp_bad"):
        return "bad_logprobs"
    if req["body"].get("prompt_logprobs") and res.get("plp_len") != pt:
        return "bad_logprobs"
    for c in ch:
        for _, args in c["tools"]:
            try:
                if not isinstance(json.loads(args), dict):
                    return "bad_tool_json"
            except (TypeError, ValueError):
                return "bad_tool_json"
    choice = req["body"].get("tool_choice")
    if choice is not None and choice not in ("auto", "none") and not ch[0]["tools"] and ch[0]["finish"] != "length":
        return "no_tool_call"
    if choice == "none" and ch[0]["tools"]:
        return "bad_tool_choice"
    rf = req["body"].get("response_format")
    if rf:
        if ch[0]["finish"] == "length":
            return "truncated_json"
        try:
            d = json.loads(ch[0]["text"])
        except ValueError:
            return "bad_json"
        if not isinstance(d, dict) or (rf["type"] == "json_schema" and not (
                isinstance(d.get("city"), str) and isinstance(d.get("population"), int))):
            return "bad_json"
    for c in ch:
        if c["text"].strip() or c["tools"]:
            continue
        if c["reasoning"].strip():
            return "reasoning_only"
        if c["finish"] == "stop" and ct <= n:
            return "eos_first"  # first token EOS: the model's own distribution at T>0 (soak finding)
        # a few whitespace/think-tag tokens are plausible; a long output with no text is not
        return "short_empty" if ct // n < 8 else "empty_output"
    return "ok"


# ---------------------------------------------------------------- the server

class Api:
    """One OpenAI-compatible vLLM server: base URL (…/v1) for the API, its root for /tokenize,
    /detokenize, /metrics and /health. Every request goes to this host:port and nowhere else."""

    def __init__(self, base_url: str, key: str = "", model: str | None = None):
        u = urllib.parse.urlsplit(base_url.rstrip("/"))
        if u.scheme != "http" or not u.port:
            raise SystemExit(f"--base-url must be http://HOST:PORT/v1, got {base_url}")
        self.host, self.port, self.path = u.hostname, u.port, u.path
        self.root = self.path[: -len("/v1")] if self.path.endswith("/v1") else self.path
        self.key = key
        self.model, self.mml = model, 200000
        if model is None:
            st, body = self.call("GET", self.path + "/models")
            if st != 200:
                raise SystemExit(f"GET {base_url}/models: HTTP {st}: {body[:200]}")
            m = json.loads(body)["data"][0]
            self.model, self.mml = m["id"], m.get("max_model_len") or self.mml

    def conn(self, timeout: float) -> http.client.HTTPConnection:
        return http.client.HTTPConnection(self.host, self.port, timeout=timeout)

    def headers(self) -> dict:
        return {"Content-Type": "application/json", **({"Authorization": f"Bearer {self.key}"} if self.key else {})}

    def call(self, method: str, path: str, body: dict | None = None, timeout: float = 60) -> tuple[int, str]:
        c = self.conn(timeout)
        try:
            c.request(method, path, json.dumps(body) if body is not None else None, self.headers())
            r = c.getresponse()
            return r.status, r.read().decode(errors="replace")
        except Exception as e:  # noqa: BLE001
            return 0, repr(e)
        finally:
            c.close()

    def kv_tokens(self) -> int | None:
        """KV cache capacity in tokens (num_gpu_blocks * block_size of vllm:cache_config_info), if exported."""
        m = re.search(r'^vllm:cache_config_info\{[^}]*\}', self.call("GET", self.root + "/metrics")[1], re.M)
        lab = dict(re.findall(r'(\w+)="([^"]*)"', m.group(0))) if m else {}
        try:
            return int(lab["num_gpu_blocks"]) * int(lab["block_size"])
        except (KeyError, ValueError):
            return None

    def tokenize(self, text: str) -> list[int]:
        st, body = self.call("POST", self.root + "/tokenize", {"model": self.model, "prompt": text, "add_special_tokens": False})
        if st != 200:
            raise RuntimeError(f"/tokenize: HTTP {st}: {body[:200]}")
        return json.loads(body)["tokens"]


class Corpus:
    """Token ids of shuffled stdlib sources, tokenized by the server in pieces of max_model_len/2
    characters (a token is at least one character, so no call exceeds max_model_len), and the pieces
    of salts and follow-up tails."""

    def __init__(self, api: Api, seed: int, n_tokens: int):
        stdlib = Path(sysconfig.get_path("stdlib"))
        files = sorted(p for p in stdlib.rglob("*.py") if "site-packages" not in p.parts)
        random.Random(seed).shuffle(files)
        self.body: list[int] = []
        for f in files:
            if len(self.body) >= n_tokens:
                break
            text, step = f"\n# ---- {f.name} ----\n" + f.read_text(errors="replace"), api.mml // 2
            for i in range(0, len(text), step):
                self.body += api.tokenize(text[i:i + step])
        self.body = self.body[:n_tokens]
        self.sha = hashlib.sha256(json.dumps(self.body).encode()).hexdigest()[:16]
        self.hex = {c: api.tokenize(c) for c in "0123456789abcdef"}
        self.salt_pre, self.salt_post = api.tokenize("[torture-salt "), api.tokenize("]\n")
        self.tail_pre, self.tail_post = api.tokenize("\n# follow-up "), api.tokenize(": what does the code above do?\n")

    def digits(self, h: str) -> list[int]:
        return [i for c in h for i in self.hex[c]]

    def prompt(self, n: int, salt: str) -> list[int]:
        s = self.salt_pre + self.digits(salt) + self.salt_post
        return s + self.body[: n - len(s)]

    def tail(self, h: str) -> list[int]:
        return self.tail_pre + self.digits(h) + self.tail_post


# ---------------------------------------------------------------- requests

class Gen:
    """Seeded request factory (one rng per phase)."""

    def __init__(self, corpus: Corpus, seed: int, model: str, max_model_len: int, priority: int):
        self.corpus, self.seed, self.model, self.mml, self.priority = corpus, seed, model, max_model_len, priority
        self.history: dict[int, list[str]] = {}  # length -> salts sent so far
        self.lock = threading.Lock()
        self.set_phase(0, PHASES["idle"])

    def set_phase(self, i: int, spec: dict) -> None:
        self.rng = random.Random(f"{self.seed}:{i}")
        self.spec = {**DEFAULTS, **spec}
        self.repeat_pool = None

    def long_prompt(self, n: int, hit: bool) -> tuple[list[int], dict]:
        r, past = self.rng, self.history.setdefault(n, [])
        n = min(n, self.mml - 64)
        if hit and past:
            salt = r.choice(past[-4:] if r.random() < 0.5 else past)  # recent (GPU) or old (CPU/fs tier)
            return self.corpus.prompt(n, salt) + self.corpus.tail(f"{r.getrandbits(32):08x}"), {"len": n, "hit": True, "salt": salt}
        salt = f"{r.getrandbits(64):016x}"
        past.append(salt)
        return self.corpus.prompt(n, salt), {"len": n, "hit": False, "salt": salt}

    def next(self, mix: str) -> dict:
        with self.lock:
            r = self.rng
            kind = mix if mix != "mixed" else r.choices(list(MIXED), list(MIXED.values()))[0]
            if kind == "repeat":
                if self.repeat_pool is None:
                    self.repeat_pool = [self._make(k) for k in ("chat", "long", "tool")]
                    for q in self.repeat_pool:
                        q["body"].update(temperature=0.0, priority=self.priority)
                q = copy.deepcopy(r.choice(self.repeat_pool))
                q.update(kind="repeat", cancel=None, timeout=LONG_TIMEOUT, planned_timeout=False)
                return q
            return self._make(kind)

    def _make(self, kind: str) -> dict:  # called under self.lock
        r, sp = self.rng, self.spec
        mt = r.choices(*MAX_TOKENS)[0]
        think = r.random() < 0.5
        base = {"model": self.model, "max_tokens": mt, "priority": self.priority + r.randint(0, 20)}
        if r.random() < 0.3:
            base["temperature"] = 0.0
        else:
            base.update({"temperature": 1.0, "top_p": 0.95, "top_k": 20} if think else
                        {"temperature": 0.7, "top_p": 0.8, "top_k": 20})
        chat = {**base, "chat_template_kwargs": {"enable_thinking": think}}
        q = {"kind": kind, "path": "/chat/completions", "stream": r.random() < sp["stream"], "meta": {"think": think}}
        if kind == "chat":
            q["body"] = {**chat, "messages": [{"role": "user", "content": r.choice(SHORT) + f" (variant {r.getrandbits(32):08x})"}]}
        elif kind == "tool":
            tc = r.choice([None, "auto", "required", "none", {"type": "function", "function": {"name": "run_shell"}}])
            q["body"] = {**chat, "messages": [{"role": "user", "content": r.choice(TOOL_PROMPTS)}], "tools": TOOLS,
                         "max_tokens": r.choice([256, 512, 1024, 2048])}
            if tc is not None:
                q["body"]["tool_choice"] = tc
        elif kind == "json":
            rf = r.choice([{"type": "json_schema", "json_schema": {"name": "city", "schema": CITY}}, {"type": "json_object"}])
            q["body"] = {**chat, "response_format": rf, "max_tokens": r.choice([64, 256, 1024]), "messages": [
                {"role": "user", "content": f"Give Denmark's capital and its population as JSON with keys city and population. (variant {r.getrandbits(32):08x})"}]}
        elif kind in ("long", "plog"):
            n = r.choice(sp["lengths"]) if kind == "long" else r.randint(2048, 4096)
            ids, meta = self.long_prompt(n, kind == "long" and r.random() < sp["hit"])
            q.update(path="/completions", meta={"think": None, **meta}, expect_prompt_tokens=len(ids))
            q["body"] = {**base, "prompt": ids, "max_tokens": max(1, min(mt, self.mml - len(ids) - 1))}
            if kind == "plog":
                q["stream"] = False
                q["body"].update(prompt_logprobs=1, max_tokens=1, temperature=0.0)
        elif kind == "api":
            v = r.choice(["n2", "stop", "seed", "logprobs", "min_tokens", "echo"])
            q["meta"]["variant"] = v
            msgs = [{"role": "user", "content": r.choice(SHORT) + f" (variant {r.getrandbits(32):08x})"}]
            q["body"] = {**chat, "messages": msgs, "max_tokens": r.choice([16, 64, 256])}
            if v == "n2":
                q["stream"] = False
                q["body"].update(n=2, temperature=1.0)
            elif v == "stop":
                q["body"]["stop"] = ["\n\n", "."]
            elif v == "seed":
                q["body"].update(seed=1234, temperature=1.0)
            elif v == "logprobs":
                q["stream"] = False
                q["body"].update(logprobs=True, top_logprobs=5)
            elif v == "min_tokens":
                q["body"].update(min_tokens=32, max_tokens=64, presence_penalty=1.2, frequency_penalty=0.3, repetition_penalty=1.1)
            else:
                q.update(path="/completions", stream=False, meta={"think": None, "variant": v})
                q["body"] = {**base, "prompt": "The capital of Denmark is", "echo": True, "logprobs": 1, "max_tokens": 4}
        q["cancel"] = None
        if q["stream"] and r.random() < sp["cancel"]:
            q["cancel"] = ["chunks", r.randint(1, 64)] if r.random() < 0.5 else ["time", round(r.uniform(0.05, 5.0), 2)]
        q["planned_timeout"] = r.random() < sp["timeout"]
        q["timeout"] = round(r.uniform(0.5, 10.0), 2) if q["planned_timeout"] else LONG_TIMEOUT
        return q


def cost(req: dict) -> int:
    """Tokens a request can hold in the KV cache: prompt + n * max_tokens (chat prompts ~300)."""
    b = req["body"]
    p = len(b["prompt"]) if isinstance(b.get("prompt"), list) else 300
    return p + b.get("n", 1) * b["max_tokens"]


class Throttle:
    """Politeness: at most max_conc requests and budget tokens (cost) in flight; a request bigger
    than the budget runs alone. --brutal drops both."""

    def __init__(self, max_conc: int, budget: int, off: bool):
        self.max_conc, self.budget, self.off = max_conc, budget, off
        self.n = self.tokens = 0
        self.cv = threading.Condition()

    def acquire(self, c: int) -> None:
        if self.off:
            return
        with self.cv:
            self.cv.wait_for(lambda: self.n == 0 or (self.n < self.max_conc and self.tokens + c <= self.budget))
            self.n += 1
            self.tokens += c

    def release(self, c: int) -> None:
        if self.off:
            return
        with self.cv:
            self.n -= 1
            self.tokens -= c
            self.cv.notify_all()


# ---------------------------------------------------------------- sending

def _finite_lp(xs) -> bool:
    return all(isinstance(x, (int, float)) and math.isfinite(x) and x <= 1e-3 for x in xs)


def _choice(c: dict) -> dict:
    """Non-streamed choice (chat or completion) -> {finish, text, reasoning, tools, lp_ok}."""
    m = c.get("message") or {}
    lp = c.get("logprobs") or {}
    vals = [e.get("logprob") for e in lp.get("content") or []]
    vals += [t.get("logprob") for e in lp.get("content") or [] for t in e.get("top_logprobs") or []]
    vals += [x for x in lp.get("token_logprobs") or [] if x is not None]
    return {"finish": c.get("finish_reason"), "text": c.get("text") or m.get("content") or "",
            "reasoning": m.get("reasoning") or m.get("reasoning_content") or "",
            "tools": [(t["function"]["name"], t["function"]["arguments"]) for t in m.get("tool_calls") or []],
            "lp_ok": _finite_lp(vals)}


LIVE: dict[int, object] = {}  # in-flight requests: id -> cut(flag), so a stopped run drops them at once


def cut_all() -> None:
    for c in list(LIVE.values()):
        c("cancelled")


def send(api: Api, req: dict) -> dict:
    """POST req; returns res: http, choices, usage, ttft/tpot (streams), cancelled/timed_out/error."""
    conn = api.conn(req["timeout"] + 30)
    body = dict(req["body"])
    if req["stream"]:
        body.update(stream=True, stream_options={"include_usage": True})
    res: dict = {}
    t0 = time.monotonic()

    def cut(flag):  # client-side cancel/timeout: drop the connection under the reader
        if res.get("done"):
            return  # the response is already complete
        res[flag] = True
        try:
            conn.sock and conn.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass

    timers = [threading.Timer(req["timeout"], cut, ("timed_out",))]
    if req.get("cancel") and req["cancel"][0] == "time":
        timers.append(threading.Timer(req["cancel"][1], cut, ("cancelled",)))
    for tm in timers:
        tm.daemon = True
        tm.start()
    LIVE[id(res)] = cut
    try:
        conn.request("POST", api.path + req["path"], json.dumps(body), api.headers())
        if res.get("cancelled") or res.get("timed_out"):
            raise ConnectionAbortedError("cut before the request was sent")
        r = conn.getresponse()
        res["http"] = r.status
        if r.status != 200:
            res["body"] = r.read()[:1000].decode(errors="replace")
        elif not req["stream"]:
            d = json.loads(r.read())
            res["done"] = True
            res["choices"] = [_choice(c) for c in d.get("choices", [])]
            res["usage"] = d.get("usage")
            plp = (d.get("choices") or [{}])[0].get("prompt_logprobs")
            if plp is not None:  # [None, {token_id: {"logprob": x, ...}}, ...]
                res["plp_len"] = len(plp)
                try:
                    vals = [v["logprob"] for e in plp[1:] for v in e.values()]
                    res["lp_bad"] = plp[0] is not None or not _finite_lp(vals) or not all(plp[1:])
                except (AttributeError, KeyError, TypeError):
                    res["lp_bad"] = True
            if any(not c.pop("lp_ok") for c in res["choices"]):
                res["lp_bad"] = True
        else:
            chs: dict[int, dict] = {}
            n = 0
            for raw in r:
                if not raw.startswith(b"data: "):
                    continue
                data = raw[6:].strip()
                if data == b"[DONE]":
                    break
                ev = json.loads(data)
                if "error" in ev:
                    res["stream_error"] = str(ev["error"])[:500]
                    break
                res["usage"] = ev.get("usage") or res.get("usage")
                for c in ev.get("choices") or []:
                    o = chs.setdefault(c.get("index", 0), {"finish": None, "text": "", "reasoning": "", "tools": {}})
                    d = c.get("delta") or {}
                    piece = (d.get("content") or "") + (c.get("text") or "")
                    rpiece = d.get("reasoning") or d.get("reasoning_content") or ""
                    for tc in d.get("tool_calls") or []:
                        t = o["tools"].setdefault(tc.get("index", 0), ["", ""])
                        f = tc.get("function") or {}
                        t[0] += f.get("name") or ""
                        t[1] += f.get("arguments") or ""
                    if piece or rpiece or d.get("tool_calls"):
                        now = time.monotonic()
                        res.setdefault("ttft", now - t0)
                        res["t_last"] = now - t0
                        n += 1
                    o["text"] += piece
                    o["reasoning"] += rpiece
                    o["finish"] = c.get("finish_reason") or o["finish"]
                if req.get("cancel") and req["cancel"][0] == "chunks" and n >= req["cancel"][1]:
                    res["cancelled"] = True
                    break
            res["done"] = not (res.get("cancelled") or res.get("timed_out"))
            res["chunks"] = n
            res["choices"] = [{**o, "tools": [tuple(t) for _, t in sorted(o["tools"].items())]} for _, o in sorted(chs.items())]
            ct = (res.get("usage") or {}).get("completion_tokens", 0)
            if "ttft" in res and ct > 1:
                res["tpot"] = (res["t_last"] - res["ttft"]) / (ct - 1)
    except Exception as e:  # noqa: BLE001  (a reset after our own cut is the cut, not an error)
        if not (res.get("cancelled") or res.get("timed_out")):
            res["error"] = repr(e)[:300]
    finally:
        LIVE.pop(id(res), None)
        for tm in timers:
            tm.cancel()
        conn.close()
    res["latency"] = time.monotonic() - t0
    return res


def sha(x) -> str:
    return hashlib.sha1(json.dumps(x, sort_keys=True).encode()).hexdigest()[:12]


def record(req: dict, res: dict, phase: dict) -> dict:
    cls = classify(req, res)
    b = req["body"]
    rec = {"t": time.time() - res["latency"], "phase": phase["i"], "ptype": phase["type"], "kind": req["kind"],
           "class": cls, "stream": req["stream"], "max_tokens": b.get("max_tokens"), "T": b.get("temperature"),
           "priority": b.get("priority"), "think": req.get("meta", {}).get("think"), "cancel": req.get("cancel"),
           "timeout": req["timeout"] if req.get("planned_timeout") else None,
           **{k: v for k, v in req.get("meta", {}).items() if k != "think"}}
    for k in ("tool_choice", "n", "prompt_logprobs"):
        if k in b:
            rec[k] = b[k]
    if "response_format" in b:
        rec["response_format"] = b["response_format"]["type"]
    for k in ("http", "error", "stream_error", "ttft", "tpot", "latency", "chunks", "body"):
        if k in res:
            rec[k] = res[k]
    u = res.get("usage") or {}
    rec.update(prompt_tokens=u.get("prompt_tokens"), completion_tokens=u.get("completion_tokens"))
    ch = res.get("choices") or []
    rec["finish"] = [c["finish"] for c in ch]
    rec["tool_calls"] = [t for c in ch for t in c["tools"]]
    keep = 300 if cls in OK_CLASSES else 8000  # keep whole text of anything suspicious
    rec["text"] = [c["text"][:keep] for c in ch]
    rec["reasoning"] = [c["reasoning"][:keep] for c in ch]
    if b.get("temperature") == 0 and ch:  # T=0 repeatability: same body, same output?
        rec["body_sha"] = sha({k: v for k, v in b.items() if k != "priority"})
        rec["text_sha"] = sha([c["text"] + "\0" + c["reasoning"] for c in ch])
    return rec


# ---------------------------------------------------------------- the run

SNAP = re.compile(r"^(vllm:(?:prefix_cache_(?:hits|queries)|external_prefix_cache_(?:hits|queries)|num_preemptions"
                  r"|corrupted_requests|spec_decode_num_(?:drafts|accepted_tokens)|request_success"
                  r"|kv_offload\w*(?:failure|failures|lost_blocks|breaker_trips|dropped)))(?:_total)?(?:\{[^}]*\})?\s+(\S+)$")


def snapshot(api: Api) -> dict:
    out: dict = {}
    for line in api.call("GET", api.root + "/metrics")[1].splitlines():
        m = SNAP.match(line)
        if m:
            out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(2))
    return out


def run(api: Api, out: Path, total_s: float, seed: int, priority: int = 100000, max_conc: int = 12,
        budget: int | None = None, brutal: bool = False, stop: threading.Event | None = None) -> None:
    """The whole schedule against api. stop (server death, Ctrl-C) ends it early and cuts what is in
    flight. budget None = half the server's KV cache (vllm:cache_config_info), else 65,536."""
    stop = stop or threading.Event()
    kv = api.kv_tokens()
    budget = budget or (kv // 2 if kv else 65536)
    out.mkdir(parents=True, exist_ok=True)
    plan = make_plan(total_s, seed)
    lengths = max(n for s in PHASES.values() for n in s.get("lengths", DEFAULTS["lengths"]))
    corpus = Corpus(api, seed, min(lengths + 64, api.mml))
    (out / "plan.json").write_text(json.dumps({"seed": seed, "total_s": total_s, "model": api.model, "max_model_len": api.mml,
                                               "priority": priority, "max_conc": max_conc, "kv_tokens": kv, "token_budget": budget, "brutal": brutal,
                                               "corpus_sha": corpus.sha, "python": platform.python_version(), "phases": plan}, indent=1))
    gen = Gen(corpus, seed, api.model, api.mml, priority)
    throttle = Throttle(max_conc, budget, brutal)
    f_load, f_phase = open(out / "load.jsonl", "a", buffering=1), open(out / "phases.jsonl", "a", buffering=1)
    wlock, cur, done = threading.Lock(), {"p": {"i": -1, "type": "start"}}, threading.Event()

    def emit(rec):
        with wlock:
            f_load.write(json.dumps(rec) + "\n")

    def harness_error(e, ph, kind):  # a bug here must show in the verdict, not silently stop a lane
        emit({"t": time.time(), "phase": ph["i"], "ptype": ph["type"], "kind": kind, "class": "harness_error", "error": repr(e)[:500]})

    def one(make, ph):  # make() builds the request
        req = {"kind": "?"}
        try:
            req = make()
            c = cost(req)
            throttle.acquire(c)
            try:
                res = send(api, req)
            finally:
                throttle.release(c)
            emit(record(req, res, ph))
        except Exception as e:  # noqa: BLE001
            harness_error(e, ph, req["kind"])

    r = random.Random(f"poll:{seed}")

    def poll():  # /tokenize -> /detokenize round trip, /metrics liveness
        ph, text, t = cur["p"], r.choice(SHORT) + f" {r.getrandbits(32):08x}", time.time()
        st, body = api.call("POST", api.root + "/tokenize", {"model": api.model, "prompt": text, "add_special_tokens": False})
        if st == 200:
            st, body = api.call("POST", api.root + "/detokenize", {"model": api.model, "tokens": json.loads(body)["tokens"]})
        cls = f"http_{st // 100}xx" if st != 200 else "ok" if json.loads(body).get("prompt") == text else "tokenize_mismatch"
        emit({"t": t, "phase": ph["i"], "ptype": ph["type"], "kind": "tokenize", "class": cls, "http": st,
              **({} if cls == "ok" else {"body": body[:500]})})
        t = time.time()
        st, body = api.call("GET", api.root + "/metrics")
        cls = f"http_{st // 100}xx" if st != 200 else "ok" if "vllm:num_requests_running" in body else "metrics_error"
        emit({"t": t, "phase": ph["i"], "ptype": ph["type"], "kind": "metrics", "class": cls, "http": st})

    def poller():
        while not done.wait(15):
            try:
                poll()
            except Exception as e:  # noqa: BLE001
                harness_error(e, cur["p"], "poll")

    def lane(mix, conc, slot, t0, dur, ph):
        while (now := time.time()) < t0 + dur and not stop.is_set():
            if slot >= active(conc, (now - t0) / dur):
                time.sleep(0.5)
                continue
            one(lambda: gen.next(mix), ph)

    def burst_lane(mix, t_end, ph):
        while time.time() < t_end and not stop.is_set():
            wave = [threading.Thread(target=one, args=(lambda: gen.next(mix), ph), daemon=True) for _ in range(BURST)]
            for th in wave:
                th.start()
            for th in wave:
                th.join()

    threading.Thread(target=poller, daemon=True).start()

    def cutter():  # a stopped run drops its in-flight requests at once
        while not done.wait(1):
            if stop.is_set():
                return cut_all()

    threading.Thread(target=cutter, daemon=True).start()
    t_run, prev = time.time(), None
    for ph in plan:
        if time.time() - t_run >= total_s or stop.is_set():
            break
        cur["p"] = ph
        spec = {**DEFAULTS, **PHASES[ph["type"]]}
        gen.set_phase(ph["i"], spec)
        m0, t0 = snapshot(api), time.time()
        probe = {"kind": "probe", "path": "/chat/completions", "stream": True, "cancel": None, "timeout": 600.0,
                 "planned_timeout": False, "meta": {"after_idle": prev == "idle"},
                 "body": {"model": api.model, "priority": priority, **PROBE}}
        one(lambda: probe, ph)
        threads = []
        for mix, conc in spec["lanes"]:
            if conc == "burst":
                threads.append(threading.Thread(target=burst_lane, args=(mix, t0 + ph["dur"], ph), daemon=True))
                continue
            for slot in range({"ramp": max(RAMP), "half": 2}.get(conc, conc)):
                threads.append(threading.Thread(target=lane, args=(mix, conc, slot, t0, ph["dur"], ph), daemon=True))
        for th in threads:
            th.start()
        while time.time() < t0 + ph["dur"] and not threads and not stop.is_set():
            time.sleep(1)  # idle
        for th in threads:
            th.join()
        f_phase.write(json.dumps({**ph, "t0": t0, "t_end": t0 + ph["dur"], "t_drained": time.time(), "metrics0": m0,
                                  "metrics1": snapshot(api), "planned_hit": spec["hit"]}) + "\n")
        prev = ph["type"]
    done.set()


def active(conc, frac: float) -> int:
    if conc == "ramp":
        return RAMP[min(int(frac * len(RAMP)), len(RAMP) - 1)]
    if conc == "half":
        return 0 if frac < 0.5 else 2
    return conc


def plan_text(total_s: float, seed: int) -> str:
    plan = make_plan(total_s, seed)
    lines = [f"{p['i']:3d}  {int(p['start'] // 3600)}:{int(p['start'] % 3600 // 60):02d}:{int(p['start'] % 60):02d}"
             f"  {p['dur'] / 60:5.1f} min  {p['type']}" for p in plan]
    by: dict[str, float] = {}
    for p in plan:
        by[p["type"]] = by.get(p["type"], 0) + p["dur"] / 60
    return "\n".join(lines + ["minutes by type: " + ", ".join(f"{k} {v:.0f}" for k, v in sorted(by.items()))])
