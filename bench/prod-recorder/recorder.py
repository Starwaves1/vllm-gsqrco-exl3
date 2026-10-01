#!/usr/bin/env python3
"""Transparent recording reverse proxy in front of the vLLM OpenAI server.

Every byte is relayed unchanged in both directions (request head and body,
response head, chunked framing, SSE events); one upstream connection per
request, closed when the client goes away so vLLM aborts the generation.
On the side it records each request to data/requests-<hour>.jsonl and runs
the 1 Hz /metrics sampler (sampler.py) to data/metrics-<hour>.jsonl.
Recording failures are counted and never touch the client's request.

    python3 recorder.py [--listen 127.0.0.1:18082] [--bind 192.168.1.5]
                        [--upstream http://127.0.0.1:18081] [--data data]
                        [--label NAME] [--save-logprobs] [--minutes N]
"""
import argparse
import asyncio
import base64
import gzip
import json
import os
import shutil
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import classify  # noqa: E402
import sampler  # noqa: E402

GEN_PATHS = ("/v1/chat/completions", "/v1/completions")
STREAM_LIMIT = 1 << 20          # max request/response head size
CAPTURE_CAP = 64 << 20          # bytes of one body kept in memory
MAX_CHUNKS = 200_000            # chunk-size list entries kept


def log(msg):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)


def parse_headers(head):
    """(first line, {lower name: value}) of a raw HTTP head (last wins)."""
    lines = head.decode("latin-1").split("\r\n")
    hdr = {}
    for line in lines[1:]:
        if ":" in line:
            k, v = line.split(":", 1)
            k = k.strip().lower()
            v = v.strip()
            hdr[k] = f"{hdr[k]}, {v}" if k in hdr and k == "connection" else v
    return lines[0], hdr


class Capture:
    """Side copy of a relayed body; never raises into the relay."""

    def __init__(self, cap=CAPTURE_CAP):
        self.buf = bytearray()
        self.chunks = []
        self.cap = cap
        self.truncated = False
        self.t_first = None

    def data(self, b):
        if self.t_first is None:
            self.t_first = time.time()
        if len(self.buf) + len(b) <= self.cap:
            self.buf += b
        else:
            self.truncated = True

    def chunk(self, n):
        if len(self.chunks) < MAX_CHUNKS:
            self.chunks.append(n)


async def relay_body(reader, writer, hdr, cap, framing_out=None):
    """Relay one message body byte-for-byte; returns the framing used."""
    te = hdr.get("transfer-encoding", "").lower()
    if "chunked" in te:
        while True:
            line = await reader.readuntil(b"\n")
            writer.write(line)
            size = int(line.split(b";", 1)[0].strip() or b"0", 16)
            if size == 0:
                while True:  # trailers, then the empty line
                    t = await reader.readuntil(b"\n")
                    writer.write(t)
                    if t in (b"\r\n", b"\n"):
                        break
                await writer.drain()
                return "chunked"
            left = size
            while left:
                data = await reader.read(min(left, 65536))
                if not data:
                    raise asyncio.IncompleteReadError(b"", left)
                writer.write(data)
                cap.data(data)
                left -= len(data)
            writer.write(await reader.readuntil(b"\n"))
            cap.chunk(size)
            await writer.drain()
    if "content-length" in hdr:
        left = int(hdr["content-length"])
        while left:
            data = await reader.read(min(left, 65536))
            if not data:
                raise asyncio.IncompleteReadError(b"", left)
            writer.write(data)
            cap.data(data)
            cap.chunk(len(data))
            left -= len(data)
            await writer.drain()
        return "length"
    if framing_out == "request":
        return "none"
    while True:  # close-delimited response
        data = await reader.read(65536)
        if not data:
            return "close"
        writer.write(data)
        cap.data(data)
        cap.chunk(len(data))
        await writer.drain()


class Recorder:
    def __init__(self, a):
        self.a = a
        host, port = a.upstream.split("://", 1)[-1].rstrip("/").rsplit(":", 1)
        self.up_host, self.up_port = host, int(port)
        self.data = Path(a.data)
        self.data.mkdir(parents=True, exist_ok=True)
        self.req_out = sampler.HourlyWriter(self.data, "requests")
        self.events = self.data / "events.jsonl"
        self.pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="rec")
        self.inflight = {}
        self.seq = 0
        self.pending = 0
        self.stats = {"requests": 0, "generations": 0, "recorded": 0, "record_errors": 0,
                      "flagged": 0, "client_aborts": 0, "upstream_errors": 0,
                      "proxy_errors": 0, "active_connections": 0}
        self.flag_counts = {}
        self.upstream = {}
        self.started = time.time()
        self.stop = asyncio.Event()
        self.conns = set()
        self.sampler = sampler.Sampler(self.up_host, self.up_port, self.data,
                                       inflight_fn=self.inflight_summary,
                                       on_sample=self.on_sample,
                                       on_restart=self.on_restart, log=log)

    # ------------------------------------------------------------ helpers
    def event(self, kind, **kw):
        try:
            with open(self.events, "a", encoding="utf-8") as f:
                f.write(json.dumps({"ts": round(time.time(), 3), "kind": kind, **kw}) + "\n")
        except Exception as e:
            log(f"event write failed: {e!r}")

    def refresh_upstream(self, why):
        try:
            info = sampler.find_upstream(self.up_port)
        except Exception as e:
            info = {"error": repr(e)}
        if self.sampler.start_ts:
            info["start_ts"] = self.sampler.start_ts
        self.upstream = info
        self.event("upstream", why=why, label=self.a.label, **info)
        log(f"upstream ({why}): pid={info.get('pid')} async={info.get('async')} "
            f"spec={json.dumps(info.get('spec'))}")

    def on_restart(self, old, new):
        self.refresh_upstream("start" if old is None else "restart")

    def spec(self):
        if self.upstream.get("spec"):
            return self.upstream["spec"]
        return {"method": "unknown", "k": None, "schedule": sampler.DEFAULT_SCHEDULE}

    def up_tag(self):
        u = self.upstream
        s = u.get("spec") or {}
        sched = s.get("schedule")
        return {"async": u.get("async", "unknown"),
                "spec": s.get("method", "unknown"),
                "k_cfg": (json.dumps(sched, separators=(",", ":")) if sched else s.get("k")),
                "start_ts": u.get("start_ts"), "pid": u.get("pid"),
                "label": self.a.label}

    def inflight_summary(self):
        s = {"n": 0, "chat": 0, "completions": 0, "stream": 0,
             "prompt_logprobs": 0, "echo": 0}
        for e in self.inflight.values():
            s["n"] += 1
            s[e["type"]] = s.get(e["type"], 0) + 1
            s["stream"] += e["stream"]
            s["prompt_logprobs"] += e["plp"]
            s["echo"] += e["echo"]
        if s["n"] <= 32:
            s["ids"] = list(self.inflight)
        return s

    def on_sample(self, sample):
        plp = sum(e["plp"] for e in self.inflight.values())
        echo = sum(e["echo"] for e in self.inflight.values())
        n = len(self.inflight)
        for e in self.inflight.values():
            e["win"].add(sample, plp - e["plp"], echo - e["echo"], n)

    def write_status(self):
        st = dict(self.stats, inflight=len(self.inflight), pending_writes=self.pending,
                  sampler_errors=self.sampler.errors, flags=self.flag_counts,
                  uptime_s=round(time.time() - self.started), pid=os.getpid(),
                  listen=self.a.listen, bind=self.a.bind, upstream=self.a.upstream,
                  label=self.a.label, save_logprobs=self.a.save_logprobs,
                  upstream_info={k: v for k, v in self.upstream.items() if k != "argv"},
                  ts=round(time.time(), 3))
        tmp = self.data / ".status.json.tmp"
        tmp.write_text(json.dumps(st, indent=1))
        tmp.replace(self.data / "status.json")

    # ------------------------------------------------------------- proxy
    async def handle(self, cr, cw):
        task = asyncio.current_task()
        self.conns.add(task)
        self.stats["active_connections"] += 1
        peer = cw.get_extra_info("peername")
        peer = f"{peer[0]}:{peer[1]}" if peer else "?"
        try:
            while not self.stop.is_set():
                try:
                    head = await cr.readuntil(b"\r\n\r\n")
                except asyncio.IncompleteReadError:
                    break
                except asyncio.LimitOverrunError:
                    cw.write(b"HTTP/1.1 431 Request Header Fields Too Large\r\n"
                             b"Content-Length: 0\r\nConnection: close\r\n\r\n")
                    break
                if not await self.handle_request(head, cr, cw, peer):
                    break
        except (ConnectionError, asyncio.IncompleteReadError, asyncio.CancelledError):
            pass
        except Exception as e:
            self.stats["proxy_errors"] += 1
            log(f"proxy error ({peer}): {e!r}")
        finally:
            self.stats["active_connections"] -= 1
            self.conns.discard(task)
            try:
                cw.close()
            except Exception:
                pass

    async def handle_request(self, head, cr, cw, peer):
        """Relay one request/response. Returns True to keep the client
        connection open for another request."""
        t0 = time.time()
        self.stats["requests"] += 1
        line, hdr = parse_headers(head)
        parts = line.split(" ")
        method, target = parts[0], parts[1] if len(parts) > 1 else "/"
        version = parts[2] if len(parts) > 2 else "HTTP/1.0"
        path = target.split("?", 1)[0]
        gen = method == "POST" and path in GEN_PATHS
        req_cap = Capture() if gen else Capture(cap=0)
        resp_cap = Capture() if gen else Capture(cap=0)
        st = {"status": None, "resp_hdr": {}, "framing": None, "aborted": False,
              "error": None, "t_head": None}
        req_close = "close" in hdr.get("connection", "").lower() or \
            (version == "HTTP/1.0" and "keep-alive" not in hdr.get("connection", "").lower())

        try:
            ur, uw = await asyncio.open_connection(self.up_host, self.up_port, limit=STREAM_LIMIT)
        except OSError as e:
            self.stats["upstream_errors"] += 1
            st.update(status=502, error=f"connect: {e}")
            msg = f"upstream {self.up_host}:{self.up_port} unreachable: {e}\n".encode()
            cw.write(b"HTTP/1.1 502 Bad Gateway\r\nContent-Type: text/plain\r\n"
                     b"Content-Length: " + str(len(msg)).encode() +
                     b"\r\nConnection: close\r\n\r\n" + msg)
            await cw.drain()
            self.finish(t0, method, target, path, peer, gen, hdr, req_cap, resp_cap, st, None)
            return False

        rid = None
        try:
            uw.write(head)
            resp_task = asyncio.create_task(self.relay_response(method, ur, cw, resp_cap, st))
            try:
                await relay_body(cr, uw, hdr, req_cap, framing_out="request")
            except (ConnectionError, asyncio.IncompleteReadError) as e:
                if not resp_task.done():
                    resp_task.cancel()
                    st["error"] = f"request body: {type(e).__name__}"
                    raise
            rid = self.register(gen, t0, req_cap, path)
            watch = asyncio.create_task(self.watch_client(cr))
            done, _ = await asyncio.wait({resp_task, watch}, return_when=asyncio.FIRST_COMPLETED)
            if resp_task in done:
                watch.cancel()
                resp_keep = resp_task.result()
            else:
                resp_task.cancel()
                try:
                    await resp_task
                except (asyncio.CancelledError, Exception):
                    pass
                st["aborted"] = True
                self.stats["client_aborts"] += 1
                resp_keep = False
        except asyncio.CancelledError:
            raise
        except (ConnectionError, asyncio.IncompleteReadError, OSError) as e:
            # upstream died mid-response, or the client vanished while we
            # were writing to it: same outcome as without the proxy
            if st["error"] is None:
                st["error"] = f"{type(e).__name__}: {e}"[:300]
            if isinstance(e, (BrokenPipeError, ConnectionResetError)) and st["status"]:
                st["aborted"] = True
                self.stats["client_aborts"] += 1
            else:
                self.stats["upstream_errors"] += 1
            resp_keep = False
        finally:
            try:
                uw.close()
            except Exception:
                pass
            self.finish(t0, method, target, path, peer, gen, hdr, req_cap, resp_cap, st, rid)
        return resp_keep and not req_close

    async def watch_client(self, cr):
        """Returns when the client has gone away (EOF or error)."""
        while True:
            await asyncio.sleep(0.25)
            if cr.at_eof() or cr.exception() is not None:
                return

    async def relay_response(self, method, ur, cw, cap, st):
        while True:
            head = await ur.readuntil(b"\r\n\r\n")
            cw.write(head)
            await cw.drain()
            line, hdr = parse_headers(head)
            status = int(line.split(" ", 2)[1])
            if 100 <= status < 200 and status != 101:
                continue
            break
        st.update(status=status, resp_hdr=hdr, t_head=time.time())
        if status == 101:
            st["framing"] = "upgrade"
            await self.tunnel(ur, cw)
            return False
        if method == "HEAD" or status in (204, 304):
            st["framing"] = "none"
        else:
            st["framing"] = await relay_body(ur, cw, hdr, cap)
        conn = hdr.get("connection", "").lower()
        return st["framing"] != "close" and "close" not in conn

    async def tunnel(self, ur, cw):
        while True:
            data = await ur.read(65536)
            if not data:
                return
            cw.write(data)
            await cw.drain()

    # --------------------------------------------------------- recording
    def register(self, gen, t0, req_cap, path):
        """Parse the request body and mark the request in flight."""
        if not gen:
            return None
        try:
            self.seq += 1
            rid = f"{int(t0 * 1000)}-{self.seq}"
            body = json.loads(bytes(req_cap.buf)) if req_cap.buf and not req_cap.truncated else {}
            rinfo = classify.request_info(body, path)
            req_cap.body = body
            req_cap.rinfo = rinfo
            self.inflight[rid] = {
                "type": rinfo["type"], "stream": int(rinfo["stream"]),
                "plp": int(rinfo["prompt_logprobs"]), "echo": int(rinfo["echo"]),
                "win": sampler.Window(self.spec(), self.sampler.latest),
            }
            return rid
        except Exception as e:
            self.stats["record_errors"] += 1
            log(f"register failed: {e!r}")
            return None

    def finish(self, t0, method, target, path, peer, gen, hdr, req_cap, resp_cap, st, rid):
        """Hand the finished exchange to a worker thread; never raises."""
        try:
            entry = self.inflight.pop(rid, None) if rid else None
            cond = entry["win"].summary() if entry else None
            base = {
                "v": 1, "rid": rid, "t0": round(t0, 3),
                "t_head": round(st["t_head"], 3) if st["t_head"] else None,
                "t_first": round(resp_cap.t_first, 3) if resp_cap.t_first else None,
                "t1": round(time.time(), 3), "client": peer, "method": method,
                "path": path, "query": target[len(path):] or None,
                "status": st["status"], "framing": st["framing"],
                "client_aborted": st["aborted"], "error": st["error"],
                "user_agent": hdr.get("user-agent"),
                "ctype": st["resp_hdr"].get("content-type"),
                "x_request_id": st["resp_hdr"].get("x-request-id"),
                "up": self.up_tag(), "cond": cond,
            }
            if gen:
                self.stats["generations"] += 1
            self.pending += 1
            fut = self.pool.submit(self.build_and_write, base, gen, req_cap, resp_cap)
            fut.add_done_callback(self._written)
        except Exception as e:
            self.stats["record_errors"] += 1
            log(f"record dropped: {e!r}")

    def _written(self, fut):
        self.pending -= 1
        exc = fut.exception()
        if exc is not None:
            self.stats["record_errors"] += 1
            log(f"record dropped: {exc!r}")
            return
        self.stats["recorded"] += 1
        flags = fut.result()
        if flags:
            self.stats["flagged"] += 1
            for f in flags:
                self.flag_counts[f] = self.flag_counts.get(f, 0) + 1

    def build_and_write(self, rec, gen, req_cap, resp_cap):
        """Worker thread: classify and append one JSON line. Returns flags."""
        flags = {}
        rec["req_bytes"] = len(req_cap.buf)
        rec["resp_bytes"] = len(resp_cap.buf)
        if gen:
            body = getattr(req_cap, "body", None)
            rinfo = getattr(req_cap, "rinfo", None) or classify.request_info(body, rec["path"])
            pinfo = classify.prompt_info(body)
            raw = bytes(resp_cap.buf)
            want_lp = self.a.save_logprobs and (rinfo.get("logprobs") or rinfo.get("prompt_logprobs")
                                                or rinfo.get("echo"))
            parsed = classify.parse_response(raw, rec["ctype"] or "", scan_logprobs=bool(want_lp))
            if rec["status"] == 200 and not (resp_cap.truncated or rec["client_aborted"]
                                             or rec["error"]):
                flags = classify.classify(parsed, rinfo, pinfo, self.allow)
            usage = parsed["usage"] or {}
            rec.update({
                "req": self.shrink(classify.sanitize(body)) if isinstance(body, dict) else None,
                "req_info": rinfo, "prompt_info": pinfo,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "usage": usage or None,
                "finish_reasons": [c["finish_reason"] for c in parsed["choices"]],
                "stop_reasons": [c["stop_reason"] for c in parsed["choices"]],
                "stop_kinds": [classify.stop_kind(c, rinfo) for c in parsed["choices"]],
                "parse": {k: parsed[k] for k in ("kind", "events", "bad_events", "utf8_valid", "lp")},
                "resp_error": parsed["error"],
                "chunk_sizes": resp_cap.chunks,
                "utf8_split": classify.utf8_split_boundaries(raw, resp_cap.chunks)
                if rec["framing"] == "chunked" else None,
                "sse_split": classify.sse_split_boundaries(raw, resp_cap.chunks)
                if rec["framing"] == "chunked" and parsed["kind"] == "sse" else None,
                "resp_truncated": resp_cap.truncated,
                "req_truncated": req_cap.truncated,
                "flags": flags, "flagged": bool(flags),
            })
            limit = self.a.max_resp_kb << 10
            if want_lp:
                limit = CAPTURE_CAP
            store = raw if len(raw) <= limit else raw[:limit]
            rec["resp_stored_truncated"] = len(store) < len(raw)
            try:
                rec["resp_text"] = store.decode("utf-8")
            except UnicodeDecodeError:
                rec["resp_b64"] = base64.b64encode(store).decode()
        self.req_out.write(rec)  # file of the hour the request ended in
        return flags

    def shrink(self, body):
        """Keep the request body under --max-req-kb by cutting long strings
        to head + tail (offsets in the stored copy are then approximate)."""
        limit = self.a.max_req_kb << 10
        if len(json.dumps(body, ensure_ascii=False)) <= limit:
            return body

        def cut(o):
            if isinstance(o, str) and len(o) > 8192:
                return o[:4096] + f"\n[... {len(o) - 8192} chars cut by recorder ...]\n" + o[-4096:]
            if isinstance(o, list):
                return [cut(x) for x in o]
            if isinstance(o, dict):
                return {k: cut(v) for k, v in o.items()}
            return o
        out = cut(body)
        out["_recorder_cut"] = True
        return out

    def compress_old(self):
        """gzip hourly files nobody writes to any more (>= 2 min idle)."""
        now = time.time()
        current = {self.req_out.path(now).name, self.sampler.out.path(now).name}
        for p in self.data.glob("*-????????-??.jsonl"):
            if p.name in current or now - p.stat().st_mtime < 120:
                continue
            with open(p, "rb") as src, gzip.open(f"{p}.gz", "ab") as dst:
                shutil.copyfileobj(src, dst)
            p.unlink()

    # --------------------------------------------------------------- run
    async def status_loop(self):
        while not self.stop.is_set():
            try:
                self.write_status()
            except Exception as e:
                log(f"status write failed: {e!r}")
            try:
                await asyncio.to_thread(self.compress_old)
            except Exception as e:
                log(f"compress failed: {e!r}")
            try:
                await asyncio.wait_for(self.stop.wait(), 5)
            except asyncio.TimeoutError:
                pass

    async def main(self):
        self.allow = frozenset(s for s in self.a.allow_scripts.split(",") if s)
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self.stop.set)
        if self.a.minutes:
            loop.call_later(self.a.minutes * 60, self.stop.set)
        servers = []
        lhost, lport = self.a.listen.rsplit(":", 1)
        addrs = [(lhost, int(lport))] + [(b, int(lport)) for b in (self.a.bind or [])]
        for h, p in addrs:
            servers.append(await asyncio.start_server(self.handle, h, p, limit=STREAM_LIMIT,
                                                      reuse_address=True))
        self.refresh_upstream("recorder start")
        self.event("start", listen=[f"{h}:{p}" for h, p in addrs], upstream=self.a.upstream,
                   label=self.a.label, save_logprobs=self.a.save_logprobs, pid=os.getpid())
        log(f"listening on {', '.join(f'{h}:{p}' for h, p in addrs)} -> {self.a.upstream}; "
            f"data in {self.data}")
        bg = [asyncio.create_task(self.sampler.run(self.stop)),
              asyncio.create_task(self.status_loop())]
        await self.stop.wait()
        log("stopping: closing listeners, draining in-flight requests")
        for s in servers:
            s.close()
        t_end = time.monotonic() + self.a.drain
        while self.inflight and time.monotonic() < t_end:
            await asyncio.sleep(0.2)
        for t in list(self.conns):
            t.cancel()
        for t in bg:
            t.cancel()
        await asyncio.gather(*bg, *list(self.conns), return_exceptions=True)
        self.pool.shutdown(wait=True)
        self.write_status()
        self.event("stop", stats=self.stats)
        log(f"stopped: {json.dumps(self.stats)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--listen", default="127.0.0.1:18082")
    ap.add_argument("--bind", action="append", default=[],
                    help="extra listen address on the same port (e.g. the LAN IP); repeatable")
    ap.add_argument("--upstream", default="http://127.0.0.1:18081")
    ap.add_argument("--data", default=str(Path(__file__).resolve().parent / "data"))
    ap.add_argument("--label", default=None, help="tag stored on every record (for --compare)")
    ap.add_argument("--save-logprobs", action="store_true",
                    help="keep the logprobs a client asked for and count NaN/inf/-9999 in them")
    ap.add_argument("--allow-scripts", default="greek",
                    help="comma list of non-Latin scripts never flagged as foreign")
    ap.add_argument("--max-req-kb", type=int, default=256)
    ap.add_argument("--max-resp-kb", type=int, default=1024)
    ap.add_argument("--drain", type=float, default=30.0,
                    help="seconds to let in-flight requests finish on stop")
    ap.add_argument("--minutes", type=float, default=0, help="stop after N minutes (0 = run until stopped)")
    a = ap.parse_args()
    asyncio.run(Recorder(a).main())


if __name__ == "__main__":
    main()
