#!/usr/bin/env python3
"""Serve the dashboard page and a small JSON API from data/ (written by collect.py and post)."""
import json, os, socket, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
RATE = 0.155  # $/h
MAX_POINTS = 1500


def load(name, default):
    try:
        return json.load(open(os.path.join(DATA, name)))
    except (OSError, ValueError):
        return default


def jsonl(name):
    try:
        with open(os.path.join(DATA, name)) as f:
            return [json.loads(l) for l in f if l.strip()]
    except OSError:
        return []


def first_ts():
    try:
        with open(os.path.join(DATA, "samples.jsonl")) as f:
            return json.loads(f.readline())["ts"]
    except (OSError, ValueError, KeyError):
        return None


def api_now(q):
    now = load("now.json", {"online": False})
    now.pop("running", None), now.pop("queued", None)
    t0 = first_ts()
    if t0:
        hours = (time.time() - t0) / 3600
        now.update(first_ts=t0, hours=round(hours, 2), cost=round(hours * RATE, 2), rate=RATE)
    return now


def api_history(q):
    since = float(q.get("since", ["0"])[0])
    rows = [s for s in jsonl("samples.jsonl") if s["ts"] >= since]
    step = max(1, -(-len(rows) // MAX_POINTS))  # thin long ranges to <= MAX_POINTS
    keys = ("ts", "util", "power", "power_limit", "sm_clock", "vram_used", "vram_total")
    return [{k: s.get(k) for k in keys} for s in rows[::step]]


def api_queue(q):
    now = load("now.json", {})
    jobs = load("jobs.json", {})
    hist = []
    for jid, j in jobs.items():
        prefix = jid.split("-", 1)[0]
        submitted = int(prefix) / 1e6 if prefix.isdigit() else None
        hist.append({"id": jid, "name": jid.split("-", 1)[-1], "submitted": submitted, **j})
    hist.sort(key=lambda j: j["id"], reverse=True)
    return {"online": now.get("online", False), "ts": now.get("ts"),
            "running": now.get("running"), "queued": now.get("queued", []), "history": hist[:200]}


def api_posts(q):
    return sorted(jsonl("posts.jsonl"), key=lambda p: p["ts"], reverse=True)


ROUTES = {"/api/now": api_now, "/api/history": api_history, "/api/queue": api_queue, "/api/posts": api_posts}


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            body, ctype = open(os.path.join(HERE, "index.html"), "rb").read(), "text/html; charset=utf-8"
        elif u.path in ROUTES:
            try:
                body, ctype = json.dumps(ROUTES[u.path](parse_qs(u.query))).encode(), "application/json"
            except ValueError as e:
                return self.send_error(400, str(e))
        else:
            return self.send_error(404)
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("192.168.1.1", 9))  # no packet sent; picks the LAN interface address
    return s.getsockname()[0]


if __name__ == "__main__":
    host = sys.argv[1] if len(sys.argv) > 1 else lan_ip()
    ThreadingHTTPServer((host, 18095), H).serve_forever()
