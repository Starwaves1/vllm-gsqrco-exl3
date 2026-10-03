#!/usr/bin/env python3
"""Serve the dashboard pages and a small JSON API from DATA (written by collect.py, post and the PR page)."""
import json, os, re, socket, subprocess, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("DASH_DATA") or os.path.expanduser("~/tools/dashboard-data")
REPO = os.path.dirname(HERE)  # this clone; collect.py fetches it every 5 min
REPORT_REFS = ("origin/upstream-prs", "origin/main")  # first ref that has docs/upstream/prs/index.json wins
REPORTS_DIR = os.environ.get("DASH_REPORTS_DIR")  # read reports from a plain directory instead (fixtures)
GITHUB = "https://github.com/Starwaves1/vllm-gsqrco-exl3"
DECISIONS = ("none", "approved", "rejected", "changes-requested")
SLUG = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
lock = threading.Lock()
RATE = 0.20  # $/h (box 2, 350 W host)
MAX_POINTS = 1500


def load(name, default):
    try:
        return json.load(open(os.path.join(DATA, name)))
    except (OSError, ValueError):
        return default


def jsonl(name):
    try:
        with open(os.path.join(DATA, name)) as f:
            lines = f.read().splitlines()
    except OSError:
        return []
    rows = []
    for l in lines:
        try:
            rows.append(json.loads(l))
        except ValueError:  # skip a torn line rather than failing the whole endpoint
            pass
    return rows


def first_ts():
    try:
        with open(os.path.join(DATA, "samples.jsonl")) as f:
            return json.loads(f.readline())["ts"]
    except (OSError, ValueError, KeyError):
        return None


def api_now(q):
    now = load("now.json", {"online": False})
    now.pop("running", None), now.pop("queued", None), now.pop("gpuq", None)
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
    return [{k: s.get(k) for k in keys} for s in rows[::-step][::-1]]  # stride from the end so the newest sample is kept


def api_queue(q):
    now = load("now.json", {})
    jobs = load("jobs.json", {})
    hist = []
    for jid, j in jobs.items():
        prefix = jid.split("-", 1)[0]
        submitted = int(prefix) / 1e6 if prefix.isdigit() else None
        hist.append({"id": jid, "name": jid.split("-", 1)[-1], "submitted": submitted, **j})
    hist.sort(key=lambda j: j["id"], reverse=True)
    return {"online": now.get("online", False), "ts": now.get("ts"), "gpuq": now.get("gpuq", []),
            "running": now.get("running"), "queued": now.get("queued", []), "history": hist[:200]}


def api_posts(q):
    return sorted(jsonl("posts.jsonl"), key=lambda p: p["ts"], reverse=True)


# ---- upstream PRs: reports (read-only, from the fetched refs of this clone) + decisions (DATA/prs.json) ----

def report_file(rel):
    """Text of docs/upstream/prs/<rel> and the ref it came from, or (None, None)."""
    if REPORTS_DIR:
        try:
            return open(os.path.join(REPORTS_DIR, rel)).read(), REPORTS_DIR
        except OSError:
            return None, None
    for ref in REPORT_REFS:
        r = subprocess.run(["git", "-C", REPO, "show", f"{ref}:docs/upstream/prs/{rel}"], capture_output=True, timeout=10)
        if r.returncode == 0:
            return r.stdout.decode("utf-8", "replace"), ref
    return None, None


def compare_url(p):
    branch = p.get("branch")
    if not branch:
        return None
    base = p.get("base")
    if not base and not REPORTS_DIR:  # fork point with main; stacked PRs should set "base" in index.json
        r = subprocess.run(["git", "-C", REPO, "merge-base", "origin/main", "origin/" + branch], capture_output=True, text=True, timeout=10)
        base = r.stdout.strip()[:12] or None
    return f"{GITHUB}/compare/{base}...{branch}" if base else f"{GITHUB}/tree/{branch}"


def pr_list():
    text, ref = report_file("index.json")
    try:
        prs = json.loads(text) if text else []
    except ValueError as e:
        return {"source": ref, "error": f"index.json: {e}", "prs": []}
    decisions = load("prs.json", {})
    fetched = None
    if not REPORTS_DIR:
        try:
            fetched = round(os.path.getmtime(os.path.join(REPO, ".git", "FETCH_HEAD")))
        except OSError:
            pass
    out = []
    for p in prs:
        if not isinstance(p, dict) or not SLUG.fullmatch(str(p.get("slug", ""))):
            continue
        d = decisions.get(p["slug"], {})
        out.append({**p, "decision": d.get("decision", "none"), "decided_at": d.get("decided_at"),
                    "note": d.get("note", ""), "comments": d.get("comments", [])})
    return {"source": ref, "fetched": fetched, "prs": out}


def api_prs(q):
    return pr_list()


def api_pr(slug):
    p = next((p for p in pr_list()["prs"] if p["slug"] == slug), None)
    if p is None:
        return None
    report, _ = report_file(f"{slug}/REPORT.md")
    return {**p, "report_md": report, "compare_url": compare_url(p)}


def record(slug, body):
    """POST /api/prs/<slug>/decision {decision, note, author} or /comment {text, author}; both land in the thread."""
    if not any(p["slug"] == slug for p in pr_list()["prs"]):
        raise LookupError(slug)
    author = str(body.get("author") or "garrett")[:40]
    entry = {"ts": round(time.time()), "author": author}
    with lock:
        decisions = load("prs.json", {})
        d = decisions.setdefault(slug, {"decision": "none", "decided_at": None, "note": "", "comments": []})
        if "decision" in body:
            if body["decision"] not in DECISIONS:
                raise ValueError(f"decision must be one of {', '.join(DECISIONS)}")
            note = str(body.get("note") or "")[:4000]
            d.update(decision=body["decision"], decided_at=entry["ts"], note=note)
            entry.update(decision=body["decision"], text=note)
        else:
            text = str(body.get("text") or "").strip()[:4000]
            if not text:
                raise ValueError("empty comment")
            entry["text"] = text
        d["comments"].append(entry)
        os.makedirs(DATA, exist_ok=True)
        tmp = os.path.join(DATA, "prs.json.tmp")
        with open(tmp, "w") as f:
            json.dump(decisions, f, indent=1)
        os.replace(tmp, os.path.join(DATA, "prs.json"))
    return d


ROUTES = {"/api/now": api_now, "/api/history": api_history, "/api/queue": api_queue, "/api/posts": api_posts,
          "/api/prs": api_prs}
PAGES = {"/": "index.html", "/index.html": "index.html", "/prs": "prs.html"}


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        u = urlparse(self.path)
        m = re.fullmatch(r"/api/prs/([^/]+)", u.path)
        if u.path in PAGES:
            body, ctype = open(os.path.join(HERE, PAGES[u.path]), "rb").read(), "text/html; charset=utf-8"
        elif u.path in ROUTES:
            try:
                body, ctype = json.dumps(ROUTES[u.path](parse_qs(u.query))).encode(), "application/json"
            except ValueError as e:
                return self.send_error(400, str(e))
        elif m and SLUG.fullmatch(m[1]) and (pr := api_pr(m[1])) is not None:
            body, ctype = json.dumps(pr).encode(), "application/json"
        else:
            return self.send_error(404)
        self.reply(body, ctype)

    def do_POST(self):
        m = re.fullmatch(r"/api/prs/([^/]+)/(decision|comment)", urlparse(self.path).path)
        if not m or not SLUG.fullmatch(m[1]):
            return self.send_error(404)
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            if not isinstance(body, dict):
                raise ValueError("body must be a JSON object")
            if m[2] == "decision" and "decision" not in body:
                raise ValueError("missing decision")
            if m[2] == "comment":
                body.pop("decision", None)
            d = record(m[1], body)
        except LookupError:
            return self.send_error(404, "unknown slug")
        except ValueError as e:
            return self.send_error(400, str(e))
        self.reply(json.dumps(d).encode(), "application/json")

    def reply(self, body, ctype):
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
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 18095  # a second port for testing next to the live one
    ThreadingHTTPServer((host, port), H).serve_forever()
