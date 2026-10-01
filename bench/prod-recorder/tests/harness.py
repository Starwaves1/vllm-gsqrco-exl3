"""Start the mock upstream and the recorder as subprocesses (tests only)."""
import glob
import json
import os
import socket
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = sys.executable


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_port(port, timeout=10):
    end = time.time() + timeout
    while time.time() < end:
        try:
            socket.create_connection(("127.0.0.1", port), 0.2).close()
            return
        except OSError:
            time.sleep(0.05)
    raise RuntimeError(f"port {port} never opened")


class Stack:
    def __init__(self, extra=(), minutes=None):
        self.tmp = tempfile.mkdtemp(prefix="prodrec-test-")
        self.data = os.path.join(self.tmp, "data")
        self.mock_port = free_port()
        self.port = free_port()
        self.extra = list(extra)
        self.minutes = minutes

    def start_mock(self):
        self.mock = subprocess.Popen([PY, os.path.join(HERE, "mock_upstream.py"), str(self.mock_port)])
        wait_port(self.mock_port)

    def start_recorder(self):
        cmd = [PY, os.path.join(ROOT, "recorder.py"), "--listen", f"127.0.0.1:{self.port}",
               "--upstream", f"http://127.0.0.1:{self.mock_port}", "--data", self.data,
               "--drain", "5"] + self.extra
        if self.minutes:
            cmd += ["--minutes", str(self.minutes)]
        self.log = open(os.path.join(self.tmp, "recorder.log"), "w")
        self.rec = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT)
        wait_port(self.port)

    def __enter__(self):
        self.start_mock()
        self.start_recorder()
        return self

    def __exit__(self, *exc):
        for p in (getattr(self, "rec", None), getattr(self, "mock", None)):
            if p and p.poll() is None:
                p.terminate()
                try:
                    p.wait(10)
                except subprocess.TimeoutExpired:
                    p.kill()
        self.log.close()

    def records(self, n=None, timeout=5):
        """All request records so far (waits until at least n exist)."""
        end = time.time() + timeout
        while True:
            out = []
            for f in sorted(glob.glob(os.path.join(self.data, "requests-*.jsonl"))):
                with open(f, encoding="utf-8") as fh:
                    out += [json.loads(line) for line in fh if line.strip()]
            if n is None or len(out) >= n or time.time() > end:
                return out
            time.sleep(0.05)

    def status(self):
        with open(os.path.join(self.data, "status.json")) as f:
            return json.load(f)


def raw_exchange(port, request, timeout=10):
    """Send raw bytes, read until the server closes; returns raw response."""
    s = socket.create_connection(("127.0.0.1", port), timeout)
    s.sendall(request)
    buf = bytearray()
    while True:
        d = s.recv(65536)
        if not d:
            break
        buf += d
    s.close()
    return bytes(buf)


def post(path, body, close=True, extra=b""):
    b = json.dumps(body).encode()
    return (f"POST {path} HTTP/1.1\r\nHost: test\r\nContent-Type: application/json\r\n"
            f"Content-Length: {len(b)}\r\n").encode() + (b"Connection: close\r\n" if close else b"") \
        + extra + b"\r\n" + b
