"""Proxy passthrough and recording against the mock upstream (CPU only)."""
import http.client
import json
import os
import socket
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import Stack, post, raw_exchange  # noqa: E402


def chat(text, stream=False, **kw):
    return dict({"model": "m", "messages": [{"role": "user", "content": text}],
                 "stream": stream, "max_tokens": 64, "priority": 7}, **kw)


class ProxyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s = Stack(extra=["--save-logprobs", "--label", "unit"])
        cls.s.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.s.__exit__(None, None, None)

    def mock_stats(self):
        return json.loads(raw_exchange(self.s.mock_port,
                                       b"GET /mock/stats HTTP/1.1\r\nHost: t\r\nConnection: close\r\n\r\n")
                          .split(b"\r\n\r\n", 1)[1])

    def by_marker(self, marker, n_expected=1):
        recs = [r for r in self.s.records(timeout=5)
                if r.get("req") and marker in json.dumps(r["req"])]
        end = time.time() + 5
        while len(recs) < n_expected and time.time() < end:
            time.sleep(0.1)
            recs = [r for r in self.s.records() if r.get("req") and marker in json.dumps(r["req"])]
        return recs

    def test_stream_byte_identical_and_request_untouched(self):
        for body in (chat("hello A1", stream=True, stream_options={"include_usage": True}),
                     chat("hello A2"),
                     {"model": "m", "prompt": "hi A3", "stream": True, "max_tokens": 5}):
            path = "/v1/completions" if "prompt" in body else "/v1/chat/completions"
            req = post(path, body)
            before = len(self.mock_stats()["requests"])
            direct = raw_exchange(self.s.mock_port, req)
            via = raw_exchange(self.s.port, req)
            self.assertEqual(direct, via)
            self.assertIn(b"\r\n\r\n", via)
            reqs = self.mock_stats()["requests"]
            # the mock saw byte-identical requests (head + body) both ways
            self.assertEqual(reqs[before], reqs[before + 1])
        self.assertIn(b"transfer-encoding: chunked", direct.lower())  # last one was a stream

    def test_utf8_split_detected_not_altered(self):
        req = post("/v1/chat/completions", chat("SPLIT3 please", stream=True))
        direct = raw_exchange(self.s.mock_port, req)
        via = raw_exchange(self.s.port, req)
        self.assertEqual(direct, via)
        lead = "閮".encode()[:1]
        self.assertIn(lead + b"\r\n", via)  # the chunk really ends mid-character
        (rec,) = self.by_marker("SPLIT3")
        self.assertEqual(rec["utf8_split"], 1)
        self.assertGreaterEqual(rec["sse_split"], 1)
        self.assertIn("閮", rec["resp_text"])
        self.assertNotIn("�", rec["resp_text"])
        self.assertIn("foreign_script", rec["flags"])
        self.assertNotIn("replacement_char", rec["flags"])

    def test_flags_end_to_end(self):
        want = {"OPENREASON": "early_stop_open_reasoning", "MIDSENT": "early_stop_mid_sentence",
                "REPEAT": "repeat_fragment", "FFFD": "replacement_char", "BLANK": "blank_only",
                "NANLP": "nonfinite_logprobs"}
        for stream in (False, True):
            for kw in want:
                body = chat(f"{kw} s={stream}", stream=stream, logprobs=(kw == "NANLP"))
                raw_exchange(self.s.port, post("/v1/chat/completions", body))
        for kw, flag in want.items():
            recs = self.by_marker(f"{kw} s=", 2)
            self.assertEqual(len(recs), 2, kw)
            for r in recs:
                self.assertIn(flag, r["flags"], (kw, r["flags"]))
                self.assertTrue(r["flagged"])
        raw_exchange(self.s.port, post("/v1/chat/completions", chat("clean-one")))
        clean = self.by_marker("clean-one")
        self.assertEqual(clean[0]["flags"], {})
        self.assertEqual(clean[0]["req_info"]["priority"], 7)
        self.assertEqual(clean[0]["up"]["label"], "unit")

    def test_keepalive_two_requests_one_connection(self):
        c = http.client.HTTPConnection("127.0.0.1", self.s.port, timeout=10)
        for i in range(2):
            c.request("POST", "/v1/chat/completions", json.dumps(chat(f"KA{i}")),
                      {"Content-Type": "application/json", "Authorization": "Bearer sk-secret"})
            r = c.getresponse()
            self.assertEqual(r.status, 200)
            self.assertEqual(json.loads(r.read())["choices"][0]["message"]["content"],
                             "Paris is the capital of France.")
        c.request("GET", "/v1/models")
        self.assertEqual(json.loads(c.getresponse().read())["data"][0]["id"], "m")
        c.close()
        self.assertEqual(len(self.by_marker("KA0")) + len(self.by_marker("KA1")), 2)
        for f in os.listdir(self.s.data):
            with open(os.path.join(self.s.data, f), "rb") as fh:
                self.assertNotIn(b"sk-secret", fh.read())

    def test_expect_100_continue(self):
        body = json.dumps(chat("EXPECT100")).encode()
        s = socket.create_connection(("127.0.0.1", self.s.port), 10)
        s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\nContent-Type: application/json\r\n"
                  b"Expect: 100-continue\r\nConnection: close\r\n"
                  + f"Content-Length: {len(body)}\r\n\r\n".encode())
        first = s.recv(100)
        self.assertTrue(first.startswith(b"HTTP/1.1 100 Continue"))
        s.sendall(body)
        buf = bytearray(first)
        while True:
            d = s.recv(65536)
            if not d:
                break
            buf += d
        s.close()
        self.assertIn(b"HTTP/1.1 200 OK", bytes(buf))
        self.assertEqual(len(self.by_marker("EXPECT100")), 1)

    def test_client_disconnect_aborts_upstream(self):
        before = self.mock_stats()["aborted"]
        s = socket.create_connection(("127.0.0.1", self.s.port), 10)
        s.sendall(post("/v1/chat/completions", chat("SLOW abort-me")))
        time.sleep(0.6)
        s.close()
        end = time.time() + 5
        while self.mock_stats()["aborted"] == before and time.time() < end:
            time.sleep(0.1)
        self.assertEqual(self.mock_stats()["aborted"], before + 1)
        (rec,) = self.by_marker("abort-me")
        self.assertTrue(rec["client_aborted"])
        self.assertEqual(rec["flags"], {})

    def test_conditions_sampled_during_generation(self):
        raw_exchange(self.s.port, post("/v1/chat/completions",
                                       chat("SLOW sampled", temperature=0, top_k=1)))
        (rec,) = self.by_marker("SLOW sampled")
        c = rec["cond"]
        self.assertGreaterEqual(c["n_samples"], 2)
        self.assertGreaterEqual(c["running_max"], 1)
        self.assertEqual(c["k_mode"], 5)  # mock has no argv -> production default schedule
        self.assertEqual(rec["req_info"]["temperature"], 0)
        self.assertEqual(rec["req_info"]["top_k"], 1)
        metrics = [f for f in os.listdir(self.s.data) if f.startswith("metrics-")]
        self.assertTrue(metrics)
        with open(os.path.join(self.s.data, metrics[0])) as fh:
            lines = [json.loads(x) for x in fh]
        self.assertTrue(any(x.get("inflight", {}).get("n", 0) >= 1 for x in lines))
        self.assertIn("running", lines[-1]["g"])

    def test_status_counts(self):
        raw_exchange(self.s.port, post("/v1/chat/completions", chat("status-probe")))
        time.sleep(5.5)
        st = self.s.status()
        self.assertGreater(st["recorded"], 0)
        self.assertEqual(st["record_errors"], 0)


class FailureTest(unittest.TestCase):
    def test_recording_failure_never_fails_client(self):
        with Stack() as s:
            # make every requests-<hour>.jsonl unopenable
            for h in range(2):
                t = time.time() + 3600 * h
                os.makedirs(os.path.join(s.data, time.strftime("requests-%Y%m%d-%H.jsonl",
                                                               time.localtime(t))), exist_ok=True)
            for i in range(3):
                out = raw_exchange(s.port, post("/v1/chat/completions", chat(f"fail{i}", stream=bool(i % 2))))
                self.assertTrue(out.startswith(b"HTTP/1.1 200 OK"), out[:80])
            time.sleep(5.5)
            st = s.status()
            self.assertEqual(st["record_errors"], 3)
            self.assertEqual(st["recorded"], 0)

    def test_upstream_down_gives_502(self):
        s = Stack()
        try:
            s.mock = None
            s.start_recorder()
            out = raw_exchange(s.port, post("/v1/chat/completions", chat("down")))
            self.assertTrue(out.startswith(b"HTTP/1.1 502"))
            (rec,) = s.records(1)
            self.assertEqual(rec["status"], 502)
        finally:
            s.__exit__(None, None, None)


if __name__ == "__main__":
    unittest.main()
