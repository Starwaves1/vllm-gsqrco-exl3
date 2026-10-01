"""`recorder.py --minutes 1` against the mock under mixed load (about 65 s)."""
import glob
import json
import os
import random
import subprocess
import sys
import threading
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import ROOT, Stack, post, raw_exchange  # noqa: E402

KWS = ["", "", "", "SPLIT3", "MIDSENT", "REPEAT", "BLANK", "SLOW"]


class Smoke(unittest.TestCase):
    def test_one_minute(self):
        with Stack(minutes=1) as s:
            stop = threading.Event()
            sent = []

            def load(seed):
                rng = random.Random(seed)
                while not stop.is_set():
                    kw = rng.choice(KWS)
                    body = {"model": "m", "max_tokens": 64, "stream": rng.random() < 0.5,
                            "temperature": rng.choice([0, 0.7, None])}
                    if rng.random() < 0.7:
                        body["messages"] = [{"role": "user", "content": f"{kw} q{len(sent)}"}]
                        path = "/v1/chat/completions"
                    else:
                        body["prompt"] = f"{kw} q{len(sent)}"
                        body["prompt_logprobs"] = 1 if rng.random() < 0.2 else None
                        path = "/v1/completions"
                    try:
                        out = raw_exchange(s.port, post(path, body), timeout=10)
                        sent.append(out.startswith(b"HTTP/1.1 200"))
                    except OSError:
                        if s.rec.poll() is not None:
                            return
                    time.sleep(rng.random() * 0.3)

            threads = [threading.Thread(target=load, args=(i,), daemon=True) for i in range(4)]
            for t in threads:
                t.start()
            rc = s.rec.wait(90)
            stop.set()
            for t in threads:
                t.join(15)
            self.assertEqual(rc, 0)
            self.assertGreater(len(sent), 50)
            self.assertTrue(all(sent), "a client request failed while the recorder was up")
            recs = s.records()
            gens = [r for r in recs if r["path"] != "/v1/models"]
            self.assertGreaterEqual(len(gens), len(sent))
            self.assertTrue(any(r["flags"] for r in gens))
            self.assertTrue(any(r["cond"]["n_samples"] >= 2 for r in gens if r.get("cond")))
            mlines = 0
            for f in glob.glob(os.path.join(s.data, "metrics-*.jsonl")):
                with open(f) as fh:
                    mlines += sum(1 for _ in fh)
            self.assertGreaterEqual(mlines, 50)
            st = s.status()
            self.assertEqual(st["record_errors"], 0)
            self.assertEqual(st["inflight"], 0)
            with open(os.path.join(s.tmp, "recorder.log")) as fh:
                log = fh.read()
            self.assertIn("stopped:", log)
            out = subprocess.run([sys.executable, os.path.join(ROOT, "report.py"), "--data", s.data],
                                 capture_output=True, text=True, check=True).stdout
            self.assertIn("generations flagged", out)
            with open(os.path.join(s.data, "report.json")) as fh:
                rep = json.load(fh)
            self.assertEqual(rep["summary"]["n"], sum(1 for r in gens if r["status"] == 200
                                                      and not r["client_aborted"]))
            size = sum(os.path.getsize(f) for f in glob.glob(os.path.join(s.data, "*.jsonl")))
            print(f"\nsmoke: {len(sent)} requests, {len(gens)} records, {size / 1024:.0f} KiB on disk")


if __name__ == "__main__":
    unittest.main()
