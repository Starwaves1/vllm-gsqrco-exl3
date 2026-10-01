"""report.py on synthetic recordings."""
import json
import os
import random
import subprocess
import sys
import tempfile
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import report  # noqa: E402


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def sse_text(content, reasoning="", finish="stop"):
    ev = [{"choices": [{"index": 0, "delta": {"reasoning": reasoning}}]}] if reasoning else []
    ev += [{"choices": [{"index": 0, "delta": {"content": content}}]},
           {"choices": [{"index": 0, "delta": {}, "finish_reason": finish, "stop_reason": None}]}]
    return "".join("data: " + json.dumps(e, ensure_ascii=False) + "\n\n" for e in ev) + "data: [DONE]\n\n"


def rec(i, t0, temp, running, flagged, label, plp=False, split=0):
    content = "The war ended in 199閮1994 after talks." if flagged else "It ended in 1994."
    flags = {"foreign_script": {"field": "content", "span": [20, 21], "detail": {}, "choice": 0}} if flagged else {}
    return {"v": 1, "rid": f"r{i}", "t0": t0, "t1": t0 + 2, "path": "/v1/chat/completions",
            "status": 200, "client_aborted": False, "error": None, "ctype": "text/event-stream",
            "framing": "chunked", "utf8_split": split, "sse_split": 0, "resp_truncated": False,
            "req_info": {"type": "chat", "temperature": temp, "top_p": None, "top_k": None, "stream": True,
                         "max_tokens": 512, "prompt_logprobs": plp, "echo": False},
            "prompt_info": {"scripts": [], "has_fffd": False, "open_think": False},
            "prompt_tokens": 100 + i, "completion_tokens": 40,
            "cond": {"running_min": running, "running_max": running, "k_mode": 5 if running <= 4 else 3,
                     "k_mixed": False, "other_prompt_logprobs_inflight": False, "preempt_delta": 0},
            "up": {"async": "off", "spec": "mtp", "k_cfg": "[[1,4,5],[5,8,3]]", "label": label},
            "resp_text": sse_text(content), "flags": flags, "flagged": bool(flags)}


class ReportTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="prodrec-report-")
        rng = random.Random(0)
        t = time.time() - 1800
        rows = []
        # A: 1000 gens, T>0 30 flagged of 600, T=0 2 of 400; B: 1000 gens, 2 flagged
        for i in range(1000):
            temp = 0 if i < 400 else 0.7
            flagged = (i < 2) or (400 <= i < 430)
            rows.append(rec(i, t + i * 0.5, temp, 1 + i % 8, flagged, "A", split=int(i == 1)))
        for i in range(1000, 2000):
            rows.append(rec(i, t + i * 0.5, 0.7, 1 + i % 8, i < 1002, "B"))
        rng.shuffle(rows)
        rows.append({"v": 1, "rid": None, "t0": t, "path": "/v1/models", "status": 200})
        path = os.path.join(self.dir, time.strftime("requests-%Y%m%d-%H.jsonl", time.localtime()))
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def run_report(self, *args):
        return subprocess.run([sys.executable, os.path.join(ROOT, "report.py"), "--data", self.dir, *args],
                              capture_output=True, text=True, check=True).stdout

    def test_report(self):
        out = self.run_report("--since", "2h")
        self.assertIn("34/2000", out)
        rep = json.loads(read(os.path.join(self.dir, "report.json")))
        self.assertEqual(rep["summary"]["flagged"], 34)
        temp = rep["bins"]["temperature"]
        self.assertEqual((temp["T=0"]["n"], temp["T=0"]["k"]), (400, 2))
        self.assertEqual((temp["T>0"]["n"], temp["T>0"]["k"]), (1600, 32))
        self.assertEqual(set(rep["bins"]["k"]), {"k=5", "k=3"})
        self.assertEqual(rep["split"]["with_utf8_split"], 1)
        self.assertEqual(len(rep["examples"]), 20)
        md = read(os.path.join(self.dir, "REPORT.md"))
        self.assertIn("⟦閮⟧", md)  # flagged span highlighted
        self.assertIn("## Temperature (T=0 vs T>0)", md)
        self.assertLess(md.index("## Temperature"), md.index("## k in use"))
        self.assertLess(md.index("## k in use"), md.index("## Running requests"))
        p, lo, hi = report.wilson(34, 2000)
        self.assertAlmostEqual(p, 0.017)
        self.assertTrue(lo < 0.017 < hi)
        self.assertIn(f"{100 * lo:.2f}%", md)

    def test_reclassify_matches(self):
        self.run_report("--reclassify")
        rep = json.loads(read(os.path.join(self.dir, "report.json")))
        self.assertEqual(rep["summary"]["flagged"], 34)

    def test_compare(self):
        self.run_report("--compare", "label=A", "label=B")
        res = json.loads(read(os.path.join(self.dir, "compare.json")))
        row = res["rows"][0]
        self.assertEqual((row["A"], row["B"]), ([32, 1000], [2, 1000]))
        self.assertLess(row["ci"][1], 0)  # B significantly lower
        md = read(os.path.join(self.dir, "COMPARE.md"))
        self.assertIn(" *", md)
        # time-range windows parse too
        self.run_report("--compare", "-2h..-10m", "-10m..now")

    def test_old_hours_gzipped_and_still_read(self):
        import types
        import recorder
        (cur,) = [f for f in os.listdir(self.dir) if f.startswith("requests-")]
        old = time.strftime("requests-%Y%m%d-%H.jsonl", time.localtime(time.time() - 7200))
        os.rename(os.path.join(self.dir, cur), os.path.join(self.dir, old))
        os.utime(os.path.join(self.dir, old), (time.time() - 600, time.time() - 600))
        r = recorder.Recorder(types.SimpleNamespace(upstream="http://127.0.0.1:9", data=self.dir, label=None))
        r.compress_old()
        self.assertEqual(sorted(os.listdir(self.dir)), [old + ".gz"])
        self.assertIn("34/2000", self.run_report())

    def test_wilson_edges(self):
        self.assertEqual(report.wilson(0, 0), (0.0, 0.0, 1.0))
        p, lo, hi = report.wilson(0, 100)
        self.assertEqual((p, lo), (0.0, 0.0))
        self.assertAlmostEqual(hi, 0.037, places=3)


if __name__ == "__main__":
    unittest.main()
