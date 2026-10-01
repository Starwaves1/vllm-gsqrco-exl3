"""bench/torture (standalone harness): schedule, classifier, throttle, monitor and report verdicts on
synthetic data, and whole runs against a mock OpenAI server (standard library only, no GPU)."""

import json
import os
import sys
import time

import pytest

HERE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "bench", "torture")
sys.path.insert(0, HERE)
import load as tl  # noqa: E402
from torture_mock import Mock  # noqa: E402
import monitor as tm  # noqa: E402
import report as tr  # noqa: E402

H12 = 12 * 3600


def test_plan_is_deterministic_and_seeded():
    assert tl.make_plan(H12, 7) == tl.make_plan(H12, 7)
    assert tl.make_plan(H12, 7) != tl.make_plan(H12, 8)


@pytest.mark.parametrize("total", [H12, 20 * 60, 10 * 60])
def test_plan_fills_the_run_and_covers_every_phase_type(total):
    plan = tl.make_plan(total, 20261001)
    assert {p["type"] for p in plan} == set(tl.TYPES) | {"idle"} == set(tl.PHASES)
    assert abs(sum(p["dur"] for p in plan) - total) < 1
    for a, b in zip(plan, plan[1:]):
        assert b["start"] == pytest.approx(a["start"] + a["dur"], abs=0.2)


def test_plan_12h_durations():
    plan = tl.make_plan(H12, 20261001)
    for p in plan[:-1]:  # the last one is cut to fit
        lo, hi = (120, 300) if p["type"] == "idle" else (600, 1200)
        assert lo <= p["dur"] <= hi, p
    assert sum(p["type"] == "idle" for p in plan) >= 10


def req(**body):
    return {"kind": "chat", "body": {"max_tokens": 256, **body}, "planned_timeout": False}


def res(finish="stop", text="hello", reasoning="", tools=(), ct=10, pt=20, n=1, **kw):
    return {"http": 200, "usage": {"prompt_tokens": pt, "completion_tokens": ct},
            "choices": [{"finish": finish, "text": text, "reasoning": reasoning, "tools": list(tools)}] * n, **kw}


SCHEMA_RF = {"type": "json_schema", "json_schema": {"name": "city", "schema": tl.CITY}}


@pytest.mark.parametrize("q,r,want", [
    (req(), res(), "ok"),
    (req(), res(cancelled=True, http=None), "cancelled"),
    ({**req(), "planned_timeout": True}, {"timed_out": True}, "client_timeout"),
    (req(), {"timed_out": True}, "stalled"),
    (req(), {"error": "ConnectionResetError()"}, "conn_error"),
    (req(), {"http": 500}, "http_5xx"),
    (req(), {"http": 400}, "http_4xx"),
    (req(), res(stream_error="boom"), "stream_error"),
    (req(), res(finish="length", text="", reasoning="thinking...", ct=256), "reasoning_only"),
    (req(), res(text="", ct=1), "eos_first"),
    (req(), res(finish="length", text=" \n", ct=256), "empty_output"),  # the soak's unexplained empties
    (req(max_tokens=4), res(finish="length", text="", ct=4), "short_empty"),
    (req(), res(finish="length", ct=255), "bad_count"),
    (req(), res(ct=300), "bad_count"),
    ({**req(), "expect_prompt_tokens": 21}, res(), "bad_count"),
    (req(min_tokens=32, max_tokens=64), res(ct=10), "bad_count"),
    (req(), res(finish="abort"), "bad_finish"),
    (req(), res(lp_bad=True), "bad_logprobs"),
    (req(prompt_logprobs=1, max_tokens=1), res(ct=1, plp_len=19), "bad_logprobs"),
    (req(prompt_logprobs=1, max_tokens=1), res(ct=1, plp_len=20), "ok"),
    (req(tool_choice="auto"), res(finish="tool_calls", text="", tools=[("read_file", '{"path": "a.py"}')]), "ok"),
    (req(tool_choice="auto"), res(finish="tool_calls", text="", tools=[("read_file", '{"path": ')]), "bad_tool_json"),
    (req(tool_choice="required"), res(), "no_tool_call"),
    (req(tool_choice="none"), res(tools=[("read_file", "{}")]), "bad_tool_choice"),
    (req(response_format=SCHEMA_RF), res(text='{"city": "Copenhagen", "population": 660000}'), "ok"),
    (req(response_format=SCHEMA_RF), res(text='{"city": "Copenhagen"}'), "bad_json"),
    (req(response_format={"type": "json_object"}), res(text="not json"), "bad_json"),
    (req(response_format=SCHEMA_RF), res(finish="length", text='{"ci', ct=256), "truncated_json"),
    (req(n=2), res(n=2, ct=20), "ok"),
    # seen on the first real smoke: EOS inside the thinking before the grammar or the tool call began
    (req(response_format=SCHEMA_RF), res(text="", reasoning="Let me just give the JSON directly.", ct=127), "reasoning_only"),
    (req(tool_choice="required"), res(text="", reasoning="I should run the tests.", ct=40), "reasoning_only"),
    # vLLM main parses a tool call out of the content and then drops it under tool_choice "none"
    (req(tool_choice="none"), res(text="", ct=26), "dropped_tool_call"),
    (req(tool_choice="none"), res(finish="length", text="", ct=256), "empty_output"),
    # raw completions may continue code with blank lines (smoke 2); a chat answer of blank lines may not
    ({**req(max_tokens=128), "path": "/completions"}, res(finish="length", text="\n" * 128, ct=128), "whitespace_completion"),
    ({**req(max_tokens=128), "path": "/chat/completions"}, res(finish="length", text="\n" * 128, ct=128), "empty_output"),
    (req(n=2), res(n=1), "bad_response"),
])
def test_classify(q, r, want):
    assert tl.classify(q, r) == want


def test_ok_classes_are_client_or_model_outcomes():
    assert tl.OK_CLASSES == {"ok", "reasoning_only", "eos_first", "short_empty", "truncated_json", "dropped_tool_call",
                             "whitespace_completion", "cancelled", "client_timeout"}


# ---------------------------------------------------------------- report on a synthetic 12 h run

def write_run(d, hours=12.0, fault=None, empty=False, drift=0.0, gpu_leak=0.0):
    total = hours * 3600
    plan = tl.make_plan(total, 1)
    t0 = 1_800_000_000.0
    (d / "plan.json").write_text(json.dumps({"seed": 1, "total_s": total, "phases": plan}))
    cols = "t,server_alive,health,gpu_mib,rss_kib,running,waiting,kv_usage,preemptions,restarts,rss_anon_kib,shmem_kib,cpu_tier_perc,fs_tier_bytes"
    rows = [cols] + [f"{t0 + 60 * k:.0f},1,200,{23480 + gpu_leak * k / 60:.0f},{31_000_000 + k},2,0,0.3,0,0,{3_000_000 + k % 7},26000000,0.5,1000"
                     for k in range(int(total // 60) + 1)]
    (d / "monitor.csv").write_text("\n".join(rows) + "\n")
    (d / "faults.log").write_text(fault + "\n" if fault else "")
    recs, phases = [], []
    for p in plan:
        s = t0 + p["start"]
        late = p["start"] > total - 3600
        phases.append({**p, "t0": s, "t_end": s + p["dur"], "t_drained": s + p["dur"] + 1, "planned_hit": 0.2,
                       "metrics0": {"vllm:prefix_cache_hits": 0, "vllm:prefix_cache_queries": 0},
                       "metrics1": {"vllm:prefix_cache_hits": 50, "vllm:prefix_cache_queries": 100}})
        recs.append({"t": s, "phase": p["i"], "ptype": p["type"], "kind": "probe", "class": "ok", "ttft": 0.05,
                     "tpot": 0.009 * (1 + drift if late else 1), "text_sha": "abc", "after_idle": False})
        for j in range(5):
            recs.append({"t": s + j, "phase": p["i"], "ptype": p["type"], "kind": "chat", "class": "ok", "ttft": 0.2, "tpot": 0.02})
        recs.append({"t": s + 6, "phase": p["i"], "ptype": p["type"], "kind": "chat", "class": "cancelled"})
    if empty:
        recs.append({"t": t0 + 9000, "phase": 3, "ptype": plan[3]["type"], "kind": "chat", "class": "empty_output", "text": [" "]})
    (d / "load.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    (d / "phases.jsonl").write_text("".join(json.dumps(p) + "\n" for p in phases))
    return d


def verdict(d):
    rep = tr.run_report(d)
    return rep["pass"], {k for k, c in rep["criteria"].items() if not c["pass"]}


def test_report_clean_run_passes(tmp_path):
    ok, failed = verdict(write_run(tmp_path))
    assert ok and not failed
    tr.markdown(tr.run_report(tmp_path))  # renders


@pytest.mark.parametrize("kw,crit", [
    ({"fault": "(EngineCore pid=1) CUDA error: an illegal memory access was encountered"}, "faults"),
    ({"empty": True}, "errors"),
    ({"drift": 0.3}, "drift"),  # probe TPOT +30% in the last hour
    ({"gpu_leak": 60.0}, "memory"),  # 60 MiB/h: ~660 MiB after warm-up
])
def test_report_injected_failures(tmp_path, kw, crit):
    ok, failed = verdict(write_run(tmp_path, **kw))
    assert not ok and failed == {crit}


def test_report_health_misses_and_blank_cells(tmp_path):
    d = write_run(tmp_path)
    lines = (d / "monitor.csv").read_text().splitlines()
    cells = [x.split(",") for x in lines[1:]]
    cells[100][2] = "0"  # one /health timeout: tolerated
    cells[200][3] = ""   # one blank GPU cell: skipped, the column is still judged
    (d / "monitor.csv").write_text("\n".join([lines[0]] + [",".join(c) for c in cells]) + "\n")
    rep = tr.run_report(d)
    assert rep["pass"] and rep["criteria"]["memory"]["value"]["gpu_mib_growth"] is not None
    cells[101][2] = "0"  # two in a row: not tolerated
    (d / "monitor.csv").write_text("\n".join([lines[0]] + [",".join(c) for c in cells]) + "\n")
    assert verdict(d) == (False, {"alive"})


def test_api_takes_plain_http_with_a_port_only():
    for bad in ("https://127.0.0.1:8000/v1", "http://127.0.0.1/v1"):
        with pytest.raises(SystemExit):
            tl.Api(bad, model="m")


def test_report_server_death_fails_alive(tmp_path):
    d = write_run(tmp_path)
    lines = (d / "monitor.csv").read_text().splitlines()
    (d / "monitor.csv").write_text("\n".join(lines[:200]) + "\n")  # monitor stopped after 3.3 h
    ok, failed = verdict(d)
    assert not ok and "alive" in failed


def test_switch_report(tmp_path):
    legs = []
    for r in (1, 2):
        for side, ns in (("a", "gsq_1"), ("b", "exl3_2")):
            name = f"r{r}-{side}"
            (tmp_path / name).mkdir()
            write_run(tmp_path / name, hours=10 / 60)
            legs.append({"round": r, "side": side, "dir": name, "serve": f"/x/serve-{side}.sh", "healthy_s": 300, "killed": 0,
                         "leftover_procs": 0, "port_free": 1, "gpu_idle_mib": 300, "gpu_mib_after": 310, "shm_left": [],
                         "tier_dirs_touched": [ns]})
    (tmp_path / "switch.jsonl").write_text("".join(json.dumps(x) + "\n" for x in legs))
    assert tr.switch_report(tmp_path)["pass"]
    legs[3]["shm_left"] = ["vllm_offload_1.mmap"]
    legs[2]["tier_dirs_touched"] = ["exl3_2"]  # model A wrote into B's namespace
    (tmp_path / "switch.jsonl").write_text("".join(json.dumps(x) + "\n" for x in legs))
    rep = tr.switch_report(tmp_path)
    assert not rep["pass"]
    assert rep["criteria"]["namespaces_disjoint"]["value"] == ["exl3_2"]
    assert [x["pass"] for x in rep["legs"]] == [True, True, True, False]
    tr.markdown(rep)


def test_throttle_caps_requests_and_tokens():
    import threading

    th, peak, lock = tl.Throttle(max_conc=3, budget=1000, off=False), {"n": 0, "tok": 0, "max_n": 0, "max_tok": 0}, threading.Lock()

    def go(c):
        th.acquire(c)
        with lock:
            peak["n"] += 1
            peak["tok"] += c
            peak["max_n"], peak["max_tok"] = max(peak["max_n"], peak["n"]), max(peak["max_tok"], peak["tok"])
        time.sleep(0.02)
        with lock:
            peak["n"] -= 1
            peak["tok"] -= c
        th.release(c)

    ts = [threading.Thread(target=go, args=(c,)) for c in [300] * 12 + [5000]]  # 5000 > budget: runs alone
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert peak["max_n"] == 3 and peak["max_tok"] <= 5000 and peak["n"] == 0


# ---------------------------------------------------------------- end to end against a mock server

@pytest.fixture
def mock(monkeypatch):
    monkeypatch.setattr(tm, "gpu_mib", lambda pids=None: None)  # never query the GPU from CPU tests
    m = Mock(0)
    yield m
    m.srv.shutdown()


def test_every_request_kind_round_trips(mock):
    api = tl.Api(mock.base)
    assert (api.model, api.mml, api.root) == ("mock", 200000, "")
    corpus = tl.Corpus(api, 3, 196064)
    assert len(corpus.body) == 196064
    gen = tl.Gen(corpus, 3, api.model, api.mml, 100000)
    seen = {}
    for i, ty in enumerate(tl.TYPES):
        gen.set_phase(i, tl.PHASES[ty])
        for mix, _ in tl.PHASES[ty]["lanes"]:
            for _ in range(25):
                q = gen.next(mix)
                q["timeout"] = min(q["timeout"], 30)
                rec = tl.record(q, tl.send(api, q), {"i": i, "type": ty})
                seen[rec["class"]] = seen.get(rec["class"], 0) + 1
                assert rec["class"] in tl.OK_CLASSES, rec
    assert seen.get("ok", 0) > 100 and seen.get("cancelled", 0) > 0, seen


def test_run_with_monitor_end_to_end(mock, tmp_path):
    log = tmp_path / "server.log"
    log.write_text("old line: Traceback before the run is not counted\n")
    api = tl.Api(mock.base)
    mon = tm.Monitor(api, tmp_path, server_pid=os.getpid(), server_log=str(log), interval=5)
    mon.start()
    log.open("a").write("(EngineCore pid=1) RuntimeError: CUDA error: an illegal memory access was encountered\n")
    tl.run(api, tmp_path, 36, seed=5, stop=mon.dead)
    mon.stop()
    recs = tr.jsonl(tmp_path / "load.jsonl")
    phases = tr.jsonl(tmp_path / "phases.jsonl")
    assert phases and all(p["metrics1"]["vllm:prefix_cache_queries"] == 10 for p in phases)
    assert {r["kind"] for r in recs} >= {"probe", "tokenize", "metrics"}
    assert all(r["class"] in tl.OK_CLASSES for r in recs), [r for r in recs if r["class"] not in tl.OK_CLASSES][:3]
    rows = (tmp_path / "monitor.csv").read_text().splitlines()
    assert rows[0].split(",") == tm.COLUMNS and len(rows) >= 3 and int(rows[1].split(",")[4]) > 0  # rss of this process
    assert (tmp_path / "faults.log").read_text().splitlines() == [
        "(EngineCore pid=1) RuntimeError: CUDA error: an illegal memory access was encountered"]
    plan = json.loads((tmp_path / "plan.json").read_text())
    assert (plan["kv_tokens"], plan["token_budget"], plan["max_conc"]) == (160000, 80000, 12)  # half the KV by default
    rep = tr.write(tmp_path)
    assert rep["criteria"]["errors"]["pass"] and not rep["criteria"]["faults"]["pass"]


def test_monitor_stops_the_run_on_a_restart(mock, tmp_path):
    api = tl.Api(mock.base)
    mon = tm.Monitor(api, tmp_path, interval=1)
    mon.start()
    time.sleep(1.5)
    mock.start_time = 2.0  # the server came back as a new process
    assert mon.dead.wait(5)
    mon.stop()
    assert "server restarted" in (tmp_path / "faults.log").read_text()


def test_cli_plan_and_prod_guard(tmp_path):
    import subprocess

    exe = os.path.join(HERE, "torture")
    out = subprocess.run([sys.executable, exe, "--plan", "--minutes", "20"], capture_output=True, text=True, check=True).stdout
    assert all(t in out for t in tl.TYPES) and "idle" in out
    r = subprocess.run([sys.executable, exe, "--base-url", "http://127.0.0.1:18081/v1", "--brutal", "--out", str(tmp_path)],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "--i-know" in r.stderr


def test_cli_switch_with_mock_servers(tmp_path):
    import socket
    import subprocess

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    mock = os.path.join(os.path.dirname(os.path.abspath(__file__)), "torture_mock.py")
    tier = tmp_path / "tier"
    cmd = [sys.executable, os.path.join(HERE, "torture"), "switch", "--port", str(port), "--rounds", "1", "--minutes", "0.35",
           "--tier-root", str(tier), "--out", str(tmp_path / "run"),
           "--a", f"{sys.executable} {mock} {port} {tier}/gsq_ns", "--b", f"{sys.executable} {mock} {port} {tier}/exl3_ns"]
    env = {**os.environ, "PATH": str(tmp_path)}  # no nvidia-smi: the GPU is never queried
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=200)
    legs = tr.jsonl(tmp_path / "run" / "switch.jsonl")
    assert [x["side"] for x in legs] == ["a", "b"], r.stdout + r.stderr
    for x, ns in zip(legs, ("gsq_ns", "exl3_ns")):
        assert x["healthy_s"] is not None and x["killed"] == 0 and x["leftover_procs"] == 0 and x["port_free"] == 1
        assert x["tier_dirs_touched"] == [ns] and x["gpu_mib_after"] is None
    rep = json.loads((tmp_path / "run" / "report.json").read_text())
    assert rep["criteria"]["namespaces_disjoint"]["pass"] and all(x["checks"]["run"] for x in rep["legs"]), r.stdout
    assert r.returncode == 0 and rep["pass"]


def test_skip_never_sends_the_skipped_kinds():
    class Corpus:  # ids only matter for length here
        def prompt(self, n, salt):
            return [1] * n

        def tail(self, h):
            return [2] * 8

    gen = tl.Gen(Corpus(), 1, "m", 200000, 100000, skip={"plog", "echo"})
    seen = set()
    for i, ty in enumerate(tl.TYPES):
        gen.set_phase(i, tl.PHASES[ty])
        for mix, _ in tl.PHASES[ty]["lanes"]:
            for _ in range(200):
                q = gen.next(mix)
                seen.add(q["kind"])
                assert not q["body"].get("prompt_logprobs") and not q["body"].get("echo")
    assert "plog" not in seen and {"chat", "long", "api"} <= seen
