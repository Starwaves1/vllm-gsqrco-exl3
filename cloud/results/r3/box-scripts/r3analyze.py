"""Round-3 analyzers (stdlib only).

  r3analyze.py instr  INSTR.jsonl [--t0 T --t1 T]      per-cycle host time by engine/scheduler/runner/
                                                     connector/sync bucket (sitecustomize timers)
  r3analyze.py trace  TRACE.json[.gz] [--steps N]      torch-profiler trace: per-step span, GPU busy/idle,
                                                     idle attributed to the host scope active at the
                                                     time, launch gaps inside the forward, D2H copies
                                                     and host syncs, kernel time by class and phase
  r3analyze.py pyspy  RAW.txt [--thread-match S]       py-spy --format raw: main-thread samples by bucket
                                                     and the top self frames
  r3analyze.py attn   LABEL=TRACE ...                  23: attention / GDN / GEMM ms per step per context
Each prints a report on stdout and, with --json FILE, writes the numbers.
"""

import argparse
import bisect
import gzip
import json
import re
import statistics
import sys
from collections import defaultdict

# ------------------------------------------------------------------ instr
# Buckets are disjoint top-level pieces of an engine cycle; the connector/tiering/sync lines are
# reported separately because they nest inside the top-level ones.
TOP = ["Scheduler.schedule", "GPUModelRunner.execute_model", "GPUModelRunner.sample_tokens",
       "Scheduler.update_from_output", "EngineCore.post_step", "EngineCoreProc._process_input_queue",
       "Scheduler.get_grammar_bitmask"]
NESTED = [
    ("runner: _update_states", "GPUModelRunner._update_states"),
    ("runner: _prepare_inputs", "GPUModelRunner._prepare_inputs"),
    ("runner: _build_attention_metadata", "GPUModelRunner._build_attention_metadata"),
    ("runner: _determine_batch_execution_and_padding", "GPUModelRunner._determine_batch_execution_and_padding"),
    ("runner: _preprocess", "GPUModelRunner._preprocess"),
    ("runner: _model_forward (launch)", "GPUModelRunner._model_forward"),
    ("runner: _sample", "GPUModelRunner._sample"),
    ("runner: propose_draft_token_ids", "GPUModelRunner.propose_draft_token_ids"),
    ("runner: _bookkeeping_sync", "GPUModelRunner._bookkeeping_sync"),
    ("runner: take_draft_token_ids", "GPUModelRunner.take_draft_token_ids"),
    ("sched: update_draft_token_ids", "Scheduler.update_draft_token_ids"),
    ("attn md: FlashInferMetadataBuilder.build (incl. seq_lens.cpu() wait)", "FlashInferMetadataBuilder.build"),
    ("attn md: GDNAttentionMetadataBuilder.build", "GDNAttentionMetadataBuilder.build"),
    ("drafter: propose", "SpecDecodeBaseProposer.propose"),
    ("drafter: per-pass metadata rebuild", "SpecDecodeBaseProposer.build_per_group_and_layer_attn_metadata"),
    ("drafter: _sample_draft_tokens", "SpecDecodeBaseProposer._sample_draft_tokens"),
    ("sync: cuda Event.synchronize", "Event.synchronize"),
    ("sync: cuda Stream.synchronize", "Stream.synchronize"),
]
CONNECTOR_PREFIX = ("OffloadingConnector.", "TieringOffloadingManager.")


def load_jsonl(path):
    rows, meta = [], None
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue  # last line of a killed process
            if r.get("meta"):
                meta = r
            else:
                rows.append(r)
    return meta, rows


def cmd_instr(a):
    meta, rows = load_jsonl(a.file)
    rows = [r for r in rows if r.get("exec") and (a.t0 is None or a.t0 <= r["t"] <= a.t1)]
    if a.nrun is not None:
        rows = [r for r in rows if r["nrun"] == a.nrun]
    if len(rows) < 10:
        sys.exit(f"instr: only {len(rows)} executed cycles in the window")
    n = len(rows)

    def ms(key):
        return [r["acc"].get(key, 0) / 1e6 for r in rows]

    def mean(v):
        return sum(v) / len(v)
    cyc = [r["cycle_ns"] / 1e6 for r in rows]
    cpu = [r["cycle_cpu_ns"] / 1e6 for r in rows]
    sync = [a_ + b_ for a_, b_ in zip(ms("Event.synchronize"), ms("Stream.synchronize"))]
    out = {"cycles": n, "cycle_ms_mean": mean(cyc), "cycle_ms_median": statistics.median(cyc),
           "cycle_cpu_ms_mean": mean(cpu), "sync_wait_ms_mean": mean(sync),
           "nrun": sorted({r["nrun"] for r in rows}), "missing_hooks": (meta or {}).get("missing")}
    lines = [f"instr: {n} executed cycles, nrun {out['nrun']}",
             f"  cycle (step start -> next step start)  mean {out['cycle_ms_mean']:.2f}  median {out['cycle_ms_median']:.2f} ms;"
             f" main-thread CPU {out['cycle_cpu_ms_mean']:.2f} ms (CUDA syncs spin, so CPU counts them)",
             f"  host blocked in CUDA event/stream sync: {out['sync_wait_ms_mean']:.2f} ms/cycle "
             f"(~GPU work the host waited for); cycle - sync = {out['cycle_ms_mean'] - out['sync_wait_ms_mean']:.2f} ms host-side",
             "  top-level (inclusive, ms/cycle mean):"]
    top_sum = 0.0
    for k in TOP:
        v = mean(ms(k))
        top_sum += v
        out[k] = v
        lines.append(f"    {k:45s} {v:7.3f}   (calls/cycle {sum(r['cnt'].get(k, 0) for r in rows) / n:.2f})")
    out["unaccounted_ms"] = out["cycle_ms_mean"] - top_sum
    out["step_call_ms_mean"] = sum(r.get("step_ns", 0) for r in rows) / n / 1e6
    lines.append(f"    {'(rest: unwrapped step() code, output queue, loop)':45s} {out['unaccounted_ms']:7.3f}")
    lines.append(f"    {'[EngineCore.step() call itself]':45s} {out['step_call_ms_mean']:7.3f}")
    lines.append("  nested (inclusive):")
    for label, k in NESTED:
        v = mean(ms(k))
        out[label] = v
        calls = sum(r["cnt"].get(k, 0) for r in rows) / n
        lines.append(f"    {label:70s} {v:7.3f}   (calls/cycle {calls:.2f})")
    conn = defaultdict(float)
    for r in rows:
        for k, v in r["acc"].items():
            if k.startswith(CONNECTOR_PREFIX):
                conn[k] += v / 1e6 / n
    out["connector"] = dict(conn)
    lines.append("  KV connector / tiering manager hooks (inclusive, ms/cycle; nested in the top-level ones):")
    for k, v in sorted(conn.items(), key=lambda x: -x[1]):
        lines.append(f"    {k:60s} {v:7.3f}")
    lines.append(f"    {'sum of OffloadingConnector.* entry points':60s} "
                 f"{sum(v for k, v in conn.items() if k.startswith('OffloadingConnector.')):7.3f}")
    print("\n".join(lines))
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


# ------------------------------------------------------------------ trace
def kclass(name: str) -> str:
    n = name.lower()
    if any(s in n for s in ("flashinfer", "batchprefill", "batchdecode", "paged_attention", "unified_attention",
                            "attention_kernel", "fmha", "flash_fwd", "mergestates", "merge_state")):
        return "attention"
    if any(s in n for s in ("gated_delta", "fused_recurrent", "chunk_", "causal_conv1d", "gdn", "fused_gdn",
                            "l2norm", "solve_tril", "recompute_w_u", "chunk_local_cumsum", "kkt")):
        return "gdn"
    if any(s in n for s in ("lcpp", "mul_mat", "mmq", "mmvq", "mma_k", "iq3", "owned", "quantize_q8", "q8_1",
                            "dequantize", "gguf")):
        return "gemm_plugin"
    if any(s in n for s in ("gemm", "cutlass", "xmma", "ampere_", "sm80_", "cublas", "s16816", "splitk")):
        return "gemm_cublas"
    if any(s in n for s in ("sampl", "topk", "top_k", "argmax", "softmax", "rejection", "multinomial", "exponential")):
        return "sampling"
    if any(s in n for s in ("rms_norm", "rmsnorm", "silu", "act_and_mul", "rotary", "rope", "fused_add")):
        return "norm_act_rope"
    return "other"


def load_trace(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        return json.load(f)["traceEvents"]


class Scopes:
    """Deepest user_annotation / python record_function scope on one thread, as a step function."""

    def __init__(self, evs):
        evs = sorted(evs, key=lambda e: (e["ts"], -e["dur"]))
        bounds, stack = [], []  # (time, name-after)
        for e in evs:
            s, f = e["ts"], e["ts"] + e["dur"]
            while stack and stack[-1][1] <= s:
                top = stack.pop()
                bounds.append((top[1], stack[-1][0] if stack else None))
            stack.append((e["name"], f))
            bounds.append((s, e["name"]))
        while stack:
            top = stack.pop()
            bounds.append((top[1], stack[-1][0] if stack else None))
        bounds.sort(key=lambda x: x[0])
        self.t = [b[0] for b in bounds]
        self.n = [b[1] for b in bounds]

    def at(self, t):
        i = bisect.bisect_right(self.t, t) - 1
        return self.n[i] if i >= 0 else None

    def segments(self, a, b):
        """[(start, end, scope)] covering [a, b]."""
        i = max(bisect.bisect_right(self.t, a) - 1, 0)
        out, cur = [], a
        while cur < b:
            name = self.n[i] if i < len(self.n) and self.t[i] <= cur else None
            nxt = self.t[i + 1] if i + 1 < len(self.t) else b
            end = min(b, nxt) if nxt > cur else b
            out.append((cur, end, name))
            cur = end
            i += 1
            if i >= len(self.t):
                if cur < b:
                    out.append((cur, b, None))
                break
        return out


def norm_scope(s):
    if s is None:
        return "(no scope: engine loop / unannotated)"
    if s.startswith("execute_context"):
        return "execute_model (unscoped part)"
    return re.sub(r"\(.*", "", s)


def analyze_trace(path, max_steps=None):
    ev = load_trace(path)
    steps = sorted((e for e in ev if e.get("cat") == "user_annotation" and e.get("name", "").startswith("execute_context")),
                   key=lambda e: e["ts"])
    if len(steps) < 3:
        sys.exit(f"trace {path}: only {len(steps)} execute_context annotations")
    main_tid = steps[0]["tid"]
    ann = [e for e in ev if e.get("tid") == main_tid and e.get("ph") == "X"
           and e.get("cat") in ("user_annotation", "python_function") and "dur" in e]
    scopes = Scopes(ann)
    rt = {e["args"]["correlation"]: e for e in ev
          if e.get("cat") in ("cuda_runtime", "cuda_driver") and "correlation" in e.get("args", {})}
    gpu = sorted((e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and e.get("ph") == "X"),
                 key=lambda e: e["ts"])
    rt_main = sorted((e for e in rt.values() if e.get("tid") == main_tid), key=lambda e: e["ts"])
    bounds = [s["ts"] for s in steps]
    nsteps = len(steps) - 1
    if max_steps:
        nsteps = min(nsteps, max_steps)
    per = []
    idle_by_scope = defaultdict(float)
    kern = defaultdict(float)  # (phase, class) -> us
    for i in range(nsteps):
        a, b = bounds[i], bounds[i + 1]
        g = [e for e in gpu if e["ts"] < b and e["ts"] + e["dur"] > a]
        busy, cur, gaps = 0.0, a, []
        for e in g:
            s, f = max(e["ts"], a), min(e["ts"] + e["dur"], b)
            if s > cur:
                gaps.append((cur, s))
            if f > cur:
                busy += f - max(s, cur)
                cur = f
        if cur < b:
            gaps.append((cur, b))
        for (x, y) in gaps:
            for (s, f, nm) in scopes.segments(x, y):
                idle_by_scope[norm_scope(nm)] += f - s
        d2h = [e for e in g if e.get("cat") == "gpu_memcpy" and "dtoh" in e["name"].lower().replace(" ", "")]
        rts = [e for e in rt_main if a <= e["ts"] < b]
        syncs = [e for e in rts if "Synchronize" in e["name"]]
        glaunch = sum(1 for e in rts if e["name"] == "cudaGraphLaunch")
        klaunch = sum(1 for e in rts if e["name"] in ("cudaLaunchKernel", "cuLaunchKernel", "cuLaunchKernelEx",
                                                       "cudaLaunchKernelExC"))
        for e in g:
            if e.get("cat") != "kernel":
                continue
            c = e.get("args", {}).get("correlation")
            l = rt.get(c)
            ph = norm_scope(scopes.at(l["ts"])) if l else "?"
            kern[(ph, kclass(e["name"]))] += e["dur"]
        per.append({"span": b - a, "busy": busy, "idle": (b - a) - busy, "gaps": len(gaps),
                    "d2h_n": len(d2h), "d2h_us": sum(e["dur"] for e in d2h),
                    "sync_n": len(syncs), "sync_host_us": sum(e["dur"] for e in syncs),
                    "graph_launches": glaunch, "kernel_launches": klaunch, "name": steps[i]["name"]})
    n = len(per)

    def m(k):
        return sum(p[k] for p in per) / n / 1000.0
    res = {"trace": path, "steps": n, "step_names": sorted({p["name"] for p in per}),
           "span_ms": m("span"), "busy_ms": m("busy"), "idle_ms": m("idle"),
           "idle_by_host_scope_ms": {k: v / n / 1000.0 for k, v in sorted(idle_by_scope.items(), key=lambda x: -x[1])},
           "d2h_per_step": sum(p["d2h_n"] for p in per) / n, "d2h_ms": m("d2h_us"),
           "host_syncs_per_step": sum(p["sync_n"] for p in per) / n, "host_sync_ms": m("sync_host_us"),
           "graph_launches_per_step": sum(p["graph_launches"] for p in per) / n,
           "kernel_launches_per_step": sum(p["kernel_launches"] for p in per) / n,
           "kernel_ms_by_phase_class": {f"{ph} | {c}": v / n / 1000.0
                                        for (ph, c), v in sorted(kern.items(), key=lambda x: -x[1])}}
    cls = defaultdict(float)
    for (ph, c), v in kern.items():
        cls[c] += v / n / 1000.0
    res["kernel_ms_by_class"] = dict(sorted(cls.items(), key=lambda x: -x[1]))
    return res


def print_trace(res):
    print(f"trace {res['trace']}: {res['steps']} steps ({', '.join(res['step_names'][:3])})")
    print(f"  per step: span {res['span_ms']:.2f}  GPU busy {res['busy_ms']:.2f}  GPU idle {res['idle_ms']:.2f} ms"
          "  (torch profiler inflates host time; compare idle shares, not absolutes)")
    print(f"  launches/step: {res['graph_launches_per_step']:.0f} cudaGraphLaunch, {res['kernel_launches_per_step']:.0f} eager kernels;"
          f"  D2H copies {res['d2h_per_step']:.1f} ({res['d2h_ms']:.3f} ms GPU);"
          f"  host sync calls {res['host_syncs_per_step']:.1f} ({res['host_sync_ms']:.2f} ms host)")
    print("  GPU idle by host scope at the time (ms/step):")
    for k, v in res["idle_by_host_scope_ms"].items():
        if v >= 0.05:
            print(f"    {k:60s} {v:7.2f}")
    print("  kernel time by class (ms/step):")
    for k, v in res["kernel_ms_by_class"].items():
        print(f"    {k:20s} {v:7.2f}")
    print("  kernel time by launching scope | class (ms/step, top 15):")
    for k, v in list(res["kernel_ms_by_phase_class"].items())[:15]:
        print(f"    {k:70s} {v:7.2f}")


def cmd_trace(a):
    res = analyze_trace(a.file, a.steps)
    print_trace(res)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


def cmd_attn(a):
    rows = {}
    for spec in a.traces:
        label, path = spec.split("=", 1)
        rows[label] = analyze_trace(path, a.steps)
    classes = ["attention", "gdn", "gemm_plugin", "gemm_cublas", "norm_act_rope", "sampling", "other"]
    print(f"{'context':14s} {'span':>7s} {'busy':>7s} {'idle':>7s} " + " ".join(f"{c:>12s}" for c in classes))
    for label, r in rows.items():
        k = r["kernel_ms_by_class"]
        print(f"{label:14s} {r['span_ms']:7.2f} {r['busy_ms']:7.2f} {r['idle_ms']:7.2f} "
              + " ".join(f"{k.get(c, 0.0):12.2f}" for c in classes))
    print("attention split by launching scope (target forward vs MTP draft passes), ms/step:")
    for label, r in rows.items():
        parts = {k.split(" | ")[0]: v for k, v in r["kernel_ms_by_phase_class"].items() if k.endswith("| attention")}
        print(f"  {label:12s} " + "  ".join(f"{p}: {v:.2f}" for p, v in sorted(parts.items(), key=lambda x: -x[1])))
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1)


# ------------------------------------------------------------------ py-spy
PYSPY_BUCKETS = [
    ("cuda sync wait (GPU busy)", ("synchronize", "cudaEventSynchronize", "cudaStreamSynchronize", "_to_list",
                                   "async_copy_ready_event")),
    ("KV connector / tiering", ("offloading", "kv_offload", "kv_connector", "tiering")),
    ("model forward launch (graphs + eager attn/GDN)", ("_model_forward", "cuda_graph", "piecewise", "flashinfer",
                                                        "gdn", "forward (vllm/model_executor")),
    ("MTP drafter", ("spec_decode", "propose")),
    ("scheduler", ("sched/scheduler.py", "kv_cache_manager", "block_pool", "single_type_kv_cache")),
    ("runner input prep / bookkeeping", ("_update_states", "_prepare_inputs", "_build_attention_metadata",
                                         "_bookkeeping_sync", "gpu_input_batch", "block_table")),
    ("engine loop / output / ZMQ / msgspec", ("engine/core.py", "msgspec", "zmq", "serial_utils")),
    ("idle / waiting for input", ("_process_input_queue", "queue.py", "wait (threading.py")),
]


def cmd_pyspy(a):
    tot = defaultdict(int)
    selfc = defaultdict(int)
    threads = defaultdict(int)
    n = 0
    for line in open(a.file):
        if line.strip() and " " in line:
            th = line.split(";", 1)[0]
            try:
                threads[th] += int(line.rsplit(" ", 1)[1])
            except ValueError:
                pass
        line = line.rstrip("\n")
        if not line or " " not in line:
            continue
        stack, cnt = line.rsplit(" ", 1)
        try:
            cnt = int(cnt)
        except ValueError:
            continue
        frames = stack.split(";")
        thread = next((f for f in frames if f.startswith("thread")), "")
        if a.thread_match and a.thread_match not in thread and a.thread_match not in stack.split(";")[0]:
            continue
        n += cnt
        for label, keys in PYSPY_BUCKETS:
            if any(k in stack for k in keys):
                tot[label] += cnt
                break
        else:
            tot["other"] += cnt
        selfc[frames[-1]] += cnt
    if n == 0:
        sys.exit("pyspy: no samples matched")
    allc = sum(threads.values())
    print(f"py-spy {a.file}: {allc} samples over {len(threads)} threads; busiest: "
          + ", ".join(f"{t} {100.0 * c / allc:.0f}%" for t, c in sorted(threads.items(), key=lambda x: -x[1])[:6]))
    print(f"  bucket shares below: {n} samples{' of ' + a.thread_match if a.thread_match else ''} (first matching bucket wins)")
    for k, v in sorted(tot.items(), key=lambda x: -x[1]):
        print(f"  {k:52s} {100.0 * v / n:5.1f} %")
    print("  top self frames:")
    for k, v in sorted(selfc.items(), key=lambda x: -x[1])[:25]:
        print(f"    {100.0 * v / n:5.1f} %  {k[:150]}")
    if a.json:
        json.dump({"samples": n, "buckets": tot, "top_self": dict(sorted(selfc.items(), key=lambda x: -x[1])[:50])},
                  open(a.json, "w"), indent=1)


SEGMENTS = [  # (name, test on the ';'-joined stack), first match wins; mirrors prod-sample-20261001.md 3b
    ("drain wait: seq_lens.cpu() at the first draft pass", lambda st: "flashinfer.py:1527" in st and "llm_base_proposer.py:568" in st),
    ("draft-loop seq_lens.cpu() waits (passes 2..k)", lambda st: "flashinfer.py:1527" in st and "llm_base_proposer.py:714" in st),
    ("seq_lens.cpu() wait elsewhere (target metadata)", lambda st: "flashinfer.py:1527" in st),
    ("target forward: eager GDN op", lambda st: "_model_forward" in st and "qwen_gdn_attention_core" in st),
    ("target forward: graph-piece replays", lambda st: "_model_forward" in st and "cuda_graph.py:256" in st),
    ("target forward: stitching + other eager ops", lambda st: "_model_forward" in st),
    ("drafter host work (outside waits)", lambda st: "llm_base_proposer.py" in st or "propose_draft_token_ids" in st
     or "_copy_draft_token_ids_to_cpu" in st),
    ("sampler + rejection sampler", lambda st: "_sample (" in st or "rejection_sampler" in st or "topk_topp" in st),
    ("runner prep (metadata, inputs, execute_model own)", lambda st: "execute_model (" in st or "_prepare_inputs" in st
     or "_build_attention_metadata" in st or "_update_states" in st),
    ("scheduler / update_from_output / post_step / loop", lambda st: "sched/scheduler.py" in st or "post_step" in st
     or "engine/core.py" in st),
]


def cmd_segments(a):
    """py-spy raw (--threads): the busiest thread's samples by step segment, in ms/step given the
    measured ms/step, plus the off-CPU share (samples / (rate x duration) without --idle)."""
    per_thread = defaultdict(lambda: defaultdict(int))
    for line in open(a.file):
        line = line.rstrip("\n")
        if not line or " " not in line:
            continue
        stack, cnt = line.rsplit(" ", 1)
        try:
            cnt = int(cnt)
        except ValueError:
            continue
        th = stack.split(";", 1)[0]
        name = next((n for n, f in SEGMENTS if f(stack)), "other")
        per_thread[th][name] += cnt
    th, segs = max(per_thread.items(), key=lambda kv: sum(kv[1].values()))
    tot = sum(segs.values())
    ticks = a.rate * a.seconds
    out = {"thread": th, "samples": tot, "ticks": ticks, "on_cpu_share": tot / ticks if ticks else None,
           "ms_per_step": a.ms_per_step, "segments": {}}
    print(f"{a.file}: main thread {th}: {tot} samples of {ticks:.0f} ticks "
          f"({100 * tot / ticks:.0f} % on-CPU / sampled), ms/step {a.ms_per_step}")
    for n in [n for n, _ in SEGMENTS] + ["other"]:
        c = segs.get(n, 0)
        ms = a.ms_per_step * c / tot if tot else 0.0
        out["segments"][n] = {"samples": c, "pct": 100.0 * c / tot if tot else 0.0, "ms_per_step": ms}
        print(f"  {n:58s} {c:6d}  {100.0 * c / tot if tot else 0:5.1f} %  {ms:6.2f} ms/step")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("instr")
    p.add_argument("file")
    p.add_argument("--t0", type=float)
    p.add_argument("--t1", type=float)
    p.add_argument("--nrun", type=int)
    p.add_argument("--json")
    p = sub.add_parser("trace")
    p.add_argument("file")
    p.add_argument("--steps", type=int)
    p.add_argument("--json")
    p = sub.add_parser("attn")
    p.add_argument("traces", nargs="+")
    p.add_argument("--steps", type=int)
    p.add_argument("--json")
    p = sub.add_parser("pyspy")
    p.add_argument("file")
    p.add_argument("--thread-match", default="")
    p.add_argument("--json")
    p = sub.add_parser("segments")
    p.add_argument("file")
    p.add_argument("--ms-per-step", type=float, required=True)
    p.add_argument("--rate", type=float, default=250)
    p.add_argument("--seconds", type=float, default=30)
    p.add_argument("--json")
    a = ap.parse_args()
    {"instr": cmd_instr, "trace": cmd_trace, "attn": cmd_attn, "pyspy": cmd_pyspy, "segments": cmd_segments}[a.cmd](a)


if __name__ == "__main__":
    main()
