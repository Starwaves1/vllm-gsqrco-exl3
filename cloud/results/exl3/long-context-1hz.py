"""Analysis behind long-context-cost-model.md section 1: the prod 1 Hz recorder log
(orchestrator-handoff/evidence/prod-exl3-deepswe-20261010/metrics-1hz.log, rec.sh format
`epoch|power,sm_mhz,util,mem,temp|vllm:k=v,...`), cut into 10 s windows.

  python long-context-1hz.py metrics-1hz.log        (needs numpy)

Per window: steps = d(iteration_tokens_total_count) (steps that produced output tokens only),
decoding requests per step = d(spec_decode_num_drafts_total) / steps (one draft per decoding
request per step at fixed k), MTP accepted per draft = d(accepted) / d(drafts). A window is
decode-only when the running gauge is constant, every running request decoded in every step
(drafts/step >= running - 0.02) and no prompt completed in it. Context proxy = kv_cache_usage x
the GPU KV pool. Segments: EXL3 3.50 bpw until 2026-10-11 02:13 ET, 4.50 bpw from 03:29 ET.
"""
import sys
import numpy as np

KEYS = ["spec_decode_num_drafts_total", "spec_decode_num_accepted_tokens_total", "num_requests_running",
        "num_requests_waiting", "kv_cache_usage_perc", "prompt_tokens_by_source_total_local_compute",
        "prompt_tokens_by_source_total_local_cache_hit", "prompt_tokens_by_source_total_external_kv_transfer",
        "generation_tokens_total", "iteration_tokens_total_count"]
DR, AC, RUN, WAIT, KV, LC, CH, EX, GEN, IC = range(6, 16)
SEGS = {"3.50 bpw": (0, 1791699180, 281885, {1: 26.82, 2: 26.91, 3: 29.3, 4: 31.76}),
        "4.50 bpw": (1791703740, 2e10, 194361, {1: 30.90, 2: 31.02, 3: 33.2, 4: 35.34})}  # ladder ms/step (+1.0 bf16 head)
SLOPE = 0.097  # ms per 1k context tokens per step, k=3 (job 23 per-call kernel times, see the doc)


def parse(path):
    rows = []
    for line in open(path):
        p = line.rstrip("\n").split("|")
        if len(p) < 3:
            continue
        try:
            g = [float(x) for x in p[1].split(",")]
        except ValueError:
            continue
        d = dict(kv.split("=", 1) for kv in p[2].split(",") if "=" in kv)
        d = {k.replace("vllm:", ""): float(v) for k, v in d.items()}
        if len(g) == 5 and KEYS[-1] in d:
            rows.append([float(p[0])] + g + [d.get(k, np.nan) for k in KEYS])
    return np.array(rows)


def windows(b, w=10):
    t0 = (b[:, 0] // w) * w
    starts = np.r_[0, np.flatnonzero(np.diff(t0)) + 1]
    out = []
    for s, e in zip(starts[:-1], starts[1:]):
        dt, d = b[e, 0] - b[s, 0], b[e] - b[s]
        if abs(dt - w) > 1 or np.any(d[[DR, AC, GEN, IC, LC, CH, EX]] < 0):
            continue  # gap or counter reset (restart)
        r = b[s:e + 1, RUN]
        out.append(dict(dt=dt, steps=d[IC], gen=d[GEN], dr=d[DR], ac=d[AC], lc=d[LC], ch=d[CH], ex=d[EX],
                        rmin=r.min(), rmax=r.max(), wait=b[s:e + 1, WAIT].mean(), kv=b[s:e, KV].mean()))
    return out


def main(path):
    a = parse(path)
    for name, (t_lo, t_hi, pool, ladder) in SEGS.items():
        b = a[(a[:, 0] >= t_lo) & (a[:, 0] < t_hi)]
        ws = windows(b)
        busy = [w for w in ws if w["steps"] > 0]
        tot = {k: sum(w[k] for w in busy) for k in ("dt", "steps", "gen", "dr", "ac", "lc")}
        print(f"== {name}: {len(ws)} windows, busy {len(busy)} ({100 * len(busy) / len(ws):.0f}%), busy {tot['dt'] / 3600:.1f} h")
        print(f"  wall/step {1000 * tot['dt'] / tot['steps']:.1f} ms, gen/step {tot['gen'] / tot['steps']:.2f}, decoding req/step "
              f"{tot['dr'] / tot['steps']:.2f}, accepted/draft {tot['ac'] / tot['dr']:.3f}, computed prompt tok {tot['lc']:.0f} "
              f"({tot['lc'] / 128 / tot['steps'] * 100:.1f}% of steps if 128-token chunks)")
        for r in (1, 2, 3):
            sel = [w for w in busy if w["rmin"] == w["rmax"] == r]
            if len(sel) < 5:
                continue
            ms = np.array([1000 * w["dt"] / w["steps"] for w in sel])
            st = sum(w["steps"] for w in sel)
            do = [w for w in sel if w["dr"] / w["steps"] >= r - 0.02 and w["lc"] == w["ch"] == w["ex"] == 0]
            msd = np.array([1000 * w["dt"] / w["steps"] for w in do])
            ctx = np.array([w["kv"] * pool / 1000 for w in do])
            print(f"  running {r}: n={len(sel)} wall/step {1000 * sum(w['dt'] for w in sel) / st:.1f} p10/50/90 "
                  f"{np.percentile(ms, 10):.1f}/{np.percentile(ms, 50):.1f}/{np.percentile(ms, 90):.1f}, gen/step "
                  f"{sum(w['gen'] for w in sel) / st:.2f}, acc/draft {sum(w['ac'] for w in sel) / sum(w['dr'] for w in sel):.3f}; "
                  f"decode-only n={len(do)} mean {1000 * sum(w['dt'] for w in do) / sum(w['steps'] for w in do):.1f} p10/50/90 "
                  f"{np.percentile(msd, 10):.1f}/{np.percentile(msd, 50):.1f}/{np.percentile(msd, 90):.1f} ctx p50 {np.median(ctx):.0f}k, "
                  f"residual at slope {SLOPE}: {np.median(msd - ladder[r] - SLOPE * ctx):.1f} ms")
        sel = [w for w in busy if w["rmin"] == w["rmax"] == 2]
        f = np.array([2 - w["dr"] / w["steps"] for w in sel])
        ms = np.array([1000 * w["dt"] / w["steps"] for w in sel])
        st = np.array([w["steps"] for w in sel])
        c, *_ = np.linalg.lstsq(np.c_[np.ones_like(f), f] * np.sqrt(st)[:, None], ms * np.sqrt(st), rcond=None)
        print(f"  running 2, by fraction f of steps with one request not decoding: ms/step = {c[0]:.1f} + {c[1]:.1f} f")
        for lo, hi in ((0, .02), (.02, .1), (.1, .2), (.2, .3), (.3, .5), (.5, .8), (.8, 2.1)):
            s = (f >= lo) & (f < hi)
            print(f"    f {lo:.2f}-{hi:.2f}: windows {s.sum():4d}  wall/step {10000 * s.sum() / max(st[s].sum(), 1):6.1f}  "
                  f"share of running-2 time {100 * s.mean():4.1f}%")


if __name__ == "__main__":
    main(sys.argv[1])
