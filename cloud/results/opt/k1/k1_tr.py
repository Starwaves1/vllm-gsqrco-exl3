import json, sys, re, collections
for path in sys.argv[1:]:
    ev = json.load(open(path))["traceEvents"]
    c = collections.defaultdict(lambda: [0, 0.0])
    for e in ev:
        if e.get("cat") != "kernel": continue
        n = e["name"]
        m = re.match(r"void (mul_mat_vec_q|own_mul_mat_vec)<\(ggml_type\)(\d+), (\d+)", n)
        if not m or m.group(2) not in ("23", "12", "22"): continue
        g = tuple(e.get("args", {}).get("grid", []))
        b = tuple(e.get("args", {}).get("block", []))
        k = (m.group(2), m.group(1)[:3], m.group(3), g[0] if g else 0, b)
        c[k][0] += 1; c[k][1] += e["dur"]
    print(path.split("/")[3])
    for k, (cnt, d) in sorted(c.items()):
        print(f"  type {k[0]} {k[1]} n{k[2]} grid.x {k[3]} block {k[4]}: {cnt} calls, {d/cnt:.1f} us/call, total {d/1000:.2f} ms")
