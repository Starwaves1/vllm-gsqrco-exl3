"""Error table from tests/gpu/test_exl3_mr.py's printed lines (job 10): per (tensor, rows, output
dtype) rel_rms and max_rel vs fp64 of exl3_gemm_mr, exl3_gemm and the dequant + fp16 GEMM, and a
count line. usage: errtab.py mr.log"""
import re
import sys

rows = []
for ln in open(sys.argv[1]):
    m = re.match(r"(\S+) m=(\d+) (bf16|fp16) mr ", ln)
    if not m:
        continue
    v = [(float(a), float(b)) for a, b in re.findall(r"'rel_rms': ([0-9.e+-]+), 'max_rel': ([0-9.e+-]+)", ln)]
    if len(v) == 3:
        rows.append((m.group(1), int(m.group(2)), m.group(3), *v))
print(f"{'tensor':10} {'rows':>4} {'out':4} {'mr_rms':>9} {'gemm_rms':>9} {'dq_rms':>9} {'mr_max':>8} {'gemm_max':>8}")
for t, m, d, a, b, c in rows:
    print(f"{t:10} {m:>4} {d:4} {a[0]:9.2e} {b[0]:9.2e} {c[0]:9.2e} {a[1]:8.2e} {b[1]:8.2e}")
if rows:
    print(f"# {len(rows)} cases; mr rms <= exl3_gemm rms in {sum(a[0] <= b[0] for *_, a, b, c in rows)}, "
          f"mr max <= exl3_gemm max in {sum(a[1] <= b[1] for *_, a, b, c in rows)}; "
          f"worst mr rms {max(a[0] for *_, a, b, c in rows):.2e}, worst mr max {max(a[1] for *_, a, b, c in rows):.2e}")
