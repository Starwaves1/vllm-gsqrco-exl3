"""Seed a new venv with hard links to the files of packages another venv already has at the
exact pinned version (same wheel, same bytes), so a second venv on a small disk costs only
the packages that differ. Every file is checked against its RECORD sha256 first; a distribution with any
mismatching file is not seeded.

  venv-seed.py SRC_VENV DST_VENV FREEZE.txt

Only name==version pins are seeded (URL pins such as vllm @ https://... are left to the
installer). Console scripts under bin/ are copied, not linked, with the interpreter line
pointed at DST_VENV. After this, `uv pip install --no-deps -r FREEZE.txt` sees those
packages as installed and fetches the rest. Idempotent: an existing file with the same bytes
as the source is replaced by a link (so a re-run dedups what an installer copied), any other
existing file is left alone.
Nothing in SRC_VENV is written (hard links share inodes; installers replace files by
rename, never in place).
"""

import base64
import csv
import filecmp
import hashlib
import os
import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from importlib import metadata


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def pins(path: str) -> dict:
    out = {}
    for line in open(path):
        m = re.match(r"^([A-Za-z0-9_.\-]+)==(\S+)\s*$", line)
        if m:
            out[norm(m.group(1))] = m.group(2)
    return out


def site(venv: str) -> str:
    return os.path.join(venv, "lib", "python3.12", "site-packages")


def sha_ok(path: str, want: str) -> bool:
    if not want:
        return True
    algo, _, b64 = want.partition("=")
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            h.update(blk)
    return base64.urlsafe_b64encode(h.digest()).rstrip(b"=").decode() == b64


def main() -> int:
    src, dst, freeze = sys.argv[1:4]
    want = pins(freeze)
    ssp, dsp = site(src), site(dst)
    src_py = os.path.realpath(os.path.join(src, "bin", "python"))
    per = {}
    for d in metadata.distributions(path=[ssp]):
        n = norm(d.metadata["Name"])
        if want.get(n) != d.version:
            continue
        files = []
        for row in csv.reader((d.read_text("RECORD") or "").splitlines()):
            if row:
                rel, h = row[0], row[1] if len(row) > 1 else ""
                files.append((os.path.normpath(os.path.join(ssp, rel)), os.path.normpath(os.path.join(dsp, rel)), h))
        per[f"{n}=={d.version}"] = files
    sbin = os.path.normpath(os.path.join(src, "bin"))
    flat = [(k, f) for k, fs in per.items() for f in fs]
    # console scripts are per-venv copies with their own interpreter line: existence only
    check = lambda kf: os.path.exists(kf[1][0]) and (os.path.dirname(kf[1][0]) == sbin or sha_ok(kf[1][0], kf[1][2]))  # noqa: E731
    with ThreadPoolExecutor(16) as ex:
        oks = list(ex.map(check, flat))
    skipped = sorted({k for (k, _), ok in zip(flat, oks) if not ok})
    for k in skipped:  # a file changed after install (e.g. an overlay): let the installer fetch it
        print(f"not seeded (files differ from RECORD): {k}")
        del per[k]
    dists = sorted(per)
    jobs = [f for k in dists for f in per[k]]
    linked = copied = relinked = 0
    for s, d, _ in jobs:
        if os.path.lexists(d):
            # an earlier install copied it: same bytes -> replace the copy with a link (atomic)
            if (os.path.dirname(s) != sbin and os.path.isfile(d) and not os.path.islink(d)
                    and not os.path.samefile(s, d) and os.path.getsize(s) == os.path.getsize(d)
                    and filecmp.cmp(s, d, shallow=False)):
                tmp = d + ".seedlink"
                os.link(s, tmp)
                os.replace(tmp, d)
                relinked += 1
            continue
        os.makedirs(os.path.dirname(d), exist_ok=True)
        if os.path.dirname(s) == sbin:  # console script: own copy, own interpreter line
            with open(s, "rb") as f:
                data = f.read()
            data = data.replace(src_py.encode(), os.path.join(dst, "bin", "python").encode())
            data = data.replace(os.path.join(src, "bin", "python").encode(), os.path.join(dst, "bin", "python").encode())
            with open(d, "wb") as f:
                f.write(data)
            shutil.copymode(s, d)
            copied += 1
        else:
            os.link(s, d)
            linked += 1
    print(f"seeded {len(dists)} distributions from {src}: {linked} files hard-linked, {relinked} identical copies "
          f"replaced by links, {copied} scripts copied, "
          f"{len(jobs)} RECORD entries verified; {len(skipped)} distributions left to the installer")
    return 0


if __name__ == "__main__":
    sys.exit(main())
