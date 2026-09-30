"""Deterministic, salted parity prompt set (token ids), generated from a seed.

Nothing large is stored in git: the text comes from the pinned venv's own vLLM source
tree (identical on any box built from env/gsq-freeze.txt plus the ba05ffab overlay), is
shuffled by the seed, prefixed with a per-sequence salt (so no prefix-cache or KV-tier
entry can match), and tokenized with the HF config dir's tokenizer. Both engines get the
same ids; neither tokenizes.

Output (--out DIR):
  seq_NNN.ids   int32 LE token ids
  seq_NNN.pos   int32 LE positions p to dump: the distribution after ids[:p+1]
  manifest.json per sequence: name, kind, n_tokens, positions count, sha256 of ids
  manifest.txt  one sequence name per line (read by llama_logits)

  python bench/parity/prompts.py --out DIR [--seed 20260927] [--tail 256] [--spread 32]
  python bench/parity/prompts.py --dry-run     tokenize, print the fingerprint, write nothing

The fingerprint (sha256 over all sequences' ids and positions) is compared with
prompts.lock.json (keyed by seed, tail, spread and the venv's vLLM version) when present, so a
changed venv or tokenizer is caught.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
if os.environ.get("GSQ_ALLOW_GPU") != "1":
    import no_gpu  # noqa: F401  (before anything that might import torch)

import numpy as np  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

HF_CONFIG = ROOT / "hf-config" / "Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp"
LOCK = Path(__file__).with_name("prompts.lock.json")

# (kind, target tokens). 100k+ contexts are the point of the set; they must still fit
# the reference: llama.cpp f16 KV on 24 GB holds ~130k with the 12.1 GB weights.
SEQUENCES = [
    ("chat", 1024), ("chat", 2048), ("code", 1536), ("code", 4096),
    ("code", 8192), ("prose", 8192), ("code", 32768), ("mixed", 32768),
    ("mixed", 65536), ("code", 102400), ("mixed", 120000),
]


def corpus_files(kind: str) -> list[Path]:
    import importlib.util

    spec = importlib.util.find_spec("vllm")  # locate only; does not import vllm
    base = Path(spec.origin).parent
    py = sorted(p for p in base.rglob("*.py") if "__pycache__" not in p.parts)
    md = sorted(base.rglob("*.md")) + sorted(base.rglob("*.jinja"))
    if kind == "code":
        return py
    if kind == "prose":
        return md or py
    return sorted(py + md)


def build_text(kind: str, n_tokens: int, rng: random.Random, tok: Tokenizer) -> list[int]:
    files = corpus_files(kind)
    rng.shuffle(files)
    salt = f"[parity-salt {rng.getrandbits(64):016x}]\n"
    parts: list[str] = []
    if kind == "chat":
        parts.append("<|im_start|>system\nYou are a careful code reviewer.<|im_end|>\n"
                     "<|im_start|>user\n" + salt + "Review this file:\n")
    else:
        parts.append(salt)
    ids: list[int] = []
    for f in files:
        parts.append(f"\n# ---- {f.name} ----\n" + f.read_text(errors="replace"))
        ids = tok.encode("".join(parts), add_special_tokens=False).ids
        if len(ids) >= n_tokens + 64:
            break
    else:
        raise SystemExit(f"corpus too small for {n_tokens} tokens ({kind})")
    ids = ids[:n_tokens]
    if kind == "chat":
        # end on a real assistant turn so the dumped tail is the model's own answer text
        tail = tok.encode("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
                          "The main issue in this file is", add_special_tokens=False).ids
        ids = ids[: n_tokens - len(tail)] + tail
    return ids


def positions(n: int, tail: int, spread: int, rng: random.Random) -> list[int]:
    pos = set(range(max(0, n - tail), n))
    if n > 4 * tail and spread:
        pos |= set(rng.sample(range(n // 16, n - tail), spread))
    return sorted(pos)


def generate(seed: int, tail: int, spread: int):
    tok = Tokenizer.from_file(str(HF_CONFIG / "tokenizer.json"))
    out = []
    for i, (kind, n) in enumerate(SEQUENCES):
        rng = random.Random(f"{seed}:{i}:{kind}:{n}")
        ids = build_text(kind, n, rng, tok)
        pos = positions(len(ids), tail, spread, rng)
        out.append((f"seq_{i:03d}", kind, np.asarray(ids, np.int32), np.asarray(pos, np.int32)))
    return out


def fingerprint(seqs) -> str:
    h = hashlib.sha256()
    for name, _, ids, pos in seqs:
        h.update(name.encode()); h.update(ids.tobytes()); h.update(pos.tobytes())
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--seed", type=int, default=20260927)
    ap.add_argument("--tail", type=int, default=256)
    ap.add_argument("--spread", type=int, default=32)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--write-lock", action="store_true", help="record the fingerprint in prompts.lock.json")
    a = ap.parse_args()
    seqs = generate(a.seed, a.tail, a.spread)
    fp = fingerprint(seqs)
    for name, kind, ids, pos in seqs:
        print(f"{name} {kind:6s} tokens={len(ids):7d} positions={len(pos):4d}")
    print(f"fingerprint {fp}")
    # the corpus is the venv's own vLLM source, so each vLLM version has its own fingerprint
    key = f"seed={a.seed},tail={a.tail},spread={a.spread},vllm={importlib.metadata.version('vllm')}"
    if LOCK.exists():
        want = json.loads(LOCK.read_text()).get(key)
        if want and want != fp:
            raise SystemExit(f"fingerprint differs from prompts.lock.json[{key}] = {want}: venv or tokenizer changed")
    if a.write_lock:
        lock = json.loads(LOCK.read_text()) if LOCK.exists() else {}
        lock[key] = fp
        LOCK.write_text(json.dumps(lock, indent=1) + "\n")
    if a.dry_run or not a.out:
        return
    a.out.mkdir(parents=True, exist_ok=True)
    man = []
    for name, kind, ids, pos in seqs:
        ids.tofile(a.out / f"{name}.ids"); pos.tofile(a.out / f"{name}.pos")
        man.append({"name": name, "kind": kind, "n_tokens": int(len(ids)), "n_pos": int(len(pos)),
                    "ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest()})
    (a.out / "manifest.json").write_text(json.dumps({"seed": a.seed, "fingerprint": fp, "sequences": man}, indent=1))
    (a.out / "manifest.txt").write_text("".join(m["name"] + "\n" for m in man))


if __name__ == "__main__":
    main()
