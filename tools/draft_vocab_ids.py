#!/usr/bin/env python3
"""Grow the MTP draft head's id list (hf-config/<model>/mtp_draft_vocab_ids.pt) to N rows.

Keeps production's list whole (40,960 ids, prepare/build_draft_vocab.py --ids) and adds the
token ids the served model emits most often outside it, counted over a corpus of its own
outputs (lines with "output_ids", e.g. cloud/results/opt-p/box-scripts/corpus_gen.py); ids the
corpus never emits follow in id order (lower id = earlier BPE merge). A token outside the list
can never be drafted, so each size is a superset of the smaller ones.

usage: draft_vocab_ids.py BASE_IDS.pt CORPUS.jsonl[.gz] N OUT.pt
The shipped list: draft_vocab_ids.py <Swift W4A16 -prepared>/mtp_draft_vocab_ids.pt
  cloud/results/opt-p/draft-corpus.jsonl.gz N hf-config/<model>/mtp_draft_vocab_ids.pt
"""
import collections
import gzip
import json
import sys

import torch

base_path, corpus, n, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
base = torch.load(base_path, weights_only=True).tolist()
vocab = 248320  # GGUF output.weight rows (hf-config config.json text_config.vocab_size)
counts = collections.Counter()
for line in (gzip.open if corpus.endswith(".gz") else open)(corpus, "rt"):
    counts.update(json.loads(line).get("output_ids") or [])
have = set(base)
seen = [t for t, _ in counts.most_common() if t not in have]
rest = [t for t in range(vocab) if t not in have and t not in counts]
ids = sorted(have | set((seen + rest)[: n - len(have)]))
emitted = sum(counts.values())
cover = lambda s: sum(c for t, c in counts.items() if t in s) / max(1, emitted)
print(f"{len(ids)} ids; corpus {emitted} tokens, {len(counts)} distinct; coverage "
      f"{cover(have) * 100:.2f}% -> {cover(set(ids)) * 100:.2f}%; {len(seen)} emitted ids outside the base")
torch.save(torch.tensor(ids, dtype=torch.int64), out)
