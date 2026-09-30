#!/usr/bin/env python3
"""Build and verify the HF config dir that vLLM uses for the Swift GSQ-RCO GGUF.

The plugin reads model dims from this dir (it only patches vocab_size and
tie_word_embeddings from the GGUF), so config.json must match the GGUF exactly.

Contents, from the Swift 1.5 W4A16 AutoRound repo (same weights family and
tokenizer as the GGUF):
  config.json            Swift's config minus quantization_config (the GGUF
                         carries its own per-tensor types)
  generation_config.json, tokenizer.json, tokenizer_config.json, vocab.json,
  merges.txt, preprocessor_config.json, processor_config.json,
  video_preprocessor_config.json      byte-identical copies
  mtp_draft_vocab_ids.pt the draft-head token ids (tools/draft_vocab_ids.py:
                         the 40,960 production's pipeline made for Swift plus
                         ids the model emits outside them); kept as is unless
                         --draft-ids names another list. With it, the vLLM
                         overlay's MTP draft scores only these rows of the
                         lm_head
  chat_template.jinja    production's template (--chat-template of the live
                         server). transformers prefers this file over the
                         chat_template entry in tokenizer_config.json
                         (tokenization_utils_base.py:1783-1799, 5.15.1).
  PROVENANCE.json        sha256 of every source and output file

Keep the dir path unique to this model: vLLM's fs KV tier namespaces its keys
by model_config.model, which the plugin sets to this dir.

usage:
  make_hf_config.py build  [--src DIR] [--template FILE] [--draft-ids FILE] [--out DIR]
  make_hf_config.py verify [--out DIR] [--gguf FILE] [--template FILE]
Run it under tools/capped with GSQ_LIGHT=1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

MODELS = Path("/home/garrett/qwen38-27b-rtx3090/models")
DEFAULT_SRC = MODELS / "Swift-1.5-Qwen3.8-27B-W4A16-AutoRound"
DEFAULT_TEMPLATE = Path("/home/garrett/models/qwen-sharp/chat_template.jinja")
DEFAULT_GGUF = (
    MODELS
    / "Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf"
)
DEFAULT_OUT = ROOT / "hf-config/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp"
DRAFT_IDS = DEFAULT_OUT / "mtp_draft_vocab_ids.pt"  # made by tools/draft_vocab_ids.py

COPIED = [
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "preprocessor_config.json",
    "processor_config.json",
    "video_preprocessor_config.json",
]

# llama.cpp b11211 src/llama-vocab.cpp:392-397 (LLAMA_VOCAB_PRE_TYPE_QWEN35):
# "original regex from tokenizer.json"
QWEN35_PRE_REGEX = (
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}|"
    " ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _repo_path(path: Path) -> str:
    """path relative to the repo when inside it (the default draft ids), else as given."""
    path = path.resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def build(src: Path, template: Path, draft_ids: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads((src / "config.json").read_text())
    config.pop("quantization_config", None)  # absent in an unquantized source (base Qwen3.8)
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    copied = [name for name in COPIED if (src / name).exists()]  # base Qwen3.8 has no processor_config.json
    for name in copied:
        shutil.copyfile(src / name, out / name)
    shutil.copyfile(template, out / "chat_template.jinja")
    if draft_ids.resolve() != (out / DRAFT_IDS.name).resolve():
        shutil.copyfile(draft_ids, out / DRAFT_IDS.name)
    provenance = {
        "source_dir": str(src),
        "chat_template_source": str(template),
        "changes": [
            "config.json: quantization_config removed; everything else identical",
            "chat_template.jinja: production's template instead of the source's stock one",
        ],
        "sources": {
            name: sha256(src / name) for name in ["config.json", *copied, "chat_template.jinja"]
        }
        | {"production chat_template.jinja": sha256(template), _repo_path(draft_ids): sha256(draft_ids)},
        "outputs": {p.name: sha256(p) for p in sorted(out.iterdir()) if p.name != "PROVENANCE.json"},
    }
    (out / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"built {out}")


def verify(out: Path, gguf_path: Path, template: Path) -> None:
    import no_gpu  # noqa: F401  (before gguf/transformers pull in torch)

    import gguf

    failures: list[str] = []

    def check(ok: bool, what: str) -> None:
        print(("ok    " if ok else "FAIL  ") + what)
        if not ok:
            failures.append(what)

    cfg = json.loads((out / "config.json").read_text())
    t = cfg["text_config"]
    check("quantization_config" not in cfg, "config.json has no quantization_config")
    check(cfg["model_type"] == "qwen3_5", "model_type qwen3_5 (vLLM's MTP override matches it)")
    check(cfg["architectures"] == ["Qwen3_5ForConditionalGeneration"], "architecture")

    reader = gguf.GGUFReader(str(gguf_path))
    meta = {k: f.contents() for k, f in reader.fields.items()
            if not k.startswith("tokenizer.ggml.") and k != "tokenizer.chat_template"}
    a = "qwen35."
    n_mtp = meta[a + "nextn_predict_layers"]
    expect = {
        "hidden_size": meta[a + "embedding_length"],
        "intermediate_size": meta[a + "feed_forward_length"],
        "num_attention_heads": meta[a + "attention.head_count"],
        "num_key_value_heads": meta[a + "attention.head_count_kv"],
        "head_dim": meta[a + "attention.key_length"],
        "num_hidden_layers": meta[a + "block_count"] - n_mtp,
        "mtp_num_hidden_layers": n_mtp,
        "max_position_embeddings": meta[a + "context_length"],
        "linear_conv_kernel_dim": meta[a + "ssm.conv_kernel"],
        "linear_key_head_dim": meta[a + "ssm.state_size"],
        "linear_num_key_heads": meta[a + "ssm.group_count"],
        "linear_num_value_heads": meta[a + "ssm.time_step_rank"],
        "full_attention_interval": meta[a + "full_attention_interval"],
    }
    for key, value in expect.items():
        check(t[key] == value, f"text_config.{key} = {t[key]} (GGUF {value})")
    check(meta[a + "attention.value_length"] == t["head_dim"], "value_length == head_dim")
    check(t["linear_num_value_heads"] * t["linear_value_head_dim"] == meta[a + "ssm.inner_size"],
          "linear_num_value_heads * linear_value_head_dim == ssm.inner_size")
    check(abs(t["rms_norm_eps"] - meta[a + "attention.layer_norm_rms_epsilon"]) < 1e-12, "rms_norm_eps")
    rope = t["rope_parameters"]
    check(rope["rope_theta"] == meta[a + "rope.freq_base"], "rope_theta")
    check(int(rope["partial_rotary_factor"] * t["head_dim"]) == meta[a + "rope.dimension_count"],
          "partial_rotary_factor * head_dim == rope.dimension_count")
    check(list(rope["mrope_section"]) == list(meta[a + "rope.dimension_sections"])[:3], "mrope_section")
    if a + "attention.recurrent_layers" in meta:
        recurrent = list(meta[a + "attention.recurrent_layers"])
    else:  # ISTA's GGUFs omit the key; llama.cpp derives it the same way (src/models/qwen35.cpp)
        fai = meta[a + "full_attention_interval"]
        recurrent = [(i + 1) % fai != 0 for i in range(meta[a + "block_count"] - n_mtp)] + [False] * n_mtp
        print(f"note  no {a}attention.recurrent_layers; derived from full_attention_interval {fai}")
    want_types = ["linear_attention" if r else "full_attention" for r in recurrent[: t["num_hidden_layers"]]]
    check(t["layer_types"] == want_types, "layer_types == GGUF recurrent_layers (first 64)")
    check(recurrent[t["num_hidden_layers"]:] == [False] * n_mtp, "MTP block is full attention")
    names = {tensor.name for tensor in reader.tensors}
    check(len(names) == 866, f"GGUF tensor count {len(names)}")
    check(t["tie_word_embeddings"] is ("output.weight" not in names), "tie_word_embeddings matches output.weight presence")

    # tokenizer: every id -> token string equal to the GGUF vocab; same merges and special ids
    tok = json.loads((out / "tokenizer.json").read_text())
    id_to_tok = {i: s for s, i in tok["model"]["vocab"].items()}
    id_to_tok.update({t_["id"]: t_["content"] for t_ in tok["added_tokens"]})
    g_tokens = reader.fields["tokenizer.ggml.tokens"].contents()
    t_vocab = t["vocab_size"]
    check(len(g_tokens) == t_vocab, f"GGUF has {len(g_tokens)} tokens == vocab_size {t_vocab}")
    mism = [i for i in range(len(g_tokens)) if i in id_to_tok and id_to_tok[i] != g_tokens[i]]
    check(not mism, f"all {len(id_to_tok)} HF token strings equal the GGUF's (mismatches: {mism[:5]})")
    extra = sorted(set(range(len(g_tokens))) - set(id_to_tok))
    check(all(g_tokens[i].startswith("[PAD") for i in extra),
          f"{len(extra)} GGUF-only ids are padding entries beyond the HF vocab")
    hf_merges = [" ".join(m) if isinstance(m, list) else m for m in tok["model"]["merges"]]
    g_merges = reader.fields["tokenizer.ggml.merges"].contents()
    check(hf_merges == list(g_merges), f"merges identical ({len(hf_merges)} vs {len(g_merges)})")
    split = tok["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]
    check(reader.fields["tokenizer.ggml.pre"].contents() == "qwen35"
          and split == QWEN35_PRE_REGEX, "pre-tokenizer regex == llama.cpp qwen35 (llama-vocab.cpp:395)")
    gen = json.loads((out / "generation_config.json").read_text())
    g_eos = reader.fields["tokenizer.ggml.eos_token_id"].contents()
    g_bos = reader.fields["tokenizer.ggml.bos_token_id"].contents()
    check(g_eos in gen["eos_token_id"] and gen["bos_token_id"] == g_bos,
          f"generation eos {gen['eos_token_id']} contains GGUF eos {g_eos}; bos {g_bos}")

    # chat template: production's; and transformers actually picks it
    check(sha256(out / "chat_template.jinja") == sha256(template), "chat_template.jinja == production's")
    from transformers import AutoTokenizer

    hf_tok = AutoTokenizer.from_pretrained(str(out))
    check(hf_tok.chat_template == template.read_text(), "AutoTokenizer loads production's chat template")
    msgs = [{"role": "user", "content": "hi"}]
    ids = hf_tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
    if hasattr(ids, "input_ids"):
        ids = ids["input_ids"]
    check(len(ids) > 0 and max(ids) < t_vocab, f"chat template renders and tokenizes ({len(ids)} ids)")

    no_gpu.assert_no_gpu_libs()
    if failures:
        sys.exit(f"{len(failures)} check(s) failed")
    print("all checks passed")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["build", "verify"])
    p.add_argument("--src", type=Path, default=DEFAULT_SRC)
    p.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    p.add_argument("--draft-ids", type=Path, default=DRAFT_IDS)
    p.add_argument("--gguf", type=Path, default=DEFAULT_GGUF)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()
    if args.cmd == "build":
        build(args.src, args.template, args.draft_ids, args.out)
    else:
        verify(args.out, args.gguf, args.template)


if __name__ == "__main__":
    main()
