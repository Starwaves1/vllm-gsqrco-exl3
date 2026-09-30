"""EXL3 checkpoint names vs vLLM's Qwen3.5 model, CPU only, metadata only.

Against turboderp/Qwen3.8-27B-exl3@3.50bpw's safetensors index and shard headers
(hf-config/Qwen3.8-27B-exl3-3.50bpw, fetched by HTTP range; no weights):
  * the format facts the plugin relies on (every EXL3 module has trellis/suh/svh/mul1 with
    consistent shapes; the bit widths per role; the unquantized linears are stored fp16);
  * the plugin's quantized/unquantized split agrees with what the checkpoint stores;
  * tools/exl3_meta_dry_run.py: vLLM's real Qwen3_5ForConditionalGeneration + Qwen3_5MTP on
    the meta device, loaded through the default loader: every param loaded, every tensor
    consumed, every EXL3 layer's parts matching its partitions (DRY RUN PASS).
"""

import collections
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "plugin-exl3"))
HF_DIR = os.path.join(ROOT, "hf-config/Qwen3.8-27B-exl3-3.50bpw")


def _tensors():
    with open(os.path.join(HF_DIR, "safetensors_headers.json")) as f:
        headers = json.load(f)
    out = {}
    for h in headers.values():
        out.update({n: (t["dtype"], tuple(t["shape"])) for n, t in h["tensors"].items()})
    with open(os.path.join(HF_DIR, "model.safetensors.index.json")) as f:
        assert set(json.load(f)["weight_map"]) == set(out)
    return out


def test_exl3_modules_complete_and_consistent():
    from vllm_exl3_plugin.format import bits_from_tile, quantized_modules

    t = _tensors()
    mods = quantized_modules(t)
    assert len(mods) == 409  # 400 decoder + lm_head + 8 MTP (fc, q/k/v/o, gate/up/down)
    bits = collections.Counter()
    for m in mods:
        tr, suh, svh, mul1 = (t[f"{m}.{s}"] for s in ("trellis", "suh", "svh", "mul1"))
        assert f"{m}.mcg" not in t and f"{m}.weight" not in t and f"{m}.bias" not in t
        assert tr[0] == "I16" and len(tr[1]) == 3
        assert suh == ("F16", (tr[1][0] * 16,)) and svh == ("F16", (tr[1][1] * 16,))
        assert mul1 == ("I32", ())
        assert tr[1][0] * 16 % 128 == 0 and tr[1][1] * 16 % 128 == 0
        role = "head" if m == "lm_head" else "mtp" if m.startswith("mtp.") else "decoder"
        bits[role, bits_from_tile(tr[1][2])] += 1
    # config.json: bits 3.5 (a K3/K4 mix, one K5), head_bits 6, mtp_bits 4
    assert bits == {("decoder", 3.0): 137, ("decoder", 4.0): 262, ("decoder", 5.0): 1,
                    ("head", 6.0): 1, ("mtp", 4.0): 8}


def test_unquantized_split_matches_checkpoint():
    """The modules the plugin keeps unquantized are the ones stored as .weight, and every
    linear stored as .weight is one of them (or a norm/conv/embedding, not a linear)."""
    from vllm_exl3_plugin.format import quantized_modules
    from vllm_exl3_plugin.weights_adapter.qwen3_5 import is_unquantized_module

    t = _tensors()
    for m in quantized_modules(t):
        assert not is_unquantized_module(m), m
    plain_linears = sorted({n.removesuffix(".weight") for n, (_, s) in t.items()
                            if n.endswith(".weight") and len(s) == 2 and "embed" not in n
                            and "visual" not in n})
    assert plain_linears and all(re.search(r"linear_attn\.in_proj_[ab]$", m) for m in plain_linears)
    assert all(t[f"{m}.weight"] == ("F16", (48, 5120)) for m in plain_linears)
    # vLLM fuses them into in_proj_ba, which the plugin keeps unquantized
    assert is_unquantized_module("language_model.model.layers.0.linear_attn.in_proj_ba")
    # the vision tower and the input embedding are bf16
    assert {d for n, (d, _) in t.items() if n.startswith("model.visual.")} == {"BF16"}
    assert t["model.language_model.embed_tokens.weight"] == ("BF16", (248320, 5120))


def test_meta_dry_run():
    p = subprocess.run(
        [sys.executable, os.path.join(ROOT, "tools/exl3_meta_dry_run.py")],
        capture_output=True, text=True, timeout=600,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith(("[main]", "[mtp]", "checkpoint", "DRY RUN"))]
    assert p.returncode == 0 and "DRY RUN PASS" in p.stdout, "\n".join(lines) + p.stderr[-3000:]
    assert "unmapped 0 []" in p.stdout
    assert "missing 0 []" in lines[0] and "EXL3 layers 257 with 401 parts" in lines[0]
    assert "missing 0 []" in lines[1] and "EXL3 layers 6 with 9 parts" in lines[1]
