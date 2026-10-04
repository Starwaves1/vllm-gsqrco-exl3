"""EXL3 checkpoint names vs vLLM's Qwen3.5 model, CPU only, metadata only.

Against the safetensors index and shard headers (hf-config/<checkpoint>, fetched by HTTP range;
no weights) of erlidev/Swift-1.5-Qwen3.8-27B-EXL3@SC_3.50bpw_H4_V6 (the primary model) and
turboderp/Qwen3.8-27B-exl3@3.50bpw (the A/B):
  * the format facts the plugin relies on (every EXL3 module has trellis/suh/svh/mul1 with
    consistent shapes; the bit widths per role; the unquantized linears are stored fp16);
  * the plugin's quantized/unquantized split agrees with what the checkpoint stores;
  * the two checkpoints differ only in the vision tower and the per-tensor bit widths;
  * erlidev's quantization_config.json bits agree with the trellis shapes;
  * tools/exl3_meta_dry_run.py: vLLM's real Qwen3_5ForConditionalGeneration + Qwen3_5MTP on
    the meta device under production's text-only argv, loaded through the default loader:
    every param loaded, every text tensor consumed, the vision tower skipped, every EXL3
    layer's parts matching its partitions (DRY RUN PASS).
"""

import collections
import glob
import json
import os
import re
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "plugin-exl3"))
SWIFT = os.path.join(ROOT, "hf-config/Swift-1.5-Qwen3.8-27B-exl3-SC_3.50bpw_H4_V6")
TURBO = os.path.join(ROOT, "hf-config/Qwen3.8-27B-exl3-3.50bpw")


def _tensors(hf_dir):
    with open(os.path.join(hf_dir, "safetensors_headers.json")) as f:
        headers = json.load(f)
    out = {}
    for h in headers.values():
        out.update({n: (t["dtype"], tuple(t["shape"])) for n, t in h["tensors"].items()})
    with open(os.path.join(hf_dir, "model.safetensors.index.json")) as f:
        assert set(json.load(f)["weight_map"]) == set(out)
    return out


def _role(m):
    return ("head" if m == "lm_head" else "mtp" if m.startswith("mtp.")
            else "visual" if m.startswith("model.visual.") else "decoder")


@pytest.mark.parametrize("hf_dir,want", [
    # bits 3.5 (a K3/K4 mix, one K5), head_bits 6, mtp_bits 4; bf16 vision tower
    (TURBO, {("decoder", 3.0): 137, ("decoder", 4.0): 262, ("decoder", 5.0): 1,
             ("head", 6.0): 1, ("mtp", 4.0): 8}),
    # sc_optimize recipe: K2..K5 decoder (K2: layers 0-1 gate/up), head_bits 4, mtp_bits 4,
    # vision_bits 6 (27 blocks x q/k/v/proj/fc1/fc2 + the merger's 2)
    (SWIFT, {("decoder", 2.0): 4, ("decoder", 3.0): 171, ("decoder", 4.0): 193,
             ("decoder", 5.0): 32, ("head", 4.0): 1, ("mtp", 4.0): 8, ("visual", 6.0): 164}),
], ids=["turboderp", "swift"])
def test_exl3_modules_complete_and_consistent(hf_dir, want):
    from vllm_exl3_plugin.format import bits_from_tile, quantized_modules

    t = _tensors(hf_dir)
    mods = quantized_modules(t)
    assert len([m for m in mods if _role(m) != "visual"]) == 409  # 400 decoder + lm_head + 8 MTP
    bits = collections.Counter()
    for m in mods:
        tr, suh, svh, mul1 = (t[f"{m}.{s}"] for s in ("trellis", "suh", "svh", "mul1"))
        assert f"{m}.mcg" not in t and f"{m}.weight" not in t
        assert f"{m}.bias" not in t or _role(m) == "visual"  # the vision tower's linears have biases
        assert tr[0] == "I16" and len(tr[1]) == 3
        assert suh == ("F16", (tr[1][0] * 16,)) and svh == ("F16", (tr[1][1] * 16,))
        assert mul1 == ("I32", ())
        assert tr[1][0] * 16 % 128 == 0 and tr[1][1] * 16 % 128 == 0
        bits[_role(m), bits_from_tile(tr[1][2])] += 1
    assert bits == want


@pytest.mark.parametrize("hf_dir", [TURBO, SWIFT], ids=["turboderp", "swift"])
def test_unquantized_split_matches_checkpoint(hf_dir):
    """The text modules the plugin keeps unquantized are the ones stored as .weight, and every
    text linear stored as .weight is one of them (or a norm/conv/embedding, not a linear)."""
    from vllm_exl3_plugin.format import quantized_modules
    from vllm_exl3_plugin.weights_adapter.qwen3_5 import is_unquantized_module

    t = _tensors(hf_dir)
    for m in quantized_modules(t):
        assert is_unquantized_module(m) == (_role(m) == "visual"), m
    plain_linears = sorted({n.removesuffix(".weight") for n, (_, s) in t.items()
                            if n.endswith(".weight") and len(s) == 2 and "embed" not in n
                            and "visual" not in n})
    assert plain_linears and all(re.search(r"linear_attn\.in_proj_[ab]$", m) for m in plain_linears)
    assert all(t[f"{m}.weight"] == ("F16", (48, 5120)) for m in plain_linears)
    # vLLM fuses them into in_proj_ba, which the plugin keeps unquantized
    assert is_unquantized_module("language_model.model.layers.0.linear_attn.in_proj_ba")
    assert t["model.language_model.embed_tokens.weight"] == ("BF16", (248320, 5120))


def test_checkpoints_differ_in_vision_and_bits_only():
    """3080 vs 2426 tensors: all of it is the vision tower. The text tensors (2054 + 39 mtp.*)
    have the same names, dtypes and shapes, except the trellis tile width (the bits)."""
    s, t = _tensors(SWIFT), _tensors(TURBO)
    assert (len(s), len(t)) == (3080, 2426)
    vis = lambda d: {n for n in d if n.startswith("model.visual.")}  # noqa: E731
    assert (len(vis(s)), len(vis(t))) == (987, 333)
    assert set(s) - vis(s) == set(t) - vis(t) and len(set(s) - vis(s)) == 2093
    for n in set(s) - vis(s):
        if n.endswith(".trellis"):
            assert s[n][0] == t[n][0] and s[n][1][:2] == t[n][1][:2], n
        else:
            assert s[n] == t[n], n
    # turboderp's tower: bf16, fused qkv, 4304-row MLP. erlidev's: EXL3 q/k/v_proj, proj, fc1, fc2
    # and merger fc1/fc2 (fp16 bias), fc1 padded to 4352 rows, fp16 norms; it also still carries
    # the bf16 fused qkv (27 weights + 27 biases). Never built under text-only serving.
    assert t["model.visual.blocks.0.mlp.linear_fc1.weight"] == ("BF16", (4304, 1152))
    assert s["model.visual.blocks.0.mlp.linear_fc1.trellis"] == ("I16", (72, 272, 96))
    assert s["model.visual.blocks.0.attn.qkv.weight"] == ("BF16", (3456, 1152))
    assert {d for n, (d, _) in t.items() if n.startswith("model.visual.")} == {"BF16"}


def test_quantization_config_json_matches_headers():
    """erlidev's quantization_config.json (exllamav3 1.5.0): bits_per_weight and stored shapes
    per text module agree with the shard headers (it lists the 707 language-model modules,
    401 EXL3; MTP and vision are not listed)."""
    from vllm_exl3_plugin.format import MUL1_MULT, bits_from_tile

    with open(os.path.join(SWIFT, "quantization_config.json")) as f:
        q = json.load(f)
    with open(os.path.join(SWIFT, "config.json")) as f:
        assert json.load(f)["quantization_config"] == {k: v for k, v in q.items() if k != "tensor_storage"}
    assert (q["version"], q["bits"], q["head_bits"], q["mtp_bits"], q["vision_bits"]) == ("1.5.0", 3.5, 4, 4, 6)
    t = _tensors(SWIFT)
    exl3 = {m: e for m, e in q["tensor_storage"].items() if e.get("quant_format") == "exl3"}
    assert (len(q["tensor_storage"]), len(exl3)) == (707, 401)
    for m, e in exl3.items():
        assert e["mul1_multiplier"] == MUL1_MULT
        assert bits_from_tile(t[f"{m}.trellis"][1][2]) == e["bits_per_weight"], m
        for name, st in e["stored_tensors"].items():
            assert tuple(st["shape"]) == t[name][1], name


# an unbuilt checkout takes phase 1's routing: EXL3_MR=2 (the default) refuses to load without _C_exl3_mr
_MR_UNBUILT = {} if glob.glob(os.path.join(ROOT, "plugin-exl3/vllm_exl3_plugin/_C_exl3_mr*.so")) else {"EXL3_MR": "0"}


@pytest.mark.parametrize("args,vision", [([], 987), (["--alt"], 333)], ids=["swift", "turboderp"])
def test_meta_dry_run(args, vision):
    p = subprocess.run(
        [sys.executable, os.path.join(ROOT, "tools/exl3_meta_dry_run.py"), *args],
        capture_output=True, text=True, timeout=600,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", **_MR_UNBUILT))
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith(("[main]", "[mtp]", "checkpoint", "DRY RUN"))]
    assert p.returncode == 0 and "DRY RUN PASS" in p.stdout, "\n".join(lines) + p.stderr[-3000:]
    assert f"vision skipped {vision} of {vision}; unmapped 0 []" in lines[2]
    assert "kept 2054 tensors (401 EXL3" in lines[0]
    assert "missing 0 []" in lines[0] and "EXL3 layers 257 with 401 parts" in lines[0]
    assert "missing 0 []" in lines[1] and "EXL3 layers 6 with 9 parts" in lines[1]
