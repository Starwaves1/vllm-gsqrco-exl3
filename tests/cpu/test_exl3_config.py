"""EXL3 plugin registration and config (plugin-exl3/vllm_exl3_plugin), CPU only.

Config parsing on turboderp/Qwen3.8-27B-exl3@3.50bpw's and erlidev's Swift SC_3.50bpw_H4_V6
config.json (hf-config/, fetched metadata), the vllm.general_plugins entry point, which quant
method each layer gets, and the refusal of an EXL3 vision tower that is being built.
"""

import json
import os
import sys
import tomllib

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "plugin-exl3"))
HF_DIR = os.path.join(ROOT, "hf-config/Qwen3.8-27B-exl3-3.50bpw")
SWIFT_DIR = os.path.join(ROOT, "hf-config/Swift-1.5-Qwen3.8-27B-exl3-SC_3.50bpw_H4_V6")


def _qcfg(hf_dir=HF_DIR):
    with open(os.path.join(hf_dir, "config.json")) as f:
        return json.load(f)["quantization_config"]


@pytest.mark.parametrize("hf_dir,want", [
    (HF_DIR, (3.5, 6.0, 4.0, "mul1", "1.4.2", None)),
    (SWIFT_DIR, (3.5, 4.0, 4.0, "mul1", "1.5.0", 6.0)),
], ids=["turboderp", "swift"])
def test_fetched_config_parses(hf_dir, want):
    from vllm_exl3_plugin.format import MUL1_MULT, EXL3QuantConfig

    q = EXL3QuantConfig.from_dict(_qcfg(hf_dir))
    assert (q.bits, q.head_bits, q.mtp_bits, q.codebook, q.version, q.vision_bits) == want
    assert (q.codebook_param, q.codebook_mult, q.mul1, q.mcg) == ("mul1", MUL1_MULT, True, False)


@pytest.mark.parametrize("codebook,param,mcg,mul1", [
    (None, None, False, False),  # absent: 3INST
    ("3inst", None, False, False),
    ("mcg", "mcg", True, False),
    ("mul1", "mul1", False, True),
])
def test_codebooks(codebook, param, mcg, mul1):
    from vllm_exl3_plugin.format import EXL3QuantConfig

    d = {"quant_method": "exl3", "bits": 4}
    if codebook:
        d["codebook"] = codebook
    q = EXL3QuantConfig.from_dict(d)
    assert (q.codebook_param, q.mcg, q.mul1) == (param, mcg, mul1)
    assert q.head_bits is None and q.mtp_bits is None


@pytest.mark.parametrize("d,msg", [
    ({"quant_method": "gptq", "bits": 4}, "not 'exl3'"),
    ({"quant_method": "exl3", "bits": 4, "codebook": "lut"}, "unknown EXL3 codebook"),
])
def test_bad_config(d, msg):
    from vllm_exl3_plugin.format import EXL3QuantConfig

    with pytest.raises(ValueError, match=msg):
        EXL3QuantConfig.from_dict(d)


@pytest.mark.parametrize("width,bits", [(16, 1.0), (48, 3.0), (64, 4.0), (80, 5.0), (96, 6.0),
                                        (128, 8.0), (24, 1.5), (40, 2.5), (56, 3.5)])
def test_bits_from_tile(width, bits):
    from vllm_exl3_plugin.format import bits_from_tile

    assert bits_from_tile(width) == bits


@pytest.mark.parametrize("width", [0, 8, 50, 72, 144])
def test_bits_from_tile_rejects(width):
    from vllm_exl3_plugin.format import bits_from_tile

    with pytest.raises(ValueError):
        bits_from_tile(width)


def test_entry_point_registers_exl3():
    """pyproject's vllm.general_plugins entry point resolves and registers quant method exl3
    (what vLLM's load_general_plugins does for an installed package)."""
    from importlib.metadata import EntryPoint

    from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS, get_quantization_config

    with open(os.path.join(ROOT, "plugin-exl3/pyproject.toml"), "rb") as f:
        eps = tomllib.load(f)["project"]["entry-points"]["vllm.general_plugins"]
    assert eps == {"exl3": "vllm_exl3_plugin:register"}
    EntryPoint(name="exl3", value=eps["exl3"], group="vllm.general_plugins").load()()
    from vllm_exl3_plugin.quantization import EXL3Config

    assert "exl3" in QUANTIZATION_METHODS
    assert get_quantization_config("exl3") is EXL3Config
    cfg = EXL3Config.from_config(_qcfg())
    assert cfg.get_name() == "exl3"
    assert cfg.get_min_capability() == 80
    assert cfg.get_config_filenames() == []


def _layer(cls):
    return object.__new__(cls)  # isinstance is all get_quant_method looks at


@pytest.mark.parametrize("cls,prefix,want", [
    ("QKVParallelLinear", "language_model.model.layers.3.self_attn.qkv_proj", "EXL3LinearMethod"),
    ("RowParallelLinear", "language_model.model.layers.3.self_attn.o_proj", "EXL3LinearMethod"),
    ("MergedColumnParallelLinear", "language_model.model.layers.0.mlp.gate_up_proj", "EXL3LinearMethod"),
    ("RowParallelLinear", "language_model.model.layers.0.mlp.down_proj", "EXL3LinearMethod"),
    ("MergedColumnParallelLinear", "language_model.model.layers.0.linear_attn.in_proj_qkvz", "EXL3LinearMethod"),
    ("RowParallelLinear", "language_model.model.layers.0.linear_attn.out_proj", "EXL3LinearMethod"),
    ("MergedColumnParallelLinear", "language_model.model.layers.0.linear_attn.in_proj_ba",
     "UnquantizedLinearMethod"),
    ("ColumnParallelLinear", "mtp.fc", "EXL3LinearMethod"),
    ("QKVParallelLinear", "mtp.layers.0.self_attn.qkv_proj", "EXL3LinearMethod"),
    ("QKVParallelLinear", "visual.blocks.0.attn.qkv", "UnquantizedLinearMethod"),
    ("RowParallelLinear", "visual.merger.linear_fc2", "UnquantizedLinearMethod"),
    ("ParallelLMHead", "language_model.lm_head", "EXL3LinearMethod"),
    ("ParallelLMHead", "lm_head", "EXL3LinearMethod"),
    ("ParallelLMHead", "mtp.draft_lm_head", None),
    # token embedding: in pinned host memory by default (EXL3_EMBED_HOST, quantization/embedding.py)
    ("VocabParallelEmbedding", "language_model.model.embed_tokens", "EXL3HostEmbeddingMethod"),
    ("VocabParallelEmbedding", "mtp.embed_tokens", None),  # the draft's copy: replaced by the target's
])
def test_quant_method_per_layer(cls, prefix, want):
    import vllm.model_executor.layers.linear as L
    import vllm.model_executor.layers.vocab_parallel_embedding as E

    from vllm_exl3_plugin.quantization import EXL3Config

    layer_cls = getattr(L, cls, None) or getattr(E, cls)
    method = EXL3Config.from_config(_qcfg()).get_quant_method(_layer(layer_cls), prefix)
    assert (type(method).__name__ if method is not None else None) == want


@pytest.mark.parametrize("limits,refused", [
    ({"image": 0, "video": 0}, False),  # production's text-only argv: the tower is not built
    ({"image": 4, "video": 0}, True),
    (None, True),  # no current vLLM config: assume the tower is built
])
def test_quantized_vision_tower_refused_unless_skipped(monkeypatch, limits, refused):
    """erlidev's 6-bit vision tower (vision_bits) cannot load into vLLM's bf16 tower; the plugin
    refuses it at construction when the tower is built, and lets vLLM skip it otherwise. Text
    layers are unaffected either way."""
    import types

    import vllm.config
    import vllm.model_executor.layers.linear as L

    from vllm_exl3_plugin.quantization import EXL3Config

    cfg = None
    if limits is not None:
        mm = types.SimpleNamespace(get_limit_per_prompt=limits.__getitem__)
        cfg = types.SimpleNamespace(model_config=types.SimpleNamespace(multimodal_config=mm))
    monkeypatch.setattr(vllm.config, "get_current_vllm_config_or_none", lambda: cfg)
    swift = EXL3Config.from_config(_qcfg(SWIFT_DIR))
    text = swift.get_quant_method(_layer(L.QKVParallelLinear), "language_model.model.layers.3.self_attn.qkv_proj")
    assert type(text).__name__ == "EXL3LinearMethod"
    vis = _layer(L.QKVParallelLinear), "visual.blocks.0.attn.qkv"
    if refused:
        with pytest.raises(NotImplementedError, match="vision_bits 6.*limit-mm-per-prompt"):
            swift.get_quant_method(*vis)
    else:
        assert type(swift.get_quant_method(*vis)).__name__ == "UnquantizedLinearMethod"
    # turboderp's bf16 tower loads unquantized whatever the limits
    assert type(EXL3Config.from_config(_qcfg()).get_quant_method(*vis)).__name__ == "UnquantizedLinearMethod"
