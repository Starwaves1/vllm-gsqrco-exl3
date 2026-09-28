"""GDN weight round trip: HF checkpoint -> llama.cpp b11211 converter -> plugin
Qwen3.5 adapter must give back the HF tensors under vLLM's names.

The converter is b11211's own `Qwen3_5TextModel.modify_tensors` (V-head
reorder, A_log -> -exp, conv1d squeeze, +1 on norms except linear_attn.norm),
built with __new__ so no checkpoint or GGUF writer is needed. Dimensions are
Swift 1.5's GDN dims (16 K heads, 48 V heads, head dims 128); hidden size is
shrunk. Values are small dyadic rationals so +1/-1 on norms is exact.
"""

import sys
from types import SimpleNamespace

import gguf
import pytest
import torch
from conftest import LLAMA_CPP

sys.path.insert(0, LLAMA_CPP)
from conversion.qwen import Qwen3_5TextModel  # noqa: E402

from vllm_gguf_plugin.quantization.layout import GGUFHeadTilingLayout  # noqa: E402
from vllm_gguf_plugin.weights_adapter.qwen3_5 import (  # noqa: E402
    Qwen35GGUFAdapter,
    Qwen35MtpGGUFAdapter,
    _map_tensor_name,
    build_qwen35_mtp_mapper,
    build_qwen35_text_mapper,
)

H = 8  # hidden size (shrunk)
NK, NV, DK, DV, CONV = 16, 48, 128, 128, 4
N_LAYERS, MTP = 64, 64
TEXT = SimpleNamespace(
    hidden_size=H, linear_num_key_heads=NK, linear_num_value_heads=NV,
    linear_key_head_dim=DK, linear_value_head_dim=DV,
)
MODEL_CONFIG = SimpleNamespace(
    model="/nonexistent",  # no mtp_draft_vocab_ids.pt: full-vocab draft head
    hf_config=SimpleNamespace(
        model_type="qwen3_5", get_text_config=lambda: TEXT,
        vision_config=SimpleNamespace(temporal_patch_size=2),
    )
)


def _converter():
    m = Qwen3_5TextModel.__new__(Qwen3_5TextModel)
    m.hparams = {
        "hidden_size": H, "linear_num_key_heads": NK, "linear_num_value_heads": NV,
        "linear_key_head_dim": DK, "linear_value_head_dim": DV,
        "linear_conv_kernel_dim": CONV, "num_hidden_layers": N_LAYERS,
    }
    m.block_count = N_LAYERS + 1
    m.tensor_map = gguf.get_tensor_name_map(gguf.MODEL_ARCH.QWEN35, N_LAYERS + 1)
    m.fuse_qkv = False
    m.fuse_gate_up_exps = False
    return m


def _rand(*shape, gen):
    return torch.randint(-64, 64, shape, generator=gen).to(torch.float32) / 64


def _hf_tensors():
    g = torch.Generator().manual_seed(0)
    qkv_rows = 2 * NK * DK + NV * DV
    L, A = "model.layers.0.", "model.layers.3."
    return {
        L + "input_layernorm.weight": _rand(H, gen=g),
        L + "post_attention_layernorm.weight": _rand(H, gen=g),
        L + "linear_attn.in_proj_qkv.weight": _rand(qkv_rows, H, gen=g),
        L + "linear_attn.in_proj_z.weight": _rand(NV * DV, H, gen=g),
        L + "linear_attn.in_proj_a.weight": _rand(NV, H, gen=g),
        L + "linear_attn.in_proj_b.weight": _rand(NV, H, gen=g),
        L + "linear_attn.A_log": _rand(NV, gen=g),
        L + "linear_attn.dt_bias": _rand(NV, gen=g),
        L + "linear_attn.conv1d.weight": _rand(qkv_rows, 1, CONV, gen=g),
        L + "linear_attn.norm.weight": _rand(DV, gen=g),
        L + "linear_attn.out_proj.weight": _rand(H, NV * DV, gen=g),
        A + "self_attn.q_norm.weight": _rand(256, gen=g),
        A + "self_attn.k_norm.weight": _rand(256, gen=g),
        A + "input_layernorm.weight": _rand(H, gen=g),
    }


def _mtp_hf_tensors():
    # Names as the converter sees them after filter_tensors (mtp.* -> layer 64).
    g = torch.Generator().manual_seed(1)
    L = f"model.layers.{MTP}."
    return {
        L + "enorm.weight": _rand(H, gen=g),
        L + "hnorm.weight": _rand(H, gen=g),
        L + "shared_head.norm.weight": _rand(H, gen=g),
        L + "eh_proj.weight": _rand(H, 2 * H, gen=g),
        L + "input_layernorm.weight": _rand(H, gen=g),
        L + "self_attn.q_norm.weight": _rand(256, gen=g),
    }


def _convert(hf):
    conv, out = _converter(), {}
    for name, t in hf.items():
        bid = int(name.split(".")[2])
        for gname, gt in conv.modify_tensors(t.clone(), name, bid):
            assert gname not in out
            out[gname] = gt.contiguous()
    return out


def _vllm_name(hf_name):
    return "model.language_model." + hf_name.removeprefix("model.")


def _run_adapter(gg, quantized=()):
    mapper = build_qwen35_text_mapper(is_multimodal=True, is_moe=False)
    name_map = {n: _map_tensor_name(mapper, n) for n in gg}
    assert None not in name_map.values(), name_map

    def stream():
        for n, t in gg.items():
            if n in quantized:  # loader order: weight_type first, then the data
                yield name_map[n].replace("weight", "weight_type"), torch.tensor(12)
            yield name_map[n], t

    out = dict(Qwen35GGUFAdapter().transform_weights(stream(), MODEL_CONFIG))
    return name_map, out


@pytest.fixture(scope="module")
def converted():
    hf = _hf_tensors()
    return hf, _convert(hf)


def test_converter_really_reorders(converted):
    hf, gg = converted
    # Sanity: the converter did change the V-head layouts, so the round trip is not trivial.
    z = gg["blk.0.attn_gate.weight"]
    assert not torch.equal(z, hf["model.layers.0.linear_attn.in_proj_z.weight"])
    assert torch.equal(gg["blk.0.attn_norm.weight"], hf["model.layers.0.input_layernorm.weight"] + 1)
    assert torch.equal(gg["blk.0.ssm_norm.weight"], hf["model.layers.0.linear_attn.norm.weight"])
    assert tuple(gg["blk.0.ssm_conv1d.weight"].shape) == (2 * NK * DK + NV * DV, CONV)


def test_dense_round_trip(converted):
    hf, gg = converted
    name_map, out = _run_adapter(gg)
    assert set(out) == {_vllm_name(n) for n in hf}
    for n, t in hf.items():
        got = out[_vllm_name(n)]
        assert got.shape == t.shape, (n, got.shape, t.shape)
        if n.endswith("A_log"):  # log(-(-exp(x))) is not bit-exact
            torch.testing.assert_close(got, t, rtol=0, atol=1e-6)
        else:
            assert torch.equal(got, t), n


def test_quantized_out_proj_uses_input_layout(converted):
    hf, gg = converted
    gname = "blk.0.ssm_out.weight"
    name_map, out = _run_adapter(gg, quantized={gname})
    vname = name_map[gname]
    # Packed columns can't be permuted, so the stored (GGML-tiled) weight is kept ...
    assert torch.equal(out[vname], gg[gname])
    # ... and the linear layer permutes activations instead.
    layouts = Qwen35GGUFAdapter().get_linear_layouts(None, MODEL_CONFIG, name_map)
    assert set(layouts) == {vname.removesuffix(".weight")}
    layout = layouts[vname.removesuffix(".weight")]
    assert layout == GGUFHeadTilingLayout(heads_per_group=NV // NK, head_dim=DV)
    x = _rand(5, NV * DV, gen=torch.Generator().manual_seed(2)).double()
    w_hf = hf["model.layers.0.linear_attn.out_proj.weight"].double()
    assert torch.equal(layout.input_to_gguf(x) @ out[vname].double().T, x @ w_hf.T)


def test_mtp_norms_round_trip():
    hf = _mtp_hf_tensors()
    gg = _convert(hf)
    mapper = build_qwen35_mtp_mapper(MTP, is_moe=False)
    name_map = {n: _map_tensor_name(mapper, n) for n in gg}
    out = dict(Qwen35MtpGGUFAdapter().transform_weights(
        ((name_map[n], t) for n, t in gg.items()), MODEL_CONFIG))
    expect = {
        "enorm.weight": "mtp.pre_fc_norm_embedding.weight",
        "hnorm.weight": "mtp.pre_fc_norm_hidden.weight",
        "shared_head.norm.weight": "mtp.norm.weight",
        "eh_proj.weight": "mtp.fc.weight",
        "input_layernorm.weight": "mtp.layers.0.input_layernorm.weight",
        "self_attn.q_norm.weight": "mtp.layers.0.self_attn.q_norm.weight",
    }
    assert set(out) == set(expect.values()), sorted(out)
    for suffix, vname in expect.items():
        assert torch.equal(out[vname], hf[f"model.layers.{MTP}.{suffix}"]), vname


def test_mtp_draft_lm_head_rows(tmp_path):
    """With mtp_draft_vocab_ids.pt in the model dir, the MTP adapter maps output.weight to
    mtp.draft_lm_head.weight and keeps exactly those rows (raw GGUF blocks, in id order)."""
    ids = torch.tensor([0, 3, 7, 8])
    torch.save(ids, tmp_path / "mtp_draft_vocab_ids.pt")
    cfg = SimpleNamespace(model=str(tmp_path), hf_config=MODEL_CONFIG.hf_config)
    w = torch.arange(10 * 144, dtype=torch.int32).to(torch.uint8).view(10, 144)  # 10 Q4_K rows
    weights = [("mtp.draft_lm_head.weight_type", torch.tensor(12)), ("mtp.draft_lm_head.weight", w)]
    out = dict(Qwen35MtpGGUFAdapter().transform_weights(iter(weights), cfg))
    assert torch.equal(out["mtp.draft_lm_head.weight"], w[ids])
    assert int(out["mtp.draft_lm_head.weight_type"]) == 12
