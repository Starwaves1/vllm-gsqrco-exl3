"""EXL3LinearMethod's loader and part assembly (vllm_exl3_plugin.quantization.linear), CPU.

Loads synthetic EXL3 tensors the way vLLM's loaders hand them over (one call per checkpoint
tensor with the shard id the model's mapper attached: None, "q"/"k"/"v", ints, or the GDN
in_proj_qkv tuple (0, 1, 2)) and checks the parts the layer ends up with, and the errors for
checkpoints that don't fit the layer.
"""

import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "plugin-exl3"))

MUL1 = 0x83DCD12D


@pytest.fixture(autouse=True)
def _phase1_routing(monkeypatch):
    """Phase 1's table (EXL3_MR=0, the hook from 17 rows): this file pins it; the multi-row
    defaults (EXL3_MR=2) are tests/cpu/test_exl3_mr.py's."""
    from vllm_exl3_plugin import ops

    monkeypatch.setattr(ops, "MR_MODE", 0)
    monkeypatch.setattr(ops, "MULTI_ROW_OP", None)
    monkeypatch.setattr(ops, "MULTI_ROW_MIN", 17)


@pytest.fixture
def method(monkeypatch):
    from vllm_exl3_plugin.format import EXL3QuantConfig
    from vllm_exl3_plugin.quantization import EXL3Config
    from vllm_exl3_plugin.quantization import linear as L

    monkeypatch.setattr(L, "get_tensor_model_parallel_world_size", lambda: 1)
    cfg = EXL3Config(EXL3QuantConfig(bits=3.5, head_bits=6, mtp_bits=4, codebook="mul1"))
    return L.EXL3LinearMethod(cfg)


def make_layer(method, k, sizes, prefix="layer"):
    layer = torch.nn.Module()
    layer.prefix = prefix
    method.create_weights(layer, k, sizes, k, sum(sizes), torch.bfloat16, weight_loader=None)
    return layer


def tensors(k, n, width=64, mult=MUL1, tag=0):
    tr = torch.full((k // 16, n // 16, width), tag, dtype=torch.int16)
    return {"trellis": tr, "suh": torch.full((k,), tag, dtype=torch.half),
            "svh": torch.full((n,), tag, dtype=torch.half),
            "mul1": torch.tensor(mult, dtype=torch.int64).to(torch.int32)}


def load(layer, ts, shard_id):
    for name, t in ts.items():
        p = getattr(layer, name)
        if shard_id is None:
            p.weight_loader(p, t)  # AutoWeightsLoader._load_param: two arguments
        else:
            p.weight_loader(p, t, shard_id)


def test_placeholders(method):
    layer = make_layer(method, 256, [128])
    assert layer.exl3_placeholders == ["trellis", "suh", "svh", "mul1"]
    assert all(getattr(layer, n).numel() == 0 for n in layer.exl3_placeholders)


def test_single_tensor(method):
    layer = make_layer(method, 256, [384])
    load(layer, tensors(256, 384, width=96), None)
    method.process_weights_after_loading(layer)
    assert layer.exl3_num_parts == 1 and layer.exl3_bits == [6.0]
    assert not hasattr(layer, "trellis") and "exl3_trellis_0" in dict(layer.named_parameters())
    assert layer.exl3_trellis_0.shape == (16, 24, 96) and layer.exl3_svh_0.shape == (384,)


def test_qkv_out_of_order(method):
    """q/k/v arrive in any order and with per-tensor bit widths; parts follow q, k, v."""
    layer = make_layer(method, 256, [512, 128, 128])
    load(layer, tensors(256, 128, width=48, tag=2), "v")
    load(layer, tensors(256, 512, width=64, tag=0), "q")
    load(layer, tensors(256, 128, width=80, tag=1), "k")
    method.process_weights_after_loading(layer)
    assert layer.exl3_num_parts == 3 and layer.exl3_bits == [4.0, 5.0, 3.0]
    assert [int(getattr(layer, f"exl3_trellis_{i}")[0, 0, 0]) for i in range(3)] == [0, 1, 2]
    assert [getattr(layer, f"exl3_svh_{i}").shape[0] for i in range(3)] == [512, 128, 128]


def test_gdn_in_proj_qkvz(method):
    """in_proj_qkv covers shards (0, 1, 2) as one tensor, in_proj_z is shard 3."""
    layer = make_layer(method, 256, [128, 128, 384, 384])
    load(layer, tensors(256, 384, tag=3), 3)
    load(layer, tensors(256, 640, tag=1), (0, 1, 2))
    method.process_weights_after_loading(layer)
    assert layer.exl3_num_parts == 2
    assert [getattr(layer, f"exl3_trellis_{i}").shape[1] * 16 for i in range(2)] == [640, 384]


def test_gate_up(method):
    layer = make_layer(method, 256, [512, 512])
    load(layer, tensors(256, 512, tag=1), 1)
    load(layer, tensors(256, 512, tag=0), 0)
    method.process_weights_after_loading(layer)
    assert [int(getattr(layer, f"exl3_suh_{i}")[0]) for i in range(2)] == [0, 1]


def test_missing_shard(method):
    layer = make_layer(method, 256, [512, 512])
    load(layer, tensors(256, 512), 0)
    with pytest.raises(ValueError, match=r"cover output shards \[0\], the layer has 2"):
        method.process_weights_after_loading(layer)


def test_missing_scale(method):
    layer = make_layer(method, 256, [512])
    ts = tensors(256, 512)
    del ts["svh"]
    load(layer, ts, None)
    with pytest.raises(ValueError, match="svh loaded for shards"):
        method.process_weights_after_loading(layer)


@pytest.mark.parametrize("bad,match", [
    ({"trellis": torch.zeros(16, 16, 64, dtype=torch.int16)}, r"expected int16 \[16, 32, 16\*K\]"),
    ({"trellis": torch.zeros(16, 32, 64, dtype=torch.int32)}, "expected int16"),
    ({"trellis": torch.zeros(16, 32, 50, dtype=torch.int16)}, "not 16\\*K"),
    ({"suh": torch.zeros(128, dtype=torch.half)}, r"suh \(128,\)"),
    ({"svh": torch.zeros(512, dtype=torch.bfloat16)}, "svh"),
])
def test_bad_shapes(method, bad, match):
    layer = make_layer(method, 256, [512])
    load(layer, tensors(256, 512) | bad, None)
    with pytest.raises(ValueError, match=match):
        method.process_weights_after_loading(layer)


def test_codebook_multiplier_checked(method):
    layer = make_layer(method, 256, [512])
    with pytest.raises(ValueError, match="multiplier 0xcbac1fed, the kernels use 0x83dcd12d"):
        load(layer, tensors(256, 512, mult=0xCBAC1FED), None)


def test_loader_copies(method):
    """The stored part is a copy, not a view of the checkpoint's (mmap) tensor."""
    layer = make_layer(method, 256, [512])
    ts = tensors(256, 512)
    load(layer, ts, None)
    ts["trellis"].fill_(7)
    method.process_weights_after_loading(layer)
    assert int(layer.exl3_trellis_0.abs().max()) == 0


def test_tensor_parallel_refused(monkeypatch, method):
    from vllm_exl3_plugin.quantization import linear as L

    monkeypatch.setattr(L, "get_tensor_model_parallel_world_size", lambda: 2)
    with pytest.raises(NotImplementedError, match="tensor parallelism"):
        make_layer(method, 256, [512])


def test_apply_concatenates_parts(monkeypatch, method):
    """apply(): one routed product per part, concatenated in shard order, bias added."""
    from vllm_exl3_plugin.quantization import linear as L

    layer = make_layer(method, 256, [512, 128, 128])
    for sid, n, tag in (("q", 512, 1), ("k", 128, 2), ("v", 128, 3)):
        load(layer, tensors(256, n, tag=tag), sid)
    method.process_weights_after_loading(layer)

    def fake_linear(x, trellis, suh, svh, mcg, mul1, out_fp32):
        assert (mcg, mul1, out_fp32, x.dtype) == (False, True, True, torch.half)  # bf16 model
        return torch.full((x.shape[0], svh.shape[0]), float(trellis[0, 0, 0]), dtype=torch.float)

    class FakeVllmOps:
        _exl3_linear = staticmethod(fake_linear)

    monkeypatch.setattr(L.torch.ops, "vllm", FakeVllmOps, raising=False)
    x = torch.zeros(2, 3, 256, dtype=torch.bfloat16)
    y = method.apply(layer, x, bias=torch.ones(768, dtype=torch.bfloat16))
    assert y.shape == (2, 3, 768) and y.dtype == torch.bfloat16
    assert y[0, 0, :512].eq(2).all() and y[0, 0, 512:640].eq(3).all() and y[0, 0, 640:].eq(4).all()
