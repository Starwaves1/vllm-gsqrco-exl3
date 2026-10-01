"""EXL3 multi-row kernel (trellis-serve's Marlin-EXL3 behind EXL3_MR), CPU only.

- routing (ops._exl3_op) with EXL3_MR 0/1/2 at 16/17/32/64/144/145 rows, per bit width;
- exl3_linear's dispatch to exl3_gemm_mr and the unpack before the dequant routes, with CPU
  stand-ins for the shim ops;
- EXL3LinearMethod._mr_prepare: which parts get repacked (K4 under EXL3_MR=2 only);
- the shim (csrc/exl3_mr_shim.cu, _C_exl3_mr): ops registered without CUDA init, every guard
  (a valid call ends in "must be CUDA tensors"), and the K4 repack against an independent
  per-word reference plus the unpack round trip, run by the real C++ ops on CPU. Skips if
  _C_exl3_mr is not built (VLLM_EXL3_BUILD=1).
"""

import glob
import json
import os
import subprocess
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "plugin-exl3"))
PKG = os.path.join(ROOT, "plugin-exl3/vllm_exl3_plugin")

GEMM, MR, RECON, RECON_HAD = "exl3_gemm", "exl3_gemm_mr", "recon_hgemm", "recon_had_hgemm"
WIDTH = {2: 32, 3: 48, 4: 64, 5: 80, 6: 96}  # tile width 16*K


@pytest.fixture
def mr_mode(monkeypatch):
    from vllm_exl3_plugin import ops

    def set_mode(mode):
        monkeypatch.setattr(ops, "MR_MODE", mode)
        monkeypatch.setattr(ops, "MULTI_ROW_OP", ops.MR_OP if mode else None)
        return ops
    return set_mode


def test_default_is_off():
    """Without EXL3_MR the phase-1 table holds (the test environment does not set it)."""
    from vllm_exl3_plugin import ops

    assert os.environ.get("EXL3_MR") is None
    assert ops.MR_MODE == 0 and ops.MULTI_ROW_OP is None


# rows -> route, for a stored K3/K5 tensor (mr_ok) under EXL3_MR=1 or 2
ROWS = [16, 17, 32, 64, 144, 145, 384, 385]


@pytest.mark.parametrize("mode", [0, 1, 2])
@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6])
def test_routing_table(mr_mode, mode, bits):
    ops = mr_mode(mode)
    width = WIDTH[bits]
    repacked = ops.mr_repacks(width, True)
    mr_ok = not repacked and ops.mr_takes(width, True)
    got = [ops._exl3_op(n, mr_ok, repacked) for n in ROWS]
    if mode == 0 or bits in (2, 6) or (bits == 4 and mode == 1):
        want = [GEMM, GEMM, GEMM, GEMM, GEMM, RECON, RECON, RECON]
    elif bits == 4:  # mode 2: repacked, every row count up to 384 on the multi-row kernel
        want = [MR, MR, MR, MR, MR, MR, MR, RECON]
        assert [ops._exl3_op(n, False, True) for n in (1, 2, 8)] == [MR] * 3
    else:  # K3/K5, modes 1 and 2
        want = [GEMM, MR, MR, MR, MR, MR, MR, RECON]
    assert got == want
    assert ops._exl3_op(1024, mr_ok, repacked) == RECON_HAD


@pytest.mark.parametrize("width,mul1,takes,repacks", [
    (32, True, False, False), (48, True, True, False), (64, True, False, True),
    (80, True, True, False), (96, True, False, False), (56, True, False, False),
    (48, False, False, False), (64, False, False, False),
])
def test_eligibility(mr_mode, width, mul1, takes, repacks):
    ops = mr_mode(2)
    assert ops.mr_takes(width, mul1) == takes
    assert ops.mr_repacks(width, mul1) == repacks
    assert not mr_mode(1).mr_repacks(width, mul1)


@pytest.mark.parametrize("mode", [1, 2])
def test_capture_sizes_never_dequant(mr_mode, mode):
    """Graph-captured sizes (to 48) only reach the two warmed kernels, never dequant + GEMM."""
    ops = mr_mode(mode)
    for mr_ok, repacked in ((False, False), (True, False), (False, True)):
        assert {ops._exl3_op(n, mr_ok, repacked) for n in range(1, 49)} <= {GEMM, MR}


@pytest.mark.parametrize("value,want", [("0", "None 0"), ("1", "exl3_gemm_mr 1"), ("2", "exl3_gemm_mr 2"),
                                        ("3", "ValueError")])
def test_env(value, want):
    code = ("import sys; sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[2]); import no_gpu\n"
            "try:\n    from vllm_exl3_plugin import ops\nexcept ValueError:\n    print('ValueError'); raise SystemExit\n"
            "print(ops.MULTI_ROW_OP, ops.MR_MODE)")
    p = subprocess.run([sys.executable, "-c", code, os.path.join(ROOT, "plugin-exl3"), os.path.join(ROOT, "tools")],
                       capture_output=True, text=True, timeout=300,
                       env=dict(os.environ, EXL3_MR=value, CUDA_VISIBLE_DEVICES=""))
    assert p.stdout.strip().splitlines()[-1:] == [want], p.stderr[-2000:]


# ---------------------------------------------------------------------------
# exl3_linear with CPU stand-ins

def ref_repack(t: torch.Tensor) -> torch.Tensor:
    """K4 repack from its definition, word by word: 32-bit word l of tile (i, 4g + j) goes to
    [i, g, l, j] (trellis-serve exl3_marlin.cu repack_trellis)."""
    kt, nt, _ = t.shape
    words = t.contiguous().view(torch.int32)  # [kt, nt, 32], little-endian int16 pairs
    out = torch.empty(kt, nt // 4, 32, 4, dtype=torch.int32)
    for i in range(kt):
        for g in range(nt // 4):
            for j in range(4):
                for lane in range(32):
                    out[i, g, lane, j] = words[i, 4 * g + j, lane]
    return out


def ref_unpack(b: torch.Tensor) -> torch.Tensor:
    kt, g = b.shape[:2]
    return b.permute(0, 1, 3, 2).contiguous().view(kt, g * 4, 32).view(torch.int16)


class FakeShim:
    def __init__(self, n_out):
        self.n_out, self.calls = n_out, []

    def _y(self, x, out_fp32):
        dtype = torch.bfloat16 if x.dtype == torch.bfloat16 else torch.float if out_fp32 else torch.half
        return torch.zeros(x.shape[0], self.n_out, dtype=dtype)

    def exl3_gemm(self, x, trellis, suh, svh, mcg, mul1, out_fp32):
        assert trellis.dtype == torch.int16
        self.calls.append(("gemm", x.shape[0]))
        return self._y(x, out_fp32)

    def exl3_gemm_mr(self, x, trellis, suh, svh, mcg, mul1, out_fp32):
        self.calls.append(("mr", x.shape[0], trellis.dtype))
        return self._y(x, out_fp32)

    def exl3_mr_unpack(self, b):
        self.calls.append(("unpack",))
        return ref_unpack(b)

    def exl3_dequant(self, trellis, suh, svh, mcg, mul1, n_start, n_count, had):
        assert trellis.dtype == torch.int16 and trellis.dim() == 3
        self.calls.append(("dequant", had))
        return torch.zeros(trellis.shape[0] * 16, n_count, dtype=torch.half)

    def exl3_hgemm(self, a, b):
        return torch.zeros(a.shape[0], b.shape[1], dtype=torch.half)

    def exl3_had_r_128(self, x, pre, post, scale):
        return x.clone()


@pytest.fixture
def fake(monkeypatch):
    shim = FakeShim(256)
    for name in ("exl3_gemm", "exl3_gemm_mr", "exl3_mr_unpack", "exl3_dequant", "exl3_hgemm", "exl3_had_r_128"):
        monkeypatch.setattr(torch.ops._C_exl3, name, getattr(shim, name), raising=False)
    return shim


@pytest.mark.parametrize("mode,bits,rows,want", [
    (0, 3, 32, [("gemm", 32)]),
    (1, 3, 16, [("gemm", 16)]),
    (1, 3, 17, [("mr", 17, torch.int16)]),
    (1, 5, 144, [("mr", 144, torch.int16)]),
    (1, 4, 64, [("gemm", 64)]),
    (1, 2, 64, [("gemm", 64)]),
    (1, 3, 145, [("mr", 145, torch.int16)]),
    (1, 3, 385, [("dequant", False)]),
    (1, 2, 145, [("dequant", False)]),
])
def test_linear_dispatch_stored(mr_mode, fake, mode, bits, rows, want):
    ops = mr_mode(mode)
    k = 128
    trellis = torch.zeros(k // 16, fake.n_out // 16, WIDTH[bits], dtype=torch.int16)
    x, s = torch.zeros(rows, k, dtype=torch.half), torch.zeros(k, dtype=torch.half)
    y = ops.exl3_linear(x, trellis, s, torch.zeros(fake.n_out, dtype=torch.half), False, True, True)
    assert y.shape == (rows, fake.n_out) and y.dtype == torch.float
    assert fake.calls == want


@pytest.mark.parametrize("rows,want", [
    (1, [("mr", 1, torch.int32)]), (16, [("mr", 16, torch.int32)]), (144, [("mr", 144, torch.int32)]),
    (145, [("mr", 145, torch.int32)]), (385, [("unpack",), ("dequant", False)]),
    (1024, [("unpack",), ("dequant", True)]),
])
def test_linear_dispatch_repacked(mr_mode, fake, rows, want):
    ops = mr_mode(2)
    k = 128
    b = torch.zeros(k // 16, fake.n_out // 64, 32, 4, dtype=torch.int32)
    x, s = torch.zeros(rows, k, dtype=torch.half), torch.zeros(k, dtype=torch.half)
    y = ops.exl3_linear(x, b, s, torch.zeros(fake.n_out, dtype=torch.half), False, True, False)
    assert y.shape == (rows, fake.n_out) and y.dtype == torch.half
    assert fake.calls == want


@pytest.mark.parametrize("bits,rows,want", [
    (3, 32, [("mr", 32, torch.int16)]), (2, 32, [("gemm", 32)]), (3, 8, [("gemm", 8)]),
    (3, 385, [("dequant", False)]),
])
def test_linear_bf16_contract(mr_mode, fake, bits, rows, want):
    """bf16 x (EXL3_MR_GLUE): bf16 out on every route; only exl3_gemm_mr sees bf16 x."""
    ops = mr_mode(2)
    seen = []
    orig = fake.exl3_gemm

    def gemm(x, *a):
        seen.append(x.dtype)
        return orig(x, *a)
    fake.exl3_gemm = gemm
    torch.ops._C_exl3.exl3_gemm = gemm
    k = 128
    trellis = torch.zeros(k // 16, fake.n_out // 16, WIDTH[bits], dtype=torch.int16)
    x, s = torch.zeros(rows, k, dtype=torch.bfloat16), torch.zeros(k, dtype=torch.half)
    y = ops.exl3_linear(x, trellis, s, torch.zeros(fake.n_out, dtype=torch.half), False, True, True)
    assert y.dtype == torch.bfloat16 and y.shape == (rows, fake.n_out)
    assert fake.calls == want and all(d == torch.half for d in seen)
    assert ops.exl3_linear_fake(x, trellis, s, s, False, True, True).dtype == torch.bfloat16


def test_glue_needs_mode_2():
    from vllm_exl3_plugin import ops

    assert ops.MR_GLUE is False  # default environment


@pytest.mark.parametrize("kt,nt,chunk", [(8, 8, 1 << 20), (24, 16, 2048), (16, 12, 4096)])
def test_repack_k4_inplace(kt, nt, chunk):
    """ops.repack_k4_ == the per-word definition, on the same storage (chunked: rows at a time)."""
    from vllm_exl3_plugin.ops import repack_k4_

    t = torch.randint(-32768, 32768, (kt, nt, 64), dtype=torch.int32, generator=torch.Generator().manual_seed(kt)).to(torch.int16)
    want = ref_repack(t.clone())
    b = repack_k4_(t, chunk_bytes=chunk)
    assert b.data_ptr() == t.data_ptr() and b.dtype == torch.int32 and b.is_contiguous()
    assert torch.equal(b, want) and torch.equal(ref_unpack(b), ref_unpack(want))


@pytest.mark.parametrize("on", [False, True])
def test_embed_host_method(monkeypatch, on):
    """EXL3_EMBED_HOST: the token embedding gets the host method; lm_head / draft head do not."""
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding

    from vllm_exl3_plugin import ops
    from vllm_exl3_plugin.format import EXL3QuantConfig
    from vllm_exl3_plugin.quantization import EXL3Config
    from vllm_exl3_plugin.quantization.embedding import EXL3HostEmbeddingMethod

    monkeypatch.setattr(ops, "EMBED_HOST", on)
    cfg = EXL3Config(EXL3QuantConfig(bits=3.5, head_bits=6, mtp_bits=4, codebook="mul1"))
    emb = VocabParallelEmbedding.__new__(VocabParallelEmbedding)
    head = ParallelLMHead.__new__(ParallelLMHead)
    m = cfg.get_quant_method(emb, "model.language_model.embed_tokens")
    assert isinstance(m, EXL3HostEmbeddingMethod) if on else m is None
    assert not isinstance(cfg.get_quant_method(head, "mtp.draft_lm_head"), EXL3HostEmbeddingMethod)
    assert not isinstance(cfg.get_quant_method(head, "lm_head"), EXL3HostEmbeddingMethod)


def test_embed_host_cpu_weight_stays():
    """On CPU (no CUDA weight) the method is a plain embedding: nothing moved."""
    from vllm_exl3_plugin.quantization.embedding import EXL3HostEmbeddingMethod

    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(torch.randn(16, 8).to(torch.bfloat16), requires_grad=False)
    m = EXL3HostEmbeddingMethod()
    m.process_weights_after_loading(layer)
    ids = torch.tensor([3, 0, 15])
    assert m.table is None and torch.equal(m.embedding(layer, ids), layer.weight[ids])


def test_fake_impl_repacked():
    from vllm_exl3_plugin.ops import exl3_linear_fake

    x = torch.empty(7, 5120, dtype=torch.half, device="meta")
    b = torch.empty(320, 16, 32, 4, dtype=torch.int32, device="meta")
    assert exl3_linear_fake(x, b, None, None, False, True, True).shape == (7, 1024)


# ---------------------------------------------------------------------------
# EXL3LinearMethod._mr_prepare (CPU tensors: repack runs, warmup is CUDA-only)

@pytest.fixture
def method(monkeypatch):
    from vllm_exl3_plugin.format import EXL3QuantConfig
    from vllm_exl3_plugin.quantization import EXL3Config
    from vllm_exl3_plugin.quantization import linear as L

    monkeypatch.setattr(L, "get_tensor_model_parallel_world_size", lambda: 1)
    return L.EXL3LinearMethod(EXL3Config(EXL3QuantConfig(bits=3.5, head_bits=6, mtp_bits=4, codebook="mul1")))


def _qkv_layer(method, g):
    """q K4, k K5, v K2 (k=128): one part per bit-width class."""
    layer = torch.nn.Module()
    layer.prefix = "qkv"
    method.create_weights(layer, 128, [256, 128, 128], 128, 512, torch.bfloat16, weight_loader=None)
    for sid, n, bits in (("q", 256, 4), ("k", 128, 5), ("v", 128, 2)):
        ts = {"trellis": torch.randint(-32768, 32767, (8, n // 16, WIDTH[bits]), dtype=torch.int16, generator=g),
              "suh": torch.ones(128, dtype=torch.half), "svh": torch.ones(n, dtype=torch.half),
              "mul1": torch.tensor(0x83DCD12D, dtype=torch.int64).to(torch.int32)}
        for name, t in ts.items():
            p = getattr(layer, name)
            p.weight_loader(p, t, sid)
    return layer


@pytest.mark.parametrize("mode", [0, 1, 2])
def test_mr_prepare(mr_mode, method, monkeypatch, mode):
    ops = mr_mode(mode)
    monkeypatch.setattr(ops, "MR_AVAILABLE", True)
    layer = _qkv_layer(method, torch.Generator().manual_seed(0))
    method.process_weights_after_loading(layer)
    stored = [getattr(layer, f"exl3_trellis_{i}") for i in range(3)]
    assert layer.exl3_bits == [4.0, 5.0, 2.0]
    if mode == 2:
        assert stored[0].dtype == torch.int32 and tuple(stored[0].shape) == (8, 4, 32, 4)
        assert torch.equal(ref_unpack(stored[0]), _qkv_layer(method, torch.Generator().manual_seed(0))
                           .trellis.exl3_parts["q"])
        assert "exl3_trellis_0" in dict(layer.named_parameters())
    else:
        assert stored[0].dtype == torch.int16
    assert [t.dtype for t in stored[1:]] == [torch.int16, torch.int16]


def test_mr_prepare_needs_library(mr_mode, method, monkeypatch):
    ops = mr_mode(1)
    monkeypatch.setattr(ops, "MR_AVAILABLE", False)
    layer = _qkv_layer(method, torch.Generator().manual_seed(0))
    with pytest.raises(RuntimeError, match="_C_exl3_mr is not built"):
        method.process_weights_after_loading(layer)


# ---------------------------------------------------------------------------
# The shim: registration, guards, repack (needs _C_exl3_mr)

OPS = ("exl3_gemm_mr", "exl3_mr_repack", "exl3_mr_unpack", "exl3_mr_warmup", "exl3_embed_host")
K_IN, N_OUT = 5120, 1024

_CHILD = r"""
import json, sys
sys.path.insert(0, sys.argv[2])
import no_gpu
import torch
torch.ops.load_library(sys.argv[1])
ops = torch.ops._C_exl3
cases = json.loads(sys.argv[3])
out = {"registered": [n for n in json.loads(sys.argv[4]) if hasattr(ops, n)]}

def t(spec):
    if spec is None:
        return None
    shape, dtype = spec["shape"], getattr(torch, spec.get("dtype", "float16"))
    off = spec.get("offset", 0)
    n = 1
    for s in shape:
        n *= s
    buf = torch.zeros(n + off + 64, dtype=dtype)
    x = buf[off:off + n].view(shape)
    if spec.get("t"):
        x = torch.zeros(list(reversed(shape[:2])) + list(shape[2:]), dtype=dtype).transpose(0, 1)
    return x

res = {}
for name, c in cases.items():
    a = c["args"]
    try:
        if c["op"] == "exl3_gemm_mr":
            ops.exl3_gemm_mr(t(a["x"]), t(a["trellis"]), t(a["suh"]), t(a["svh"]), a["mcg"], a["mul1"], a["out_fp32"])
        elif c["op"] == "exl3_mr_warmup":
            ops.exl3_mr_warmup(t(a["trellis"]), t(a["suh"]), t(a["svh"]), False, True, a["rows"], True)
        elif c["op"] == "exl3_mr_repack":
            ops.exl3_mr_repack(t(a["trellis"]))
        elif c["op"] == "exl3_mr_unpack":
            ops.exl3_mr_unpack(t(a["trellis"]))
        elif c["op"] == "exl3_embed_host":
            ops.exl3_embed_host(t(a["ids"]), a["ptr"], a["rows"], a["cols"])
        res[name] = "no error"
    except RuntimeError as e:
        res[name] = str(e).splitlines()[0][:300]
out["results"] = res

# repack on synthetic blocks: every int16 value, odd/even lanes, several tiles
g = torch.Generator().manual_seed(0)
for kt, nt in ((8, 8), (16, 24)):
    tr = torch.randint(-32768, 32768, (kt, nt, 64), dtype=torch.int32, generator=g).to(torch.int16)
    b = ops.exl3_mr_repack(tr)
    torch.save({"tr": tr, "b": b, "back": ops.exl3_mr_unpack(b)}, sys.argv[5] + f"/repack-{kt}x{nt}.pt")
no_gpu.assert_no_gpu_libs()
out["cuda_initialized"] = torch.cuda.is_initialized()
print(json.dumps(out))
"""


def T(*shape, dtype="float16", **kw):
    return dict(shape=list(shape), dtype=dtype, **kw)


def tr(k=K_IN, n=N_OUT, width=48, **kw):
    return T(k // 16, n // 16, width, dtype=kw.pop("dtype", "int16"), **kw)


def rp(k=K_IN, n=N_OUT, **kw):
    return T(k // 16, n // 64, 32, 4, dtype="int32", **kw)


def gemm(x=None, trellis=None, suh=None, svh=None, mcg=False, mul1=True, out_fp32=True):
    return dict(op="exl3_gemm_mr", args=dict(x=x or T(32, K_IN), trellis=trellis or tr(), suh=suh or T(K_IN),
                                             svh=svh or T(N_OUT), mcg=mcg, mul1=mul1, out_fp32=out_fp32))


CUDA = "must be CUDA tensors"
CASES = {
    "gemm-valid-K3": (gemm(), CUDA),
    "gemm-valid-K5": (gemm(trellis=tr(width=80)), CUDA),
    "gemm-valid-K4-repacked": (gemm(trellis=rp()), CUDA),
    "gemm-valid-out-fp16": (gemm(out_fp32=False), CUDA),
    "gemm-valid-1-row": (gemm(x=T(1, K_IN)), CUDA),
    "gemm-valid-144-rows": (gemm(x=T(144, K_IN)), CUDA),
    "gemm-valid-0-rows": (gemm(x=T(0, K_IN)), CUDA),
    "gemm-K4-stored": (gemm(trellis=tr(width=64)), "K4 needs exl3_mr_repack"),
    "gemm-K2": (gemm(trellis=tr(width=32)), "must be K3 or K5"),
    "gemm-K6": (gemm(trellis=tr(width=96)), "must be K3 or K5"),
    "gemm-K3.5": (gemm(trellis=tr(width=56)), "must be K3 or K5"),
    "gemm-3inst": (gemm(mul1=False), "only the mul1 codebook is built"),
    "gemm-mcg": (gemm(mcg=True, mul1=False), "only the mul1 codebook is built"),
    "gemm-mcg-and-mul1": (gemm(mcg=True, mul1=True), "mcg and mul1 are exclusive"),
    "gemm-trellis-2d": (gemm(trellis=T(320, 3072, dtype="int16")), "must be K3 or K5"),
    "gemm-trellis-int32-3d": (gemm(trellis=tr(dtype="int32")), "exl3_mr_repack's int32 K4"),
    "gemm-trellis-float": (gemm(trellis=tr(dtype="float16")), "exl3_mr_repack's int32 K4"),
    "gemm-repacked-bad-inner": (gemm(trellis=T(320, 16, 4, 32, dtype="int32")), "exl3_mr_repack's int32 K4"),
    "gemm-trellis-noncontig": (gemm(trellis=tr(t=True)), "trellis must be contiguous"),
    "gemm-trellis-misaligned": (gemm(trellis=tr(offset=1)), "trellis must be 16-byte aligned"),
    "gemm-n-not-128": (gemm(trellis=tr(n=64), svh=T(64)), "multiples of 128"),
    "gemm-k-not-128": (gemm(x=T(4, 5104), trellis=tr(k=5104), suh=T(5104)), "multiples of 128"),
    "gemm-repacked-k-not-128": (gemm(x=T(4, 5104), trellis=rp(k=5104), suh=T(5104)), "multiples of 128"),
    "gemm-x-k-mismatch": (gemm(x=T(4, 4096)), "columns, the weight has k=5120"),
    "gemm-valid-x-bf16": (gemm(x=T(4, K_IN, dtype="bfloat16")), CUDA),
    "gemm-x-fp32": (gemm(x=T(4, K_IN, dtype="float32")), "x must be fp16 or bf16"),
    "gemm-x-1d": (gemm(x=T(K_IN)), "x must be 2-D"),
    "gemm-x-noncontig": (gemm(x=T(4, K_IN, t=True)), "x must be contiguous"),
    "gemm-x-misaligned": (gemm(x=T(4, K_IN, offset=1)), "x must be 16-byte aligned"),
    "gemm-suh-size": (gemm(suh=T(4096)), "suh must be 1-D of size 5120"),
    "gemm-suh-bf16": (gemm(suh=T(K_IN, dtype="bfloat16")), "suh must be fp16"),
    "gemm-svh-size": (gemm(svh=T(K_IN)), "svh must be 1-D of size 1024"),
    "gemm-svh-misaligned": (gemm(svh=T(N_OUT, offset=1)), "svh must be 16-byte aligned"),
    "gemm-repacked-svh-size": (gemm(trellis=rp(n=2048)), "svh must be 1-D of size 2048"),
    "warmup-valid": (dict(op="exl3_mr_warmup", args=dict(trellis=tr(), suh=T(K_IN), svh=T(N_OUT), rows=[17, 144])),
                     CUDA),
    "warmup-valid-repacked": (dict(op="exl3_mr_warmup", args=dict(trellis=rp(), suh=T(K_IN), svh=T(N_OUT),
                                                                  rows=list(range(1, 145)))), CUDA),
    "warmup-rows-0": (dict(op="exl3_mr_warmup", args=dict(trellis=tr(), suh=T(K_IN), svh=T(N_OUT), rows=[0])),
                      "row counts must be >= 1"),
    "warmup-K4-stored": (dict(op="exl3_mr_warmup", args=dict(trellis=tr(width=64), suh=T(K_IN), svh=T(N_OUT),
                                                             rows=[17])), "K4 needs exl3_mr_repack"),
    "repack-valid": (dict(op="exl3_mr_repack", args=dict(trellis=tr(width=64))), "no error"),
    "repack-K3": (dict(op="exl3_mr_repack", args=dict(trellis=tr(width=48))), "only a K4 int16 trellis"),
    "repack-K6": (dict(op="exl3_mr_repack", args=dict(trellis=tr(width=96))), "only a K4 int16 trellis"),
    "repack-int32": (dict(op="exl3_mr_repack", args=dict(trellis=tr(width=64, dtype="int32"))),
                     "only a K4 int16 trellis"),
    "repack-noncontig": (dict(op="exl3_mr_repack", args=dict(trellis=tr(width=64, t=True))), "must be contiguous"),
    "repack-n-not-128": (dict(op="exl3_mr_repack", args=dict(trellis=tr(n=64, width=64))), "multiples of 128"),
    "unpack-valid": (dict(op="exl3_mr_unpack", args=dict(trellis=rp())), "no error"),
    "embed-valid": (dict(op="exl3_embed_host", args=dict(ids=T(6, dtype="int64"), ptr=4096, rows=248320, cols=5120)),
                    "ids must be a CUDA tensor"),
    "embed-ids-int32": (dict(op="exl3_embed_host", args=dict(ids=T(6, dtype="int32"), ptr=4096, rows=8, cols=64)),
                        "ids must be contiguous 1-D int64"),
    "embed-ids-2d": (dict(op="exl3_embed_host", args=dict(ids=T(2, 3, dtype="int64"), ptr=4096, rows=8, cols=64)),
                     "ids must be contiguous 1-D int64"),
    "embed-null": (dict(op="exl3_embed_host", args=dict(ids=T(6, dtype="int64"), ptr=0, rows=8, cols=64)),
                   "16-byte aligned bf16"),
    "embed-misaligned": (dict(op="exl3_embed_host", args=dict(ids=T(6, dtype="int64"), ptr=4100, rows=8, cols=64)),
                         "16-byte aligned bf16"),
    "embed-cols": (dict(op="exl3_embed_host", args=dict(ids=T(6, dtype="int64"), ptr=4096, rows=8, cols=60)),
                   "16-byte aligned bf16"),
    "unpack-int16": (dict(op="exl3_mr_unpack", args=dict(trellis=tr(width=64))), "b must be exl3_mr_repack's"),
    "unpack-bad-inner": (dict(op="exl3_mr_unpack", args=dict(trellis=T(320, 16, 4, 32, dtype="int32"))),
                         "b must be exl3_mr_repack's"),
}


def _so():
    found = glob.glob(os.path.join(PKG, "_C_exl3_mr*.so"))
    return found[0] if found else None


@pytest.fixture(scope="module")
def child(tmp_path_factory):
    so = _so()
    if so is None:
        pytest.skip(f"not built: {PKG}/_C_exl3_mr*.so (VLLM_EXL3_BUILD=1)")
    out = tmp_path_factory.mktemp("mr")
    p = subprocess.run(
        [sys.executable, "-c", _CHILD, so, os.path.join(ROOT, "tools"),
         json.dumps({k: v[0] for k, v in CASES.items()}), json.dumps(OPS), str(out)],
        capture_output=True, text=True, timeout=300, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    assert p.returncode == 0, p.stderr[-3000:]
    res = json.loads(p.stdout.strip().splitlines()[-1])
    res["dir"] = out
    return res


def test_ops_registered_without_cuda_init(child):
    assert child["registered"] == list(OPS)
    assert child["cuda_initialized"] is False


@pytest.mark.parametrize("name", list(CASES))
def test_guard(child, name):
    got = child["results"][name]
    assert CASES[name][1] in got, got


@pytest.mark.parametrize("shape", ["8x8", "16x24"])
def test_repack_round_trip(child, shape):
    """The C++ repack equals the per-word definition and unpack restores every bit."""
    d = torch.load(child["dir"] / f"repack-{shape}.pt")
    assert d["b"].dtype == torch.int32 and d["b"].shape == ref_repack(d["tr"]).shape
    assert torch.equal(d["b"], ref_repack(d["tr"]))
    assert d["back"].dtype == torch.int16 and torch.equal(d["back"], d["tr"])
