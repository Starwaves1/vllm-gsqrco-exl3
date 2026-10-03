# SPDX-License-Identifier: Apache-2.0
"""EXL3 linear method: dense linears and the lm_head.

A fused vLLM layer (qkv_proj, gate_up_proj, GDN in_proj_qkvz) is several checkpoint tensors,
and EXL3 tensors cannot be concatenated: each has its own bit width K and its own suh (the
quantizer folds per-tensor RMS into it), and the input Hadamard runs on x * suh. So the layer
keeps one part per checkpoint tensor (the GGUF plugin's per-shard pattern) and apply() runs one
product per part and concatenates. A checkpoint tensor can cover several vLLM shards at once:
GDN in_proj_qkv arrives with shard id (0, 1, 2), in_proj_z with 3 (Qwen3_5Model's mapper).

Loading: create_weights registers placeholder params (trellis, suh, svh and the codebook flag
the config implies, mul1 or mcg) whose weight_loader stores each loaded tensor under its shard
id; no vLLM loader is patched. process_weights_after_loading checks the parts against the
layer's partition sizes and replaces the placeholders with per-part params
exl3_{trellis,suh,svh}_{i}. On CUDA it then runs exl3_warmup for each new shape (outside any
graph capture: model loading precedes vLLM's profiling and capture). With EXL3_MR set
(ops.py) it then repacks the K4 trellises (EXL3_MR=2) and warms exl3_gemm_mr for every part
the multi-row kernel takes, at every row count routed to it.
"""

from __future__ import annotations

import torch
from torch.nn import Parameter
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.utils import set_weight_attrs

from .. import ops
from ..format import as_uint32, bits_from_tile

_QKV = {"q": 0, "k": 1, "v": 2}
_TENSORS = ("trellis", "suh", "svh")
# Row counts exl3_warmup runs per shape, one per vendored bucket (exl3_shim.cu row_bucket):
# 1 (GEMV m == 1), 2, 4, 8 (GEMV to 8 rows, autotune buckets), 16 (every m >= 9).
WARMUP_ROWS = [1, 2, 4, 8, 16]
_WARMED: set[tuple] = set()


def _shard_indices(shard_id, num_shards: int) -> list[int]:
    """vLLM output shards a checkpoint tensor with this shard id covers."""
    if shard_id is None:
        return list(range(num_shards))
    if isinstance(shard_id, tuple):
        return list(shard_id)
    if isinstance(shard_id, str):
        return [_QKV[shard_id]]
    return [int(shard_id)]


def _make_loader(codebook_param: str | None, codebook_mult: int | None):
    def weight_loader(param: Parameter, loaded_weight: torch.Tensor, loaded_shard_id=None):
        """Store one checkpoint tensor of an EXL3 layer under its shard id (int, str, tuple
        or None). The codebook flag is checked against the multiplier the kernels are
        compiled with, then stored like the rest."""
        if param.exl3_name == codebook_param:
            value = as_uint32(int(loaded_weight.reshape(-1)[0].item()))
            if value != codebook_mult:
                raise ValueError(
                    f"EXL3 {codebook_param} multiplier {value:#x}, the kernels use {codebook_mult:#x}"
                )
        # copy off the checkpoint's (mmap) tensor onto the parameter's device
        stored = torch.empty_like(loaded_weight, device=param.device)
        stored.copy_(loaded_weight)
        key = loaded_shard_id if not isinstance(loaded_shard_id, list) else tuple(loaded_shard_id)
        param.exl3_parts[key] = stored

    return weight_loader


class EXL3LinearMethod(LinearMethodBase):
    """Linear method for EXL3 tensors (see the module docstring)."""

    def __init__(self, quant_config) -> None:
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        del input_size, output_size, extra_weight_attrs  # our own loader, not vLLM's
        if get_tensor_model_parallel_world_size() > 1:
            raise NotImplementedError("EXL3 plugin: tensor parallelism is not supported yet")
        quant = self.quant_config.quant
        loader = _make_loader(quant.codebook_param, quant.codebook_mult)
        names = [*_TENSORS] + ([quant.codebook_param] if quant.codebook_param else [])
        for name in names:
            param = Parameter(torch.empty(0), requires_grad=False)
            param.exl3_name = name
            param.exl3_parts = {}
            set_weight_attrs(param, {"weight_loader": loader})
            layer.register_parameter(name, param)
        layer.exl3_placeholders = names
        layer.exl3_in = input_size_per_partition
        layer.exl3_out_sizes = list(output_partition_sizes)
        layer.exl3_dtype = params_dtype

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        quant = self.quant_config.quant
        where = getattr(layer, "prefix", type(layer).__name__)
        placeholders = {n: getattr(layer, n) for n in layer.exl3_placeholders}
        keys = list(placeholders["trellis"].exl3_parts)
        for name, p in placeholders.items():
            if set(p.exl3_parts) != set(keys):
                raise ValueError(f"EXL3 {where}: {name} loaded for shards {sorted(map(str, p.exl3_parts))}, "
                                 f"trellis for {sorted(map(str, keys))}")
        sizes = layer.exl3_out_sizes
        keys.sort(key=lambda k: _shard_indices(k, len(sizes))[0])
        covered = [i for k in keys for i in _shard_indices(k, len(sizes))]
        if covered != list(range(len(sizes))):
            raise ValueError(f"EXL3 {where}: checkpoint tensors cover output shards {covered}, "
                             f"the layer has {len(sizes)}")

        parts = []
        for key in keys:
            trellis = placeholders["trellis"].exl3_parts[key]
            suh = placeholders["suh"].exl3_parts[key]
            svh = placeholders["svh"].exl3_parts[key]
            n = sum(sizes[i] for i in _shard_indices(key, len(sizes)))
            k = layer.exl3_in
            if (trellis.dim() != 3 or trellis.dtype != torch.int16
                    or tuple(trellis.shape[:2]) != (k // 16, n // 16)):
                raise ValueError(f"EXL3 {where} shard {key}: trellis {tuple(trellis.shape)} "
                                 f"{trellis.dtype}, expected int16 [{k // 16}, {n // 16}, 16*K]")
            bits_from_tile(trellis.shape[2])
            for vec, size, label in ((suh, k, "suh"), (svh, n, "svh")):
                if tuple(vec.shape) != (size,) or vec.dtype != torch.half:
                    raise ValueError(f"EXL3 {where} shard {key}: {label} {tuple(vec.shape)} {vec.dtype}, "
                                     f"expected fp16 [{size}]")
            parts.append((trellis, suh, svh))

        for name in placeholders:
            delattr(layer, name)
        for i, tensors in enumerate(parts):
            for name, t in zip(_TENSORS, tensors):
                layer.register_parameter(f"exl3_{name}_{i}", Parameter(t, requires_grad=False))
        layer.exl3_num_parts = len(parts)
        layer.exl3_bits = [bits_from_tile(t.shape[2]) for t, _, _ in parts]

        if parts[0][0].device.type == "cuda":
            self._warmup(layer, parts, quant)
        if ops.MR_MODE:
            self._mr_prepare(layer, quant)

    def _warmup(self, layer, parts, quant) -> None:
        if not ops.OPS_AVAILABLE:
            raise RuntimeError("EXL3 plugin: _C_exl3 is not built (VLLM_EXL3_BUILD=1, see EXL3.md)")
        for trellis, suh, svh in parts:
            out_fp32 = layer.exl3_dtype != torch.half
            key = (trellis.device.index, tuple(trellis.shape), quant.codebook, out_fp32)
            if key in _WARMED:
                continue
            torch.ops._C_exl3.exl3_warmup(trellis, suh, svh, quant.mcg, quant.mul1, WARMUP_ROWS, out_fp32)
            _WARMED.add(key)

    def _mr_prepare(self, layer, quant) -> None:
        """EXL3_MR: under 2, K4 trellises become exl3_mr_repack's layout in place (ops.repack_k4_,
        after exl3_warmup, which reads the stored one). Then exl3_mr_warmup
        for every part exl3_gemm_mr takes, at each routed row count (the capture guard is
        per row count)."""
        if not ops.MR_AVAILABLE:
            raise RuntimeError(f"EXL3_MR={ops.MR_MODE} but _C_exl3_mr is not built (VLLM_EXL3_BUILD=1)")
        mr = torch.ops._C_exl3
        out_fp32 = layer.exl3_dtype != torch.half
        if ops.MR_CONCAT and self._mr_concat(layer, quant, out_fp32):
            return
        for i in range(layer.exl3_num_parts):
            trellis = getattr(layer, f"exl3_trellis_{i}")
            if ops.mr_repacks(trellis.shape[2], quant.mul1):
                trellis = ops.repack_k4_(trellis)
                setattr(layer, f"exl3_trellis_{i}", Parameter(trellis, requires_grad=False))
                rows = range(1, ops.GEMM_MAX_ROWS + 1)  # graphs stop at 48 rows; above 144 eager
            elif ops.mr_takes(trellis.shape[2], quant.mul1):
                rows = range(ops.MULTI_ROW_MIN, ops.GEMM_MAX_ROWS + 1)
            else:
                continue
            key = ("mr", trellis.device.index, tuple(trellis.shape), trellis.dtype, out_fp32)
            if trellis.device.type != "cuda" or key in _WARMED:
                continue
            mr.exl3_mr_warmup(trellis, getattr(layer, f"exl3_suh_{i}"), getattr(layer, f"exl3_svh_{i}"),
                              quant.mcg, quant.mul1, list(rows), out_fp32)
            _WARMED.add(key)

    def _mr_concat(self, layer, quant, out_fp32: bool) -> bool:
        """EXL3_MR_CONCAT: a fused layer whose parts all take exl3_gemm_mr at one K becomes one
        concatenated trellis (exl3_cat_*, the parts dropped: no second copy). Returns whether it did."""
        n = layer.exl3_num_parts
        ts = [getattr(layer, f"exl3_trellis_{i}") for i in range(n)]
        if n < 2 or len({t.shape[2] for t in ts}) != 1 or not (
                ops.mr_repacks(ts[0].shape[2], quant.mul1) or ops.mr_takes(ts[0].shape[2], quant.mul1)):
            return False
        if ops.mr_repacks(ts[0].shape[2], quant.mul1):
            ts = [ops.repack_k4_(t) for t in ts]
        widths = [ops.out_features(t) for t in ts]
        tensors = {"trellis": torch.cat(ts, dim=1),
                   "suh": torch.cat([getattr(layer, f"exl3_suh_{i}") for i in range(n)]),
                   "svh": torch.cat([getattr(layer, f"exl3_svh_{i}") for i in range(n)])}
        for i in range(n):
            for name in _TENSORS:
                delattr(layer, f"exl3_{name}_{i}")
        del ts
        for name, t in tensors.items():
            layer.register_parameter(f"exl3_cat_{name}", Parameter(t, requires_grad=False))
        layer.exl3_cat_widths = widths
        t = tensors["trellis"]
        key = ("mr-cat", t.device.index, tuple(t.shape), t.dtype, tuple(widths), out_fp32)
        if t.device.type == "cuda" and key not in _WARMED:
            ends = [sum(widths[:i + 1]) for i in range(n - 1)]
            torch.ops._C_exl3.exl3_mr_warmup_multi(t, tensors["suh"], tensors["svh"], ends, quant.mcg, quant.mul1,
                                                   list(range(1, ops.GEMM_MAX_ROWS + 1)), out_fp32)
            _WARMED.add(key)
        return True

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        quant = self.quant_config.quant
        # one opaque op per layer (ops.exl3_linear_parts: the parts and their concatenation). The
        # kernels take fp16: cast once per layer; with a bf16 model they write fp32 (no second fp16
        # rounding), cast once. Glue (ops.MR_GLUE): bf16 straight through instead, made contiguous as
        # the cast did (the compiled graph asserts the op's input strides)
        xh = x.reshape(-1, x.shape[-1])
        # A wide single-part layer (the lm_head) takes bf16 in every mode: a bf16 result, no fp32 copy of
        # the full-vocab logits when prompt_logprobs sends whole prompt chunks
        wide = layer.exl3_num_parts == 1 and ops.out_features(layer.exl3_trellis_0) > ops.RECON_SLICE_N
        glue = x.dtype == torch.bfloat16 and (ops.MR_GLUE or wide)
        xh = xh.contiguous() if glue else xh.to(torch.half)
        if hasattr(layer, "exl3_cat_widths"):  # EXL3_MR_CONCAT
            out = torch.ops.vllm._exl3_linear_cat(xh, layer.exl3_cat_trellis, layer.exl3_cat_suh, layer.exl3_cat_svh,
                                                  layer.exl3_cat_widths, quant.mcg, quant.mul1, x.dtype != torch.half)
            return self._finish(out, x, bias)
        n = range(layer.exl3_num_parts)
        out = torch.ops.vllm._exl3_linear_parts(
            xh,
            [getattr(layer, f"exl3_trellis_{i}") for i in n],
            [getattr(layer, f"exl3_suh_{i}") for i in n],
            [getattr(layer, f"exl3_svh_{i}") for i in n],
            quant.mcg,
            quant.mul1,
            x.dtype != torch.half,
        )
        return self._finish(out, x, bias)

    @staticmethod
    def _finish(out: torch.Tensor, x: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        out = out.to(x.dtype).reshape(*x.shape[:-1], out.shape[-1])
        if bias is not None:
            out = out + bias
        return out
