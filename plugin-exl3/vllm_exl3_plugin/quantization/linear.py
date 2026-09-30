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
graph capture: model loading precedes vLLM's profiling and capture).
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
# Row counts exl3_warmup runs per shape: 1..16 each (autotune bucket and GEMV choice depend
# on them) and 17, which stands for every count above 16 (one autotune bucket).
WARMUP_ROWS = list(range(1, 18))
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

    def _warmup(self, layer, parts, quant) -> None:
        if not ops.OPS_AVAILABLE:
            raise RuntimeError("EXL3 plugin: _C_exl3 is not built (VLLM_EXL3_BUILD=1, see EXL3.md)")
        for trellis, suh, svh in parts:
            key = (trellis.device.index, tuple(trellis.shape), quant.codebook, layer.exl3_dtype)
            if key in _WARMED:
                continue
            torch.ops._C_exl3.exl3_warmup(trellis, suh, svh, quant.mcg, quant.mul1, WARMUP_ROWS, layer.exl3_dtype)
            _WARMED.add(key)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        quant = self.quant_config.quant
        x2 = x.reshape(-1, x.shape[-1])
        outs = [
            torch.ops.vllm._exl3_linear(
                x2,
                getattr(layer, f"exl3_trellis_{i}"),
                getattr(layer, f"exl3_suh_{i}"),
                getattr(layer, f"exl3_svh_{i}"),
                quant.mcg,
                quant.mul1,
            )
            for i in range(layer.exl3_num_parts)
        ]
        out = outs[0] if len(outs) == 1 else torch.cat(outs, dim=-1)
        out = out.reshape(*x.shape[:-1], out.shape[-1])
        if bias is not None:
            out = out + bias
        return out
