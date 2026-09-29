# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from itertools import groupby

import gguf
import torch
from gguf import GGMLQuantizationType as WeightType
from vllm import envs
from vllm.model_executor.layers.linear import (
    LinearMethodBase,
    UnquantizedLinearMethod,
    register_weight_loader_v2_supported_method,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

from .. import ops
from .layout import GGUFLinearLayout
from .params import (
    GGUFUninitializedWeightParameter,
    GGUFUninitializedWeightTypeParameter,
    GGUFWeightParameter,
    _gguf_ordered_shard_ids,
    _materialize_gguf_weight_parameter,
    _materialize_gguf_weight_type_parameter,
    _resolve_gguf_weight_loader,
    _resolve_gguf_weight_type_loader,
)
from .utils import (
    DEQUANT_TYPES,
    IMATRIX_QUANT_TYPES,
    MMQ_QUANT_TYPES,
    MMVQ_QUANT_TYPES,
    UNQUANTIZED_TYPES,
)


def _lcpp_reads_q8_1(n: int, weight_type: int) -> bool:
    """Route L runs n rows of this type through a kernel that reads X as q8_1
    blocks (MMVQ or the IQ3 kernel), not MMQ. MMQ is faster than MMVQ from 8
    rows (cloud/results/phase2/micro/micro.tsv); IQ3_S / IQ3_XXS up to 8 rows
    take the shim's own kernel, faster than both (cloud/results/phase3/item5)."""
    return n < 8 or (n == 8 and weight_type in (WeightType.IQ3_S, WeightType.IQ3_XXS))


def _fused_mul_mat_gguf(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_type: int,
    x_q8: torch.Tensor | None = None,
) -> torch.Tensor:
    """x @ weight.T. x_q8: x already quantized by _quantize_x_q8_1 for this
    product and others on the same x; used if this product reads q8_1."""
    if weight_type in IMATRIX_QUANT_TYPES:
        mmvq_safe = 8 if weight.shape[0] > 5120 else 16
    else:
        mmvq_safe = 2 if weight.shape[0] > 5120 else 6
    if x.shape[0] == 0:
        return torch.empty(x.shape[0], weight.shape[0], dtype=x.dtype, device=x.device)
    if weight_type in UNQUANTIZED_TYPES:
        return x @ weight.T
    if ops.LCPP_ENABLED and weight_type in ops.LCPP_QUANT_TYPES:
        C = torch.ops._C_gguf
        if not _lcpp_reads_q8_1(x.shape[0], weight_type):
            return C.lcpp_mul_mat_q(weight, x, weight_type, weight.shape[0])
        iq3 = weight_type in (WeightType.IQ3_S, WeightType.IQ3_XXS)
        op = C.lcpp_mul_mat_vec_iq3 if iq3 else C.lcpp_mul_mat_vec_q
        return op(weight, x, weight_type, weight.shape[0], x_q8)
    if x.shape[0] <= mmvq_safe and weight_type in MMVQ_QUANT_TYPES:
        y = ops.ggml_mul_mat_vec_a8(weight, x, weight_type, weight.shape[0])
    elif weight_type in MMQ_QUANT_TYPES:
        y = ops.ggml_mul_mat_a8(weight, x, weight_type, weight.shape[0])
    elif weight_type in DEQUANT_TYPES:
        block_size, type_size = gguf.GGML_QUANT_SIZES[weight_type]
        shape = (weight.shape[0], weight.shape[1] // type_size * block_size)
        weight = ops.ggml_dequantize(weight, weight_type, *shape, x.dtype)
        y = x @ weight.T
    else:
        weight_type = WeightType(weight_type)
        raise NotImplementedError(f"Unsupported GGUF quantization type: {weight_type}")
    return y


def _shard_weight(
    weight: torch.Tensor, start: int, end: int, size: int
) -> torch.Tensor:
    """Rows start:end, first `size` bytes, of a padded multi-shard weight.

    With VLLM_GGUF_LCPP=1 each run of adjacent same-type shards is stored
    contiguously from the start of its first shard's region (see
    _create_padded_weight_param), so for a run this is a view, not a copy.
    Under that layout the offsets of a shard inside a run do not locate its
    bytes: address runs (_shard_runs), not single shards."""
    if ops.LCPP_ENABLED:
        rows = end - start
        return weight[start:end].view(-1)[: rows * size].view(rows, size)
    return weight[start:end, :size].contiguous()


def _shard_runs(weight: torch.Tensor, shard_ids: list, weight_types: list[int]):
    """(rows, type) for each run of adjacent same-type shards of a padded
    multi-shard weight: one product per run instead of one per shard. Only
    VLLM_GGUF_LCPP=1 stores runs contiguously; otherwise every shard is its own
    run, as before (stock routing depends on each product's row count)."""
    offsets = weight.shard_offset_map
    key = (lambda p: p[1]) if ops.LCPP_ENABLED else (lambda p: p[0])
    for _, run in groupby(zip(shard_ids, weight_types), key=key):
        ids, types = zip(*run)
        weight_type = types[0]
        start, _, size = offsets[ids[0]]
        yield _shard_weight(weight, start, offsets[ids[-1]][1], size), weight_type


def _quantize_x_q8_1(x: torch.Tensor, weight_types: list[int]) -> torch.Tensor:
    """x as q8_1 blocks, quantized once for all products of weight_types on x
    that read q8_1 (Route L; the same bytes each would make). Left unfilled
    when none of them does: MMQ quantizes x itself, in its own layout."""
    n = x.shape[0]
    q8_1 = [t for t in weight_types if t in ops.LCPP_QUANT_TYPES and _lcpp_reads_q8_1(n, t)]
    if n and q8_1:
        return torch.ops._C_gguf.lcpp_quantize_q8_1(x, q8_1[0], False, False)
    return _quantize_x_q8_1_fake(x, weight_types)


def _fused_mul_mat_gguf_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_type: int,
    x_q8: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.empty(x.shape[0], weight.shape[0], dtype=x.dtype, device=x.device)


def _unquantized_gemm(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    """x @ weight.T (+ bias) for a GGUF F32/F16/BF16 tensor. Up to 8 rows as a
    batched gemv: for 2..8 rows x ~100 columns (GDN in_proj_ba) cuBLAS picks a
    GEMM with a few 1-warp CTAs, 25-36 us instead of 4-7
    (cloud/results/opt-p/micro-bf16b.txt)."""
    if x.shape[0] <= 8:
        y = torch.bmm(x.unsqueeze(1), weight.T.expand(x.shape[0], -1, -1)).squeeze(1)
        return y if bias is None else y + bias
    return torch.nn.functional.linear(x, weight, bias)


def _unquantized_gemm_fake(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    return x.new_empty(x.shape[0], weight.shape[0])


def _quantize_x_q8_1_fake(x: torch.Tensor, weight_types: list[int]) -> torch.Tensor:
    # block_q8_1: 32 int8 values + a half2 (scale, sum) = 36 bytes
    return torch.empty(x.shape[0] * x.shape[1] // 32 * 36, dtype=torch.uint8, device=x.device)


try:
    direct_register_custom_op(
        op_name="_fused_mul_mat_gguf",
        op_func=_fused_mul_mat_gguf,
        fake_impl=_fused_mul_mat_gguf_fake,
    )
    direct_register_custom_op(
        op_name="_quantize_x_q8_1",
        op_func=_quantize_x_q8_1,
        fake_impl=_quantize_x_q8_1_fake,
    )
    direct_register_custom_op(
        op_name="_gguf_unquantized_gemm",
        op_func=_unquantized_gemm,
        fake_impl=_unquantized_gemm_fake,
    )
    fused_mul_mat_gguf = torch.ops.vllm._fused_mul_mat_gguf
    quantize_x_q8_1 = torch.ops.vllm._quantize_x_q8_1
    unquantized_gemm = torch.ops.vllm._gguf_unquantized_gemm
except AttributeError as error:
    raise error


class GGUFUnquantizedLinearMethod(UnquantizedLinearMethod):
    """vLLM's unquantized linear for the GGUF's F32/F16/BF16 tensors, with the
    product in a custom op, so its row-count choice (_unquantized_gemm) is made
    per call, not fixed when the model is traced."""

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.dim() != 2 or not current_platform.is_cuda() or envs.VLLM_BATCH_INVARIANT:
            return super().apply(layer, x, bias)
        return unquantized_gemm(x, layer.weight, bias)


@register_weight_loader_v2_supported_method
class GGUFLinearMethod(LinearMethodBase):
    """Linear method for GGUF."""

    def __init__(
        self,
        quant_config,
        layout: GGUFLinearLayout | None = None,
    ) -> None:
        self.quant_config = quant_config
        self.layout = layout

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
        del output_size
        self.params_dtype = params_dtype
        output_size_per_partition = sum(output_partition_sizes)
        fallback_weight_loader = extra_weight_attrs.pop("weight_loader", None)
        weight_loader = _resolve_gguf_weight_loader(layer, fallback_weight_loader)
        assert weight_loader is not None

        tensor_shape = (output_size_per_partition, input_size_per_partition)
        weight = GGUFUninitializedWeightParameter(requires_grad=False)
        set_weight_attrs(
            weight,
            {
                "weight_loader": weight_loader,
                "input_dim": 1,
                "output_dim": 0,
                "tensor_shape": tensor_shape,
                "data_container": [],
                "shard_id": [],
                "shard_id_map": {},
            },
        )
        set_weight_attrs(weight, extra_weight_attrs)
        layer.register_parameter("weight", weight)

        weight_loader_type = _resolve_gguf_weight_type_loader(
            layer, fallback_weight_loader
        )
        assert weight_loader_type is not None
        weight_type = GGUFUninitializedWeightTypeParameter(requires_grad=False)
        set_weight_attrs(
            weight_type,
            {
                "weight_loader": weight_loader_type,
                "weight_type": 0,
                "shard_weight_type": {},
                "num_elements": len(output_partition_sizes),
                "ignore_warning": True,
            },
        )
        set_weight_attrs(weight_type, extra_weight_attrs)
        layer.register_parameter("weight_type", weight_type)

        if self.layout is not None:
            set_weight_attrs(
                weight,
                {
                    "gguf_layout": self.layout,
                    "gguf_logical_input_size": input_size,
                    "gguf_weight_type_parameter": weight_type,
                },
            )

    def process_weights_after_loading(self, layer: torch.nn.Module):
        self._materialize_gguf_parameters(layer)
        weight_type = layer.weight_type.weight_type
        if not (weight_type in UNQUANTIZED_TYPES or weight_type in DEQUANT_TYPES):
            weight_type = WeightType(weight_type)
            raise ValueError(
                f"Unsupported GGUF quantization type {weight_type} in layer {layer}."
            )
        self._create_padded_weight_param(layer)

    def _materialize_gguf_parameters(self, layer: torch.nn.Module) -> None:
        self._materialize_weight(layer)
        self._materialize_weight_type(layer)

    def _materialize_weight(self, layer: torch.nn.Module) -> None:
        _materialize_gguf_weight_parameter(layer, "weight")

    def _materialize_weight_type(self, layer: torch.nn.Module) -> None:
        _materialize_gguf_weight_type_parameter(layer, "weight_type")

    def _create_padded_weight_param(self, layer: torch.nn.Module):
        """Create padded weight parameter for GGUF MergedLinear layer."""
        weight = layer.weight
        shard_id_map = weight.shard_id_map
        shard_id = weight.shard_id
        if len(data_container := weight.data_container) > 1:
            dtype = {data.dtype for data in data_container}
            assert len(dtype) == 1, ValueError(
                f"Data container has mixed dtypes: {dtype}"
            )
            dtype = next(iter(dtype))
            padded_side = max(x.size(1) for x in data_container)
            concat_side = sum(x.size(0) for x in data_container)
            padded_data = torch.zeros(
                (concat_side, padded_side), dtype=dtype, device=weight.device
            )
            shard_offset_map = dict[str, tuple[int, int, int]]()
            ordered_shard_ids = _gguf_ordered_shard_ids(shard_id)
            types = layer.weight_type.shard_weight_type
            current_offset = run_start = 0
            run_type = None
            for idx in ordered_shard_ids:
                id_in_container = shard_id_map[idx]
                start = current_offset
                end = start + data_container[id_in_container].size(0)
                size = data_container[id_in_container].size(1)
                if ops.LCPP_ENABLED:  # same-type runs contiguous, see _shard_weight
                    wtype = types.get(idx, layer.weight_type.weight_type)
                    if wtype != run_type:
                        run_start, run_type = start, wtype
                    at = run_start * padded_side + (start - run_start) * size
                    padded_data.view(-1)[at : at + (end - start) * size] = (
                        data_container[id_in_container].reshape(-1)
                    )
                else:
                    padded_data[start:end, :size] = data_container[id_in_container]
                shard_offset_map[idx] = (start, end, size)
                current_offset = end
            padded_param = GGUFWeightParameter(
                data=padded_data,
                weight_loader=weight.weight_loader,
                input_dim=weight.input_dim,
                output_dim=weight.output_dim,
                tensor_shape=weight.tensor_shape,
            )
            padded_param.data_container = []
            padded_param.shard_id = ordered_shard_ids
            padded_param.shard_id_map = dict(weight.shard_id_map)
            if hasattr(weight, "ignore_warning"):
                padded_param.ignore_warning = weight.ignore_warning
            set_weight_attrs(padded_param, {"shard_offset_map": shard_offset_map})
            weight.data_container.clear()
            weight.shard_id.clear()
            weight.shard_id_map.clear()
            if weight.data.numel() > 0:
                weight.data = torch.empty(0, dtype=weight.dtype, device=weight.device)
            layer.register_parameter("weight", padded_param)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from . import fused_mul_mat_gguf as fused_mul_mat_gguf_op

        if self.layout is not None:
            x = self.layout.input_to_gguf(x)

        shard_id = layer.weight.shard_id
        if shard_id:
            shard_id = ["q", "k", "v"] if "q" in shard_id else shard_id
            weight = layer.weight
            fallback_wtype = layer.weight_type.weight_type
            shard_weight_types = [
                layer.weight_type.shard_weight_type.get(idx, fallback_wtype)
                for idx in shard_id
            ]
            if len(set(shard_weight_types)) == 1:
                out = fused_mul_mat_gguf_op(x, weight, shard_weight_types[0])
                if bias is not None:
                    out.add_(bias)
                return out
            runs = list(_shard_runs(weight, shard_id, shard_weight_types))
            # Route L: the runs share one q8_1 quantization of x (the cat stays
            # outside the ops, where inductor folds it into a following split)
            x_q8 = quantize_x_q8_1(x, [t for _, t in runs]) if ops.LCPP_ENABLED else None
            out = torch.cat(
                [fused_mul_mat_gguf_op(x, w, t, x_q8) for w, t in runs], axis=1
            )
        else:
            weight = layer.weight
            weight_type = layer.weight_type.weight_type
            out = fused_mul_mat_gguf_op(x, weight, weight_type)
        if bias is not None:
            out.add_(bias)
        return out
