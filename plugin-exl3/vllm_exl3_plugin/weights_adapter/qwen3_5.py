# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5 / Qwen3.8 (Qwen3_5ForConditionalGeneration + Qwen3_5MTP) in exllamav3's layout.

Name mapping. exllamav3 writes the HF names (model.language_model.layers.N..., lm_head,
mtp.*) with each quantized linear's `.weight` replaced by `.trellis/.suh/.svh/.mul1`
(format.py). vLLM's own mappers do the rest, unchanged:
  * Qwen3_5ForConditionalGeneration: model.language_model. -> language_model.model.,
    lm_head. -> language_model.lm_head., model.visual. -> visual., mtp. -> dropped;
  * Qwen3NextModel / Qwen3_5Model stacking: q/k/v_proj -> qkv_proj ("q", "k", "v"),
    gate/up_proj -> gate_up_proj (0, 1), GDN in_proj_qkv -> in_proj_qkvz ((0, 1, 2), one
    tensor), in_proj_z -> in_proj_qkvz (3), in_proj_b/in_proj_a -> in_proj_ba (0, 1);
  * Qwen3_5MTP: mtp. -> model., keeps embed_tokens / lm_head, same stacking.
The stacked name keeps the suffix (".q_proj.trellis" -> ".qkv_proj.trellis"), so the EXL3
linear method's params are named trellis/suh/svh/mul1 and its loader stores each tensor
under the shard id the mapper attached. tools/exl3_meta_dry_run.py checks the result against
the checkpoint's safetensors index: every model param loaded, every tensor consumed.

What stays unquantized (bf16/fp16 in the checkpoint, vLLM's unquantized methods): the input
embedding, norms, GDN conv1d/A_log/dt_bias, and the linears below.
"""

from __future__ import annotations

# GDN in_proj_a / in_proj_b (vLLM in_proj_ba): fp16 in the checkpoint; exllamav3 does not
# quantize them (qmap None; 48 outputs, not a multiple of the 128-wide Hadamard).
UNQUANTIZED_LINEARS = ("in_proj_ba",)
# The overlay's pruned MTP draft head: bf16 rows written by tools/exl3_draft_head.py.
DRAFT_HEAD = "draft_lm_head"
DRAFT_HEAD_TENSOR = "mtp.draft_lm_head.weight"
DRAFT_HEAD_FILE = "mtp_draft_head.safetensors"
DRAFT_IDS_FILE = "mtp_draft_vocab_ids.pt"


def is_unquantized_module(prefix: str) -> bool:
    """Whether the vLLM module at prefix is stored unquantized in an EXL3 Qwen3.5 checkpoint
    (plain bf16 vision tower assumed: exllamav3's "V" variants with a 6-bit tower are not
    supported)."""
    parts = prefix.split(".")
    return parts[-1] in (*UNQUANTIZED_LINEARS, DRAFT_HEAD) or "visual" in parts
