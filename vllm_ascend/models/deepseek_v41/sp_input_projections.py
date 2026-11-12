# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental lossless BF16 transport for replicated input projections."""

import torch
import torch.nn.functional as F


def projection_layout(attn):
    layout = [
        ("q_a", torch.bfloat16, attn.q_lora_rank),
        ("kv", torch.bfloat16, attn.head_dim),
    ]
    if attn.role.is_kv_source:
        dtype = torch.float32 if attn.role.compress_ratio == 2 else torch.bfloat16
        layout.append(("compressor_kv", dtype, attn.head_dim))
        if attn.role.compress_ratio == 2:
            layout.append(("compressor_score", torch.float32, attn.head_dim))
    if attn.indexer is not None:
        layout.append(("indexer_weights", torch.float32, attn.indexer.n_heads))
    return layout


def pack_projections(values, layout):
    parts = []
    for name, dtype, width in layout:
        value = values[name]
        assert value.dtype == dtype and value.shape[-1] == width
        # Keep every field start and the row stride aligned for FP32 recovery.
        bits = value.contiguous().view(torch.bfloat16)
        if bits.shape[-1] % 2:
            bits = F.pad(bits, (0, 1))
        parts.append(bits)
    return torch.cat(parts, dim=-1)


def unpack_projections(packed, layout):
    values = {}
    offset = 0
    for name, dtype, width in layout:
        words = width * (2 if dtype == torch.float32 else 1)
        padded_words = words + words % 2
        bits = packed[:, offset : offset + words].contiguous()
        values[name] = bits.view(dtype)
        offset += padded_words
    assert offset == packed.shape[-1]
    return values


def project_and_pack(attn, hidden_states):
    values = {"q_a": attn.wq_a(hidden_states), "kv": attn.wkv(hidden_states)}
    if attn.role.is_kv_source:
        source = hidden_states.float() if attn.role.compress_ratio == 2 else hidden_states
        values["compressor_kv"] = attn.compressor.wkv(source)
        if attn.role.compress_ratio == 2:
            values["compressor_score"] = attn.compressor.wgate(source)
    if attn.indexer is not None:
        values["indexer_weights"] = attn.indexer.project_weights(hidden_states)
    return pack_projections(values, projection_layout(attn))


def local_projection_rejection_reason(attn):
    """Allow only inspected token-local schemes and replicated local calls."""
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    from vllm_ascend.ops.cv_linear import CVLinearWrapper
    from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod
    from vllm_ascend.quantization.methods.w4a8.w4a8 import AscendW4A8DynamicLinearMethod
    from vllm_ascend.quantization.methods.w8a8.w8a8_dynamic import AscendW8A8DynamicLinearMethod

    allowed = (
        UnquantizedLinearMethod,
        AscendUnquantizedLinearMethod,
        AscendW4A8DynamicLinearMethod,
        AscendW8A8DynamicLinearMethod,
    )
    linears = [("wq_a", attn.wq_a), ("wkv", attn.wkv)]
    if attn.indexer is not None:
        linears.append(("indexer.weights_proj", attn.indexer.weights_proj))
    for name, linear in linears:
        if CVLinearWrapper._detect_communication(linear):
            return f"{name} has an embedded collective"
        method = linear.quant_method
        method = getattr(method, "quant_method", method)
        if type(method) not in allowed:
            return f"{name}: unsupported scheme {type(method).__name__}"
    return None
