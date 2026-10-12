# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Platform-neutral reference operators for DeepSeek-V4 SM80 sparse MLA.

This module intentionally has no ROCm/AITER dependency.  The attention math
is expressed with PyTorch while the packed ``fp8_ds_mla`` cache is decoded by
the shared DSV4 Triton gather kernel, which preserves the exact E4M3FN/UE8M0
layout used by every platform.
"""

import math

import torch

from vllm._fp8e4m3_sm80 import (
    _e4m3fn_to_f32_reference,
    e4m3fn_to_f32,
)
from vllm.triton_utils import tl, triton

NOPE_DIM = 448
ROPE_DIM = 64
HEAD_DIM = 512
TOKEN_DATA_BYTES = 576
TOKEN_SCALE_BYTES = 8
PREFILL_QUERY_CHUNK_SIZE = 512
PREFILL_ATTENTION_WORKSPACE_CAP_BYTES = 2 * 1024**3
_E4M3_LUT_CACHE: dict[torch.device, torch.Tensor] = {}


def _check_dims(q, head_dim, nope_head_dim, rope_head_dim):
    if (q.shape[-1], head_dim, nope_head_dim, rope_head_dim) != (
        HEAD_DIM,
        HEAD_DIM,
        NOPE_DIM,
        ROPE_DIM,
    ):
        raise ValueError("DeepSeek-V4 sparse MLA requires 448 NoPE + 64 RoPE")


def _decode_e8m0_scales(scale):
    if scale.dtype == torch.float8_e8m0fnu:
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            _upcast_e8m0_to_fp32,
        )

        return _upcast_e8m0_to_fp32(scale).contiguous()
    return scale.to(torch.float32)


def _decode_e4m3_weight(weight: torch.Tensor) -> torch.Tensor:
    lut = _E4M3_LUT_CACHE.get(weight.device)
    if lut is None:
        lut = torch.tensor(
            [_e4m3fn_to_f32_reference(i) for i in range(256)],
            dtype=torch.float32,
            device=weight.device,
        )
        _E4M3_LUT_CACHE[weight.device] = lut
    return lut[weight.view(torch.uint8).long()].view(weight.shape)


def _expand_scales(scale, rows, cols):
    scale = _decode_e8m0_scales(scale)
    rb, cb = scale.shape[-2:]
    scale = torch.repeat_interleave(scale, math.ceil(rows / rb), -2)[..., :rows, :]
    return torch.repeat_interleave(scale, math.ceil(cols / cb), -1)[..., :, :cols]


def triton_inv_rope_einsum(
    rotary_emb, o, positions, rope_head_dim, n_local_groups, o_lora_rank, wo_a
):
    """Inverse GPT-J RoPE and BF16 reference WO_A projection for SM80."""
    rotated, _ = rotary_emb.forward_native(positions, o, inverse=True)
    rotated = rotated.view(o.shape[0], n_local_groups, -1)
    weight = wo_a.weight
    scale = getattr(wo_a, "weight_scale_inv", None)
    if weight.dtype in (torch.uint8, torch.float8_e4m3fn):
        weight = _decode_e4m3_weight(weight)
    else:
        weight = weight.to(torch.float32)
    weight = weight.view(n_local_groups, o_lora_rank, -1)
    if scale is not None:
        weight = weight * _expand_scales(
            scale.view(n_local_groups, -1, scale.shape[-1]),
            o_lora_rank,
            rotated.shape[-1],
        )
    return torch.einsum("tgd,grd->tgr", rotated.float(), weight).to(torch.bfloat16)


@triton.jit
def _decode_cache_rows_kernel(
    cache_ptr,
    slots_ptr,
    out_ptr,
    cache_block_stride,
    slots_stride,
    out_stride0,
    out_stride1,
    block_size: tl.constexpr,
    width: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    BLOCK_NOPE: tl.constexpr,
    BLOCK_ROPE: tl.constexpr,
    TOKEN_DATA_BYTES: tl.constexpr,
    TOKEN_SCALE_BYTES: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1)
    slot = tl.load(slots_ptr + row * slots_stride + col)
    valid = (slot >= 0) & (slot < NUM_BLOCKS * block_size)
    safe_slot = tl.where(valid, slot, 0)
    block_idx = safe_slot // block_size
    pos_in_block = safe_slot % block_size
    block_ptr = cache_ptr + block_idx.to(tl.int64) * cache_block_stride
    token_ptr = block_ptr + pos_in_block * TOKEN_DATA_BYTES
    scale_ptr = (
        block_ptr + block_size * TOKEN_DATA_BYTES + pos_in_block * TOKEN_SCALE_BYTES
    )
    out_row = out_ptr + row * out_stride0 + col * out_stride1

    nope_offsets = tl.arange(0, BLOCK_NOPE)
    nope_mask = valid & (nope_offsets < NOPE_DIM)
    encoded = tl.load(token_ptr + nope_offsets, mask=nope_mask, other=0)
    scales = tl.load(
        scale_ptr + nope_offsets // 64,
        mask=nope_mask,
        other=127,
    )
    decoded = e4m3fn_to_f32(encoded) * tl.exp2(scales.to(tl.float32) - 127.0)
    tl.store(out_row + nope_offsets, decoded.to(tl.bfloat16), mask=nope_mask)

    rope_offsets = tl.arange(0, BLOCK_ROPE)
    rope_mask = valid & (rope_offsets < ROPE_DIM)
    rope_ptr = (token_ptr + NOPE_DIM).to(tl.pointer_type(tl.bfloat16))
    rope = tl.load(rope_ptr + rope_offsets, mask=rope_mask, other=0.0)
    tl.store(out_row + NOPE_DIM + rope_offsets, rope, mask=rope_mask)


def _decode_cache_rows(cache, slots):
    """Gather global slots from the DSV4 packed E4M3FN/UE8M0 cache."""
    assert cache.dtype == torch.uint8
    assert cache.stride(-1) == 1
    block_size = cache.shape[1]
    slots = slots.to(torch.int32).contiguous()
    num_rows, width = slots.shape
    out = torch.empty(
        (num_rows, width, HEAD_DIM), dtype=torch.bfloat16, device=cache.device
    )
    if num_rows == 0 or width == 0:
        return out
    _decode_cache_rows_kernel[(num_rows, width)](
        cache,
        slots,
        out,
        cache.stride(0),
        slots.stride(0),
        out.stride(0),
        out.stride(1),
        block_size=block_size,
        width=width,
        NOPE_DIM=NOPE_DIM,
        ROPE_DIM=ROPE_DIM,
        BLOCK_NOPE=triton.next_power_of_2(NOPE_DIM),
        BLOCK_ROPE=triton.next_power_of_2(ROPE_DIM),
        TOKEN_DATA_BYTES=TOKEN_DATA_BYTES,
        TOKEN_SCALE_BYTES=TOKEN_SCALE_BYTES,
        NUM_BLOCKS=cache.shape[0],
    )
    return out


@triton.jit
def _gather_cache_rows_float_kernel(
    rows_ptr,
    indices_ptr,
    lengths_ptr,
    out_ptr,
    indices_stride,
    rows_stride,
    width,
    num_rows,
    head_dim: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    HAS_LENGTHS: tl.constexpr,
):
    pair = tl.program_id(0)
    query = pair // width
    column = pair % width
    index = tl.load(indices_ptr + query * indices_stride + column)
    valid = (index >= 0) & (index < num_rows)
    if HAS_LENGTHS:
        valid &= column < tl.load(lengths_ptr + query)
    safe_index = tl.where(valid, index, 0)

    offsets = tl.arange(0, BLOCK_DIM)
    values = tl.load(
        rows_ptr + safe_index.to(tl.int64) * rows_stride + offsets,
        mask=valid & (offsets < head_dim),
        other=0.0,
    )
    tl.store(
        out_ptr + pair * head_dim + offsets,
        values.to(tl.float32),
        mask=offsets < head_dim,
    )


def _gather_cache_rows_float(rows, indices, lengths=None):
    """Gather BF16 cache rows directly into a bounded FP32 workspace."""
    assert rows.dtype == torch.bfloat16
    assert rows.stride(-1) == 1
    assert indices.ndim == 2
    num_queries, width = indices.shape
    out = torch.empty(
        (num_queries, width, rows.shape[-1]),
        dtype=torch.float32,
        device=rows.device,
    )
    if num_queries == 0 or width == 0:
        return out
    lengths_arg = indices if lengths is None else lengths
    _gather_cache_rows_float_kernel[(num_queries * width,)](
        rows,
        indices,
        lengths_arg,
        out,
        indices.stride(0),
        rows.stride(0),
        width=width,
        num_rows=rows.shape[0],
        head_dim=rows.shape[-1],
        BLOCK_DIM=triton.next_power_of_2(rows.shape[-1]),
        HAS_LENGTHS=lengths is not None,
        num_warps=8,
    )
    return out


def _prefill_query_chunk_size(q, width):
    """Choose a query chunk from tensor shape and current allocator headroom."""
    # CUDA allocator queries are host-side synchronization points and are not
    # legal inside graph capture.  Use the bounded static chunk there; the
    # surrounding graph already fixes tensor shapes and addresses.
    if torch.cuda.is_current_stream_capturing():
        return PREFILL_QUERY_CHUNK_SIZE
    score_bytes = q.shape[1] * torch.float32.itemsize
    row_bytes = q.shape[-1] * torch.float32.itemsize
    output_bytes = q.shape[1] * q.shape[-1] * torch.float32.itemsize
    bytes_per_query = width * (row_bytes + score_bytes + 1) + output_bytes

    free_bytes, _ = torch.cuda.mem_get_info(q.device)
    reusable_bytes = max(
        torch.cuda.memory_reserved(q.device) - torch.cuda.memory_allocated(q.device),
        0,
    )
    # Keep half of currently available allocator capacity for the surrounding
    # model forward and cap this operator's temporary allocation at 2 GiB.
    workspace_bytes = min(
        PREFILL_ATTENTION_WORKSPACE_CAP_BYTES,
        (free_bytes + reusable_bytes) // 2,
    )
    return max(
        1,
        min(PREFILL_QUERY_CHUNK_SIZE, workspace_bytes // bytes_per_query),
    )


def _attend(q, rows, valid, scale, sink):
    # Gathered cache rows for invalid/padded slots are not populated by the
    # masked Triton stores.  Zero them before the dot product so an
    # uninitialized (or stale NaN) row cannot poison the weighted sum even
    # though its softmax probability is masked to zero below.
    rows.masked_fill_(~valid.unsqueeze(-1), 0)
    rows_float = rows.float()
    scores = torch.einsum("thd,tkd->thk", q.float(), rows_float) * float(scale)
    scores = scores.masked_fill(~valid[:, None], -torch.inf)
    if sink is not None:
        sink_scores = sink[: q.shape[1]].view(1, -1, 1).expand(q.shape[0], -1, -1)
        scores = torch.cat((scores, sink_scores), -1)
    probs = torch.softmax(scores, -1)
    probs = torch.nan_to_num(probs, nan=0.0)
    return torch.einsum("thk,tkd->thd", probs[..., : rows.shape[1]], rows_float)


def _attend_float_rows(q, rows_float, valid, scale, sink):
    """Attention over pre-gathered FP32 rows with in-place normalization."""
    # The gather kernel intentionally skips invalid slots.  Clear those rows
    # before the weighted sum because ``0 * NaN`` would otherwise propagate
    # stale/uninitialized values even though their softmax probability is zero.
    rows_float.masked_fill_(~valid.unsqueeze(-1), 0)
    scores = torch.einsum("thd,tkd->thk", q.float(), rows_float)
    scores.mul_(float(scale)).masked_fill_(~valid[:, None], -torch.inf)

    max_score = scores.amax(-1, keepdim=True)
    sink_score = None
    if sink is not None:
        sink_score = sink[: q.shape[1]].view(1, -1, 1)
        max_score = torch.maximum(max_score, sink_score)
    max_score.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)

    scores.sub_(max_score).exp_()
    denominator = scores.sum(-1, keepdim=True)
    if sink_score is not None:
        denominator.add_(torch.exp(sink_score - max_score))
    denominator.clamp_min_(torch.finfo(torch.float32).tiny)
    scores.div_(denominator)
    return torch.einsum("thk,tkd->thd", scores, rows_float)


def _run_prefill_query_chunk(*, q, rows, indices, lengths, valid, scale, sink, output):
    gathered = _gather_cache_rows_float(rows, indices, lengths)
    output.copy_(_attend_float_rows(q, gathered, valid, scale, sink))


def _attend_groups(q, row_groups, valid_groups, scale, sink):
    """Attend over multiple cache groups without concatenating full KV rows."""
    score_groups = []
    float_groups = []
    for rows, valid in zip(row_groups, valid_groups):
        rows.masked_fill_(~valid.unsqueeze(-1), 0)
        rows_float = rows.float()
        scores = torch.einsum("thd,tkd->thk", q.float(), rows_float)
        scores = scores.mul_(float(scale)).masked_fill_(~valid[:, None], -torch.inf)
        score_groups.append(scores)
        float_groups.append(rows_float)

    scores = torch.cat(score_groups, -1)
    if sink is not None:
        sink_scores = sink[: q.shape[1]].view(1, -1, 1).expand(q.shape[0], -1, -1)
        scores = torch.cat((scores, sink_scores), -1)
    probs = torch.softmax(scores, -1)
    probs = torch.nan_to_num(probs, nan=0.0)

    output = torch.zeros_like(q, dtype=torch.float32)
    start = 0
    for rows_float in float_groups:
        end = start + rows_float.shape[1]
        output.add_(torch.einsum("thk,tkd->thd", probs[..., start:end], rows_float))
        start = end
    return output


def _apply_lengths(indices, lengths, capacity):
    valid = (indices >= 0) & (indices < capacity)
    if lengths is not None:
        width = indices.shape[-1]
        valid &= torch.arange(width, device=indices.device)[None] < lengths[:, None]
    return valid


def triton_sparse_attn_prefill(
    *,
    q,
    kv,
    indices,
    topk_length,
    max_topk_length=None,
    scale,
    head_dim,
    nope_head_dim,
    rope_head_dim,
    attn_sink,
    output,
):
    _check_dims(q, head_dim, nope_head_dim, rope_head_dim)
    if q.shape[0] == 0:
        return
    dense = indices.reshape(indices.shape[0], -1)
    width = dense.shape[-1]
    if max_topk_length is not None:
        width = min(width, int(max_topk_length))
        dense = dense[:, :width]
    rows = kv.reshape(-1, head_dim)
    query_chunk_size = _prefill_query_chunk_size(q, width)
    columns = torch.arange(width, device=q.device)
    for start in range(0, q.shape[0], query_chunk_size):
        end = min(start + query_chunk_size, q.shape[0])
        chunk_indices = dense[start:end]
        valid = (chunk_indices >= 0) & (chunk_indices < rows.shape[0])
        chunk_lengths = None
        if topk_length is not None:
            chunk_lengths = topk_length[start:end]
            valid &= columns[None] < chunk_lengths[:, None]
        _run_prefill_query_chunk(
            q=q[start:end],
            rows=rows,
            indices=chunk_indices,
            lengths=chunk_lengths,
            valid=valid,
            scale=scale,
            sink=attn_sink,
            output=output[start:end],
        )


def triton_sparse_attn_decode(
    *,
    q,
    kv_cache,
    swa_k_cache,
    swa_only,
    topk_indices,
    topk_lens,
    swa_indices,
    swa_lens,
    attn_sink,
    scale,
    head_dim,
    nope_head_dim,
    rope_head_dim,
    output,
):
    _check_dims(q, head_dim, nope_head_dim, rope_head_dim)
    if q.shape[0] == 0:
        return
    swa_slots = swa_indices.reshape(q.shape[0], -1).to(torch.long)
    swa_rows = _decode_cache_rows(swa_k_cache, swa_slots)
    swa_valid = _apply_lengths(
        swa_slots, swa_lens, swa_k_cache.shape[0] * swa_k_cache.shape[1]
    )
    if not swa_only:
        assert kv_cache is not None and topk_indices is not None
        top_slots = topk_indices.reshape(q.shape[0], -1).to(torch.long)
        top_rows = _decode_cache_rows(kv_cache, top_slots)
        top_valid = _apply_lengths(
            top_slots, topk_lens, kv_cache.shape[0] * kv_cache.shape[1]
        )
        result = _attend_groups(
            q,
            (top_rows, swa_rows),
            (top_valid, swa_valid),
            scale,
            attn_sink,
        )
    else:
        result = _attend(q, swa_rows, swa_valid, scale, attn_sink)
    output.copy_(result)
