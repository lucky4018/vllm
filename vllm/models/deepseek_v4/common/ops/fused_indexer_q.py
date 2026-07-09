# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.import_utils import has_cutedsl

from vllm.models.deepseek_v4.platform_utils import use_reference_impl
# SM80 (A800) software e4m3fn codec: Triton on cap 8.0 lacks tl.float8e4nv.
from vllm.models.deepseek_v4.common.ops.fp8e4m3_sm80 import f32_to_e4m3fn

# MXFP4: 32 elements per block, packed 2 nibbles per byte, ue8m0 block scale.
MXFP4_BLOCK_SIZE = 32


@triton.jit
def _get_cos_sin(
    cos_sin_cache_ptr,
    cos_sin_cache_stride,
    pos,
    HALF_ROT_DIM: tl.constexpr,
):
    block = tl.arange(0, HALF_ROT_DIM)
    cos = tl.load(cos_sin_cache_ptr + pos * cos_sin_cache_stride + block)
    cos = cos.to(tl.float32)
    sin = tl.load(cos_sin_cache_ptr + pos * cos_sin_cache_stride + block + HALF_ROT_DIM)
    sin = sin.to(tl.float32)
    return cos, sin


@triton.jit
def _fp32x2_to_fp4x2(x_lo, x_hi):
    return tl.inline_asm_elementwise(
        """
        {
            .reg .b8 tmp;
            cvt.rn.satfinite.e2m1x2.f32 tmp, $1, $2;
            cvt.u32.u8 $0, tmp;
        }
        """,
        constraints="=r,f,f",
        args=[x_hi, x_lo],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    ).to(tl.uint8)


@triton.jit
def _quantize_mxfp4_pair(x_lo, x_hi):
    amax = tl.maximum(tl.max(tl.abs(x_lo)), tl.max(tl.abs(x_hi)))
    amax = tl.maximum(amax, 6.0 * (2**-126))
    log2_ratio = tl.math.ceil(tl.math.log2(amax * (1.0 / 6.0)))
    log2_ratio = tl.minimum(tl.maximum(log2_ratio, -127.0), 127.0)
    scale = tl.math.exp2(log2_ratio)
    ue8m0 = (log2_ratio + 127.0).to(tl.uint8)
    inv_scale = 1.0 / scale
    packed = _fp32x2_to_fp4x2(x_lo * inv_scale, x_hi * inv_scale)
    return packed, ue8m0


@triton.jit
def _fused_indexer_q_rope_quant_kernel(
    pos_ptr,
    index_q_ptr,
    index_q_stride0,
    index_q_stride1,
    index_q_cos_sin_ptr,
    index_q_cos_sin_stride,
    INDEX_Q_HALF_ROT_DIM: tl.constexpr,
    index_q_fp8_ptr,
    index_q_fp8_stride0,
    index_q_fp8_stride1,
    INDEX_Q_HEAD_DIM: tl.constexpr,
    index_weights_ptr,
    index_weights_stride,
    index_weights_softmax_scale,
    index_weights_head_scale,
    index_weights_out_ptr,
    index_weights_out_stride,
):
    INDEX_Q_ROT_DIM: tl.constexpr = 2 * INDEX_Q_HALF_ROT_DIM
    INDEX_Q_NOPE_DIM: tl.constexpr = INDEX_Q_HEAD_DIM - INDEX_Q_ROT_DIM
    tl.static_assert(INDEX_Q_NOPE_DIM >= 0)

    tok_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    pos = tl.load(pos_ptr + tok_idx)
    cos, sin = _get_cos_sin(
        index_q_cos_sin_ptr,
        index_q_cos_sin_stride,
        pos,
        INDEX_Q_HALF_ROT_DIM,
    )
    half_offset = tl.arange(0, INDEX_Q_HALF_ROT_DIM)
    base_ptr = index_q_ptr + tok_idx * index_q_stride0 + head_idx * index_q_stride1

    rot_base = base_ptr + INDEX_Q_NOPE_DIM
    x_even = tl.load(rot_base + half_offset * 2).to(tl.float32)
    x_odd = tl.load(rot_base + half_offset * 2 + 1).to(tl.float32)
    r_even = x_even * cos - x_odd * sin
    r_odd = x_odd * cos + x_even * sin

    r_even = r_even.to(tl.bfloat16).to(tl.float32)
    r_odd = r_odd.to(tl.bfloat16).to(tl.float32)

    amax = tl.maximum(tl.max(tl.abs(r_even)), tl.max(tl.abs(r_odd)))
    if INDEX_Q_NOPE_DIM > 0:
        nope_offset = tl.arange(0, INDEX_Q_NOPE_DIM)
        x_nope = tl.load(base_ptr + nope_offset).to(tl.float32)
        amax = tl.maximum(amax, tl.max(tl.abs(x_nope)))
    index_q_scale = tl.div_rn(tl.maximum(amax, 1e-4), 448.0)
    index_q_scale = tl.math.exp2(tl.math.ceil(tl.math.log2(index_q_scale)))

    fp8_base_ptr = (
        index_q_fp8_ptr + tok_idx * index_q_fp8_stride0 + head_idx * index_q_fp8_stride1
    )
    if INDEX_Q_NOPE_DIM > 0:
        tl.store(
            fp8_base_ptr + nope_offset,
            f32_to_e4m3fn(tl.clamp(tl.div_rn(x_nope, index_q_scale), -448.0, 448.0)),
        )
    fp8_rot_base = fp8_base_ptr + INDEX_Q_NOPE_DIM
    tl.store(
        fp8_rot_base + half_offset * 2,
        f32_to_e4m3fn(tl.clamp(tl.div_rn(r_even, index_q_scale), -448.0, 448.0)),
    )
    tl.store(
        fp8_rot_base + half_offset * 2 + 1,
        f32_to_e4m3fn(tl.clamp(tl.div_rn(r_odd, index_q_scale), -448.0, 448.0)),
    )

    index_weights = tl.load(
        index_weights_ptr + tok_idx * index_weights_stride + head_idx
    )
    index_weights = index_weights.to(tl.float32)
    index_weights *= index_q_scale
    index_weights *= index_weights_softmax_scale
    index_weights *= index_weights_head_scale
    tl.store(
        index_weights_out_ptr + tok_idx * index_weights_out_stride + head_idx,
        index_weights,
    )


@triton.jit
def _fused_indexer_q_rope_mxfp4_kernel(
    pos_ptr,
    index_q_ptr,
    index_q_stride0,
    index_q_stride1,
    index_q_cos_sin_ptr,
    index_q_cos_sin_stride,
    INDEX_Q_HALF_ROT_DIM: tl.constexpr,
    index_q_mxfp4_ptr,
    index_q_mxfp4_stride0,
    index_q_mxfp4_stride1,
    index_q_scale_ptr,
    index_q_scale_stride0,
    index_q_scale_stride1,
    INDEX_Q_HEAD_DIM: tl.constexpr,
    MXFP4_BLOCK: tl.constexpr,
    index_weights_ptr,
    index_weights_stride,
    index_weights_softmax_scale,
    index_weights_head_scale,
    index_weights_out_ptr,
    index_weights_out_stride,
):
    INDEX_Q_ROT_DIM: tl.constexpr = 2 * INDEX_Q_HALF_ROT_DIM
    INDEX_Q_NOPE_DIM: tl.constexpr = INDEX_Q_HEAD_DIM - INDEX_Q_ROT_DIM
    NUM_NOPE_BLOCKS: tl.constexpr = INDEX_Q_NOPE_DIM // MXFP4_BLOCK
    NUM_ROPE_BLOCKS: tl.constexpr = INDEX_Q_ROT_DIM // MXFP4_BLOCK
    HALF_BLOCK: tl.constexpr = MXFP4_BLOCK // 2
    tl.static_assert(INDEX_Q_NOPE_DIM >= 0)
    tl.static_assert(INDEX_Q_NOPE_DIM % MXFP4_BLOCK == 0)
    tl.static_assert(INDEX_Q_ROT_DIM % MXFP4_BLOCK == 0)
    tl.static_assert(MXFP4_BLOCK % 2 == 0)

    tok_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    pos = tl.load(pos_ptr + tok_idx)

    q_base = index_q_ptr + tok_idx * index_q_stride0 + head_idx * index_q_stride1
    out_base = (
        index_q_mxfp4_ptr
        + tok_idx * index_q_mxfp4_stride0
        + head_idx * index_q_mxfp4_stride1
    )
    scale_base = (
        index_q_scale_ptr
        + tok_idx * index_q_scale_stride0
        + head_idx * index_q_scale_stride1
    )

    half_off = tl.arange(0, HALF_BLOCK)

    for b in tl.static_range(NUM_NOPE_BLOCKS):
        base = b * MXFP4_BLOCK
        x_lo = tl.load(q_base + base + half_off * 2).to(tl.float32)
        x_hi = tl.load(q_base + base + half_off * 2 + 1).to(tl.float32)
        packed, ue8m0 = _quantize_mxfp4_pair(x_lo, x_hi)
        tl.store(out_base + base // 2 + half_off, packed)
        tl.store(scale_base + b, ue8m0)

    rot_q_base = q_base + INDEX_Q_NOPE_DIM
    for b in tl.static_range(NUM_ROPE_BLOCKS):
        pair_off = b * HALF_BLOCK + half_off
        cos_b = tl.load(
            index_q_cos_sin_ptr + pos * index_q_cos_sin_stride + pair_off
        ).to(tl.float32)
        sin_b = tl.load(
            index_q_cos_sin_ptr
            + pos * index_q_cos_sin_stride
            + pair_off
            + INDEX_Q_HALF_ROT_DIM
        ).to(tl.float32)
        x_even = tl.load(rot_q_base + pair_off * 2).to(tl.float32)
        x_odd = tl.load(rot_q_base + pair_off * 2 + 1).to(tl.float32)
        r_even = x_even * cos_b - x_odd * sin_b
        r_odd = x_odd * cos_b + x_even * sin_b
        r_even = r_even.to(tl.bfloat16).to(tl.float32)
        r_odd = r_odd.to(tl.bfloat16).to(tl.float32)
        packed, ue8m0 = _quantize_mxfp4_pair(r_even, r_odd)
        rope_byte_off = (INDEX_Q_NOPE_DIM + b * MXFP4_BLOCK) // 2
        tl.store(out_base + rope_byte_off + half_off, packed)
        tl.store(scale_base + NUM_NOPE_BLOCKS + b, ue8m0)

    index_weights = tl.load(
        index_weights_ptr + tok_idx * index_weights_stride + head_idx
    ).to(tl.float32)
    index_weights *= index_weights_softmax_scale
    index_weights *= index_weights_head_scale
    tl.store(
        index_weights_out_ptr + tok_idx * index_weights_out_stride + head_idx,
        index_weights,
    )


def fused_indexer_q_rope_quant(
    positions: torch.Tensor,
    index_q: torch.Tensor,
    index_q_cos_sin_cache: torch.Tensor,
    index_weights: torch.Tensor,
    index_weights_softmax_scale: float,
    index_weights_head_scale: float,
    use_fp4: bool = False,
):
    assert positions.ndim == 1
    assert index_q.ndim == 3
    assert index_q_cos_sin_cache.ndim == 2

    num_tokens = positions.shape[0]
    num_index_q_heads = index_q.shape[1]
    index_q_head_dim = index_q.shape[2]

    index_weights_out = torch.empty_like(index_weights, dtype=torch.float32)

    if use_fp4:
        assert index_q_head_dim % MXFP4_BLOCK_SIZE == 0
        num_scale_blocks = index_q_head_dim // MXFP4_BLOCK_SIZE
        index_q_packed = torch.empty(
            (num_tokens, num_index_q_heads, index_q_head_dim // 2),
            dtype=torch.uint8,
            device=index_q.device,
        )
        index_q_scale = torch.empty(
            (num_tokens, num_index_q_heads, num_scale_blocks),
            dtype=torch.uint8,
            device=index_q.device,
        )
        if has_cutedsl() and not use_reference_impl():
            from vllm.models.deepseek_v4.nvidia.ops.fused_indexer_q_cutedsl import (
                fused_indexer_q_rope_quant_mxfp4_cutedsl,
            )
            fused_indexer_q_rope_quant_mxfp4_cutedsl(
                positions,
                index_q,
                index_q_cos_sin_cache,
                index_weights,
                index_weights_softmax_scale,
                index_weights_head_scale,
                index_q_packed,
                index_q_scale,
                index_weights_out,
            )
        else:
            _fused_indexer_q_rope_mxfp4_kernel[(num_tokens, num_index_q_heads)](
                positions,
                index_q,
                index_q.stride(0),
                index_q.stride(1),
                index_q_cos_sin_cache,
                index_q_cos_sin_cache.stride(0),
                index_q_cos_sin_cache.shape[-1] // 2,
                index_q_packed,
                index_q_packed.stride(0),
                index_q_packed.stride(1),
                index_q_scale,
                index_q_scale.stride(0),
                index_q_scale.stride(1),
                index_q_head_dim,
                MXFP4_BLOCK_SIZE,
                index_weights,
                index_weights.stride(0),
                index_weights_softmax_scale,
                index_weights_head_scale,
                index_weights_out,
                index_weights_out.stride(0),
                num_warps=1,
            )
        return (
            index_q_packed,
            index_q_scale.view(torch.int32).squeeze(-1),
        ), index_weights_out

    index_q_fp8 = torch.empty_like(index_q, dtype=torch.float8_e4m3fn)
    if has_cutedsl() and not use_reference_impl():
        from vllm.models.deepseek_v4.nvidia.ops.fused_indexer_q_cutedsl import (
            fused_indexer_q_rope_quant_fp8_cutedsl,
        )
        fused_indexer_q_rope_quant_fp8_cutedsl(
            positions,
            index_q,
            index_q_cos_sin_cache,
            index_weights,
            index_weights_softmax_scale,
            index_weights_head_scale,
            index_q_fp8,
            index_weights_out,
        )
    else:
        _fused_indexer_q_rope_quant_kernel[(num_tokens, num_index_q_heads)](
            positions,
            index_q,
            index_q.stride(0),
            index_q.stride(1),
            index_q_cos_sin_cache,
            index_q_cos_sin_cache.stride(0),
            index_q_cos_sin_cache.shape[-1] // 2,
            index_q_fp8,
            index_q_fp8.stride(0),
            index_q_fp8.stride(1),
            index_q_head_dim,
            index_weights,
            index_weights.stride(0),
            index_weights_softmax_scale,
            index_weights_head_scale,
            index_weights_out,
            index_weights_out.stride(0),
            num_warps=1,
        )
    return index_q_fp8, index_weights_out
