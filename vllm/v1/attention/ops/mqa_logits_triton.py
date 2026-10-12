# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton fallback for DeepGEMM's fp8_mqa_logits / fp8_paged_mqa_logits."""

import os

import torch

from vllm._fp8e4m3_sm80 import _e4m3fn_to_f32_reference
from vllm.triton_utils import tl, triton

# Paged decode: num_warps=4 dominated on A100/SM80 across {2,4,8}; the others
# were 1.5–1.7× slower at (num_heads=32, head_dim=128, block_size=64), so
# narrow the sweep to keep autotune from latching onto a bad pick under noise.
_PAGED_AUTOTUNE_CONFIGS = [
    triton.Config({}, num_warps=4, num_stages=ns) for ns in (2, 4)
]

# Prefill kernel adds BLOCK_N as a free tile axis. num_warps=8 was 1.5–3×
# worse than {2,4} across the sweep; keep BLOCK_N ∈ {32, 64, 128} so autotune
# can pick per shape (BN=128 wins for GLM-5.1 long chunks).
_PREFILL_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_N": bn}, num_warps=nw, num_stages=ns)
    for bn in (32, 64, 128)
    for nw in (2, 4)
    for ns in (2, 4)
]

# Warmup shape mirrors the chunked-prefill regime (small M, long N) so
# autotune picks a tile sized for real serving rather than a launch-overhead-
# dominated dummy grid.
_PREFILL_WARMUP_M = 8
_PREFILL_WARMUP_N = 8192


_E4M3FN_BF16_LUT_CACHE: dict[torch.device, torch.Tensor] = {}


def _get_e4m3fn_bf16_lut(device: torch.device) -> torch.Tensor:
    lut = _E4M3FN_BF16_LUT_CACHE.get(device)
    if lut is not None:
        return lut
    lut = torch.tensor(
        [_e4m3fn_to_f32_reference(i) for i in range(256)],
        dtype=torch.float32,
        device=device,
    ).to(torch.bfloat16)
    _E4M3FN_BF16_LUT_CACHE[device] = lut
    return lut


def _as_e4m3fn_bytes(tensor: torch.Tensor, name: str) -> torch.Tensor:
    """Return one raw E4M3FN byte per tensor element without requantizing."""
    if tensor.dtype == torch.uint8:
        return tensor
    if tensor.dtype == torch.float8_e4m3fn:
        return tensor.view(torch.uint8)
    raise TypeError(
        f"{name} must use torch.float8_e4m3fn or raw torch.uint8 storage, "
        f"got {tensor.dtype}"
    )


@triton.jit
def _decode_e4m3fn_bf16_lut(u, lut_ptr):
    return tl.load(lut_ptr + u.to(tl.uint32))


@triton.jit
def _kv_page_offset(block_idx, page_stride):
    # Packed DeepSeek-V4 pools stride across several cache-layer page types.
    # A valid physical page ID can therefore address beyond 2 GiB even though
    # the block table itself stores int32 IDs.
    return block_idx.to(tl.int64) * page_stride


@triton.autotune(
    configs=_PAGED_AUTOTUNE_CONFIGS,
    key=["num_heads", "head_dim", "block_size"],
)
@triton.jit
def _fp8_paged_mqa_logits_kernel(
    q_ptr,
    kv_fp8_ptr,
    kv_scale_ptr,
    weights_ptr,
    fp8_lut_ptr,
    seq_lens_ptr,
    decode_lens_ptr,
    block_tables_ptr,
    logits_ptr,
    max_model_len,
    num_kv_blocks,
    stride_q_b,
    stride_q_n,
    stride_q_h,
    stride_q_d,
    stride_kvf_block,
    stride_kvf_s,
    stride_kvf_d,
    stride_kvs_block,
    stride_kvs_s,
    stride_w_t,
    stride_w_h,
    stride_sl_b,
    stride_sl_n,
    stride_bt_b,
    stride_bt_k,
    stride_l_t,
    stride_l_n,
    next_n: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    token_id = tl.program_id(0)
    block_rk = tl.program_id(1)

    batch_id = token_id // next_n
    next_n_id = token_id % next_n

    decode_len = tl.load(decode_lens_ptr + batch_id)
    query_seq_len = tl.load(
        seq_lens_ptr + batch_id * stride_sl_b + next_n_id * stride_sl_n
    )
    context_len = tl.load(
        seq_lens_ptr + batch_id * stride_sl_b + (decode_len - 1) * stride_sl_n,
        mask=decode_len > 0,
        other=0,
    )
    if block_rk * block_size >= context_len:
        return

    block_idx = tl.load(
        block_tables_ptr + batch_id * stride_bt_b + block_rk * stride_bt_k
    )
    valid_block = (block_idx >= 0) & (block_idx < num_kv_blocks)

    offs_h = tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, BLOCK_D)
    offs_n = tl.arange(0, BLOCK_N)
    mask_h = offs_h < num_heads
    mask_d = offs_d < head_dim
    mask_n = offs_n < block_size

    q_base = q_ptr + batch_id * stride_q_b + next_n_id * stride_q_n
    q_byte = tl.load(
        q_base + offs_h[:, None] * stride_q_h + offs_d[None, :] * stride_q_d,
        mask=mask_h[:, None] & mask_d[None, :],
        other=0,
    )
    q = _decode_e4m3fn_bf16_lut(q_byte, fp8_lut_ptr)

    page_offset = _kv_page_offset(block_idx, stride_kvf_block)
    kvf_base = kv_fp8_ptr + page_offset
    k_byte = tl.load(
        kvf_base + offs_n[:, None] * stride_kvf_s + offs_d[None, :] * stride_kvf_d,
        mask=valid_block & mask_n[:, None] & mask_d[None, :],
        other=0,
    )
    kvs_base = kv_scale_ptr + _kv_page_offset(block_idx, stride_kvs_block)
    k_scale = tl.load(
        kvs_base + offs_n * stride_kvs_s,
        mask=valid_block & mask_n,
        other=0.0,
    )
    k = _decode_e4m3fn_bf16_lut(k_byte, fp8_lut_ptr)
    # Scale in fp32 after the dot to avoid an extra bf16 round-trip on K.
    s = tl.dot(q, tl.trans(k)) * k_scale[None, :]

    w = tl.load(
        weights_ptr + token_id * stride_w_t + offs_h * stride_w_h,
        mask=mask_h,
        other=0.0,
    )
    s = tl.where(s > 0, s, 0.0) * w[:, None]
    out = tl.sum(s, axis=0)

    k_offset = block_rk * block_size + offs_n
    valid_query = (next_n_id < decode_len) & (query_seq_len > 0)
    valid = mask_n & valid_block & valid_query & (k_offset < query_seq_len)
    out = tl.where(valid, out, float("-inf"))

    tl.store(
        logits_ptr + token_id * stride_l_t + k_offset * stride_l_n,
        out,
        mask=mask_n & (k_offset < max_model_len),
    )


def fp8_paged_mqa_logits_triton(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    decode_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
    clean_logits: bool = True,
) -> torch.Tensor:
    """Triton implementation of DeepGEMM's fp8_paged_mqa_logits.

    Args:
        q:             [B, next_n, H, D] native E4M3FN or raw uint8 bytes
        kv_cache:      [num_blocks, block_size, 1, D+4] uint8; each page
            stores all FP8 values followed by all fp32 scales
        weights:       [B*next_n, H] float32
        seq_lens:      [B, next_n] int32 per-query effective context lengths
        decode_lens:   [B] int32 number of valid queries in each request
        block_tables:  [B, max_blocks] int32
        max_model_len: output width. Caller passes the active batch max so
            the logits buffer and grid stay tight.
        clean_logits: when False, skip the -inf pre-fill of the output
            (indexer top-k reads only `[:context_len]` per row).
    Returns:
        logits:        [B*next_n, max_model_len] float32
    """
    B, next_n, num_heads, head_dim = q.shape
    _, block_size, one, d_plus_4 = kv_cache.shape
    assert one == 1
    assert d_plus_4 == head_dim + 4
    assert seq_lens.shape == (B, next_n)
    assert decode_lens.shape == (B,)
    if max_model_len > block_tables.shape[1] * block_size:
        raise ValueError(
            f"max_model_len ({max_model_len}) exceeds block-table capacity "
            f"({block_tables.shape[1] * block_size})"
        )

    # Both indexer cache writers pack values first, then scales within each
    # page. The logical D+4 dimension is not a physical per-token record.
    # Preserve the packed pool's page stride and this layer's storage offset.
    num_blocks = kv_cache.shape[0]
    page_stride = kv_cache.stride(0)
    kv_byte = torch.as_strided(
        kv_cache,
        (num_blocks, block_size, head_dim),
        (page_stride, head_dim, 1),
    )
    kv_scale_bytes = torch.as_strided(
        kv_cache,
        (num_blocks, block_size, 4),
        (page_stride, 4, 1),
        storage_offset=kv_cache.storage_offset() + block_size * head_dim,
    )
    kv_scale = kv_scale_bytes.view(torch.float32).squeeze(-1)
    assert kv_byte.shape == (num_blocks, block_size, head_dim)
    assert kv_scale.shape == (num_blocks, block_size)
    q_byte = _as_e4m3fn_bytes(q, "q")

    if os.getenv("VLLM_VALIDATE_PAGED_MQA_BLOCK_TABLE", "0") == "1":
        # Synchronize before inspecting metadata so a preceding asynchronous
        # kernel failure is attributed to its producer, not this kernel launch.
        torch.cuda.synchronize(q.device)
        pages_per_req = torch.div(
            seq_lens[:, -1] + block_size - 1,
            block_size,
            rounding_mode="floor",
        )
        page_ids = torch.arange(block_tables.shape[1], device=q.device)
        active = page_ids.unsqueeze(0) < pages_per_req.unsqueeze(1)
        invalid = active & ((block_tables < 0) | (block_tables >= num_blocks))
        if invalid.any().item():
            bad = invalid.nonzero()[:8].cpu().tolist()
            bad_values = block_tables[invalid][:8].cpu().tolist()
            raise ValueError(
                "paged MQA received invalid active KV pages: "
                f"locations={bad}, values={bad_values}, "
                f"num_kv_blocks={num_blocks}, pages_per_req="
                f"{pages_per_req.cpu().tolist()}"
            )

    if clean_logits:
        logits = torch.full(
            (B * next_n, max_model_len),
            float("-inf"),
            dtype=torch.float32,
            device=q.device,
        )
    else:
        logits = torch.empty(
            (B * next_n, max_model_len), dtype=torch.float32, device=q.device
        )

    BLOCK_H = max(16, triton.next_power_of_2(num_heads))
    BLOCK_D = triton.next_power_of_2(head_dim)
    BLOCK_N = triton.next_power_of_2(block_size)

    fp8_lut = _get_e4m3fn_bf16_lut(q.device)
    grid = (B * next_n, block_tables.shape[1])
    _fp8_paged_mqa_logits_kernel[grid](
        q_byte,
        kv_byte,
        kv_scale,
        weights,
        fp8_lut,
        seq_lens,
        decode_lens,
        block_tables,
        logits,
        max_model_len,
        num_blocks,
        q_byte.stride(0),
        q_byte.stride(1),
        q_byte.stride(2),
        q_byte.stride(3),
        kv_byte.stride(0),
        kv_byte.stride(1),
        kv_byte.stride(2),
        kv_scale.stride(0),
        kv_scale.stride(1),
        weights.stride(0),
        weights.stride(1),
        seq_lens.stride(0),
        seq_lens.stride(1),
        block_tables.stride(0),
        block_tables.stride(1),
        logits.stride(0),
        logits.stride(1),
        next_n=next_n,
        num_heads=num_heads,
        head_dim=head_dim,
        block_size=block_size,
        BLOCK_H=BLOCK_H,
        BLOCK_D=BLOCK_D,
        BLOCK_N=BLOCK_N,
    )
    return logits


@triton.autotune(
    configs=_PREFILL_AUTOTUNE_CONFIGS,
    # Per-program work is N-independent; key on (heads, dim) only so chunked
    # prefill with varying N doesn't re-tune on every new chunk size.
    key=["num_heads", "head_dim"],
)
@triton.jit
def _fp8_mqa_logits_kernel(
    q_ptr,
    k_ptr,
    k_scale_ptr,
    weights_ptr,
    ks_ptr,
    ke_ptr,
    logits_ptr,
    stride_q_m,
    stride_q_h,
    stride_q_d,
    stride_k_n,
    stride_k_d,
    stride_w_m,
    stride_w_h,
    stride_l_m,
    stride_l_n,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    N,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # bf16 q/k inputs: the wrapper pre-decodes FP8 → bf16. At compute-bound
    # prefill this is ~2× the in-kernel LUT (LUT lookups contend with the
    # matmul for ALU/regs). Paged-decode keeps the LUT path.
    m = tl.program_id(0)
    n_block = tl.program_id(1)

    n_start = n_block * BLOCK_N
    offs_n = n_start + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    # Early-exit when this row's `[ks, ke)` range doesn't overlap the tile.
    # Chunked prefill produces many such all-masked tiles per row.
    ks = tl.load(ks_ptr + m)
    ke = tl.load(ke_ptr + m)
    if (n_start >= ke) | (n_start + BLOCK_N <= ks):
        # When `clean_logits=False` the caller skipped the -inf pre-fill, so
        # write -inf here for the early-exit tile.
        tl.store(
            logits_ptr + m * stride_l_m + offs_n * stride_l_n,
            tl.full([BLOCK_N], float("-inf"), dtype=tl.float32),
            mask=mask_n,
        )
        return

    offs_h = tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, BLOCK_D)
    mask_h = offs_h < num_heads
    mask_d = offs_d < head_dim

    q = tl.load(
        q_ptr
        + m * stride_q_m
        + offs_h[:, None] * stride_q_h
        + offs_d[None, :] * stride_q_d,
        mask=mask_h[:, None] & mask_d[None, :],
        other=0.0,
    )

    k = tl.load(
        k_ptr + offs_n[:, None] * stride_k_n + offs_d[None, :] * stride_k_d,
        mask=mask_n[:, None] & mask_d[None, :],
        other=0.0,
    )
    k_scale = tl.load(k_scale_ptr + offs_n, mask=mask_n, other=0.0)
    s = tl.dot(q, tl.trans(k)) * k_scale[None, :]

    w = tl.load(
        weights_ptr + m * stride_w_m + offs_h * stride_w_h,
        mask=mask_h,
        other=0.0,
    )
    s = tl.where(s > 0, s, 0.0) * w[:, None]
    out = tl.sum(s, axis=0)

    valid = mask_n & (offs_n >= ks) & (offs_n < ke)
    out = tl.where(valid, out, float("-inf"))

    tl.store(
        logits_ptr + m * stride_l_m + offs_n * stride_l_n,
        out,
        mask=mask_n,
    )


def fp8_mqa_logits_triton(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    clean_logits: bool = True,
) -> torch.Tensor:
    """Triton implementation of DeepGEMM's fp8_mqa_logits.

    Args:
        q:            [M, H, D] native E4M3FN or raw uint8 bytes
        kv:           (K [N, D] native E4M3FN or raw uint8, scales [N])
        weights:      [M, H] float32
        cu_seqlen_ks: [M] int32
        cu_seqlen_ke: [M] int32
        clean_logits: when False, skip the -inf pre-fill of the output
            (indexer top-k reads only `[ks, ke)` per row). Matches DeepGEMM.
    Returns:
        logits:       [M, N] float32
    """
    k_fp8, k_scales = kv
    k_scales = k_scales.reshape(-1)

    M, num_heads, head_dim = q.shape
    N = k_fp8.shape[0]

    if clean_logits:
        logits = torch.full((M, N), float("-inf"), dtype=torch.float32, device=q.device)
    else:
        logits = torch.empty((M, N), dtype=torch.float32, device=q.device)

    BLOCK_H = max(16, triton.next_power_of_2(num_heads))
    BLOCK_D = triton.next_power_of_2(head_dim)

    # Decode raw E4M3FN bytes through the software LUT; SM80 does not support
    # the native float8 cast used by Hopper kernels.
    fp8_lut = _get_e4m3fn_bf16_lut(q.device)
    q_bf16 = fp8_lut[_as_e4m3fn_bytes(q, "q").long()]
    k_bf16 = fp8_lut[_as_e4m3fn_bytes(k_fp8, "k_fp8").long()]

    # Grid depends on the autotuned BLOCK_N.
    grid = lambda meta: (M, triton.cdiv(N, meta["BLOCK_N"]))  # noqa: E731
    _fp8_mqa_logits_kernel[grid](
        q_bf16,
        k_bf16,
        k_scales,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        logits,
        q_bf16.stride(0),
        q_bf16.stride(1),
        q_bf16.stride(2),
        k_bf16.stride(0),
        k_bf16.stride(1),
        weights.stride(0),
        weights.stride(1),
        logits.stride(0),
        logits.stride(1),
        num_heads=num_heads,
        head_dim=head_dim,
        N=N,
        BLOCK_H=BLOCK_H,
        BLOCK_D=BLOCK_D,
    )
    return logits


def warmup_fp8_mqa_logits_triton(
    num_heads: int,
    head_dim: int,
    device: torch.device,
) -> None:
    """Prime the prefill `@triton.autotune` cache so first-call doesn't pay
    the inline sweep (~5–8 s on A100 SM80). N is a runtime scalar, so one
    small-M / long-N shape covers all chunk lengths."""
    max_block_n = max(c.kwargs["BLOCK_N"] for c in _PREFILL_AUTOTUNE_CONFIGS)
    m = _PREFILL_WARMUP_M
    n = max(_PREFILL_WARMUP_N, max_block_n)
    q = torch.zeros(m, num_heads, head_dim, dtype=torch.uint8, device=device)
    k = torch.zeros(n, head_dim, dtype=torch.uint8, device=device)
    scales = torch.zeros(n, dtype=torch.float32, device=device)
    weights = torch.zeros(m, num_heads, dtype=torch.float32, device=device)
    ks = torch.zeros(m, dtype=torch.int32, device=device)
    ke = torch.full((m,), n, dtype=torch.int32, device=device)
    fp8_mqa_logits_triton(q, (k, scales), weights, ks, ke)


def warmup_fp8_paged_mqa_logits_triton(
    num_heads: int,
    head_dim: int,
    block_size: int,
    device: torch.device,
) -> None:
    """Prime the paged-decode `@triton.autotune` cache for the indexer's
    logits kernel (see `warmup_fp8_mqa_logits_triton` for rationale).
    """
    num_blocks = 2
    q = torch.zeros(1, 1, num_heads, head_dim, dtype=torch.uint8, device=device)
    kv_cache = torch.zeros(
        num_blocks, block_size, 1, head_dim + 4, dtype=torch.uint8, device=device
    )
    weights = torch.zeros(1, num_heads, dtype=torch.float32, device=device)
    seq_lens = torch.tensor([[block_size]], dtype=torch.int32, device=device)
    decode_lens = torch.ones(1, dtype=torch.int32, device=device)
    block_tables = torch.zeros(1, 1, dtype=torch.int32, device=device)
    fp8_paged_mqa_logits_triton(
        q,
        kv_cache,
        weights,
        seq_lens,
        decode_lens,
        block_tables,
        max_model_len=block_size,
    )
