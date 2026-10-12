# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.mqa_logits_triton import (
    _kv_page_offset,
    fp8_mqa_logits_triton,
    fp8_paged_mqa_logits_triton,
)

pytestmark = [
    pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA only"),
]


@triton.jit
def _page_offset_test_kernel(block_idx_ptr, output_ptr, page_stride):
    block_idx = tl.load(block_idx_ptr)
    tl.store(output_ptr, _kv_page_offset(block_idx, page_stride))


def test_paged_mqa_page_offset_uses_int64_for_packed_cache_stride():
    block_idx = torch.tensor([3459], dtype=torch.int32, device="cuda")
    output = torch.empty(1, dtype=torch.int64, device="cuda")
    page_stride = 620_864

    _page_offset_test_kernel[(1,)](block_idx, output, page_stride)
    torch.cuda.synchronize()

    expected = block_idx.item() * page_stride
    assert expected > 2**31 - 1
    assert output.item() == expected


def _run_paged_mqa(
    block_table: list[int],
    seq_len: int,
    *,
    num_kv_blocks: int = 3,
    block_size: int = 64,
    max_model_len: int | None = None,
) -> torch.Tensor:
    device = torch.device("cuda")
    num_heads = 64
    head_dim = 128
    if max_model_len is None:
        max_model_len = seq_len

    q = torch.zeros(1, 1, num_heads, head_dim, dtype=torch.uint8, device=device)
    kv_cache = torch.zeros(
        num_kv_blocks,
        block_size,
        1,
        head_dim + 4,
        dtype=torch.uint8,
        device=device,
    )
    weights = torch.ones(1, num_heads, dtype=torch.float32, device=device)
    seq_lens = torch.tensor([[seq_len]], dtype=torch.int32, device=device)
    decode_lens = torch.ones(1, dtype=torch.int32, device=device)
    block_tables = torch.tensor([block_table], dtype=torch.int32, device=device)

    logits = fp8_paged_mqa_logits_triton(
        q,
        kv_cache,
        weights,
        seq_lens,
        decode_lens,
        block_tables,
        max_model_len=max_model_len,
    )
    torch.cuda.synchronize(device)
    return logits


def test_paged_mqa_valid_pages_and_partial_tail():
    logits = _run_paged_mqa([2, 0, 1], 130)

    assert logits.shape == (1, 130)
    torch.testing.assert_close(logits, torch.zeros_like(logits))


@pytest.mark.parametrize("invalid_page", [-1, 3])
def test_paged_mqa_masks_invalid_physical_pages(invalid_page: int):
    logits = _run_paged_mqa([0, invalid_page, 1], 130)

    torch.testing.assert_close(logits[:, :64], torch.zeros_like(logits[:, :64]))
    assert torch.isneginf(logits[:, 64:128]).all()
    torch.testing.assert_close(logits[:, 128:], torch.zeros_like(logits[:, 128:]))


def test_paged_mqa_ignores_inactive_entries_in_wide_block_table():
    block_table = [0, 1] + [-1] * 2046
    logits = _run_paged_mqa(block_table, 65)

    torch.testing.assert_close(logits, torch.zeros_like(logits))


def test_paged_mqa_rejects_width_beyond_block_table_capacity():
    with pytest.raises(ValueError, match="exceeds block-table capacity"):
        _run_paged_mqa([0], 64, max_model_len=65)


def test_paged_mqa_diagnostic_rejects_invalid_active_page(monkeypatch):
    monkeypatch.setenv("VLLM_VALIDATE_PAGED_MQA_BLOCK_TABLE", "1")

    with pytest.raises(ValueError, match="invalid active KV pages"):
        _run_paged_mqa([0, 3], 65)


@pytest.mark.parametrize("page_padding,storage_offset", [(0, 0), (576, 0), (576, 256)])
@pytest.mark.parametrize("clean_logits", [True, False])
def test_paged_mqa_matches_prefill_for_real_indexer_cache(
    page_padding, storage_offset, clean_logits
):
    """Use the production cache writer/gatherer, including padded pool pages."""
    device = torch.device("cuda")
    block_size, head_dim, num_heads = 64, 128, 64
    num_blocks, seq_len = 3, 130
    page_bytes = block_size * (head_dim + 4)
    backing = torch.zeros(
        num_blocks * (page_bytes + page_padding) + storage_offset,
        dtype=torch.uint8,
        device=device,
    )
    cache = torch.as_strided(
        backing,
        (num_blocks, block_size, 1, head_dim + 4),
        (page_bytes + page_padding, head_dim + 4, head_dim + 4, 1),
        storage_offset=storage_offset,
    )
    block_table = torch.tensor([[2, 0, 1]], dtype=torch.int32, device=device)
    positions = torch.arange(seq_len, device=device)
    slots = (
        block_table[0, positions // block_size].long() * block_size
        + positions % block_size
    )
    # Powers of two are represented exactly by the production quantizer.
    values = torch.tensor([0.5, 1.0, -2.0, 4.0], device=device)
    keys = values[positions % len(values), None].expand(-1, head_dim)
    keys = keys.to(torch.bfloat16).contiguous()
    ops.indexer_k_quant_and_cache(keys, cache.squeeze(-2), slots, head_dim, "ue8m0")
    gathered = torch.empty((seq_len, head_dim), dtype=torch.uint8, device=device)
    scales = torch.empty((seq_len, 4), dtype=torch.uint8, device=device)
    ops.cp_gather_indexer_k_quant_cache(
        cache.squeeze(-2),
        gathered,
        scales,
        block_table,
        torch.tensor([0, seq_len], dtype=torch.int32, device=device),
    )
    query = torch.full((1, num_heads, head_dim), 0x38, dtype=torch.uint8, device=device)
    weights = torch.full((1, num_heads), 1 / num_heads, device=device)
    ends = torch.tensor([seq_len], dtype=torch.int32, device=device)
    expected = fp8_mqa_logits_triton(
        query,
        (gathered, scales.view(torch.float32)),
        weights,
        torch.zeros_like(ends),
        ends,
    )
    torch.testing.assert_close(expected[0], keys[:, 0].float().clamp_min(0) * head_dim)
    actual = fp8_paged_mqa_logits_triton(
        query.unsqueeze(1),
        cache,
        weights,
        ends.unsqueeze(0),
        torch.ones_like(ends),
        block_table,
        max_model_len=seq_len,
        clean_logits=clean_logits,
    )
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("num_heads", [16, 64])
def test_paged_mqa_real_cache_matches_torch_for_padded_multi_token_decode(num_heads):
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(7319)
    block_size, head_dim = 64, 128
    num_blocks, page_stride, storage_offset = 6, 9024, 256
    backing = torch.zeros(
        num_blocks * page_stride + storage_offset, dtype=torch.uint8, device=device
    )
    cache = torch.as_strided(
        backing,
        (num_blocks, block_size, 1, head_dim + 4),
        (page_stride, head_dim + 4, head_dim + 4, 1),
        storage_offset=storage_offset,
    )
    block_tables = torch.tensor(
        [[4, 2, -1], [1, 5, 3]], device=device, dtype=torch.int32
    )
    lengths = [67, 130]
    slots = torch.cat(
        [
            block_tables[batch, torch.arange(length, device=device) // block_size]
            .long()
            .mul(block_size)
            .add(torch.arange(length, device=device) % block_size)
            for batch, length in enumerate(lengths)
        ]
    )
    keys = torch.randn(sum(lengths), head_dim, generator=generator, device=device).to(
        torch.bfloat16
    )
    ops.indexer_k_quant_and_cache(keys, cache.squeeze(-2), slots, head_dim, "ue8m0")
    gathered = torch.empty_like(keys, dtype=torch.uint8)
    scales = torch.empty((sum(lengths), 4), device=device, dtype=torch.uint8)
    ops.cp_gather_indexer_k_quant_cache(
        cache.squeeze(-2),
        gathered,
        scales,
        block_tables,
        torch.tensor([0, 67, 197], device=device, dtype=torch.int32),
    )
    q = torch.randn(2, 3, num_heads, head_dim, generator=generator, device=device).to(
        torch.float8_e4m3fn
    )
    weights = torch.rand(6, num_heads, generator=generator, device=device) / num_heads
    seq_lens = torch.tensor(
        [[65, 66, 67], [129, 130, 0]], device=device, dtype=torch.int32
    )
    decode_lens = torch.tensor([3, 2], device=device, dtype=torch.int32)
    actual = fp8_paged_mqa_logits_triton(
        q, cache, weights, seq_lens, decode_lens, block_tables, max_model_len=130
    )
    decoded_keys = gathered.view(torch.float8_e4m3fn).float()
    decoded_scales = scales.view(torch.float32).flatten()
    expected = torch.full_like(actual, -torch.inf)
    start = 0
    for batch, length in enumerate(lengths):
        for next_id in range(int(decode_lens[batch])):
            row = batch * 3 + next_id
            end = int(seq_lens[batch, next_id])
            dots = q[batch, next_id].float() @ decoded_keys[start : start + end].T
            dots = dots * decoded_scales[start : start + end]
            expected[row, :end] = (dots.clamp_min(0) * weights[row, :, None]).sum(0)
        start += length
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
