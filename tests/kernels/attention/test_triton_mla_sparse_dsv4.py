# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the DeepSeek-V4 SM80 packed-cache decoder."""

import weakref

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.ops import triton_mla_sparse as sparse_ops
from vllm.v1.attention.ops.triton_mla_sparse import (
    _decode_cache_rows,
    _gather_cache_rows_float,
    triton_sparse_attn_prefill,
)


@pytest.mark.parametrize("block_size", [4, 256])
def test_decode_cache_rows_compiles_and_preserves_packed_layout(block_size: int):
    capability = current_platform.get_device_capability()
    if capability is None or capability.major != 8:
        pytest.skip("DeepSeek-V4 Triton packed-cache decoder requires SM80")

    num_blocks = 2
    token_data_bytes = 576
    token_scale_bytes = 8
    block_stride = block_size * (token_data_bytes + token_scale_bytes)
    cache = torch.zeros(
        (num_blocks, block_size, token_data_bytes + token_scale_bytes),
        dtype=torch.uint8,
        device="cuda",
    )
    flat = cache.view(-1)

    slots = torch.tensor(
        [[0, block_size - 1, block_size, 2 * block_size - 1]],
        dtype=torch.int32,
        device="cuda",
    )
    expected = torch.empty(
        (1, slots.shape[1], 512), dtype=torch.bfloat16, device="cuda"
    )

    for column, slot in enumerate(slots[0].tolist()):
        block_idx, pos_in_block = divmod(slot, block_size)
        token_offset = block_idx * block_stride + pos_in_block * token_data_bytes
        scale_offset = (
            block_idx * block_stride
            + block_size * token_data_bytes
            + pos_in_block * token_scale_bytes
        )

        # E4M3FN 0x38 is 1.0. E8M0 127 represents a unit scale.
        flat[token_offset : token_offset + 448] = 0x38
        flat[scale_offset : scale_offset + 8] = 127

        rope = torch.arange(64, dtype=torch.bfloat16, device="cuda") + column
        flat[token_offset + 448 : token_offset + 576].view(torch.bfloat16).copy_(
            rope
        )
        expected[0, column, :448] = 1
        expected[0, column, 448:] = rope

    actual = _decode_cache_rows(cache, slots)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_decode_cache_rows_masks_slots_outside_physical_cache():
    capability = current_platform.get_device_capability()
    if capability is None or capability.major != 8:
        pytest.skip("DeepSeek-V4 Triton packed-cache decoder requires SM80")

    block_size = 4
    num_blocks = 2
    cache = torch.zeros(
        (num_blocks, block_size, 584), dtype=torch.uint8, device="cuda"
    )
    slots = torch.tensor(
        [[-1, num_blocks * block_size, num_blocks * block_size + 17]],
        dtype=torch.int32,
        device="cuda",
    )

    actual = _decode_cache_rows(cache, slots)
    torch.cuda.synchronize()

    # Invalid rows are intentionally left unwritten and are zeroed by the
    # attention consumer. This assertion verifies the launch shape and, most
    # importantly, that neither negative nor upper-tail slots access the cache.
    assert actual.shape == (1, slots.shape[1], 512)


@pytest.mark.parametrize("swa_only", [False, True])
@pytest.mark.parametrize("has_lengths", [False, True])
@pytest.mark.parametrize("has_sink", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_sparse_decode_masks_unwritten_out_of_range_rows(
    monkeypatch, swa_only, has_lengths, has_sink, device
):
    if device == "cuda":
        capability = current_platform.get_device_capability()
        if capability is None or capability.major != 8:
            pytest.skip("DeepSeek-V4 Triton packed-cache decoder requires SM80")

    swa_cache = torch.zeros((2, 4, 584), dtype=torch.uint8, device=device)
    top_cache = torch.zeros((3, 4, 584), dtype=torch.uint8, device=device)
    for cache, value, encoded in [(swa_cache, 1, 0x38), (top_cache, 3, 0x44)]:
        pages = cache.view(cache.shape[0], -1)
        tokens = pages[:, : 4 * 576].view(cache.shape[0], 4, 576)
        tokens[..., :448] = encoded
        tokens[..., 448:].view(torch.bfloat16).fill_(value)
        pages[:, 4 * 576 :] = 127

    def poisoned_decode(cache, slots):
        valid = (slots >= 0) & (slots < cache.shape[0] * cache.shape[1])
        if device == "cuda":
            rows = _decode_cache_rows(cache, slots)
        else:
            rows = torch.full((*slots.shape, 512), torch.nan)
            rows[valid] = 1 if cache is swa_cache else 3
        # Deterministically expose any invalid row still used by attention.
        rows[~valid] = torch.nan
        return rows

    monkeypatch.setattr(sparse_ops, "_decode_cache_rows", poisoned_decode)
    q = torch.zeros((2, 1, 512), device=device)
    swa_indices = torch.tensor(
        [[0, -1, 8, 25, 1], [-1, 8, 25, -1, 9]], device=device
    )
    top_indices = torch.tensor(
        [[8, -1, 12, 29, 9], [-1, 12, 29, -1, 13]], device=device
    )
    lengths = torch.tensor([4, 4], device=device) if has_lengths else None
    sink = torch.zeros(1, device=device) if has_sink else None
    output = torch.empty_like(q)

    sparse_ops.triton_sparse_attn_decode(
        q=q,
        kv_cache=None if swa_only else top_cache,
        swa_k_cache=swa_cache,
        swa_only=swa_only,
        topk_indices=None if swa_only else top_indices,
        topk_lens=lengths,
        swa_indices=swa_indices,
        swa_lens=lengths,
        attn_sink=sink,
        scale=512**-0.5,
        head_dim=512,
        nope_head_dim=448,
        rope_head_dim=64,
        output=output,
    )

    rows_per_group = 1 if has_lengths else 2
    count = rows_per_group * (1 if swa_only else 2)
    total = rows_per_group * (1 if swa_only else 4)
    expected = torch.zeros_like(q)
    expected[0] = total / (count + int(has_sink))
    torch.testing.assert_close(output, expected, rtol=0, atol=1e-6)


def test_prefill_gather_masks_lengths_and_out_of_range_indices():
    capability = current_platform.get_device_capability()
    if capability is None or capability.major != 8:
        pytest.skip("DeepSeek-V4 Triton prefill gather requires SM80")

    rows = torch.arange(6 * 512, dtype=torch.float32, device="cuda")
    rows = rows.reshape(6, 512).to(torch.bfloat16)
    indices = torch.tensor(
        [[2, 5, -1, 99], [4, 1, 0, 3]],
        dtype=torch.int32,
        device="cuda",
    )
    lengths = torch.tensor([2, 3], dtype=torch.int32, device="cuda")

    actual = _gather_cache_rows_float(rows, indices, lengths)
    expected = torch.zeros_like(actual)
    expected[0, 0] = rows[2].float()
    expected[0, 1] = rows[5].float()
    expected[1, 0] = rows[4].float()
    expected[1, 1] = rows[1].float()
    expected[1, 2] = rows[0].float()

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_sparse_prefill_matches_fp32_reference_with_padded_wide_indices():
    capability = current_platform.get_device_capability()
    if capability is None or capability.major != 8:
        pytest.skip("DeepSeek-V4 Triton sparse prefill requires SM80")

    generator = torch.Generator(device="cuda").manual_seed(7)
    q = torch.randn(
        (3, 2, 512),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    kv = torch.randn(
        (7, 1, 512),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    indices = torch.full((3, 8192), -1, dtype=torch.int32, device="cuda")
    indices[:, :7] = torch.tensor(
        [[0, 1, 2, -1, 5, 6, 4],
         [6, 5, 4, 3, 2, 99, 0],
         [1, 3, 5, 0, 2, 4, 6]],
        dtype=torch.int32,
        device="cuda",
    )
    lengths = torch.tensor([3, 5, 7], dtype=torch.int32, device="cuda")
    sink = torch.tensor([-0.5, 0.25], dtype=torch.float32, device="cuda")
    output = torch.empty_like(q)
    scale = 512**-0.5

    triton_sparse_attn_prefill(
        q=q,
        kv=kv,
        indices=indices,
        topk_length=lengths,
        max_topk_length=7,
        scale=scale,
        head_dim=512,
        nope_head_dim=448,
        rope_head_dim=64,
        attn_sink=sink,
        output=output,
    )

    active = indices[:, :7]
    valid = (active >= 0) & (active < kv.shape[0])
    valid &= torch.arange(7, device="cuda")[None] < lengths[:, None]
    safe = active.clamp(0, kv.shape[0] - 1).long()
    rows = kv.reshape(-1, 512)[safe].float()
    rows.masked_fill_(~valid[..., None], 0)
    scores = torch.einsum("thd,tkd->thk", q.float(), rows) * scale
    scores.masked_fill_(~valid[:, None], -torch.inf)
    scores = torch.cat((scores, sink.view(1, 2, 1).expand(3, -1, -1)), -1)
    probs = torch.softmax(scores, -1)
    expected = torch.einsum("thk,tkd->thd", probs[..., :7], rows)

    torch.testing.assert_close(
        output.float(), expected, rtol=2e-2, atol=2e-2
    )


def test_sparse_prefill_releases_gather_workspace_between_chunks(monkeypatch):
    capability = current_platform.get_device_capability()
    if capability is None or capability.major != 8:
        pytest.skip("DeepSeek-V4 Triton sparse prefill requires SM80")

    references = []
    original_gather = sparse_ops._gather_cache_rows_float

    def tracked_gather(rows, indices, lengths=None):
        assert all(reference() is None for reference in references)
        gathered = original_gather(rows, indices, lengths)
        references.append(weakref.ref(gathered))
        return gathered

    monkeypatch.setattr(sparse_ops, "_prefill_query_chunk_size", lambda *_: 1)
    monkeypatch.setattr(sparse_ops, "_gather_cache_rows_float", tracked_gather)

    q = torch.randn((3, 1, 512), dtype=torch.bfloat16, device="cuda")
    kv = torch.randn((4, 1, 512), dtype=torch.bfloat16, device="cuda")
    indices = torch.tensor(
        [[0, 1], [1, 2], [2, 3]], dtype=torch.int32, device="cuda"
    )
    lengths = torch.full((3,), 2, dtype=torch.int32, device="cuda")
    output = torch.empty_like(q)

    sparse_ops.triton_sparse_attn_prefill(
        q=q,
        kv=kv,
        indices=indices,
        topk_length=lengths,
        max_topk_length=2,
        scale=512**-0.5,
        head_dim=512,
        nope_head_dim=448,
        rope_head_dim=64,
        attn_sink=None,
        output=output,
    )

    assert all(reference() is None for reference in references)
