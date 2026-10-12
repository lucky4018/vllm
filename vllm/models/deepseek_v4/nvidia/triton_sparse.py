# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4 sparse MLA attention for NVIDIA SM80."""

from typing import TYPE_CHECKING, cast

import torch

from vllm.forward_context import get_forward_context
from vllm.models.deepseek_v4.common.ops import (
    combine_topk_swa_indices,
    compute_global_topk_indices_and_lens,
    dequantize_and_gather_k_cache,
)
from vllm.models.deepseek_v4.nvidia.flashmla import DeepseekV4FlashMLAAttention
from vllm.models.deepseek_v4.sparse_mla import DeepseekV4FlashMLABackend
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWABackend
from vllm.v1.attention.ops.triton_mla_sparse import (
    triton_inv_rope_einsum,
    triton_sparse_attn_decode,
    triton_sparse_attn_prefill,
)
from vllm.v1.worker.workspace import current_workspace_manager

if TYPE_CHECKING:
    from vllm.models.deepseek_v4.sparse_mla import DeepseekV4FlashMLAMetadata
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata


class DeepseekV4TritonMLASparseBackend(DeepseekV4FlashMLABackend):
    @staticmethod
    def get_name() -> str:
        return "TRITON_MLA_SPARSE_DSV4"

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 8


class DeepseekV4TritonMLASparseAttention(DeepseekV4FlashMLAAttention):
    """NVIDIA-only SM80 attention; execution is delegated to Triton DSV4 ops."""

    backend_cls = DeepseekV4TritonMLASparseBackend
    # FlashMLA's SWA builder calls the SM90-only scheduler.  The portable
    # sparse SWA backend selects the reference Triton builder on SM80.
    swa_backend_cls = DeepseekSparseSWABackend

    @classmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        return num_heads

    def _o_proj(self, o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # DeepGEMM's FP8 einsum is SM90+ only.  Keep Ampere on the BF16
        # inverse-RoPE + WO_A reference and use the regular WO_B projection.
        z = triton_inv_rope_einsum(
            self.rotary_emb,
            o,
            positions,
            self.rope_head_dim,
            self.n_local_groups,
            self.o_lora_rank,
            self.wo_a,
        )
        return self.wo_b(z.flatten(1))

    def forward_mqa(self, q, kv, positions, output) -> None:
        assert output.shape == q.shape
        assert output.dtype == q.dtype
        context = get_forward_context().attn_metadata
        if context is None:
            output.zero_()
            return
        assert isinstance(context, dict)
        swa = cast("DeepseekSparseSWAMetadata", context[self.swa_cache_layer.prefix])
        dsv4 = cast("DeepseekV4FlashMLAMetadata | None", context.get(self.prefix))
        n = swa.num_decode_tokens
        if swa.num_prefills:
            self._triton_prefill(
                q[n:],
                self.kv_cache if self.compress_ratio > 1 else None,
                self.swa_cache_layer.kv_cache,
                output[n:],
                dsv4,
                swa,
            )
        if n == 0:
            return
        topk_indices = topk_lens = None
        if self.compress_ratio > 1:
            assert dsv4 is not None and swa.is_valid_token is not None
            if self.compress_ratio == 4:
                assert self.topk_indices_buffer is not None
                topk_indices, topk_lens = compute_global_topk_indices_and_lens(
                    self.topk_indices_buffer[:n],
                    swa.token_to_req_indices,
                    dsv4.block_table[: swa.num_decodes],
                    dsv4.block_size // self.compress_ratio,
                    swa.is_valid_token[:n],
                )
                topk_indices = topk_indices.view(n, 1, -1)
            else:
                topk_indices = dsv4.c128a_global_decode_topk_indices
                topk_lens = dsv4.c128a_decode_topk_lens
        triton_sparse_attn_decode(
            q=q[:n],
            kv_cache=self.kv_cache if self.compress_ratio > 1 else None,
            swa_k_cache=self.swa_cache_layer.kv_cache,
            swa_only=self.compress_ratio <= 1,
            topk_indices=topk_indices,
            topk_lens=topk_lens,
            swa_indices=swa.decode_swa_indices,
            swa_lens=swa.decode_swa_lens,
            attn_sink=self.attn_sink,
            scale=self.scale,
            head_dim=self.head_dim,
            nope_head_dim=self.nope_head_dim,
            rope_head_dim=self.rope_head_dim,
            output=output[:n],
        )

    def _triton_prefill(
        self, q, compressed_cache, swa_cache, output, attn_metadata, swa
    ) -> None:
        seq_lens = swa.prefill_seq_lens
        gather_lens = swa.prefill_gather_lens
        qloc = swa.query_start_loc
        qloc_cpu = swa.query_start_loc_cpu
        assert seq_lens is not None and gather_lens is not None
        assert qloc is not None and qloc_cpu is not None
        nd = swa.num_decodes
        nt = swa.num_decode_tokens
        np = swa.num_prefill_tokens
        base = qloc_cpu[nd]
        swa_only = attn_metadata is None
        if not swa_only:
            if self.compress_ratio == 4:
                assert self.topk_indices_buffer is not None
                topk = self.topk_indices_buffer[nt : nt + np]
            else:
                assert attn_metadata is not None
                topk = attn_metadata.c128a_prefill_topk_indices
            assert topk is not None
            top_k = topk.shape[-1]
        else:
            assert self.topk_indices_buffer is not None
            topk, top_k = self.topk_indices_buffer[nt:], 0
        for cs, ce, chunk_n, chunk_m in swa.get_prefill_chunk_plan(
            compress_ratio=self.compress_ratio,
            prefill_chunk_size=self.PREFILL_CHUNK_SIZE,
        ):
            chunk_size = ce - cs
            kv = current_workspace_manager().get_simultaneous(
                ((chunk_size, chunk_m, q.shape[-1]), torch.bfloat16)
            )[0]
            if not swa_only:
                assert attn_metadata is not None and compressed_cache is not None
                dequantize_and_gather_k_cache(
                    kv[:chunk_size],
                    compressed_cache,
                    seq_lens=seq_lens[cs:ce] // self.compress_ratio,
                    gather_lens=None,
                    block_table=attn_metadata.block_table[nd:][cs:ce],
                    block_size=attn_metadata.block_size // self.compress_ratio,
                    offset=0,
                    use_fnuz=False,
                )
            dequantize_and_gather_k_cache(
                kv[:chunk_size],
                swa_cache,
                seq_lens=seq_lens[cs:ce],
                gather_lens=gather_lens[cs:ce],
                block_table=swa.block_table[nd:][cs:ce],
                block_size=swa.block_size,
                offset=chunk_n,
                use_fnuz=current_platform.is_fp8_fnuz(),
            )
            qs = qloc_cpu[nd + cs] - base
            qe = qloc_cpu[nd + ce] - base
            combined, lengths = combine_topk_swa_indices(
                topk[qs:qe],
                qloc[nd + cs : nd + ce + 1],
                seq_lens[cs:ce],
                gather_lens[cs:ce],
                self.window_size,
                self.compress_ratio,
                top_k,
                chunk_m,
                chunk_n,
            )
            triton_sparse_attn_prefill(
                q=q[qs:qe],
                kv=kv.view(-1, 1, q.shape[-1]),
                indices=combined,
                topk_length=lengths,
                max_topk_length=(min(top_k, chunk_n) + self.window_size),
                scale=self.scale,
                head_dim=self.head_dim,
                nope_head_dim=self.nope_head_dim,
                rope_head_dim=self.rope_head_dim,
                attn_sink=self.attn_sink,
                output=output[qs:qe],
            )
