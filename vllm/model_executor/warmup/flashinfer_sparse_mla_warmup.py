# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warmup and autotune helpers for FlashInfer sparse MLA backends."""

from typing import TYPE_CHECKING, cast

import torch

from vllm.logger import init_logger
from vllm.model_executor.warmup.flashinfer_autotune_cache import (
    resolve_flashinfer_autotune_file,
    write_flashinfer_autotune_cache,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import autotune as flashinfer_autotune
from vllm.utils.flashinfer import has_flashinfer
from vllm.v1.worker.gpu.warmup import run_mixed_prefill_decode_warmup

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner as V2GPUModelRunner
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)

_DEEPSEEK_V4_SPARSE_MLA_BACKENDS = frozenset(
    {
        "FLASHMLA_SPARSE_DSV4",
        "FLASHINFER_MLA_SPARSE_DSV4",
        "ROCM_FLASHMLA_SPARSE_DSV4",
        "DEEPSEEK_SPARSE_SWA",
        "TRITON_MLA_SPARSE_DSV4",
    }
)
_FLASHINFER_MLA_SPARSE_BACKENDS = frozenset({"FLASHINFER_MLA_SPARSE_SM120"})
_DEEPSEEK_V4_FLASHINFER_MLA_SPARSE_BACKENDS = frozenset({"FLASHINFER_MLA_SPARSE_DSV4"})

_FLASHINFER_SM120_SPARSE_MLA_DECODE_LABELS = {
    "FLASHINFER_MLA_SPARSE_SM120": "DSv3.2",
    "FLASHINFER_MLA_SPARSE_DSV4": "DSv4",
}

_SPARSE_MLA_MIXED_WARMUP_TOKENS = 16


def _attention_backend_name(backend: object) -> str | None:
    get_name = getattr(backend, "get_name", None)
    if get_name is None:
        return None
    try:
        return get_name()
    except NotImplementedError:
        return None


def _has_deepseek_v4_sparse_mla_backend(runner: "GPUModelRunner") -> bool:
    for groups in getattr(runner, "attn_groups", []) or ():
        for group in groups:
            name = _attention_backend_name(getattr(group, "backend", None))
            if name in _DEEPSEEK_V4_SPARSE_MLA_BACKENDS:
                return True
    return False


def _has_attention_backend(runner: "GPUModelRunner", expected_name: str) -> bool:
    for groups in getattr(runner, "attn_groups", []) or ():
        for group in groups:
            name = _attention_backend_name(getattr(group, "backend", None))
            if name == expected_name:
                return True
    return False


def _find_indexer_block_size(runner: "GPUModelRunner") -> int | None:
    for groups in getattr(runner, "attn_groups", []) or ():
        for group in groups:
            name = _attention_backend_name(getattr(group, "backend", None))
            if name == "DEEPSEEK_V4_INDEXER":
                return int(group.kv_cache_spec.block_size)
    return None


def _warmup_sm80_dsv4_indexer(worker: "Worker") -> bool:
    """Compile the SM80 Triton indexer kernels before graph capture."""
    runner = worker.model_runner
    if not _has_attention_backend(runner, "TRITON_MLA_SPARSE_DSV4"):
        return False
    if not current_platform.is_cuda():
        return False
    capability = current_platform.get_device_capability()
    if capability is None or capability.major >= 9:
        return False

    hf_config = worker.model_config.hf_config
    num_heads = int(hf_config.index_n_heads)
    head_dim = int(hf_config.index_head_dim)
    block_size = _find_indexer_block_size(runner)
    if block_size is None:
        logger.warning(
            "Skipping DeepSeek V4 SM80 Triton Indexer warmup because the "
            "DEEPSEEK_V4_INDEXER cache group was not found."
        )
        return False

    from vllm.v1.attention.ops.mqa_logits_triton import (
        warmup_fp8_mqa_logits_triton,
        warmup_fp8_paged_mqa_logits_triton,
    )

    logger.info(
        "Warming up DeepSeek V4 SM80 Triton Indexer MQA kernels "
        "(heads=%d, head_dim=%d, block_size=%d).",
        num_heads,
        head_dim,
        block_size,
    )
    with torch.inference_mode():
        warmup_fp8_mqa_logits_triton(num_heads, head_dim, worker.device)
        warmup_fp8_paged_mqa_logits_triton(
            num_heads, head_dim, block_size, worker.device
        )
    return True


def _flashinfer_sparse_mla_decode_label(
    runner: "GPUModelRunner",
    allowed_backends: frozenset[str],
) -> str | None:
    for groups in getattr(runner, "attn_groups", []) or ():
        for group in groups:
            name = _attention_backend_name(getattr(group, "backend", None))
            if name in allowed_backends:
                return _FLASHINFER_SM120_SPARSE_MLA_DECODE_LABELS.get(name)
    return None


def _clamp_warmup_tokens(num_tokens: int, max_tokens: int) -> int:
    return max(0, min(num_tokens, max_tokens))


def _uses_v2_model_runner(runner: "GPUModelRunner") -> bool:
    vllm_config = getattr(runner, "vllm_config", None)
    return bool(getattr(vllm_config, "use_v2_model_runner", False))


def _run_flashinfer_sparse_mla_decode_autotune(
    worker: "Worker",
    num_tokens: int,
    allowed_backends: frozenset[str],
) -> bool:
    """Autotune FlashInfer's SM120 sparse-MLA decode path."""
    runner = worker.model_runner
    log_label = _flashinfer_sparse_mla_decode_label(runner, allowed_backends)
    if log_label is None:
        return False
    if worker.vllm_config.kernel_config.enable_flashinfer_autotune is not True:
        return False
    if not has_flashinfer() or not current_platform.is_device_capability_family(120):
        return False

    try:
        from flashinfer.autotuner import AutoTuner
    except ImportError:
        logger.warning(
            "Skipping FlashInfer SM120 sparse MLA decode autotune because "
            "FlashInfer autotuner is unavailable."
        )
        return False

    from vllm.distributed.parallel_state import get_world_group

    world = get_world_group()
    is_leader = world.rank_in_group == 0
    cache_path = resolve_flashinfer_autotune_file(runner)

    dummy_run_kwargs = dict(
        num_tokens=num_tokens,
        skip_eplb=True,
        is_profile=True,
        force_attention=True,
        create_mixed_batch=True,
    )

    if is_leader:
        logger.info(
            "Autotuning FlashInfer SM120 sparse MLA %s decode with cache: %s",
            log_label,
            cache_path,
        )

    with torch.inference_mode():
        warmup_executed = True
        if is_leader:
            if _uses_v2_model_runner(runner) and runner.max_num_reqs >= 2:
                v2_runner = cast("V2GPUModelRunner", runner)
                warmup_executed = run_mixed_prefill_decode_warmup(
                    v2_runner,
                    worker.execute_model,
                    worker.sample_tokens,
                    num_tokens,
                    mixed_step_context=flashinfer_autotune(True, cache=str(cache_path)),
                    req_id_prefix="_sparse_mla_v2_warmup",
                )
            else:
                with flashinfer_autotune(True, cache=str(cache_path)):
                    runner._dummy_run(**dummy_run_kwargs)
        else:
            if _uses_v2_model_runner(runner) and runner.max_num_reqs >= 2:
                v2_runner = cast("V2GPUModelRunner", runner)
                warmup_executed = run_mixed_prefill_decode_warmup(
                    v2_runner,
                    worker.execute_model,
                    worker.sample_tokens,
                    num_tokens,
                    req_id_prefix="_sparse_mla_v2_warmup",
                )
            else:
                runner._dummy_run(**dummy_run_kwargs)

    if not warmup_executed:
        return False

    tune_results: bytes | None = None
    if is_leader and cache_path.exists():
        with open(cache_path, "rb") as f:
            tune_results = f.read()

    tune_results = world.broadcast_object(tune_results, src=0)
    if tune_results is None:
        logger.warning(
            "No FlashInfer SM120 sparse MLA %s decode autotune cache entries found. "
            "Falling back to FlashInfer's default tactic heuristic.",
            log_label,
        )
        world.barrier()
        return True

    write_flashinfer_autotune_cache(cache_path, tune_results)
    world.barrier()

    AutoTuner.get().load_configs(str(cache_path))
    logger.info(
        "FlashInfer SM120 sparse MLA %s decode autotune cache loaded on rank %d "
        "from %s.",
        log_label,
        world.rank_in_group,
        cache_path,
    )
    return True


def _flashinfer_sparse_mla_decode_autotune(
    worker: "Worker",
    num_tokens: int,
) -> bool:
    return _run_flashinfer_sparse_mla_decode_autotune(
        worker, num_tokens, _FLASHINFER_MLA_SPARSE_BACKENDS
    )


def _deepseek_v4_sparse_mla_decode_autotune(
    worker: "Worker",
    num_tokens: int,
) -> bool:
    return _run_flashinfer_sparse_mla_decode_autotune(
        worker, num_tokens, _DEEPSEEK_V4_FLASHINFER_MLA_SPARSE_BACKENDS
    )


def flashinfer_sparse_mla_decode_autotune_warmup(worker: "Worker") -> None:
    """Autotune generic FlashInfer sparse MLA decode when selected."""
    runner = worker.model_runner
    if runner.is_pooling_model:
        return

    max_tokens = worker.scheduler_config.max_num_batched_tokens
    mixed_tokens = _clamp_warmup_tokens(_SPARSE_MLA_MIXED_WARMUP_TOKENS, max_tokens)
    if mixed_tokens <= 0:
        return
    _flashinfer_sparse_mla_decode_autotune(worker, mixed_tokens)


def deepseek_v4_sparse_mla_attention_warmup(worker: "Worker") -> None:
    """Warm DSv4 sparse-MLA mixed prefill+decode attention."""
    runner = worker.model_runner
    if runner.is_pooling_model or not _has_deepseek_v4_sparse_mla_backend(runner):
        return
    if _warmup_sm80_dsv4_indexer(worker):
        return

    max_tokens = worker.scheduler_config.max_num_batched_tokens
    mixed_tokens = _clamp_warmup_tokens(_SPARSE_MLA_MIXED_WARMUP_TOKENS, max_tokens)
    if mixed_tokens <= 0:
        return

    logger.info(
        "Warming up DeepSeek V4 sparse MLA attention for mixed tokens=%s.",
        mixed_tokens,
    )
    mixed_warmup_done = _deepseek_v4_sparse_mla_decode_autotune(worker, mixed_tokens)
    if not mixed_warmup_done:
        if _uses_v2_model_runner(runner) and runner.max_num_reqs >= 2:
            v2_runner = cast("V2GPUModelRunner", runner)
            run_mixed_prefill_decode_warmup(
                v2_runner,
                worker.execute_model,
                worker.sample_tokens,
                mixed_tokens,
                req_id_prefix="_sparse_mla_v2_warmup",
            )
        else:
            runner._dummy_run(
                num_tokens=mixed_tokens,
                skip_eplb=True,
                is_profile=True,
                force_attention=True,
                create_mixed_batch=True,
            )
