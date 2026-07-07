# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pre-Hopper (SM80/A800) detection for DeepSeek-V4.

vLLM v0.22.1's nvidia DeepSeek-V4 path requires FlashMLA (SM90+) and DeepGEMM
MegaMoE (SM100). On pre-Hopper CUDA GPUs (e.g. A800, capability 8.0) we reuse
the portable AMD/ROCm reference path (plain PyTorch/Triton, runs on any CUDA).

The device class is a runtime constant (it cannot change within a process), so
the result is cached. This caching is REQUIRED for torch.compile / CUDA-graph:
calling current_platform.get_device_capability() (an lru_cache-wrapped C
function) inside a compiled forward triggers a Dynamo graph break
("can't handle functions not implemented in python"). With the value cached,
the compiled region only reads a plain Python bool constant. The cache is
warmed during eager model construction (this is called from __init__.py import
and get_builder_cls) well before graph capture.
"""
from vllm.platforms import current_platform

_USE_REFERENCE_IMPL: "bool | None" = None


def use_reference_impl() -> bool:
    """True on ROCm or pre-Hopper CUDA (compute capability major < 9). Cached."""
    global _USE_REFERENCE_IMPL
    if _USE_REFERENCE_IMPL is None:
        if current_platform.is_rocm():
            _USE_REFERENCE_IMPL = True
        elif current_platform.is_cuda():
            cap = current_platform.get_device_capability()
            _USE_REFERENCE_IMPL = bool(cap is not None and cap.major < 9)
        else:
            _USE_REFERENCE_IMPL = False
    return _USE_REFERENCE_IMPL
