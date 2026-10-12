# SPDX-License-Identifier: Apache-2.0
"""Dependency-free contracts for the v0.29 DeepSeek-V4 SM80 port."""

import ast
from pathlib import Path

ROOT = Path(__file__).parents[2]


def _source(relative: str) -> str:
    return (ROOT / relative).read_text()


def _function(relative: str, name: str) -> ast.FunctionDef:
    module = ast.parse(_source(relative), filename=relative)
    return next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def test_sm80_backend_is_registered_and_selected():
    registry = _source("vllm/v1/attention/backends/registry.py")
    model = _source("vllm/models/deepseek_v4/nvidia/model.py")
    assert "TRITON_MLA_SPARSE_DSV4" in registry
    assert "DeepseekV4TritonMLASparseAttention" in model
    assert "device_capability.major == 8" in model


def test_explicit_flashmla_backend_keeps_v029_dispatch():
    function = _function(
        "vllm/models/deepseek_v4/nvidia/model.py", "_select_dsv4_attn_cls"
    )
    flashmla_branch = next(
        node
        for node in function.body
        if isinstance(node, ast.If)
        and any(
            isinstance(item, ast.Attribute)
            and item.attr == "FLASHMLA_SPARSE"
            for item in ast.walk(node.test)
        )
    )
    returns = [
        node
        for node in ast.walk(flashmla_branch)
        if isinstance(node, ast.Return)
    ]
    assert any(
        isinstance(node.value, ast.Name)
        and node.value.id == "DeepseekV4FlashMLAAttention"
        for node in returns
    )
    assert not any(
        isinstance(node.value, ast.Name)
        and node.value.id == "DeepseekV4TritonMLASparseAttention"
        for node in returns
    )


def test_sparse_indexer_has_triton_mqa_fallback():
    path = "vllm/model_executor/layers/sparse_attn_indexer.py"
    source = _source(path)
    function = _function(path, "sparse_attn_indexer")
    called = {
        node.func.id
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "fp8_mqa_logits_triton" in source
    assert "fp8_paged_mqa_logits_triton" in source
    assert {"fp8_mqa_logits_triton", "fp8_paged_mqa_logits_triton"} <= called


def test_streaming_state_preserves_logprobs_and_tool_ids():
    source = _source("vllm/entrypoints/openai/responses/streaming_events.py")
    assert "accumulated_logprobs" in source
    assert "state.accumulated_logprobs.extend" in source
    assert 'call_id or f"call_{random_uuid()}"' in source


def test_deepseek_tokenizer_defaults_to_thinking_and_normalizes_effort():
    source = _source("vllm/tokenizers/deepseek_v4.py")
    assert "thinking_enabled = True" in source
    assert 'reasoning_effort in ("low", "minimal", "medium")' in source
