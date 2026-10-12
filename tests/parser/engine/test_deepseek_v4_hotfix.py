# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for DeepSeek V4-specific parser engine semantics."""


import pytest

from tests.parser.engine.conftest import make_mock_tokenizer
from tests.parser.engine.replay_harness import (
    DUMMY_TOOLS,
    _test_request,
)
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.parser.abstract_parser import DelegatingParser
from vllm.parser.deepseek_v4 import (
    DSML_THINK_END,
    DSML_THINK_START,
    DSML_TOOL_END,
    DSML_TOOL_START,
    DeepSeekV4Parser,
)
from vllm.parser.parser_manager import ParserManager

_THINK_START_ID = 50
_THINK_END_ID = 51

_PARAM_OPEN = '｜DSML｜parameter name="{name}" string="{is_str}">'
_PARAM_CLOSE = "</｜DSML｜parameter>"


def _hotfix_request(api="chat", overrides=None, include_reasoning=True):
    common = dict(
        model="deepseek-v4-flash",
        chat_template_kwargs=overrides,
        include_reasoning=include_reasoning,
    )
    if api == "responses":
        return ResponsesRequest(input="test", **common)
    return ChatCompletionRequest(
        messages=[{"role": "user", "content": "test"}], **common
    )


def _hotfix_parser(path, settings=None):
    tokenizer = make_mock_tokenizer(
        {
            DSML_THINK_START: 200,
            DSML_THINK_END: 201,
            DSML_TOOL_START: 202,
            DSML_TOOL_END: 203,
        }
    )
    cls = DeepSeekV4Parser
    if path == "adapter":
        cls = ParserManager.get_parser(
            reasoning_parser_name="deepseek_v4",
            tool_parser_name="deepseek_v4",
            enable_auto_tools=True,
        )
        assert issubclass(cls, DelegatingParser)
    return cls(tokenizer, chat_template_kwargs=settings or {})


def _hotfix_stream(parser, text, request, reason, split, terminal_empty=False):
    if request.tools:
        # The serving layer selects automatic tool parsing when tools are
        # supplied.  Keep this focused parser harness explicit so a request
        # with tools does not accidentally exercise the "no tools" branch.
        request.tool_choice = "auto"
    chunks = (
        [text]
        if split == 0
        else [text[i : i + split] for i in range(0, len(text), split)]
    )
    if terminal_empty:
        chunks.append("")
    if not chunks:
        chunks = [""]
    deltas = []
    for i, chunk in enumerate(chunks):
        finished = i == len(chunks) - 1
        delta = parser.parse_delta(
            chunk,
            [],
            request,
            finished=finished,
            finish_reason=reason if finished else None,
        )
        if delta is not None:
            deltas.append(delta)
    return (
        "".join(d.reasoning or "" for d in deltas),
        "".join(d.content or "" for d in deltas),
        [tc for d in deltas for tc in (d.tool_calls or [])],
    )


class TestHotfixFallback:
    """Replay generated output through real engines; no model or GPU required.

    The contract is field routing after termination, not generation control.
    Compare chunked output with the full parse reused by Responses completion.
    """

    @pytest.mark.parametrize("path", ["direct", "adapter"])
    @pytest.mark.parametrize("api", ["chat", "responses"])
    @pytest.mark.parametrize(
        "reason", ["stop", "length", "abort", "error", None, "unknown"]
    )
    @pytest.mark.parametrize("split", [0, 1, 7])
    @pytest.mark.parametrize("terminal_empty", [False, True])
    def test_finish_reason_and_pending_tail(
        self, path, api, reason, split, terminal_empty
    ):
        request = _hotfix_request(api)
        text = "答案是 42，比较符号 <"
        parser = _hotfix_parser(path)
        reasoning, content, tools = _hotfix_stream(
            parser, text, request, reason, split, terminal_empty
        )
        expected = text if reason == "stop" else ""
        assert reasoning == text
        assert content == expected
        assert not tools
        # Reparse on the same instance, as response.completed does.
        full_reasoning, full_content, full_tools = parser.parse_with_finish_reason(
            text, request, finish_reason=reason
        )
        assert full_reasoning == reasoning
        assert (full_content or "") == content
        assert not full_tools

    @pytest.mark.parametrize("path", ["direct", "adapter"])
    @pytest.mark.parametrize(
        "server,override,enabled",
        [
            ({}, None, True),
            ({"force_nonempty_content": False}, None, False),
            ({"force_nonempty_content": False}, {"force_nonempty_content": True}, True),
            (
                {"force_nonempty_content": True},
                {"force_nonempty_content": False},
                False,
            ),
            ({"force_nonempty_content": False}, {"thinking": True}, False),
            *[
                ({}, {"force_nonempty_content": value}, False)
                for value in [False, "false", "true", 0, 1, None, [], {}]
            ],
            *[
                ({"force_nonempty_content": value}, None, False)
                for value in ["false", 0, 1, None]
            ],
        ],
    )
    def test_request_overrides_server_and_only_boolean_true_enables(
        self, path, server, override, enabled
    ):
        request = _hotfix_request(overrides=override)
        expected = "answer" if enabled else ""
        assert (
            _hotfix_stream(_hotfix_parser(path, server), "answer", request, "stop", 1)[
                1
            ]
            == expected
        )
        result = _hotfix_parser(path, server).parse_with_finish_reason(
            "answer", request, finish_reason="stop"
        )
        assert result[0] == "answer"
        assert (result[1] or "") == expected

    @pytest.mark.parametrize("path", ["direct", "adapter"])
    @pytest.mark.parametrize("split", [0, 1, 3])
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("thought</think>", ""),
            ("thought</think>answer", "answer"),
            ("<think>answer", "answer"),
            ("", ""),
            ("   \n", ""),
            ("thought<｜DSML｜tool_calls>", ""),
            ("thought<｜DSML｜invoke name=broken", ""),
            ("thought<｜DS", ""),
            ("thought</｜D", ""),
        ],
    )
    def test_real_boundaries_and_dsml_never_promote(self, path, split, text, expected):
        request = _hotfix_request()
        streamed = _hotfix_stream(_hotfix_parser(path), text, request, "stop", split)
        assert streamed[1] == expected
        full = _hotfix_parser(path).parse_with_finish_reason(
            text, request, finish_reason="stop"
        )
        assert (full[1] or "") == expected

    @pytest.mark.parametrize("path", ["direct", "adapter"])
    def test_special_token_end_marker_does_not_promote(self, path):
        parser = _hotfix_parser(path)
        request = _hotfix_request()
        first = parser.parse_delta("thought", [300], request, finished=False)
        last = parser.parse_delta(
            "</think>", [201], request, finished=True, finish_reason="stop"
        )
        assert first.reasoning == "thought"
        assert last is None or not last.content

    @pytest.mark.parametrize("path", ["direct", "adapter"])
    @pytest.mark.parametrize("split", [0, 1])
    def test_include_reasoning_false_still_supplies_fallback(self, path, split):
        result = _hotfix_stream(
            _hotfix_parser(path),
            "answer",
            _hotfix_request(include_reasoning=False),
            "stop",
            split,
        )
        assert result[:2] == ("", "answer")

    @pytest.mark.parametrize("path", ["direct", "adapter"])
    def test_thinking_disabled_keeps_normal_content(self, path):
        settings = {"thinking": False}
        request = _hotfix_request(overrides=settings)
        result = _hotfix_stream(
            _hotfix_parser(path, settings), "answer", request, "stop", 1
        )
        assert result[:2] == ("", "answer")

    @pytest.mark.parametrize("path", ["direct", "adapter"])
    def test_full_parse_reset_does_not_reuse_previous_generation(self, path):
        parser = _hotfix_parser(path)
        request = _hotfix_request()
        assert (
            parser.parse_with_finish_reason(
                "x</think>first", request, finish_reason="stop"
            )[1]
            == "first"
        )
        assert parser.parse_with_finish_reason("second", request, finish_reason="stop")[
            :2
        ] == ("second", "second")
        assert parser.parse("third", request)[1] is None

    @pytest.mark.parametrize("path", ["direct", "adapter"])
    @pytest.mark.parametrize("split", [0, 1, 9])
    def test_valid_tool_output_is_not_promoted(self, path, split):
        request = _hotfix_request()
        request.tools = _test_request(DUMMY_TOOLS).tools
        text = (
            'thought<｜DSML｜tool_calls><｜DSML｜invoke name="stub">'
            "</｜DSML｜invoke></｜DSML｜tool_calls>"
        )
        result = _hotfix_stream(_hotfix_parser(path), text, request, "stop", split)
        assert result[0] == "thought"
        assert not result[1]
        assert any(tc.function and tc.function.name == "stub" for tc in result[2])
        full = _hotfix_parser(path).parse_with_finish_reason(
            text, request, enable_auto_tools=True, finish_reason="stop"
        )
        assert full[0] == "thought"
        assert not full[1]
        assert full[2][0].name == "stub"
