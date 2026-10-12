# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4 parser: ``<think>``/``</think>``
reasoning plus DSML tool calls in a single state machine.

DeepSeek V4 output format::

    <think>
    ...reasoning...
    </think>
    <｜DSML｜tool_calls>
    <｜DSML｜invoke name="func_name">
    <｜DSML｜parameter name="location" string="true">杭州</｜DSML｜parameter>
    <｜DSML｜parameter name="count" string="false">5</｜DSML｜parameter>
    </｜DSML｜invoke>
    </｜DSML｜tool_calls>

The model is prompted with an opened ``<think>`` (via custom Python prompt
logic, not the chat template -- see ``vllm/tokenizers/deepseek_v4_encoding.py``),
so ``thinking=True`` starts the parser in ``REASONING``. The model does not
always close it: short replies sometimes skip deliberation and answer
directly, so generation can end with no ``</think>`` ever seen. Without a
fallback, that entire reply is classified as reasoning and ``content`` stays
empty (vLLM issue #48645).

For DeepSeek-V4 this compatibility fallback is enabled by default, because
existing OpenAI-compatible clients do not send a parser-specific request
flag. Request settings override server template defaults; only boolean
``True`` enables an explicitly supplied flag. An unterminated reasoning
block is copied to content only after a natural stop, with no observed
reasoning boundary, content, or tool markup. Reasoning is retained.
"""

from __future__ import annotations

import contextlib
import functools
import json
from collections.abc import Sequence
from typing import TYPE_CHECKING

import regex as re

from vllm.parser.engine.events import EventType
from vllm.parser.engine.parser_engine import ParserEngine
from vllm.parser.engine.parser_engine_config import (
    ParserEngineConfig,
    ParserState,
    Transition,
)
from vllm.tool_parsers.utils import find_tool_properties

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )
    from vllm.entrypoints.openai.engine.protocol import DeltaMessage
    from vllm.entrypoints.openai.responses.protocol import (
        FunctionCall,
        ResponsesRequest,
    )
    from vllm.parser.engine.events import SemanticEvent
    from vllm.tokenizers import TokenizerLike
    from vllm.tool_parsers.abstract_tool_parser import Tool

_DSML = "｜DSML｜"

DSML_THINK_START = "<think>"
DSML_THINK_END = "</think>"
DSML_TOOL_START = f"<{_DSML}tool_calls>"
DSML_TOOL_END = f"</{_DSML}tool_calls>"
DSML_INVOKE_PREFIX = f'<{_DSML}invoke name="'
DSML_INVOKE_NAME_END = '">'
DSML_INVOKE_END = f"</{_DSML}invoke>"
DSML_PARAM_CLOSE = f"</{_DSML}parameter>"
_DSML_INCOMPLETE_PREFIXES = tuple(
    marker[:length]
    for marker in (f"<{_DSML}", f"</{_DSML}")
    for length in range(3 if marker.startswith("</") else 2, len(marker) + 1)
)

_ESCAPED_DSML = re.escape(_DSML)
_PARAM_RE = re.compile(
    rf'<{_ESCAPED_DSML}parameter\s+name="([^"]+)"\s+string="(true|false)">'
    rf"(.*?)</{_ESCAPED_DSML}parameter>",
    re.DOTALL,
)
_PARTIAL_PARAM_RE = re.compile(
    rf'<{_ESCAPED_DSML}parameter\s+name="([^"]+)"\s+string="(true|false)">'
    rf"(.*)$",
    re.DOTALL,
)


def _dsml_arg_converter(raw_args: str, partial: bool) -> str:
    params: dict[str, object] = {}

    last_end = 0
    for m in _PARAM_RE.finditer(raw_args):
        name, is_str, value = m.group(1), m.group(2), m.group(3)
        if is_str == "true":
            params[name] = value
        else:
            try:
                params[name] = json.loads(value)
            except (json.JSONDecodeError, ValueError):
                params[name] = value
        last_end = m.end()

    if partial:
        pm = _PARTIAL_PARAM_RE.search(raw_args, last_end)
        if pm:
            name, is_str, value = pm.group(1), pm.group(2), pm.group(3)
            if is_str == "true":
                params[name] = value
            else:
                with contextlib.suppress(json.JSONDecodeError, ValueError):
                    params[name] = json.loads(value)

    return json.dumps(params, ensure_ascii=False)


def _unwrap_wrapper_args(
    args_json: str,
    tools: list[Tool] | None,
    func_name: str | None,
) -> str:
    if not tools or not func_name:
        return args_json
    try:
        args = json.loads(args_json)
    except (json.JSONDecodeError, ValueError):
        return args_json
    if not isinstance(args, dict):
        return args_json
    properties = find_tool_properties(tools, func_name)
    if not properties:
        return args_json
    allowed = set(properties.keys())
    for wrapper in ("arguments", "input"):
        if set(args.keys()) != {wrapper} or wrapper in allowed:
            continue
        inner = args[wrapper]
        if isinstance(inner, str):
            try:
                inner = json.loads(inner)
            except json.JSONDecodeError:
                return args_json
        if isinstance(inner, dict) and set(inner.keys()).issubset(allowed):
            return json.dumps(inner, ensure_ascii=False)
    return args_json


@functools.cache
def deepseek_v4_config(thinking: bool = False) -> ParserEngineConfig:
    return ParserEngineConfig(
        name="deepseek_v4",
        initial_state=ParserState.REASONING if thinking else ParserState.CONTENT,
        terminals={
            "THINK_START": DSML_THINK_START,
            "THINK_END": DSML_THINK_END,
            "TOOL_START": DSML_TOOL_START,
            "TOOL_END": DSML_TOOL_END,
            "INVOKE_PREFIX": DSML_INVOKE_PREFIX,
            "INVOKE_NAME_END": DSML_INVOKE_NAME_END,
            "INVOKE_END": DSML_INVOKE_END,
            "PARAM_CLOSE": DSML_PARAM_CLOSE,
        },
        token_id_terminals={
            "THINK_START": DSML_THINK_START,
            "THINK_END": DSML_THINK_END,
            "TOOL_START": DSML_TOOL_START,
            "TOOL_END": DSML_TOOL_END,
        },
        transitions={
            (ParserState.CONTENT, "THINK_START"): Transition(
                ParserState.REASONING,
                (EventType.REASONING_START,),
            ),
            # Absorb a bare </think> with no prior <think>
            (ParserState.CONTENT, "THINK_END"): Transition(
                ParserState.CONTENT,
                (),
            ),
            # Absorb a duplicate <think> while already reasoning
            (ParserState.REASONING, "THINK_START"): Transition(
                ParserState.REASONING,
                (),
            ),
            (ParserState.REASONING, "THINK_END"): Transition(
                ParserState.CONTENT,
                (EventType.REASONING_END,),
            ),
            # Tool call beginning while still inside <think>
            (ParserState.REASONING, "TOOL_START"): Transition(
                ParserState.TOOL_PREAMBLE,
                (EventType.REASONING_END,),
            ),
            (ParserState.CONTENT, "TOOL_START"): Transition(
                ParserState.TOOL_PREAMBLE,
                (),
            ),
            (ParserState.TOOL_PREAMBLE, "INVOKE_PREFIX"): Transition(
                ParserState.TOOL_NAME,
                (EventType.TOOL_CALL_START,),
            ),
            (ParserState.TOOL_NAME, "INVOKE_NAME_END"): Transition(
                ParserState.TOOL_ARGS,
                (),
            ),
            (ParserState.TOOL_ARGS, "INVOKE_END"): Transition(
                ParserState.TOOL_BETWEEN,
                (EventType.TOOL_CALL_END,),
            ),
            (ParserState.TOOL_ARGS, "TOOL_END"): Transition(
                ParserState.CONTENT,
                (EventType.TOOL_CALL_END,),
            ),
            # Parallel tool calls
            (ParserState.TOOL_BETWEEN, "INVOKE_PREFIX"): Transition(
                ParserState.TOOL_NAME,
                (EventType.TOOL_CALL_START,),
            ),
            (ParserState.TOOL_BETWEEN, "TOOL_END"): Transition(
                ParserState.CONTENT,
                (),
            ),
        },
        content_events={
            ParserState.CONTENT: EventType.TEXT_CHUNK,
            ParserState.REASONING: EventType.REASONING_CHUNK,
            ParserState.TOOL_NAME: EventType.TOOL_NAME,
            ParserState.TOOL_ARGS: EventType.ARG_VALUE_CHUNK,
        },
        arg_converter=_dsml_arg_converter,
        arg_structural_chars=frozenset(">"),
        strip_content_whitespace_with_tools=False,
        tool_args_json=False,
    )


class DeepSeekV4Parser(ParserEngine):
    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> None:
        chat_kwargs = kwargs.pop("chat_template_kwargs", None) or {}
        self._force_nonempty_content = (
            chat_kwargs.get("force_nonempty_content", True) is True
        )
        thinking = bool(
            chat_kwargs.get("thinking") or chat_kwargs.get("enable_thinking")
        )
        if "thinking" not in chat_kwargs and "enable_thinking" not in chat_kwargs:
            thinking = True
        thinking = thinking and chat_kwargs.get("reasoning_effort") != "none"
        super().__init__(
            tokenizer,
            tools,
            parser_engine_config=deepseek_v4_config(thinking=thinking),
            **kwargs,
        )
        self._arg_converter = self._convert_args
        self._streamed_reasoning: list[str] = []
        self._fallback_reasoning_ended = False
        self._fallback_has_content = False
        self._fallback_has_tools = False
        self._fallback_has_dsml = False
        self._fallback_text_tail = ""

    def _convert_args(self, raw_args: str, partial: bool) -> str:
        result = _dsml_arg_converter(raw_args, partial)
        if not self._tools:
            return result
        func_name = next((s.name for s in self._tool_slots if s.args == raw_args), None)
        return _unwrap_wrapper_args(result, self._tools, func_name)

    def _reset(self, initial_state: ParserState | None = None) -> None:
        super()._reset(initial_state=initial_state)
        self._streamed_reasoning = []
        self._fallback_reasoning_ended = False
        self._fallback_has_content = False
        self._fallback_has_tools = False
        self._fallback_has_dsml = False
        self._fallback_text_tail = ""

    def _record_fallback_events(self, events: list[SemanticEvent]) -> None:
        for event in events:
            # Real transitions carry the matched marker. finish() synthesizes
            # a REASONING_END with an empty value for an unfinished block.
            if event.type == EventType.REASONING_END and event.value:
                self._fallback_reasoning_ended = True
            elif event.type == EventType.TEXT_CHUNK and event.value.strip():
                self._fallback_has_content = True
            elif event.type == EventType.TOOL_CALL_START:
                self._fallback_has_tools = True

    def _feed(
        self, delta_text: str, delta_token_ids: Sequence[int]
    ) -> list[SemanticEvent]:
        combined = self._fallback_text_tail + delta_text
        self._fallback_has_dsml |= "｜DSML" in combined
        self._fallback_text_tail = combined[-len(DSML_TOOL_START) :]
        events = super()._feed(delta_text, delta_token_ids)
        self._record_fallback_events(events)
        return events

    def _events_to_delta(
        self,
        events: list[SemanticEvent],
        finished: bool = False,
    ) -> DeltaMessage | None:
        self._record_fallback_events(events)
        delta = super()._events_to_delta(events, finished=finished)
        if delta is not None and delta.reasoning is not None:
            self._streamed_reasoning.append(delta.reasoning)
        return delta

    def _should_force_content(
        self,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> bool:
        chat_template_kwargs = getattr(request, "chat_template_kwargs", None)
        if chat_template_kwargs and "force_nonempty_content" in chat_template_kwargs:
            return chat_template_kwargs["force_nonempty_content"] is True
        return self._force_nonempty_content

    def _can_promote(
        self,
        request: ChatCompletionRequest | ResponsesRequest,
        finish_reason: str | None,
    ) -> bool:
        return (
            self._should_force_content(request)
            and finish_reason == "stop"
            and not self._fallback_reasoning_ended
            and not self._fallback_has_content
            and not self._fallback_has_tools
            and not self._tool_slots
            and not self._fallback_has_dsml
            and not self._fallback_text_tail.endswith(_DSML_INCOMPLETE_PREFIXES)
        )

    def prepare_streaming_fallback(self) -> DeltaMessage | None:
        """Drain pending text before the adapter evaluates terminal fallback."""
        return self._strip_trailing_reasoning(self.finish_streaming())

    def get_streaming_fallback_content(
        self,
        text: str,
        request: ChatCompletionRequest | ResponsesRequest,
        finish_reason: str | None = None,
    ) -> str | None:
        """Copy eligible reasoning after pending lexer text has been drained."""
        if not self._can_promote(request, finish_reason):
            return None
        content = "".join(self._streamed_reasoning)
        if self._strip_trailing_reasoning_ws:
            content = content.rstrip()
        return content if content.strip() else None

    def extract_reasoning(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> tuple[str | None, str | None]:
        """Legacy callers have no terminal metadata, so do not promote."""
        return self.extract_reasoning_with_finish_reason(model_output, request)

    def extract_reasoning_with_finish_reason(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
        *,
        finish_reason: str | None = None,
    ) -> tuple[str | None, str | None]:
        self._reset()
        events = self._feed(model_output, [])
        events.extend(self._engine.finish())
        self._record_fallback_events(events)

        reasoning_parts: list[str] = []
        content_parts: list[str] = []
        for event in events:
            if event.type == EventType.REASONING_CHUNK:
                reasoning_parts.append(event.value)
            elif event.type == EventType.TEXT_CHUNK:
                content_parts.append(event.value)
            elif event.type == EventType.REASONING_END:
                self._reasoning_ended = True

        raw_reasoning = "".join(reasoning_parts)
        if self._strip_trailing_reasoning_ws:
            raw_reasoning = raw_reasoning.rstrip()
        reasoning = raw_reasoning or None
        content = "".join(content_parts) or None

        if (
            self._can_promote(request, finish_reason)
            and reasoning
            and reasoning.strip()
        ):
            content = reasoning
        return reasoning, content

    def parse_with_finish_reason(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
        enable_auto_tools: bool = False,
        model_output_token_ids: Sequence[int] = (),
        *,
        finish_reason: str | None = None,
    ) -> tuple[str | None, str | None, list[FunctionCall] | None]:
        reasoning, content, tool_calls = super().parse(
            model_output, request, enable_auto_tools, model_output_token_ids
        )
        if (
            self._can_promote(request, finish_reason)
            and reasoning
            and reasoning.strip()
        ):
            content = reasoning
        return reasoning, content, tool_calls
