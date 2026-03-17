from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any, Literal, cast

import openai
from agents import models
from agents.agent_output import AgentOutputSchemaBase
from agents.handoffs import Handoff
from agents.items import TResponseInputItem
from agents.model_settings import ModelSettings
from agents.models.chatcmpl_converter import Converter
from agents.models.chatcmpl_helpers import ChatCmplHelpers
from agents.models.fake_id import FAKE_RESPONSES_ID
from agents.models.interface import Model, ModelTracing
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from agents.models.openai_responses import Converter as OpenAIResponsesConverter
from agents.tool import Tool
from agents.tracing.span_data import GenerationSpanData
from agents.tracing.spans import Span
from agents.util._json import _to_dump_compatible
from hiddenlayer import AsyncHiddenLayer
from openai import AsyncOpenAI, AsyncStream, Omit, omit
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.responses import Response
from openai.types.responses.response_prompt_param import ResponsePromptParam

from hiddenlayer_openai_guardrails import InputBlockedError, OutputBlockedError

from ._types import HiddenLayerParams
from ._v2_client import REQUEST_EVALUATION_PATH, RESPONSE_EVALUATION_PATH, evaluate

logger = logging.getLogger(__name__)


class _StreamingResponseEvaluator:
    """Wraps an AsyncStream, collecting text content and firing a fire-and-forget HL response evaluation when the stream is exhausted."""

    def __init__(
        self,
        stream: AsyncStream[ChatCompletionChunk],
        *,
        client: AsyncHiddenLayer,
        params: HiddenLayerParams,
        roundtrip_id: str,
    ) -> None:
        self._stream = stream
        self._client = client
        self._params = params
        self._roundtrip_id = roundtrip_id
        self._content_parts: list[str] = []
        self._model: str | None = None
        self._finish_reason: str | None = None

    def __aiter__(self) -> AsyncIterator[ChatCompletionChunk]:
        return self

    async def __anext__(self) -> ChatCompletionChunk:
        try:
            chunk = await self._stream.__anext__()
            if self._model is None:
                self._model = chunk.model
            for choice in chunk.choices:
                if choice.delta.content:
                    self._content_parts.append(choice.delta.content)
                if choice.finish_reason:
                    self._finish_reason = choice.finish_reason
            return chunk
        except StopAsyncIteration:
            await self._fire_evaluation()
            raise

    async def _fire_evaluation(self) -> None:
        content = "".join(self._content_parts)
        payload = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "model": self._model or "unknown",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": self._finish_reason or "stop",
                }
            ],
        }
        if content:
            await evaluate(
                self._client,
                RESPONSE_EVALUATION_PATH,
                payload,
                self._params,
                roundtrip_id=self._roundtrip_id,
            )


def _omit_none_and_sentinel(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None and value is not omit}


def _default_requester_id() -> str:
    return os.getenv("HIDDENLAYER_REQUESTER_ID", "hiddenlayer-openai-integration")


def _extract_model_and_client(model_value: str | Model | None) -> tuple[str, AsyncOpenAI]:
    default_model = models.get_default_model()
    default_model_name = getattr(default_model, "model", None) or str(default_model)

    if isinstance(model_value, OpenAIChatCompletionsModel):
        return str(model_value.model), model_value._get_client()

    if model_value is None:
        return default_model_name, AsyncOpenAI()

    if isinstance(model_value, str):
        return model_value, AsyncOpenAI()

    maybe_model_name = getattr(model_value, "model", None)
    maybe_client = getattr(model_value, "_client", None)

    if maybe_model_name is not None:
        client = maybe_client if isinstance(maybe_client, AsyncOpenAI) else AsyncOpenAI()
        return str(maybe_model_name), client

    return str(model_value), AsyncOpenAI()


class HiddenLayerProtectedModel(OpenAIChatCompletionsModel):
    """Wrap OpenAI chat completions with inline HiddenLayer v2 scans."""

    def __init__(
        self,
        model: str,
        openai_client: AsyncOpenAI,
        *,
        hiddenlayer_params: HiddenLayerParams,
        hiddenlayer_client: AsyncHiddenLayer | None,
    ) -> None:
        super().__init__(model=model, openai_client=openai_client)
        self._hiddenlayer_params = hiddenlayer_params
        self._hiddenlayer_client = hiddenlayer_client or AsyncHiddenLayer()

    def _build_request_payload(
        self,
        *,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        stream: bool,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        if hasattr(self, "_validate_official_openai_input_content_types"):
            self._validate_official_openai_input_content_types(input)  # not in older versions

        converted_messages = Converter.items_to_messages(input, model=self.model)
        if system_instructions:
            converted_messages.insert(
                0,
                {
                    "content": system_instructions,
                    "role": "system",
                },
            )
        converted_messages = _to_dump_compatible(converted_messages)

        if model_settings.parallel_tool_calls and tools:
            parallel_tool_calls: bool | Omit = True
        elif model_settings.parallel_tool_calls is False:
            parallel_tool_calls = False
        else:
            parallel_tool_calls = omit

        tool_choice = Converter.convert_tool_choice(model_settings.tool_choice)
        response_format = Converter.convert_response_format(output_schema)
        converted_tools = [Converter.tool_to_openai(tool) for tool in tools] if tools else []
        for handoff in handoffs:
            converted_tools.append(Converter.convert_handoff_tool(handoff))
        converted_tools = _to_dump_compatible(converted_tools)

        reasoning_effort = model_settings.reasoning.effort if model_settings.reasoning else None
        store = ChatCmplHelpers.get_store_param(self._get_client(), model_settings)
        stream_options = ChatCmplHelpers.get_stream_options_param(self._get_client(), model_settings, stream=stream)
        stream_param: Literal[True] | Omit = True if stream else omit

        request_payload = _omit_none_and_sentinel(
            {
                "model": self.model,
                "messages": converted_messages,
                "tools": converted_tools if converted_tools else omit,
                "temperature": self._non_null_or_omit(model_settings.temperature),
                "top_p": self._non_null_or_omit(model_settings.top_p),
                "frequency_penalty": self._non_null_or_omit(model_settings.frequency_penalty),
                "presence_penalty": self._non_null_or_omit(model_settings.presence_penalty),
                "max_tokens": self._non_null_or_omit(model_settings.max_tokens),
                "tool_choice": tool_choice,
                "response_format": response_format,
                "parallel_tool_calls": parallel_tool_calls,
                "stream": cast(Any, stream_param),
                "stream_options": self._non_null_or_omit(stream_options),
                "store": self._non_null_or_omit(store),
                "reasoning_effort": self._non_null_or_omit(reasoning_effort),
                "verbosity": self._non_null_or_omit(model_settings.verbosity),
                "top_logprobs": self._non_null_or_omit(model_settings.top_logprobs),
                "prompt_cache_retention": self._non_null_or_omit(model_settings.prompt_cache_retention),
                "metadata": self._non_null_or_omit(model_settings.metadata),
            }
        )

        extra_args = _to_dump_compatible(model_settings.extra_args or {})
        extra_body = _to_dump_compatible(model_settings.extra_body or {})
        request_payload.update(extra_args)
        if isinstance(extra_body, dict):
            request_payload.update(extra_body)

        request_options = {
            "extra_headers": self._merge_headers(model_settings),
            "extra_query": model_settings.extra_query,
        }

        debug_options = {
            "messages": converted_messages,
            "tools": converted_tools,
            "tool_choice": tool_choice,
            "response_format": response_format,
        }

        return request_payload, request_options, debug_options

    @staticmethod
    def _chat_completion_from_payload(payload: dict[str, Any]) -> ChatCompletion:
        return ChatCompletion(**payload)

    async def _fetch_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        span: Span[GenerationSpanData],
        tracing: ModelTracing,
        stream: bool = False,
        prompt: ResponsePromptParam | None = None,
    ) -> ChatCompletion | tuple[Response, AsyncStream[ChatCompletionChunk]]:
        del prompt

        request_payload, request_options, debug_options = self._build_request_payload(
            system_instructions=system_instructions,
            input=input,
            model_settings=model_settings,
            tools=tools,
            output_schema=output_schema,
            handoffs=handoffs,
            stream=stream,
        )

        if tracing.include_data():
            span.span_data.input = debug_options["messages"]

        logger.debug(
            "HiddenLayer v2 request payload prepared. tools=%s stream=%s tool_choice=%s response_format=%s",
            json.dumps(debug_options["tools"], ensure_ascii=False),
            stream,
            debug_options["tool_choice"],
            json.dumps(debug_options["response_format"], ensure_ascii=False)
            if debug_options["response_format"] is not omit
            else None,
        )

        roundtrip_id = str(uuid.uuid4())
        request_eval = await evaluate(
            self._hiddenlayer_client,
            REQUEST_EVALUATION_PATH,
            request_payload,
            self._hiddenlayer_params,
            roundtrip_id=roundtrip_id,
        )

        if request_eval.blocked:
            raise InputBlockedError("Blocked by HiddenLayer due to model input policy violation.")
            # return self._chat_completion_from_payload(request_eval.payload)

        effective_request_payload = request_eval.payload or request_payload

        ret = await self._get_client().chat.completions.create(
            **effective_request_payload,
            extra_headers=request_options["extra_headers"],
            extra_query=request_options["extra_query"],
        )

        if isinstance(ret, openai.AsyncStream):
            ret = _StreamingResponseEvaluator(
                ret,
                client=self._hiddenlayer_client,
                params=self._hiddenlayer_params,
                roundtrip_id=roundtrip_id,
            )

        if isinstance(ret, ChatCompletion):
            response_eval = await evaluate(
                self._hiddenlayer_client,
                RESPONSE_EVALUATION_PATH,
                ret.model_dump(),
                self._hiddenlayer_params,
                roundtrip_id=roundtrip_id,
            )
            if response_eval.blocked:
                raise OutputBlockedError("Blocked by HiddenLayer due to model output policy violation.")

            if response_eval.payload:
                return self._chat_completion_from_payload(response_eval.payload)

            return ret

        responses_tool_choice = OpenAIResponsesConverter.convert_tool_choice(model_settings.tool_choice)
        if responses_tool_choice is None or responses_tool_choice is omit:
            responses_tool_choice = "auto"

        response = Response(
            id=FAKE_RESPONSES_ID,
            created_at=time.time(),
            model=self.model,
            object="response",
            output=[],
            tool_choice=responses_tool_choice,  # type: ignore[arg-type]
            top_p=model_settings.top_p,
            temperature=model_settings.temperature,
            tools=[],
            parallel_tool_calls=bool(effective_request_payload.get("parallel_tool_calls", False)),
            reasoning=model_settings.reasoning,
        )
        return response, ret  # type: ignore[return-value]
