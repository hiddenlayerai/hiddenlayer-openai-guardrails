# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Drop-in replacement for the OpenAI Agents SDK `Agent` that automatically routes every chat-completion roundtrip through HiddenLayer's v2 detection endpoints to catch prompt-injection and policy violations. Unlike the Agents SDK's native guardrail decorator hooks, this library plugs into the Agents SDK by wrapping the underlying chat-completions `Model`, so a single interception point sees the full request/response payload (messages, tools, tool calls, tool results, streamed content).

## Commands

```bash
# Install dependencies (uses uv)
uv sync

# Run all tests
pytest tests

# Unit tests only (mocked, no network)
pytest tests -m unit

# Live integration tests (require credentials + OPENAI_API_KEY)
RUN_LIVE_INTEGRATION_TESTS=1 pytest tests -m integration

# Run a single test
pytest tests/test_v2_model.py::test_top_level_v2_agent_reuses_model_wrapper -v

# Lint / import-order check (ruff: E, F, I selected; line length 120)
uv run ruff check .
uv run ruff format .
```

Note: `pytest.ini_options` sets `asyncio_mode = "auto"`, so async tests do not need an explicit `@pytest.mark.asyncio` marker (though many tests include one anyway).

## Architecture

The public surface is intentionally tiny: `Agent`, `HiddenLayerParams`, `InputBlockedError`, `OutputBlockedError`, `V2EvaluationResult` (all re-exported from `__init__.py`). Implementation lives in four modules:

- `agents.py` (`Agent`): factory class whose `__new__` returns a stock `agents.Agent`. Its only job is to resolve the user's `model` argument (string, `Model` instance, or pre-wrapped `HiddenLayerProtectedModel`) into a `HiddenLayerProtectedModel` and pass it through. The Agents SDK then drives that model normally.
- `_v2_model.py` (`HiddenLayerProtectedModel`): subclass of `agents.models.openai_chatcompletions.OpenAIChatCompletionsModel`. The whole guardrail integration is implemented here by overriding `_fetch_response`. Also contains `_extract_model_and_client`, which understands the several shapes `agents` accepts for `model` (str, `OpenAIChatCompletionsModel`, anything with `.model`/`._client`, or `None`), and `_StreamingResponseEvaluator`, which proxies `AsyncStream[ChatCompletionChunk]` and runs a fire-and-forget response evaluation when the stream finishes.
- `_v2_client.py`: thin async wrapper that POSTs to `/detection/v2/request-evaluations` and `/detection/v2/response-evaluations` via an `AsyncHiddenLayer` client, attaches `Hl-Roundtrip-Id` / `HL-Project-Id` / `hl-requester-id` / `hl-runtime-session-id` headers, and returns a `V2EvaluationResult` with `blocked` derived from the `hl-runtime-action` response header (`BLOCK` => blocked). The endpoint may also return a modified `payload` that callers are expected to use instead of the original.
- `_types.py`: `HiddenLayerParams` (Pydantic; reads `HIDDENLAYER_PROJECT_ID` / `HIDDENLAYER_REQUESTER_ID` from env via `Field(default_factory=...)`) and the two block exceptions. Fields: `model`, `project_id`, `requester_id`, `session_id` (per-conversation correlation ID; forwarded as `hl-runtime-session-id`), `exclude_tools_from_evaluation` (when True, `tools`/`tool_choice` are stripped from the HL eval payload but preserved on the downstream OpenAI call).

### Request/response flow inside `HiddenLayerProtectedModel._fetch_response`

1. `_build_request_payload` runs the same conversion `OpenAIChatCompletionsModel` would (system instructions + `Converter.items_to_messages` + tool conversion + handoffs + sentinel-aware option assembly) to produce the chat-completions payload that *would* have been sent to OpenAI.
2. If `params.exclude_tools_from_evaluation` is set and the payload has `tools`, a copy with `tools` and `tool_choice` stripped is built as the HL-bound `eval_payload`; otherwise `eval_payload is request_payload`.
3. A fresh `roundtrip_id` (UUID) is generated and `eval_payload` is POSTed to `REQUEST_EVALUATION_PATH` with `session_id=params.session_id`. If `blocked`, `InputBlockedError` is raised. Otherwise `request_eval.payload or eval_payload` becomes the effective payload (HL may rewrite messages, e.g. for redaction).
4. If `exclude_tools_from_evaluation` is set, `tools` and `tool_choice` from the original `request_payload` are re-attached to the effective payload before it goes to OpenAI, so any HL-side modifications to `messages` are preserved while tool definitions remain untouched.
5. The effective payload is sent to OpenAI via `self._get_client().chat.completions.create(...)`.
6. **Non-streaming**: the returned `ChatCompletion` is POSTed to `RESPONSE_EVALUATION_PATH` (also with `session_id`). `blocked` => `OutputBlockedError`; otherwise an HL-returned payload (if present) is rehydrated via `ChatCompletion(**payload)` and returned instead of the original.
7. **Streaming**: the `AsyncStream` is wrapped in `_StreamingResponseEvaluator` (constructed with `session_id`), which accumulates `delta.content` and posts the assembled completion to `RESPONSE_EVALUATION_PATH` after the stream exhausts. Because streaming results have already been delivered to the caller by then, the streaming response eval is effectively fire-and-forget logging, **not** blocking. A synthetic `Response` is also returned alongside the wrapped stream to satisfy the Agents SDK's `_fetch_response` return contract.

### Things to be careful about when editing

- `Agent.__new__` returns `agents.Agent`, not `Agent` (`-> Any` lies on purpose). Do not change it to return `Agent` or to instantiate via `super().__new__`; downstream `isinstance(agent, agents.Agent)` checks rely on this.
- If `model` is already a `HiddenLayerProtectedModel`, reuse it verbatim. Don't re-wrap (this would double-evaluate and double-charge).
- `HiddenLayerProtectedModel._build_request_payload` mirrors the upstream Agents SDK's payload assembly. When the Agents SDK adds/renames a `ModelSettings` field, this method must follow. The `_validate_official_openai_input_content_types` hasattr guard exists because that method only exists on newer SDK versions.
- `_to_dump_compatible` / `_omit_none_and_sentinel` / the `omit` sentinel are imported from `openai`/`agents` internals. The Agents SDK uses an `Omit` sentinel rather than `None` to distinguish "explicitly unset" from "use server default"; preserve that distinction when adding new fields.
- The `_StreamingResponseEvaluator._fire_evaluation` path POSTs to the response-evaluation endpoint after the stream is exhausted but **discards the result**, so a `BLOCK` action on streamed output cannot actually halt anything. This is a known bug, tracked separately from the request-side block path which does enforce `OutputBlockedError`.
- When adding a new `HiddenLayerParams` field that needs to be forwarded to HL, you must thread it through **all four** `evaluate()` call sites in `_v2_model.py`: request eval, non-streaming response eval, the `_StreamingResponseEvaluator` constructor, and `_StreamingResponseEvaluator._fire_evaluation`. Forgetting one is the most likely class of bug in this file.

## Environment Variables

- `HIDDENLAYER_CLIENT_ID`, `HIDDENLAYER_CLIENT_SECRET` (required) - consumed by `AsyncHiddenLayer()` from the `hiddenlayer-sdk` package.
- `HIDDENLAYER_PROJECT_ID` (optional) - default value for `HiddenLayerParams.project_id`; emitted as the `HL-Project-Id` header for policy routing.
- `HIDDENLAYER_REQUESTER_ID` (optional, defaults to `"hiddenlayer-openai-integration"`) - emitted as the `hl-requester-id` header.
- `OPENAI_API_KEY` - required for any test that touches the OpenAI client; unit tests set it via `monkeypatch.setenv`.
- `RUN_LIVE_INTEGRATION_TESTS=1` - gates the live integration test path.
