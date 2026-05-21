from unittest.mock import AsyncMock, MagicMock

import pytest

from hiddenlayer_openai_guardrails import Agent as TopLevelAgent, HiddenLayerParams
from hiddenlayer_openai_guardrails._v2_client import V2EvaluationResult
from hiddenlayer_openai_guardrails._v2_model import (
    HiddenLayerProtectedModel,
    _extract_model_and_client,
    _StreamingResponseEvaluator,
)


@pytest.mark.unit
def test_top_level_v2_agent_reuses_model_wrapper(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")

    agent = TopLevelAgent(
        name="Assistant",
        model="gpt-4o-mini",
        hiddenlayer_params=HiddenLayerParams(
            model="gpt-4o-mini",
            project_id="project-123",
        ),
    )

    assert isinstance(agent.model, HiddenLayerProtectedModel)
    assert not agent.input_guardrails
    assert not agent.output_guardrails


@pytest.mark.unit
def test_agent_sets_model_name_from_string(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")

    agent = TopLevelAgent(
        name="Assistant",
        model="gpt-4o",
        hiddenlayer_params=HiddenLayerParams(project_id="proj"),
    )

    assert isinstance(agent.model, HiddenLayerProtectedModel)
    assert agent.model.model == "gpt-4o"


@pytest.mark.unit
def test_agent_infers_model_name_into_params_when_not_set(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")

    params = HiddenLayerParams(project_id="proj")
    agent = TopLevelAgent(name="Assistant", model="gpt-4o-mini", hiddenlayer_params=params)

    assert agent.model._hiddenlayer_params.model == "gpt-4o-mini"


@pytest.mark.unit
def test_agent_reuses_existing_protected_model(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")

    from openai import AsyncOpenAI

    existing_model = HiddenLayerProtectedModel(
        model="gpt-4o",
        openai_client=AsyncOpenAI(api_key="test"),
        hiddenlayer_params=HiddenLayerParams(project_id="proj"),
        hiddenlayer_client=None,
    )

    agent = TopLevelAgent(name="Assistant", model=existing_model)

    assert agent.model is existing_model


@pytest.mark.unit
def test_extract_model_and_client_with_string(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")

    model_name, client = _extract_model_and_client("gpt-4o")

    assert model_name == "gpt-4o"


@pytest.mark.unit
def test_extract_model_and_client_with_none(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")

    model_name, client = _extract_model_and_client(None)

    assert isinstance(model_name, str)
    assert model_name != ""


@pytest.mark.unit
def test_extract_model_and_client_with_openai_chat_completions_model(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")

    from openai import AsyncOpenAI
    from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel

    openai_client = AsyncOpenAI(api_key="test")
    model = OpenAIChatCompletionsModel(model="gpt-4", openai_client=openai_client)

    model_name, client = _extract_model_and_client(model)

    assert model_name == "gpt-4"
    assert client is openai_client


# ---------------------------------------------------------------------------
# Helpers for driving _fetch_response in unit tests
# ---------------------------------------------------------------------------


def _build_fake_completion(model: str = "gpt-4o-mini", content: str = "ok"):
    from openai.types.chat import ChatCompletion
    from openai.types.chat.chat_completion import Choice
    from openai.types.chat.chat_completion_message import ChatCompletionMessage

    return ChatCompletion(
        id="chatcmpl-test",
        object="chat.completion",
        created=0,
        model=model,
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(role="assistant", content=content),
                finish_reason="stop",
            )
        ],
    )


def _build_protected_model(
    *,
    hiddenlayer_params: HiddenLayerParams,
    completion_response=None,
):
    """Construct a HiddenLayerProtectedModel with a stubbed OpenAI client.

    The returned tuple is `(model, openai_create_mock)` so tests can assert on
    what was sent to OpenAI.
    """
    from openai import AsyncOpenAI

    openai_client = AsyncOpenAI(api_key="test")
    create_mock = AsyncMock(return_value=completion_response or _build_fake_completion())
    openai_client.chat.completions.create = create_mock

    model = HiddenLayerProtectedModel(
        model=hiddenlayer_params.model or "gpt-4o-mini",
        openai_client=openai_client,
        hiddenlayer_params=hiddenlayer_params,
        hiddenlayer_client=MagicMock(),
    )
    return model, create_mock


async def _run_fetch_response(model: HiddenLayerProtectedModel, *, tools=None, stream: bool = False):
    from agents.model_settings import ModelSettings
    from agents.models.interface import ModelTracing

    return await model._fetch_response(
        system_instructions=None,
        input="hello",
        model_settings=ModelSettings(),
        tools=tools or [],
        output_schema=None,
        handoffs=[],
        span=MagicMock(),
        tracing=ModelTracing.DISABLED,
        stream=stream,
    )


def _patch_evaluate(monkeypatch):
    """Replace _v2_model.evaluate with a capturing stub. Returns the captured list."""
    captured = []

    async def fake_evaluate(client, path, payload, params, *, roundtrip_id=None, session_id=None):
        captured.append(
            {
                "path": path,
                "payload": payload,
                "session_id": session_id,
                "roundtrip_id": roundtrip_id,
            }
        )
        return V2EvaluationResult(action=None, payload={}, blocked=False, roundtrip_id=roundtrip_id or "")

    monkeypatch.setattr("hiddenlayer_openai_guardrails._v2_model.evaluate", fake_evaluate)
    return captured


# ---------------------------------------------------------------------------
# session_id forwarding
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.unit
async def test_session_id_forwarded_to_request_and_response_evaluations(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    captured = _patch_evaluate(monkeypatch)

    model, _ = _build_protected_model(
        hiddenlayer_params=HiddenLayerParams(model="gpt-4o-mini", session_id="conv-abc"),
    )

    await _run_fetch_response(model)

    assert len(captured) == 2
    assert [c["path"] for c in captured] == [
        "/detection/v2/request-evaluations",
        "/detection/v2/response-evaluations",
    ]
    assert all(c["session_id"] == "conv-abc" for c in captured)


@pytest.mark.asyncio
@pytest.mark.unit
async def test_session_id_absent_means_none_passed_to_evaluate(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    captured = _patch_evaluate(monkeypatch)

    model, _ = _build_protected_model(
        hiddenlayer_params=HiddenLayerParams(model="gpt-4o-mini"),
    )

    await _run_fetch_response(model)

    assert all(c["session_id"] is None for c in captured)


@pytest.mark.asyncio
@pytest.mark.unit
async def test_streaming_response_evaluator_forwards_session_id():
    captured = []

    async def fake_evaluate(client, path, payload, params, *, roundtrip_id=None, session_id=None):
        captured.append({"path": path, "session_id": session_id, "roundtrip_id": roundtrip_id})
        return V2EvaluationResult(action=None, payload={}, blocked=False, roundtrip_id=roundtrip_id or "")

    # Build a stream that yields a single chunk with text content, then ends.
    fake_chunk = MagicMock()
    fake_chunk.model = "gpt-4o-mini"
    choice = MagicMock()
    choice.delta.content = "hello"
    choice.finish_reason = "stop"
    fake_chunk.choices = [choice]

    class _FakeStream:
        def __init__(self):
            self._chunks = iter([fake_chunk])

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self._chunks)
            except StopIteration:
                raise StopAsyncIteration

    evaluator = _StreamingResponseEvaluator(
        _FakeStream(),
        client=MagicMock(),
        params=HiddenLayerParams(model="gpt-4o-mini", session_id="stream-sess"),
        roundtrip_id="rt-1",
        session_id="stream-sess",
    )

    # Replace the module-level evaluate so _fire_evaluation calls the stub.
    import hiddenlayer_openai_guardrails._v2_model as v2_model

    original = v2_model.evaluate
    v2_model.evaluate = fake_evaluate
    try:
        async for _ in evaluator:
            pass
    finally:
        v2_model.evaluate = original

    assert len(captured) == 1
    assert captured[0]["path"] == "/detection/v2/response-evaluations"
    assert captured[0]["session_id"] == "stream-sess"


# ---------------------------------------------------------------------------
# exclude_tools_from_evaluation
# ---------------------------------------------------------------------------


def _function_tool():
    from agents import function_tool

    @function_tool
    def lookup_secret(query: str) -> str:
        """Execute a shell command and return its output."""
        return f"result for {query}"

    return lookup_secret


@pytest.mark.asyncio
@pytest.mark.unit
async def test_exclude_tools_strips_tools_from_eval_payload_only(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    captured = _patch_evaluate(monkeypatch)

    model, create_mock = _build_protected_model(
        hiddenlayer_params=HiddenLayerParams(
            model="gpt-4o-mini",
            exclude_tools_from_evaluation=True,
        ),
    )

    await _run_fetch_response(model, tools=[_function_tool()])

    request_eval = next(c for c in captured if c["path"] == "/detection/v2/request-evaluations")
    assert "tools" not in request_eval["payload"]
    assert "tool_choice" not in request_eval["payload"]

    openai_kwargs = create_mock.call_args.kwargs
    assert "tools" in openai_kwargs
    assert len(openai_kwargs["tools"]) == 1


@pytest.mark.asyncio
@pytest.mark.unit
async def test_exclude_tools_false_default_sends_tools_to_hiddenlayer(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    captured = _patch_evaluate(monkeypatch)

    model, create_mock = _build_protected_model(
        hiddenlayer_params=HiddenLayerParams(model="gpt-4o-mini"),
    )

    await _run_fetch_response(model, tools=[_function_tool()])

    request_eval = next(c for c in captured if c["path"] == "/detection/v2/request-evaluations")
    assert "tools" in request_eval["payload"]
    assert request_eval["payload"]["tools"] == create_mock.call_args.kwargs["tools"]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_exclude_tools_preserves_hl_redacted_messages(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")

    redacted_messages = [{"role": "user", "content": "REDACTED"}]

    async def fake_evaluate(client, path, payload, params, *, roundtrip_id=None, session_id=None):
        if path == "/detection/v2/request-evaluations":
            modified = {**payload, "messages": redacted_messages}
            return V2EvaluationResult(
                action="REDACT", payload=modified, blocked=False, roundtrip_id=roundtrip_id or ""
            )
        return V2EvaluationResult(action=None, payload={}, blocked=False, roundtrip_id=roundtrip_id or "")

    monkeypatch.setattr("hiddenlayer_openai_guardrails._v2_model.evaluate", fake_evaluate)

    model, create_mock = _build_protected_model(
        hiddenlayer_params=HiddenLayerParams(
            model="gpt-4o-mini",
            exclude_tools_from_evaluation=True,
        ),
    )

    await _run_fetch_response(model, tools=[_function_tool()])

    openai_kwargs = create_mock.call_args.kwargs
    assert openai_kwargs["messages"] == redacted_messages
    assert "tools" in openai_kwargs
    assert len(openai_kwargs["tools"]) == 1


@pytest.mark.asyncio
@pytest.mark.unit
async def test_exclude_tools_and_session_id_combined(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    captured = _patch_evaluate(monkeypatch)

    model, create_mock = _build_protected_model(
        hiddenlayer_params=HiddenLayerParams(
            model="gpt-4o-mini",
            session_id="combo-sess",
            exclude_tools_from_evaluation=True,
        ),
    )

    await _run_fetch_response(model, tools=[_function_tool()])

    request_eval = next(c for c in captured if c["path"] == "/detection/v2/request-evaluations")
    assert request_eval["session_id"] == "combo-sess"
    assert "tools" not in request_eval["payload"]
    assert "tools" in create_mock.call_args.kwargs
