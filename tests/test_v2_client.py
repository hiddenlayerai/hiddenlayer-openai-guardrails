import pytest
from unittest.mock import AsyncMock, MagicMock

from hiddenlayer_openai_guardrails._types import HiddenLayerParams
from hiddenlayer_openai_guardrails._v2_client import (
    REQUEST_EVALUATION_PATH,
    RESPONSE_EVALUATION_PATH,
    V2EvaluationResult,
    evaluate,
)


@pytest.mark.unit
def test_v2_evaluation_result_blocked():
    result = V2EvaluationResult(action="BLOCK", payload={}, blocked=True)
    assert result.blocked is True
    assert result.action == "BLOCK"


@pytest.mark.unit
def test_v2_evaluation_result_not_blocked():
    result = V2EvaluationResult(action=None, payload={"choices": []}, blocked=False)
    assert result.blocked is False
    assert result.action is None


@pytest.mark.asyncio
@pytest.mark.unit
async def test_evaluate_blocked():
    mock_response = MagicMock()
    mock_response.headers.get.return_value = "BLOCK"
    mock_response.json.return_value = {}

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    params = HiddenLayerParams(model="gpt-4", project_id="proj-1")
    result = await evaluate(mock_client, REQUEST_EVALUATION_PATH, {"model": "gpt-4"}, params)

    assert result.blocked is True
    assert result.action == "BLOCK"


@pytest.mark.asyncio
@pytest.mark.unit
async def test_evaluate_not_blocked():
    mock_response = MagicMock()
    mock_response.headers.get.return_value = None
    mock_response.json.return_value = {"choices": [{"message": {"content": "hello"}}]}

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    params = HiddenLayerParams(model="gpt-4")
    result = await evaluate(mock_client, RESPONSE_EVALUATION_PATH, {}, params)

    assert result.blocked is False
    assert result.action is None
    assert result.payload == {"choices": [{"message": {"content": "hello"}}]}


@pytest.mark.asyncio
@pytest.mark.unit
async def test_evaluate_sends_project_id_header():
    mock_response = MagicMock()
    mock_response.headers.get.return_value = None
    mock_response.json.return_value = {}

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    params = HiddenLayerParams(model="gpt-4", project_id="my-project")
    await evaluate(mock_client, REQUEST_EVALUATION_PATH, {}, params)

    headers = mock_client.post.call_args.kwargs["options"]["headers"]
    assert headers["HL-Project-Id"] == "my-project"


@pytest.mark.asyncio
@pytest.mark.unit
async def test_evaluate_uses_provided_roundtrip_id():
    mock_response = MagicMock()
    mock_response.headers.get.return_value = None
    mock_response.json.return_value = {}

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    params = HiddenLayerParams(model="gpt-4")
    result = await evaluate(mock_client, REQUEST_EVALUATION_PATH, {}, params, roundtrip_id="fixed-id")

    assert result.roundtrip_id == "fixed-id"
    headers = mock_client.post.call_args.kwargs["options"]["headers"]
    assert headers["Hl-Roundtrip-Id"] == "fixed-id"


@pytest.mark.asyncio
@pytest.mark.unit
async def test_evaluate_generates_roundtrip_id_when_not_provided():
    mock_response = MagicMock()
    mock_response.headers.get.return_value = None
    mock_response.json.return_value = {}

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    params = HiddenLayerParams(model="gpt-4")
    result = await evaluate(mock_client, REQUEST_EVALUATION_PATH, {}, params)

    assert result.roundtrip_id != ""


@pytest.mark.asyncio
@pytest.mark.unit
async def test_evaluate_sends_session_id_header_when_provided():
    mock_response = MagicMock()
    mock_response.headers.get.return_value = None
    mock_response.json.return_value = {}

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    params = HiddenLayerParams(model="gpt-4")
    await evaluate(mock_client, REQUEST_EVALUATION_PATH, {}, params, session_id="sess-abc")

    headers = mock_client.post.call_args.kwargs["options"]["headers"]
    assert headers["hl-runtime-session-id"] == "sess-abc"


@pytest.mark.asyncio
@pytest.mark.unit
async def test_evaluate_omits_session_id_header_when_not_provided():
    mock_response = MagicMock()
    mock_response.headers.get.return_value = None
    mock_response.json.return_value = {}

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    params = HiddenLayerParams(model="gpt-4")
    await evaluate(mock_client, REQUEST_EVALUATION_PATH, {}, params)

    headers = mock_client.post.call_args.kwargs["options"]["headers"]
    assert "hl-runtime-session-id" not in headers
