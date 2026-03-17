import pytest

from hiddenlayer_openai_guardrails._v2_client import V2EvaluationResult
from hiddenlayer_openai_guardrails._v2_payloads import (
    build_openai_request_payload,
    build_openai_response_payload,
    build_tool_call_request_payload,
    build_tool_result_response_payload,
    extract_content_from_payload,
)


@pytest.mark.unit
def test_build_openai_request_payload():
    msgs = [{"role": "user", "content": "hello"}]
    payload = build_openai_request_payload(msgs, "gpt-4")
    assert payload == {"model": "gpt-4", "messages": msgs}


@pytest.mark.unit
def test_build_openai_request_payload_default_model():
    payload = build_openai_request_payload([{"role": "user", "content": "hi"}])
    assert payload["model"] == "unknown"


@pytest.mark.unit
def test_build_openai_response_payload():
    payload = build_openai_response_payload("The answer is 42.", "gpt-4")
    assert payload["model"] == "gpt-4"
    assert payload["choices"][0]["message"]["role"] == "assistant"
    assert payload["choices"][0]["message"]["content"] == "The answer is 42."
    assert payload["choices"][0]["finish_reason"] == "stop"


@pytest.mark.unit
def test_build_tool_call_request_payload():
    payload = build_tool_call_request_payload("calculator", '{"x": 1}', "gpt-4")
    assert payload["model"] == "gpt-4"
    msg = payload["messages"][0]
    assert msg["role"] == "assistant"
    assert msg["tool_calls"][0]["function"]["name"] == "calculator"
    assert msg["tool_calls"][0]["function"]["arguments"] == '{"x": 1}'


@pytest.mark.unit
def test_build_tool_result_response_payload():
    payload = build_tool_result_response_payload("calculator", "42", "gpt-4")
    assert payload["model"] == "gpt-4"
    msg = payload["messages"][0]
    assert msg["role"] == "tool"
    assert msg["name"] == "calculator"
    assert msg["content"] == "42"


@pytest.mark.unit
def test_extract_content_from_response_payload():
    payload = build_openai_response_payload("Hello world")
    assert extract_content_from_payload(payload, "assistant") == "Hello world"


@pytest.mark.unit
def test_extract_content_from_request_payload():
    payload = build_openai_request_payload([{"role": "user", "content": "Hi"}])
    assert extract_content_from_payload(payload, "user") == "Hi"


@pytest.mark.unit
def test_extract_content_from_empty_payload():
    assert extract_content_from_payload({}, "assistant") is None
    assert extract_content_from_payload({}, "user") is None


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

