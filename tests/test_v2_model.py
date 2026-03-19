import pytest

from hiddenlayer_openai_guardrails import Agent as TopLevelAgent, HiddenLayerParams
from hiddenlayer_openai_guardrails._v2_model import HiddenLayerProtectedModel, _extract_model_and_client


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
