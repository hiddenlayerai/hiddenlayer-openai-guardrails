import pytest

from hiddenlayer_openai_guardrails import Agent as TopLevelAgent, HiddenLayerParams
from hiddenlayer_openai_guardrails._v2_model import HiddenLayerProtectedModel
from hiddenlayer_openai_guardrails.v2 import Agent as V2Agent


@pytest.mark.unit
def test_v2_agent_factory_wraps_model_without_guardrails(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("HIDDENLAYER_PROJECT_ID", "project-123")

    agent = V2Agent(name="Assistant", model="gpt-4o-mini")

    assert isinstance(agent.model, HiddenLayerProtectedModel)
    assert not agent.input_guardrails
    assert not agent.output_guardrails


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
