import pytest

from hiddenlayer_openai_guardrails import (
    Agent,
    HiddenLayerParams,
)
from hiddenlayer_openai_guardrails._v2_model import HiddenLayerProtectedModel


@pytest.fixture
def v2_params() -> HiddenLayerParams:
    return HiddenLayerParams(
        model="gpt-4o-mini",
        project_id="test-project",
    )


@pytest.mark.unit
def test_params_accepts_project_id():
    params = HiddenLayerParams(
        model="gpt-4o-mini",
        project_id="my-project",
    )
    assert params.project_id == "my-project"



@pytest.mark.unit
def test_agent_uses_model_wrapper_without_guardrails(v2_params, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")

    agent = Agent(name="test", instructions="Help.", hiddenlayer_params=v2_params)

    assert isinstance(agent.model, HiddenLayerProtectedModel)
    assert agent.input_guardrails == []
    assert agent.output_guardrails == []
