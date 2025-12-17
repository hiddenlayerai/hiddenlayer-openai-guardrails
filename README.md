## HiddenLayer Guardrails for OpenAI Agents

Drop-in replacement for the Agents SDK `Agent` that wires HiddenLayer guardrails into agent and tool execution. Agent input/output and every tool call are sent through HiddenLayer's analyze endpoint so prompt-injection and policy violations are caught automatically.

### Installation

`pip install hiddenlayer-openai-guardrails`

### Configuration

- Set `HIDDENLAYER_CLIENT_ID` and `HIDDENLAYER_CLIENT_SECRET` in your environment so the SDK can authenticate.

### Usage

The agent mirrors `agents.Agent` but adds HiddenLayer guardrails to the agent and all tools. Example adapted from `tests/test_agents.py`:

```python
from agents import Runner, function_tool
from agents.run import RunConfig
from hiddenlayer_openai_guardrails.agents import Agent


@function_tool
def get_weather(city: str) -> str:
    """returns weather info for the specified city."""
    return f"The weather in {city} is sunny"


agent = Agent(
    name="Haiku agent",
    instructions="Always respond in haiku form",
    model="gpt-5-nano",
    tools=[get_weather],  # tool input/output are screened by HiddenLayer
)

result = Runner.run_sync(
    agent,
    "What's the weather in Toronto",
    run_config=RunConfig(tracing_disabled=True),
)
print(result.final_output)
```

### How it works

- `hiddenlayer_openai_guardrails.agents.Agent` returns a regular `agents.Agent` configured with:
  - Agent-level input/output guardrails that analyze user and assistant messages.
  - Tool-level guardrails that inspect arguments before execution and outputs afterward.
- Guardrails rely on `AsyncHiddenLayer.interactions.analyze` and will raise when HiddenLayer signals a blocking action.

### Development

- Run tests after installing dev deps (`pytest` and `pytest-asyncio`): `pytest tests`
- Code lives in `src/hiddenlayer_openai_guardrails/agents.py`; tests are in `tests/test_agents.py`.
