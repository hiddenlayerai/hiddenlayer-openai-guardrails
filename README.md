## HiddenLayer Guardrails for OpenAI Agents (Beta)

Drop-in replacement for the Agents SDK `Agent` that wires HiddenLayer guardrails into agent and tool execution. Agent input/output and every tool call are sent through HiddenLayer's analyze endpoint so prompt-injection and policy violations are caught automatically.

> **Note:** OpenAI's native guardrails only support blocking content—they do not support redacting sensitive information from input or output. This library provides redaction capabilities through HiddenLayer's `REDACT` action, allowing you to sanitize content while still allowing the request to proceed.

### Installation

`pip install hiddenlayer-openai-guardrails`

### Configuration

#### Environment Variables

The following environment variables must be set for authentication:

- `HIDDENLAYER_CLIENT_ID` - HiddenLayer API client ID (required)
- `HIDDENLAYER_CLIENT_SECRET` - HiddenLayer API client secret (required)

Optional environment variables:

- `HIDDENLAYER_PROJECT_ID` - HiddenLayer project ID for policy routing
- `HIDDENLAYER_REQUESTER_ID` - Identifier for tracking requests (default: "hiddenlayer-openai-integration")

```bash
# Required
export HIDDENLAYER_CLIENT_ID="your-client-id"
export HIDDENLAYER_CLIENT_SECRET="your-client-secret"

# Optional
export HIDDENLAYER_PROJECT_ID="your-project-id"
export HIDDENLAYER_REQUESTER_ID="your-app-name"
```

#### HiddenLayerParams

Configure HiddenLayer behavior using the `HiddenLayerParams` object:

```python
from hiddenlayer_openai_guardrails import HiddenLayerParams

params = HiddenLayerParams(
    project_id="my-project",       # Optional: HiddenLayer project ID for policy routing
    model="gpt-4o-mini",            # Optional: Model name for tracking (auto-detected from agent if not set)
    requester_id="my-app-v1",      # Optional: Identifier for tracking requests (default: "hiddenlayer-openai-integration")
)
```

All fields are optional. If `model` is not provided, it will be automatically detected from the agent's model configuration.

### Usage

#### Basic Agent with Guardrails

The `Agent` class mirrors `agents.Agent` but adds HiddenLayer guardrails to the agent and all tools. Guardrails automatically block malicious content:

```python
from agents import Runner, function_tool
from agents.run import RunConfig
from hiddenlayer_openai_guardrails import Agent, HiddenLayerParams


@function_tool
def get_weather(city: str) -> str:
    """returns weather info for the specified city."""
    return f"The weather in {city} is sunny"


# Configure HiddenLayer parameters
params = HiddenLayerParams(
    project_id="my-project",  # optional: for policy routing
)

agent = Agent(
    name="Haiku agent",
    instructions="Always respond in haiku form",
    model="gpt-4o-mini",
    tools=[get_weather],  # tool input/output are screened by HiddenLayer
    hiddenlayer_params=params,  # optional: defaults will be used if not provided
)

result = Runner.run_sync(
    agent,
    "What's the weather in Toronto",
    run_config=RunConfig(tracing_disabled=True),
)
print(result.final_output)
```

#### Redacting Input and Output

Since OpenAI's guardrails can only block (not redact), this library provides helper functions for content redaction:

```python
from agents import Runner
from hiddenlayer_openai_guardrails import (
    Agent,
    HiddenLayerParams,
    redact_input,
    redact_output,
    InputBlockedError,
    OutputBlockedError,
)

# Configure HiddenLayer parameters
params = HiddenLayerParams(project_id="my-project")

agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant.",
    hiddenlayer_params=params,
)

try:
    # Redact sensitive info from user input before processing
    safe_input = await redact_input(
        user_input,
        hiddenlayer_params=params,
    )

    # Run agent (guardrails will block malicious content)
    result = await Runner.run(agent, safe_input)

    # Redact sensitive info from output before showing to user
    safe_output = await redact_output(
        result.final_output,
        hiddenlayer_params=params,
    )
    print(safe_output)

except InputBlockedError:
    print("Input was blocked by HiddenLayer")
except OutputBlockedError:
    print("Output was blocked by HiddenLayer")
```

#### Safe Streaming Output

For streaming responses, use `safe_stream` to stream event objects while scanning the final output through HiddenLayer guardrails:

```python
from agents import Runner
from hiddenlayer_openai_guardrails import Agent, HiddenLayerParams, safe_stream

# Configure HiddenLayer parameters
params = HiddenLayerParams(project_id="my-project")

agent = Agent(
    name="Assistant",
    instructions="Help users",
    hiddenlayer_params=params,
)
result = Runner.run_streamed(agent, user_input)

async for event in safe_stream(result, hiddenlayer_params=params):
    # event is an Agents SDK stream event (not plain text)
    print(event)
```

If HiddenLayer returns a `BLOCK` action for the final streamed output, `safe_stream` raises `OutputBlockedError` after streaming completes.

### How it works

- `hiddenlayer_openai_guardrails.agents.Agent` returns a regular `agents.Agent` configured with:
  - Agent-level input/output guardrails that analyze user and assistant messages.
  - Tool-level guardrails that inspect tool arguments before execution and tool output afterward.
- Guardrails rely on `AsyncHiddenLayer.interactions.analyze` and will raise when HiddenLayer signals a blocking action.
- Input guardrails normalize and forward the full conversation message list so HiddenLayer sees the same message flow the model receives.
- Tool guardrails currently enforce block-only behavior; `REDACT` actions are not applied in tool hooks.
- Tool output scanning is sent through the output phase with assistant-role message semantics.

### Development

- Run tests after installing dev deps (`pytest` and `pytest-asyncio`): `pytest tests`
- Public API lives in `src/hiddenlayer_openai_guardrails/agents.py` (facade); implementation is split across:
  - `src/hiddenlayer_openai_guardrails/_hiddenlayer.py` (HiddenLayer client + analyze calls)
  - `src/hiddenlayer_openai_guardrails/_guardrails.py` (agent/tool/MCP guardrail wiring)
  - `src/hiddenlayer_openai_guardrails/_normalize.py` (payload normalization)
  - `src/hiddenlayer_openai_guardrails/_analysis.py` (response parsing)
  - `src/hiddenlayer_openai_guardrails/_redaction.py` and `src/hiddenlayer_openai_guardrails/_streaming.py`
- Tests are in `tests/test_agents_unit.py` and `tests/test_agents_integration.py`.
