## HiddenLayer Guardrails for OpenAI Agents (Beta)

Drop-in replacement for the OpenAI Agents SDK `Agent` that routes every chat-completion roundtrip through HiddenLayer's v2 detection endpoints. User input, system instructions, tool definitions, model output, and streamed content are all evaluated for prompt-injection and policy violations. When HiddenLayer signals a block, the library raises `InputBlockedError` or `OutputBlockedError` and the call never reaches the user (input side) or the caller (output side, non-streaming).

This is implemented by wrapping the chat-completions `Model` the Agents SDK uses internally, so a single interception point sees the full request and response payloads. There is no separate decorator API to attach.

### Installation

```bash
pip install hiddenlayer-openai-guardrails
```

### Configuration

Authentication is read from environment variables by the underlying `hiddenlayer-sdk` client:

| Variable | Required | Purpose |
| --- | --- | --- |
| `HIDDENLAYER_CLIENT_ID` | yes | HiddenLayer API client ID |
| `HIDDENLAYER_CLIENT_SECRET` | yes | HiddenLayer API client secret |
| `HIDDENLAYER_PROJECT_ID` | no | Default value for `HiddenLayerParams.project_id`; routes evaluations to a specific HL policy project |
| `HIDDENLAYER_REQUESTER_ID` | no | Default value for `HiddenLayerParams.requester_id`; tags the source of evaluations (defaults to `"hiddenlayer-openai-integration"`) |

Per-call behavior is configured via `HiddenLayerParams`:

```python
from hiddenlayer_openai_guardrails import HiddenLayerParams

params = HiddenLayerParams(
    project_id="my-project",                # optional: HL policy routing
    requester_id="my-app-v1",               # optional: source tag
    model="gpt-4o-mini",                    # optional: auto-detected from agent.model if unset
    session_id="conversation-7f3a",         # optional: per-conversation correlation ID
    exclude_tools_from_evaluation=False,    # optional: see "Excluding tool definitions" below
)
```

All fields are optional. `model` will be inferred from the agent's model configuration when not provided.

### Usage

#### Basic agent

`Agent` mirrors `agents.Agent` and adds HiddenLayer evaluation transparently. The returned object is a standard `agents.Agent`, so it works with `Runner.run`, `Runner.run_sync`, and `Runner.run_streamed` the same way as the upstream SDK.

```python
from agents import Runner, function_tool
from agents.run import RunConfig
from hiddenlayer_openai_guardrails import Agent, HiddenLayerParams


@function_tool
def get_weather(city: str) -> str:
    """Return weather info for the specified city."""
    return f"The weather in {city} is sunny"


agent = Agent(
    name="Haiku agent",
    instructions="Always respond in haiku form.",
    model="gpt-4o-mini",
    tools=[get_weather],
    hiddenlayer_params=HiddenLayerParams(project_id="my-project"),
)

result = Runner.run_sync(
    agent,
    "What's the weather in Toronto?",
    run_config=RunConfig(tracing_disabled=True),
)
print(result.final_output)
```

If HiddenLayer blocks the input or output, `Runner` propagates `InputBlockedError` / `OutputBlockedError` from `hiddenlayer_openai_guardrails`:

```python
from hiddenlayer_openai_guardrails import InputBlockedError, OutputBlockedError

try:
    result = await Runner.run(agent, user_input)
except InputBlockedError:
    print("Request blocked by HiddenLayer policy.")
except OutputBlockedError:
    print("Response blocked by HiddenLayer policy.")
```

#### Streaming

Streaming works through `Runner.run_streamed` unchanged. The library accumulates streamed assistant content and posts it to the response-evaluation endpoint after the stream completes for logging and downstream policy enforcement. Note that streamed content has already been delivered to the caller by the time the response evaluation fires, so a `BLOCK` action on a streamed response is currently observed-only and not enforced inline.

#### Session correlation

Set `session_id` on `HiddenLayerParams` to tag every request- and response-evaluation HTTP call with `hl-runtime-session-id`. This is what the HiddenLayer console uses to group multiple turns of a conversation together. Without it, evaluations from a multi-turn agent appear as unrelated rows.

```python
agent = Agent(
    name="support-bot",
    instructions="Help the user.",
    hiddenlayer_params=HiddenLayerParams(
        project_id="support",
        session_id=f"conv-{conversation_id}",
    ),
)
```

`session_id` is a static string for the lifetime of the `HiddenLayerParams` instance. If you need a different session per turn, construct a new `HiddenLayerParams` (and a new `Agent`) for each turn.

#### Excluding tool definitions from evaluation

By default, tool definitions are included in the payload posted to the request-evaluation endpoint along with the user messages. This is usually correct: HiddenLayer can inspect what tools the model is allowed to call.

In some integrations, tool *descriptions* themselves trigger false positives in injection detectors (for example a tool named `execute_command` with a description that mentions running shell commands). Set `exclude_tools_from_evaluation=True` to strip `tools` and `tool_choice` from the HiddenLayer payload while keeping them intact on the downstream OpenAI call:

```python
agent = Agent(
    name="cli-helper",
    instructions="Help the user with shell tasks.",
    tools=[execute_command, tail_file],
    hiddenlayer_params=HiddenLayerParams(
        project_id="cli",
        exclude_tools_from_evaluation=True,
    ),
)
```

When this flag is set, any HiddenLayer-side modifications to the request (for example, redacted messages) are still honored; the original tool list is re-attached on top before the request is sent to OpenAI.

### How it works

`hiddenlayer_openai_guardrails.Agent` is a factory whose `__new__` returns a regular `agents.Agent` configured with a `HiddenLayerProtectedModel` in place of the standard chat-completions model. On each Agents SDK roundtrip, `HiddenLayerProtectedModel._fetch_response`:

1. Builds the chat-completions request the way the upstream SDK would (system instructions, converted messages, tools, tool choice, response format, etc.).
2. POSTs that payload to `/detection/v2/request-evaluations`. A `BLOCK` action raises `InputBlockedError`. A modified payload (for example, redacted messages) is used in place of the original.
3. Sends the resulting payload to OpenAI.
4. For non-streaming responses, POSTs the completion to `/detection/v2/response-evaluations`. A `BLOCK` action raises `OutputBlockedError`; a modified payload replaces the model's original response.
5. For streaming responses, wraps the stream so that streamed `delta.content` is buffered and posted to the response-evaluation endpoint after the stream exhausts.

All HiddenLayer calls carry `Hl-Roundtrip-Id` (request and response evaluations for a single turn share the same value), `HL-Project-Id`, `hl-requester-id`, and optionally `hl-runtime-session-id`. The `hl-runtime-action` response header drives block / pass behavior.

### Development

```bash
# Install dependencies (uses uv)
uv sync

# Run unit tests (mocked, no network)
pytest tests -m unit

# Run live integration tests (requires HL credentials + OPENAI_API_KEY)
RUN_LIVE_INTEGRATION_TESTS=1 pytest tests -m integration

# Run all tests
pytest tests

# Lint and import-order check (ruff: E, F, I; line length 120)
uv run ruff check .
```

Implementation lives in four files under `src/hiddenlayer_openai_guardrails/`:

- `agents.py`: `Agent` factory; wraps the user-supplied model into `HiddenLayerProtectedModel` and hands a stock `agents.Agent` back to the caller.
- `_v2_model.py`: `HiddenLayerProtectedModel` (subclass of `OpenAIChatCompletionsModel`) and `_StreamingResponseEvaluator`. All HL evaluation logic lives here.
- `_v2_client.py`: async wrapper around `AsyncHiddenLayer.post(...)` for the two v2 endpoints; assembles correlation headers and returns a `V2EvaluationResult`.
- `_types.py`: `HiddenLayerParams` (Pydantic), `InputBlockedError`, `OutputBlockedError`.

Tests live in `tests/test_v2_client.py`, `tests/test_v2_model.py`, and `tests/test_v2_integration.py`.
