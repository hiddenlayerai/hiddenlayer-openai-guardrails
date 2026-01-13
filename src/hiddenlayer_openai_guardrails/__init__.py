from hiddenlayer_openai_guardrails.agents import (
    Agent,
    InputBlockedError,
    OutputBlockedError,
    redact_input,
    redact_output,
    redact_streamed_output,
)

__all__ = [
    "Agent",
    "InputBlockedError",
    "OutputBlockedError",
    "redact_input",
    "redact_output",
    "redact_streamed_output",
]


def hello() -> str:
    return "Hello from hiddenlayer-openai-guardrails!"
