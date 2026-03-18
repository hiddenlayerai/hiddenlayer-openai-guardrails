import logging
from collections.abc import Callable
from typing import Any

from agents import Agent as OpenAIAgent, models
from agents.models.interface import Model
from hiddenlayer import AsyncHiddenLayer

from hiddenlayer_openai_guardrails._types import HiddenLayerParams

from ._v2_model import HiddenLayerProtectedModel, _extract_model_and_client

logger = logging.getLogger(__name__)


def _parse_model(model: str | Model | None):
    if not model:
        return models.get_default_model()

    return str(model)


class Agent:
    """Drop-in replacement for Agents SDK Agent with HiddenLayer guardrails.

    This class acts as a factory that creates a regular Agents SDK Agent instance
    with HiddenLayer guardrails automatically configured. Guardrails analyze input
    and output for policy violations and will block execution when violations are
    detected.

    Guardrails are applied at multiple levels:
    - Agent input: Checks user input before the agent processes it
    - Agent output: Checks agent responses before returning to user
    - Tool input: Checks tool calls before execution
    - Tool output: Checks tool results after execution

    Note: Guardrails only BLOCK on policy violations. For content redaction,
    use the separate `redact_input()` and `redact_output()` functions before
    and after calling `Runner.run()`.

    Example:
        ```python
        from hiddenlayer_openai_guardrails import Agent, HiddenLayerParams, redact_input, redact_output
        from agents import Runner, function_tool

        @function_tool
        def get_weather(location: str) -> str:
            return f"Weather in {location}: Sunny"

        # Configure HiddenLayer parameters
        params = HiddenLayerParams(project_id="my-project")

        agent = Agent(
            name="Weather Assistant",
            instructions="You help with weather information.",
            tools=[get_weather],
            hiddenlayer_params=params,
        )

        # Optional: redact sensitive content from input
        user_input = await redact_input(raw_input, hiddenlayer_params=params)

        # Run agent - guardrails will block malicious content
        result = await Runner.run(agent, user_input)

        # Optional: redact sensitive content from output
        final_output = await redact_output(result.final_output, hiddenlayer_params=params)
        ```
    """

    def __new__(
        cls,
        name: str,
        instructions: str | Callable[[Any, Any], Any] | None = None,
        hiddenlayer_params: HiddenLayerParams | None = None,
        hiddenlayer_client: AsyncHiddenLayer | None = None,
        **agent_kwargs: Any,
    ) -> Any:  # Returns agents.Agent
        """Create a new Agent instance with HiddenLayer guardrails.

        Args:
            name: Agent name
            instructions: Agent instructions. Can be a string, a callable that dynamically
                generates instructions, or None. If a callable, it will receive the context
                and agent instance and must return a string.
            hiddenlayer_params: Optional HiddenLayerParams object for configuration. If not provided,
                defaults will be used.
            hiddenlayer_client: Optional AsyncHiddenLayer client instance
            **agent_kwargs: All other arguments passed to Agent constructor (model, tools, etc.)

        Returns:
            agents.Agent: A fully configured Agent instance with HiddenLayer guardrails
        """
        # Apply tool-level guardrails
        model = agent_kwargs.get("model", None)

        if isinstance(model, HiddenLayerProtectedModel):
            wrapped_model = model
        else:
            model_name, openai_client = _extract_model_and_client(model)
            if hiddenlayer_params is None:
                hiddenlayer_params = HiddenLayerParams(model=model_name)

            if not hiddenlayer_params.model:
                hiddenlayer_params.model = model_name
            wrapped_model = HiddenLayerProtectedModel(
                model=model_name,
                openai_client=openai_client,
                hiddenlayer_params=hiddenlayer_params,
                hiddenlayer_client=hiddenlayer_client,
            )

        agent_kwargs["model"] = wrapped_model

        return OpenAIAgent(
            name=name,
            instructions=instructions,
            **agent_kwargs,
        )
