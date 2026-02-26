import logging
from collections.abc import AsyncIterator, Callable
from typing import Any, Literal

import agents
from agents import (
    Agent as OpenAIAgent,
    models,
)
from agents.guardrail import InputGuardrail, OutputGuardrail
from agents.models.interface import Model
from agents.result import RunResultStreaming
from agents.stream_events import StreamEvent
from hiddenlayer import AsyncHiddenLayer

from . import _analysis as _analysis, _normalize as _normalize, _types as _types
from ._guardrails import (
    attach_guardrail_to_tools as _attach_guardrail_to_tools_impl,
    create_input_output_guardrail as _create_input_output_guardrail_impl,
    create_tool_guardrail as _create_tool_guardrail_impl,
    scan_tool_definition as _scan_tool_definition_impl,
    wrap_mcp_tools_with_guardrails as _wrap_mcp_tools_with_guardrails_impl,
)
from ._hiddenlayer import analyze_content as _analyze_content
from ._redaction import redact_content as _redact_content_impl
from ._streaming import safe_stream as _safe_stream_impl

logger = logging.getLogger(__name__)

AnalysisResult = _types.AnalysisResult
HiddenLayerParams = _types.HiddenLayerParams
HiddenlayerActions = _types.HiddenlayerActions
InputBlockedError = _types.InputBlockedError
OutputBlockedError = _types.OutputBlockedError

# Preserve historically-imported module paths for public types.
AnalysisResult.__module__ = __name__
HiddenLayerParams.__module__ = __name__
HiddenlayerActions.__module__ = __name__
InputBlockedError.__module__ = __name__
OutputBlockedError.__module__ = __name__

_normalize_content = _normalize.normalize_content
_normalize_input_messages = _normalize.normalize_input_messages
_safe_json_dumps = _normalize.safe_json_dumps
_safe_parse_tool_arguments = _normalize.safe_parse_tool_arguments

_parse_analysis = _analysis.parse_analysis


def _create_tool_guardrail(
    guardrail_type: str,
    hiddenlayer_params: HiddenLayerParams,
    client: AsyncHiddenLayer | None = None,
) -> Callable:
    return _create_tool_guardrail_impl(
        guardrail_type,
        hiddenlayer_params,
        client,
        analyze_content=_analyze_content,
    )


def _create_input_output_guardrail(
    guardrail_type: str,
    hiddenlayer_params: HiddenLayerParams,
    client: AsyncHiddenLayer | None = None,
) -> InputGuardrail | OutputGuardrail:
    return _create_input_output_guardrail_impl(
        guardrail_type,
        hiddenlayer_params,
        client,
        analyze_content=_analyze_content,
    )


async def _scan_tool_definition(
    tool: Any,
    hiddenlayer_params: HiddenLayerParams,
    client: AsyncHiddenLayer | None = None,
) -> None:
    await _scan_tool_definition_impl(
        tool,
        hiddenlayer_params,
        client,
        analyze_content=_analyze_content,
    )


def _attach_guardrail_to_tools(
    tools: list[Any],
    guardrail: Callable,
    guardrail_type: str,
) -> None:
    _attach_guardrail_to_tools_impl(tools, guardrail, guardrail_type)


def _wrap_mcp_tools_with_guardrails(
    agent: agents.Agent,
    tool_input_guardrail: Callable,
    tool_output_guardrail: Callable,
    hiddenlayer_params: HiddenLayerParams,
    hiddenlayer_client: AsyncHiddenLayer | None = None,
) -> Any:
    return _wrap_mcp_tools_with_guardrails_impl(
        agent,
        tool_input_guardrail,
        tool_output_guardrail,
        hiddenlayer_params,
        hiddenlayer_client,
        analyze_content=_analyze_content,
        scan_tool_definition_fn=_scan_tool_definition,
    )


def _parse_model(model: str | Model | None):
    if not model:
        return models.get_default_model()

    return str(model)


async def _redact_content(
    content: str,
    role: Literal["user", "assistant"],
    params: HiddenLayerParams,
    client: AsyncHiddenLayer | None = None,
) -> str:
    return await _redact_content_impl(content, role, params, client, analyze_content=_analyze_content)


async def redact_input(
    input: str,
    hiddenlayer_params: HiddenLayerParams,
    client: AsyncHiddenLayer | None = None,
) -> str:
    """Redact input through HiddenLayer before agent execution.

    This function allows you to analyze and potentially redact user input
    before passing it to Runner.run(). Unlike input guardrails which can
    only block or report, this function can return redacted content.

    Args:
        input: The user input string to analyze
        hiddenlayer_params: HiddenLayerParams for configuration.
        client: Optional AsyncHiddenLayer client instance. If omitted, a shared lazy client is used.

    Returns:
        The original input if no issues, or redacted input if REDACT action

    Raises:
        InputBlockedError: If HiddenLayer returns BLOCK action

    Example:
        ```python
        from hiddenlayer_openai_guardrails import redact_input, HiddenLayerParams

        params = HiddenLayerParams(project_id="my-project")
        redacted = await redact_input(user_input, hiddenlayer_params=params)
        result = await Runner.run(agent, redacted)
        ```
    """
    return await _redact_content(input, "user", hiddenlayer_params, client)


async def redact_output(
    output: str,
    hiddenlayer_params: HiddenLayerParams,
    client: AsyncHiddenLayer | None = None,
) -> str:
    """Redact agent output through HiddenLayer.

    This function allows you to analyze and potentially redact agent output
    after Runner.run() completes. Unlike output guardrails which can only
    block or report, this function can return redacted content.

    Args:
        output: The agent output string to analyze
        hiddenlayer_params: HiddenLayerParams for configuration.
        client: Optional AsyncHiddenLayer client instance. If omitted, a shared lazy client is used.

    Returns:
        The original output if no issues, or redacted output if REDACT action

    Raises:
        OutputBlockedError: If HiddenLayer returns BLOCK action

    Example:
        ```python
        from hiddenlayer_openai_guardrails import redact_output, HiddenLayerParams

        result = await Runner.run(agent, user_input)
        params = HiddenLayerParams(project_id="my-project")
        redacted = await redact_output(result.final_output, hiddenlayer_params=params)
        ```
    """
    return await _redact_content(output, "assistant", hiddenlayer_params, client)


async def safe_stream(
    streaming_result: RunResultStreaming,
    hiddenlayer_params: HiddenLayerParams,
    client: AsyncHiddenLayer | None = None,
) -> AsyncIterator[StreamEvent]:
    """Stream agent output while scanning it through HiddenLayer guardrails.

    This function consumes a streaming result from Runner.run_streamed(),
    yielding stream events in real-time. Once streaming completes, the full
    output is scanned through HiddenLayer to detect policy violations.

    Args:
        streaming_result: Result from Runner.run_streamed()
        hiddenlayer_params: HiddenLayerParams for configuration.
        client: Optional AsyncHiddenLayer client instance. If omitted, a shared lazy client is used.

    Yields:
        Stream events as they arrive from the underlying stream

    Raises:
        OutputBlockedError: If HiddenLayer blocks the output

    Example:
        ```python
        from hiddenlayer_openai_guardrails import Agent, safe_stream, HiddenLayerParams
        from agents import Runner

        agent = Agent(name="Assistant", instructions="Help users")
        result = Runner.run_streamed(agent, user_input)

        params = HiddenLayerParams(project_id="my-proj")
        async for event in safe_stream(result, hiddenlayer_params=params):
            print(event)
        ```
    """
    async for event in _safe_stream_impl(
        streaming_result,
        hiddenlayer_params,
        client,
        analyze_content=_analyze_content,
    ):
        yield event


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
        tools = agent_kwargs.get("tools", [])
        model = agent_kwargs.get("model", None)

        model = _parse_model(model)

        # Use provided params or create defaults
        if hiddenlayer_params is None:
            hiddenlayer_params = HiddenLayerParams(model=model)

        if not hiddenlayer_params.model:
            hiddenlayer_params.model = model

        tool_input_gr = _create_tool_guardrail("input", hiddenlayer_params, hiddenlayer_client)
        tool_output_gr = _create_tool_guardrail("output", hiddenlayer_params, hiddenlayer_client)
        _attach_guardrail_to_tools(tools, tool_input_gr, "input")
        _attach_guardrail_to_tools(tools, tool_output_gr, "output")

        # Create input/output guardrails with HiddenLayer params
        input_gr: InputGuardrail = _create_input_output_guardrail("input", hiddenlayer_params, hiddenlayer_client)
        output_gr: OutputGuardrail = _create_input_output_guardrail("output", hiddenlayer_params, hiddenlayer_client)

        # Create the base agent with guardrails
        agent = OpenAIAgent(
            name=name,
            instructions=instructions,
            input_guardrails=[input_gr],
            output_guardrails=[output_gr],
            **agent_kwargs,
        )

        if agent_kwargs.get("mcp_servers"):
            agent = _wrap_mcp_tools_with_guardrails(
                agent,
                tool_input_gr,
                tool_output_gr,
                hiddenlayer_params,
                hiddenlayer_client,
            )

        return agent
