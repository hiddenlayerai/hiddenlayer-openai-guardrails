from openai.types.realtime.input_audio_buffer_timeout_triggered import InputAudioBufferTimeoutTriggered
from dataclasses import dataclass
from enum import Enum
from typing import Any, AsyncIterator, Callable, Literal
import json
import logging

from agents import models
from agents import (
    Agent as OpenAIAgent,
    GuardrailFunctionOutput,
    RunContextWrapper,
    TResponseInputItem,
    ToolGuardrailFunctionOutput,
    ToolInputGuardrailData,
    ToolOutputGuardrailData,
    output_guardrail,
    tool_input_guardrail,
    tool_output_guardrail,
)
import agents
from agents.guardrail import input_guardrail, InputGuardrail, OutputGuardrail
from agents.result import RunResultStreaming

from agents.models.interface import Model
from hiddenlayer import AsyncHiddenLayer
from pydantic import BaseModel


logger = logging.getLogger(__name__)


class HiddenLayerParams(BaseModel):
    """HiddenLayer request metadata and policy routing parameters."""

    model: str | None = None
    project_id: str | None = None
    requester_id: str = "hiddenlayer-openai-integration"


class HiddenlayerActions(str, Enum):
    BLOCK = "Block"
    REDACT = "Redact"


class InputBlockedError(Exception):
    """Raised when HiddenLayer blocks the input."""

    pass


class OutputBlockedError(Exception):
    """Raised when HiddenLayer blocks the output."""

    pass


@dataclass
class AnalysisResult:
    """Unified result from HiddenLayer analysis."""

    block: bool
    redact: bool
    redacted_content: str | None


async def _analyze_content(
    content: str,
    role: Literal["user", "assistant"],
    hiddenlayer_params: HiddenLayerParams,
    client: AsyncHiddenLayer | None = None,
) -> Any:
    """Single location for all HiddenLayer API calls.

    Args:
        content: The content to analyze
        role: "user" for input analysis, "assistant" for output analysis
        hiddenlayer_params: HiddenLayer configuration parameters
        client: Optional AsyncHiddenLayer client instance

    Returns:
        Raw HiddenLayer analysis response
    """
    if client is None:
        client = AsyncHiddenLayer()

    # Build metadata - only include model if it's set
    metadata: dict[str, Any] = {"requester_id": hiddenlayer_params.requester_id}
    if hiddenlayer_params.model:
        metadata["model"] = hiddenlayer_params.model

    message = {"messages": [{"role": role, "content": content}]}

    kwargs: dict[str, Any] = {"metadata": metadata}
    if hiddenlayer_params.project_id:
        kwargs["hl_project_id"] = hiddenlayer_params.project_id

    if role == "user":
        kwargs["input"] = message
    else:
        kwargs["output"] = message

    return await client.interactions.analyze(**kwargs)


def _parse_analysis(response: Any, role: Literal["user", "assistant"]) -> AnalysisResult:
    """Parse HiddenLayer response into unified result.

    Args:
        response: Raw HiddenLayer analysis response
        role: "user" for input analysis, "assistant" for output analysis

    Returns:
        AnalysisResult with block, redact, and optional redacted_content
    """
    action = response.evaluation.action if response.evaluation else None
    block = action == HiddenlayerActions.BLOCK
    redact = action == HiddenlayerActions.REDACT

    redacted_content = None
    if redact and response.modified_data:
        messages = response.modified_data.input.messages if role == "user" else response.modified_data.output.messages
        if messages:
            redacted_content = messages[-1].content

    return AnalysisResult(block, redact, redacted_content)


def _create_tool_guardrail(
    guardrail_type: str,
    hiddenlayer_params: HiddenLayerParams,
    client: AsyncHiddenLayer | None = None,
) -> Callable:
    """Create a generic tool-level guardrail wrapper.

    Args:
        guardrail_type: "input" (before tool execution) or "output" (after tool execution)
        hiddenlayer_params: HiddenLayer configuration parameters
        client: Optional AsyncHiddenLayer client instance

    Returns:
        Tool guardrail function decorated with @tool_input_guardrail or @tool_output_guardrail
    """

    if guardrail_type == "input":

        @tool_input_guardrail
        async def tool_input_gr(data: ToolInputGuardrailData) -> ToolGuardrailFunctionOutput:
            """Check tool call before execution."""
            # The tool input guardrail data doesn't have tool description but that can be an attack vector
            # We get the current tool here by filtering all tools and then parse the description where possible
            tools = await data.agent.get_all_tools(data.context)

            curr_tool = None
            for tool in tools:
                if tool.name == data.context.tool_name:
                    curr_tool = tool

            check_data = json.dumps(
                {
                    "tool_name": data.context.tool_name,
                    "tool_description": curr_tool.description if hasattr(curr_tool, "description") else "",
                    "arguments": data.context.tool_arguments,
                    "call_id": getattr(data.context, "tool_call_id", None),
                }
            )

            print(check_data)

            analysis = await _analyze_content(check_data, "user", hiddenlayer_params, client)

            if analysis.evaluation and analysis.evaluation.action == HiddenlayerActions.BLOCK:
                return ToolGuardrailFunctionOutput.raise_exception(
                    output_info="Tool call was violative of policy and was blocked by Hiddenlayer"
                )

            return ToolGuardrailFunctionOutput(output_info="Guardrail check passed")

        return tool_input_gr

    else:  # output

        @tool_output_guardrail
        async def tool_output_gr(data: ToolOutputGuardrailData) -> ToolGuardrailFunctionOutput:
            """Check tool output after execution."""
            check_data = json.dumps(
                {
                    "tool_name": data.context.tool_name,
                    "arguments": data.context.tool_arguments,
                    "output": str(data.output),
                    "call_id": getattr(data.context, "tool_call_id", None),
                }
            )

            analysis = await _analyze_content(check_data, "user", hiddenlayer_params, client)

            if analysis.evaluation and analysis.evaluation.action == HiddenlayerActions.BLOCK:
                return ToolGuardrailFunctionOutput.raise_exception(
                    output_info="Tool call was violative of policy and was blocked by Hiddenlayer"
                )

            return ToolGuardrailFunctionOutput(output_info="Guardrail check passed")

        return tool_output_gr


def _create_input_output_guardrail(
    guardrail_type: str,
    hiddenlayer_params: HiddenLayerParams,
    client: AsyncHiddenLayer | None = None,
):
    """Create an input or output guardrail with HiddenLayer parameters.

    Args:
        guardrail_type: "input" or "output" to determine which guardrail to create
        hiddenlayer_params: HiddenLayer configuration parameters
        client: Optional AsyncHiddenLayer client instance

    Returns:
        Guardrail function decorated with @input_guardrail or @output_guardrail
    """
    if guardrail_type == "input":

        @input_guardrail
        async def hiddenlayer_input_guardrail(
            ctx: RunContextWrapper[None], agent: OpenAIAgent, input: str | list[TResponseInputItem]
        ) -> GuardrailFunctionOutput:
            content = input if isinstance(input, str) else input[0]["content"]

            response = await _analyze_content(content, "user", hiddenlayer_params, client)
            result = _parse_analysis(response, "user")

            return GuardrailFunctionOutput(
                output_info="Blocked by HiddenLayer." if result.block else "Nothing detected.",
                tripwire_triggered=result.block,
            )

        return hiddenlayer_input_guardrail

    else:  # output

        @output_guardrail
        async def hiddenlayer_output_guardrail(
            ctx: RunContextWrapper[None], agent: OpenAIAgent, output: str | list[TResponseInputItem]
        ) -> GuardrailFunctionOutput:
            response = await _analyze_content(str(output), "assistant", hiddenlayer_params, client)
            result = _parse_analysis(response, "assistant")

            return GuardrailFunctionOutput(
                output_info="Blocked by HiddenLayer." if result.block else "Nothing detected.",
                tripwire_triggered=result.block,
            )

        return hiddenlayer_output_guardrail


def _attach_guardrail_to_tools(
    tools: list[Any],
    guardrail: Callable,
    guardrail_type: str,
) -> None:
    """Attach a guardrail to all tools in the list.

    Args:
        tools: List of tool objects to attach the guardrail to
        guardrail: The guardrail function to attach
        guardrail_type: Either "input" or "output" to determine which list to append to
    """
    attr_name = "tool_input_guardrails" if guardrail_type == "input" else "tool_output_guardrails"

    for tool in tools:
        if not hasattr(tool, attr_name) or getattr(tool, attr_name) is None:
            setattr(tool, attr_name, [])

        # Check if this guardrail is already attached (idempotency)
        existing_guardrails = getattr(tool, attr_name)
        if guardrail not in existing_guardrails:
            existing_guardrails.append(guardrail)


def _wrap_mcp_tools_with_guardrails(
    agent: Any,  # agents.Agent type
    tool_input_guardrail: Callable,
    tool_output_guardrail: Callable,
) -> Any:
    """Wrap agent's get_mcp_tools method to attach guardrails to MCP tools.

    Uses defensive monkey patching to gracefully handle OpenAI SDK changes.
    If the method signature changes or doesn't exist, logs a warning and
    returns the agent unchanged (MCP tools won't have guardrails but won't crash).

    MCP tools are discovered dynamically at runtime, so we intercept the
    get_mcp_tools() method to attach guardrails after discovery but before
    tool execution.

    Args:
        agent: The OpenAI Agent instance to wrap
        tool_input_guardrail: Input guardrail to attach to MCP tools
        tool_output_guardrail: Output guardrail to attach to MCP tools

    Returns:
        The same agent instance, with wrapped get_mcp_tools if successful
    """

    if not hasattr(agent, "get_mcp_tools"):
        logger.warning(
            "Agent does not have 'get_mcp_tools' method. "
            "MCP tools will not have guardrails attached. "
            "This may indicate an OpenAI SDK version incompatibility."
        )
        return agent

    original_get_mcp_tools = agent.get_mcp_tools

    # safe wrap the mcp tools function so that if something fails, we don't crash the app
    async def get_mcp_tools_with_guardrails(run_context):
        try:
            mcp_tools = await original_get_mcp_tools(run_context)

            if not isinstance(mcp_tools, list):
                logger.warning(
                    f"Agent.get_mcp_tools returned {type(mcp_tools)}, expected list. "
                    "Signature may have changed. Returning tools without guardrails."
                )
                return mcp_tools

            _attach_guardrail_to_tools(mcp_tools, tool_input_guardrail, "input")
            _attach_guardrail_to_tools(mcp_tools, tool_output_guardrail, "output")

            return mcp_tools

        except TypeError as e:
            logger.error(
                f"Failed to call Agent.get_mcp_tools - signature may have changed: {e}. "
                "MCP tools will not have guardrails. Please check OpenAI SDK compatibility."
            )
            try:
                return await original_get_mcp_tools()
            except Exception:
                return []

        except Exception as e:
            logger.error(f"Unexpected error in MCP tool guardrail wrapper: {e}. " "Falling back to original method.")
            return await original_get_mcp_tools(run_context)

    # Replace method on agent instance
    agent.get_mcp_tools = get_mcp_tools_with_guardrails

    return agent


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
    """Shared redaction logic for both input and output.

    Args:
        content: The content to analyze/redact
        role: "user" for input, "assistant" for output
        params: HiddenLayer configuration parameters
        client: Optional AsyncHiddenLayer client instance

    Returns:
        Original content if clean, redacted content if modified

    Raises:
        InputBlockedError: If input is blocked (role="user")
        OutputBlockedError: If output is blocked (role="assistant")
    """
    response = await _analyze_content(content, role, params, client)
    result = _parse_analysis(response, role)

    if result.block:
        if role == "user":
            raise InputBlockedError("Input blocked by HiddenLayer.")
        else:
            raise OutputBlockedError("Output blocked by HiddenLayer.")

    return result.redacted_content if result.redact and result.redacted_content else content


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
        params: Optional HiddenLayerParams for configuration. If not provided, defaults will be used.
        client: Optional AsyncHiddenLayer client instance

    Returns:
        The original input if no issues, or redacted input if REDACT action

    Raises:
        InputBlockedError: If HiddenLayer returns BLOCK action

    Example:
        ```python
        from hiddenlayer_openai_guardrails import redact_input, HiddenLayerParams

        params = HiddenLayerParams(project_id="my-project")
        redacted = await redact_input(user_input, params=params)
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
        params: Optional HiddenLayerParams for configuration. If not provided, defaults will be used.
        client: Optional AsyncHiddenLayer client instance

    Returns:
        The original output if no issues, or redacted output if REDACT action

    Raises:
        OutputBlockedError: If HiddenLayer returns BLOCK action

    Example:
        ```python
        from hiddenlayer_openai_guardrails import redact_output, HiddenLayerParams

        result = await Runner.run(agent, user_input)
        params = HiddenLayerParams(project_id="my-project")
        redacted = await redact_output(result.final_output, params=params)
        ```
    """
    return await _redact_content(output, "assistant", hiddenlayer_params, client)


async def redact_streamed_output(
    streaming_result: RunResultStreaming,
    hiddenlayer_params: HiddenLayerParams,
    client: AsyncHiddenLayer | None = None,
) -> AsyncIterator[str]:
    """Buffer streamed output, scan it, then replay events if clean.

    This function consumes a streaming result from Runner.run_streamed(),
    buffers all text deltas, scans the complete output through HiddenLayer,
    and if clean, replays the original text deltas as they came in.
    If redaction is needed, yields the redacted content instead.

    Args:
        streaming_result: Result from Runner.run_streamed()
        params: Optional HiddenLayerParams for configuration. If not provided, defaults will be used.
        client: Optional AsyncHiddenLayer client instance

    Yields:
        Original text deltas if clean, or redacted output if modified

    Raises:
        OutputBlockedError: If HiddenLayer blocks the output

    Example:
        ```python
        from hiddenlayer_openai_guardrails import Agent, redact_streamed_output, HiddenLayerParams
        from agents import Runner

        agent = Agent(name="Assistant", instructions="Help users")
        result = Runner.run_streamed(agent, user_input)

        params = HiddenLayerParams(project_id="my-proj")
        async for chunk in redact_streamed_output(result, params=params):
            print(chunk, end="", flush=True)
        ```
    """
    # Buffer all text deltas as they come in
    events = []

    async for event in streaming_result.stream_events():
        events.append(event)

    # Get the complete output
    output = str(streaming_result.final_output) if streaming_result.final_output else ""

    # Scan/redact using existing function (raises OutputBlockedError if blocked)
    redacted = await redact_output(output, hiddenlayer_params=hiddenlayer_params, client=client)

    # If output is clean (unchanged), replay original deltas
    if redacted == output:
        for event in events:
            yield event
    else:
        # Output was redacted, yield the redacted version
        yield redacted


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
        user_input = await redact_input(raw_input, params=params)

        # Run agent - guardrails will block malicious content
        result = await Runner.run(agent, user_input)

        # Optional: redact sensitive content from output
        final_output = await redact_output(result.final_output, params=params)
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

        Raises:
            ImportError: If agents package is not available
        """
        try:
            from agents import Agent
        except ImportError as e:
            raise ImportError(
                "The 'agents' package is required to use GuardrailAgent. Please install it with: pip install openai-agents"
            ) from e

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
        openai_agent = OpenAIAgent(
            name=name,
            instructions=instructions,
            input_guardrails=[input_gr],
            output_guardrails=[output_gr],
            **agent_kwargs,
        )

        # If MCP servers are provided, wrap get_mcp_tools to attach guardrails
        if agent_kwargs.get("mcp_servers"):
            openai_agent = _wrap_mcp_tools_with_guardrails(
                openai_agent,
                tool_input_gr,
                tool_output_gr,
            )

        return openai_agent
