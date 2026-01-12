from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable
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
from agents.guardrail import input_guardrail

from agents.models.interface import Model
from hiddenlayer import AsyncHiddenLayer
from pydantic import BaseModel


client = AsyncHiddenLayer()
logger = logging.getLogger(__name__)


class HiddenlayerParams(BaseModel):
    model: str
    project_id: str | None
    requester_id: str


class HiddenlayerActions(str, Enum):
    BLOCK = "Block"
    REDACT = "Redact"


def _create_tool_guardrail(
    guardrail_type: str,
    # context: Any,
    # raise_guardrail_errors: bool,
    # block_on_violations: bool,
    hiddenlayer_params: HiddenlayerParams,
) -> Callable:
    """Create a generic tool-level guardrail wrapper.

    Args:
        guardrail: The configured guardrail
        guardrail_type: "input" (before tool execution) or "output" (after tool execution)
        context: Guardrail context for LLM client
        raise_guardrail_errors: Whether to raise on errors
        block_on_violations: If True, use raise_exception (halt). If False, use reject_content (continue).

    Returns:
        Tool guardrail function decorated with @tool_input_guardrail or @tool_output_guardrail
    """
    try:
        from agents import (
            ToolGuardrailFunctionOutput,
            ToolInputGuardrailData,
            ToolOutputGuardrailData,
            tool_input_guardrail,
            tool_output_guardrail,
        )
    except ImportError as e:
        raise ImportError(
            "The 'agents' package is required for tool guardrails. Please install it with: pip install openai-agents"
        ) from e

    if guardrail_type == "input":

        @tool_input_guardrail
        async def tool_input_gr(data: ToolInputGuardrailData) -> ToolGuardrailFunctionOutput:
            """Check tool call before execution."""

            check_data = json.dumps(
                {
                    "tool_name": data.context.tool_name,
                    "arguments": data.context.tool_arguments,
                    "call_id": getattr(data.context, "tool_call_id", None),
                }
            )

            if hiddenlayer_params.project_id:
                analysis = await client.interactions.analyze(
                    metadata={"model": hiddenlayer_params.model, "requester_id": hiddenlayer_params.requester_id},
                    hl_project_id=hiddenlayer_params.project_id,
                    input={"messages": [{"role": "tool", "content": check_data}]},
                )
            else:
                analysis = await client.interactions.analyze(
                    metadata={"model": hiddenlayer_params.model, "requester_id": hiddenlayer_params.requester_id},
                    input={"messages": [{"role": "tool", "content": check_data}]},
                )

            if (
                analysis.evaluation
                and analysis.modified_data.input.messages
                and analysis.evaluation.action == HiddenlayerActions.REDACT
            ):
                output_info = analysis.modified_data.input.messages[-1].content

            if analysis.evaluation and analysis.evaluation.action == HiddenlayerActions.BLOCK:
                message = f"Tool call was violative of policy and was blocked by Hiddenlayer"

                return ToolGuardrailFunctionOutput.raise_exception(output_info=message)
                # else:
                #     return ToolGuardrailFunctionOutput.reject_content(message=message, output_info=result.info)

            return ToolGuardrailFunctionOutput(output_info="Guardrail check passed")

        return tool_input_gr

    else:  # output

        @tool_output_guardrail
        async def tool_output_gr(data: ToolOutputGuardrailData) -> ToolGuardrailFunctionOutput:
            """Check tool call before execution."""

            check_data = json.dumps(
                {
                    "tool_name": data.context.tool_name,
                    "arguments": data.context.tool_arguments,
                    "output": str(data.output),
                    "call_id": getattr(data.context, "tool_call_id", None),
                }
            )

            analysis = await client.interactions.analyze(
                metadata={"model": hiddenlayer_params.model, "requester_id": hiddenlayer_params.requester_id},
                input={"messages": [{"role": "tool", "content": check_data}]},
            )

            if (
                analysis.evaluation
                and analysis.modified_data.input.messages
                and analysis.evaluation.action == HiddenlayerActions.REDACT
            ):
                output_info = analysis.modified_data.input.messages[-1].content

            if analysis.evaluation and analysis.evaluation.action == HiddenlayerActions.BLOCK:
                message = f"Tool call was violative of policy and was blocked by Hiddenlayer"

                return ToolGuardrailFunctionOutput.raise_exception(output_info=message)

            return ToolGuardrailFunctionOutput(output_info="Guardrail check passed")

        return tool_output_gr


def _create_input_guardrail(
    hiddenlayer_params: HiddenlayerParams,
) -> Callable:
    """Create an input guardrail with HiddenLayer parameters.

    Args:
        hiddenlayer_params: HiddenLayer configuration parameters

    Returns:
        Input guardrail function decorated with @input_guardrail
    """

    @input_guardrail
    async def hiddenlayer_input_guardrail(
        ctx: RunContextWrapper[None], agent: OpenAIAgent, input: str | list[TResponseInputItem]
    ) -> GuardrailFunctionOutput:
        if isinstance(input, str):
            hl_input = {"messages": [{"role": "user", "content": input}]}
        # This gets hit during streaming
        else:
            hl_input = {"messages": [{"role": "user", "content": input[0]["content"]}]}

        if hiddenlayer_params.project_id:
            analysis = await client.interactions.analyze(
                metadata={"model": hiddenlayer_params.model, "requester_id": hiddenlayer_params.requester_id},
                hl_project_id=hiddenlayer_params.project_id,
                input=hl_input,
            )
        else:
            analysis = await client.interactions.analyze(
                metadata={"model": hiddenlayer_params.model, "requester_id": hiddenlayer_params.requester_id},
                input=hl_input,
            )

        output_info = "Nothing detected."

        if analysis.evaluation and analysis.evaluation.action == HiddenlayerActions.BLOCK:
            output_info = "Blocked by Hiddenlayer."

        if (
            analysis.evaluation
            and analysis.modified_data.input.messages
            and analysis.evaluation.action == HiddenlayerActions.REDACT
        ):
            output_info = analysis.modified_data.input.messages[-1].content

        return GuardrailFunctionOutput(
            output_info=output_info,
            tripwire_triggered=(analysis.evaluation and analysis.evaluation.action == HiddenlayerActions.BLOCK)
            or False,
        )

    return hiddenlayer_input_guardrail


def _create_output_guardrail(
    hiddenlayer_params: HiddenlayerParams,
) -> Callable:
    """Create an output guardrail with HiddenLayer parameters.

    Args:
        hiddenlayer_params: HiddenLayer configuration parameters

    Returns:
        Output guardrail function decorated with @output_guardrail
    """

    @output_guardrail
    async def hiddenlayer_output_guardrail(
        ctx: RunContextWrapper[None], agent: OpenAIAgent, output: str | list[TResponseInputItem]
    ) -> GuardrailFunctionOutput:
        if hiddenlayer_params.project_id:
            analysis = await client.interactions.analyze(
                metadata={"model": hiddenlayer_params.model, "requester_id": hiddenlayer_params.requester_id},
                hl_project_id=hiddenlayer_params.project_id,
                output={"messages": [{"role": "assistant", "content": str(output)}]},
            )
        else:
            analysis = await client.interactions.analyze(
                metadata={"model": hiddenlayer_params.model, "requester_id": hiddenlayer_params.requester_id},
                output={"messages": [{"role": "assistant", "content": str(output)}]},
            )

        output_info = "Nothing detected."

        if analysis.evaluation and analysis.evaluation.action == HiddenlayerActions.BLOCK:
            output_info = "Blocked by Hiddenlayer."

        if (
            analysis.evaluation
            and analysis.modified_data.input.messages
            and analysis.evaluation.action == HiddenlayerActions.REDACT
        ):
            output_info = analysis.modified_data.input.messages[-1].content

        return GuardrailFunctionOutput(
            output_info=output_info,
            tripwire_triggered=(analysis.evaluation and analysis.evaluation.action == HiddenlayerActions.BLOCK)
            or False,
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
        getattr(tool, attr_name).append(guardrail)


def _parse_model(model: str | Model | None):
    if not model:
        return models.get_default_model()

    return str(model)


class Agent:
    """Drop-in replacement for Agents SDK Agent with automatic guardrails integration.


    This class acts as a factory that creates a regular Agents SDK Agent instance
    with guardrails automatically configured from a pipeline configuration.


    Prompt Injection Detection guardrails are applied at the tool level (before and
    after each tool call), while other guardrails run at the agent level.


    When you supply an Agents Session via ``Runner.run(..., session=...)`` the
    guardrails automatically read the persisted conversation history. Without a
    session, guardrails operate on the conversation passed to ``Runner.run`` for
    the current turn.


    Example:
        ```python
        from guardrails import GuardrailAgent
        from agents import Runner, function_tool




        @function_tool
        def get_weather(location: str) -> str:
            return f"Weather in {location}: Sunny"




        agent = GuardrailAgent(
            config="guardrails_config.json",
            name="Weather Assistant",
            instructions="You help with weather information.",
            tools=[get_weather],
        )


        # Use with Agents SDK Runner - prompt injection checks run on each tool call
        result = await Runner.run(agent, "What's the weather in Tokyo?")
        ```
    """

    def __new__(
        cls,
        name: str,
        instructions: str | Callable[[Any, Any], Any] | None = None,
        hiddenlayer_project_id: str | None = None,
        hiddenlayer_requester_id: str = "openai-agent-sdk",
        **agent_kwargs: Any,
    ) -> Any:  # Returns agents.Agent
        """Create a new Agent instance with guardrails automatically configured.


        This method acts as a factory that:
        1. Loads the pipeline configuration
        2. Separates tool-level from agent-level guardrails
        3. Applies agent-level guardrails as input/output guardrails
        4. Applies tool-level guardrails (e.g., Prompt Injection Detection) to all tools:
           - pre_flight + input stages → tool_input_guardrail (before tool execution)
           - output stage → tool_output_guardrail (after tool execution)
        5. Returns a regular Agent instance ready for use with Runner.run()


        Args:
            config: Pipeline configuration (file path, dict, or JSON string)
            name: Agent name
            instructions: Agent instructions. Can be a string, a callable that dynamically
                generates instructions, or None. If a callable, it will receive the context
                and agent instance and must return a string.
            raise_guardrail_errors: If True, raise exceptions when guardrails fail to execute.
                If False (default), treat guardrail errors as safe and continue execution.
            block_on_tool_violations: If True, tool guardrail violations raise exceptions (halt execution).
                If False (default), violations use reject_content (agent can continue and explain).
                Note: Agent-level input/output guardrails always block regardless of this setting.
            **agent_kwargs: All other arguments passed to Agent constructor (including tools)


        Returns:
            agents.Agent: A fully configured Agent instance with guardrails


        Raises:
            ImportError: If agents package is not available
            ConfigError: If configuration is invalid
            Exception: If raise_guardrail_errors=True and a guardrail fails to execute
        """
        try:
            from agents import Agent
        except ImportError as e:
            raise ImportError(
                "The 'agents' package is required to use GuardrailAgent. Please install it with: pip install openai-agents"
            ) from e

        # Apply tool-level guardrails
        tools = agent_kwargs.get("tools", [])
        model = agent_kwargs.pop(
            "model",
        )

        model = _parse_model(model)

        hiddenlayer_params = HiddenlayerParams(
            project_id=hiddenlayer_project_id, requester_id=hiddenlayer_requester_id, model=model
        )
        tool_input_gr = _create_tool_guardrail("input", hiddenlayer_params)
        tool_output_gr = _create_tool_guardrail("output", hiddenlayer_params)
        _attach_guardrail_to_tools(tools, tool_input_gr, "input")
        _attach_guardrail_to_tools(tools, tool_output_gr, "output")

        # Create input/output guardrails with HiddenLayer params
        input_gr = _create_input_guardrail(hiddenlayer_params)
        output_gr = _create_output_guardrail(hiddenlayer_params)

        # Create and return a regular Agent instance with guardrails configured

        return OpenAIAgent(
            name=name,
            instructions=instructions,
            input_guardrails=[input_gr],
            output_guardrails=[output_gr],
            **agent_kwargs,
        )
