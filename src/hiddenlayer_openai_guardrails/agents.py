from pathlib import Path
from typing import Any, Callable
import json

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
from agents.guardrail import input_guardrail

from hiddenlayer import AsyncHiddenLayer


client = AsyncHiddenLayer()


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

    analysis = await client.interactions.analyze(
        metadata={"model": "openai-guardrails", "requester_id": "openai-guardrail-example"},
        input={"messages": [{"role": "tool", "content": check_data}]},
    )

    if analysis.evaluation and analysis.modified_data.input.messages and analysis.evaluation.action == "Redact":
        output_info = analysis.modified_data.input.messages[-1].content

    if analysis.evaluation and analysis.evaluation.action == "Block":
        message = f"Tool call was violative of policy and was blocked by Hiddenlayer"

        return ToolGuardrailFunctionOutput.raise_exception(output_info=message)
        # else:
        #     return ToolGuardrailFunctionOutput.reject_content(message=message, output_info=result.info)

    return ToolGuardrailFunctionOutput(output_info="Guardrail check passed")


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
        metadata={"model": "openai-guardrails", "requester_id": "openai-guardrail-example"},
        input={"messages": [{"role": "tool", "content": check_data}]},
    )

    if analysis.evaluation and analysis.modified_data.input.messages and analysis.evaluation.action == "Redact":
        output_info = analysis.modified_data.input.messages[-1].content

    if analysis.evaluation and analysis.evaluation.action == "Block":
        message = f"Tool call was violative of policy and was blocked by Hiddenlayer"

        return ToolGuardrailFunctionOutput.raise_exception(output_info=message)
        # else:
        #     return ToolGuardrailFunctionOutput.reject_content(message=message, output_info=result.info)

    return ToolGuardrailFunctionOutput(output_info="Guardrail check passed")


@input_guardrail
async def hiddenlayer_input_guardrail(
    ctx: RunContextWrapper[None], agent: OpenAIAgent, input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    # if project_id:
    #     analysis = hl_client.interactions.analyze(
    #         metadata={"model": model, "requester_id": requester_id},
    #         hl_project_id=project_id,
    #         input={"messages": [{"role": role, "content": content}]},
    #     )
    # else:
    analysis = await client.interactions.analyze(
        metadata={"model": "openai-guardrails", "requester_id": "openai-guardrail-example"},
        input={"messages": [{"role": "user", "content": input}]},
    )
    output_info = "Nothing detected."

    if analysis.evaluation and analysis.evaluation.action == "Block":
        output_info = "Blocked by Hiddenlayer."

    if analysis.evaluation and analysis.modified_data.input.messages and analysis.evaluation.action == "Redact":
        output_info = analysis.modified_data.input.messages[-1].content

    return GuardrailFunctionOutput(
        output_info=output_info,
        tripwire_triggered=(analysis.evaluation and analysis.evaluation.action == "Block") or False,
    )


@output_guardrail
async def hiddenlayer_output_guardrail(
    ctx: RunContextWrapper[None], agent: OpenAIAgent, output: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    # if project_id:
    #     analysis = hl_client.interactions.analyze(
    #         metadata={"model": model, "requester_id": requester_id},
    #         hl_project_id=project_id,
    #         input={"messages": [{"role": role, "content": content}]},
    #     )
    # else:
    analysis = await client.interactions.analyze(
        metadata={"model": "openai-guardrails", "requester_id": "openai-guardrail-example"},
        output={"messages": [{"role": "assistant", "content": str(output)}]},
    )
    output_info = "Nothing detected."

    if analysis.evaluation and analysis.evaluation.action == "Block":
        output_info = "Blocked by Hiddenlayer."

    if analysis.evaluation and analysis.modified_data.input.messages and analysis.evaluation.action == "Redact":
        output_info = analysis.modified_data.input.messages[-1].content

    return GuardrailFunctionOutput(
        output_info=output_info,
        tripwire_triggered=(analysis.evaluation and analysis.evaluation.action == "Block") or False,
    )


def _attach_guardrail_to_tools(tools: list[Any], guardrail: Callable, guardrail_type: str) -> None:
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
        raise_guardrail_errors: bool = False,
        block_on_tool_violations: bool = False,
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
        # try:
        #     from agents import Agent
        # except ImportError as e:
        #     raise ImportError(
        #         "The 'agents' package is required to use GuardrailAgent. Please install it with: pip install openai-agents"
        #     ) from e

        # _ensure_agent_runner_patch()

        # Separate tool-level from agent-level guardrails in each stage
        # preflight_tool, preflight_agent = _separate_tool_level_from_agent_level(stage_guardrails.get("pre_flight", []))
        # input_tool, input_agent = _separate_tool_level_from_agent_level(stage_guardrails.get("input", []))
        # output_tool, output_agent = _separate_tool_level_from_agent_level(stage_guardrails.get("output", []))

        # Create agent-level INPUT guardrails from config
        # input_guardrails = []

        # Add agent-level guardrails from pre_flight and input stages
        # agent_input_stages = []
        # if preflight_agent:
        #     agent_input_stages.append("pre_flight")
        # if input_agent:
        #     agent_input_stages.append("input")

        # if agent_input_stages:
        #     input_guardrails.extend(
        #         _create_agents_guardrails_from_config(
        #             config=config,
        #             stages=agent_input_stages,
        #             guardrail_type="input",
        #             raise_guardrail_errors=raise_guardrail_errors,
        #         )
        #     )

        # # Merge with user-provided input guardrails (config ones run first, then user ones)
        # input_guardrails.extend(user_input_guardrails)

        # Create agent-level OUTPUT guardrails from config
        # output_guardrails = []
        # if output_agent:
        #     output_guardrails = _create_agents_guardrails_from_config(
        #         config=config,
        #         stages=["output"],
        #         guardrail_type="output",
        #         raise_guardrail_errors=raise_guardrail_errors,
        #     )

        # # Merge with user-provided output guardrails (config ones run first, then user ones)
        # output_guardrails.extend(user_output_guardrails)

        # Apply tool-level guardrails
        tools = agent_kwargs.get("tools", [])

        # # Map pipeline stages to tool guardrails:
        # # - pre_flight + input stages → tool_input_guardrail (checks BEFORE tool execution)
        # # - output stage → tool_output_guardrail (checks AFTER tool execution)
        # if tools and (preflight_tool or input_tool or output_tool):
        #     context = _create_default_tool_context()

        #     # pre_flight + input stages → tool_input_guardrail
        #     for guardrail in preflight_tool + input_tool:
        #         tool_input_gr = _create_tool_guardrail(
        #             guardrail=guardrail,
        #             guardrail_type="input",
        #             context=context,
        #             raise_guardrail_errors=raise_guardrail_errors,
        #             block_on_violations=block_on_tool_violations,
        #         )
        _attach_guardrail_to_tools(tools, tool_input_gr, "input")

        #     # output stage → tool_output_guardrail
        #     for guardrail in output_tool:
        #         tool_output_gr = _create_tool_guardrail(
        #             guardrail=guardrail,
        #             guardrail_type="output",
        #             context=context,
        #             raise_guardrail_errors=raise_guardrail_errors,
        #             block_on_violations=block_on_tool_violations,
        #         )
        _attach_guardrail_to_tools(tools, tool_output_gr, "output")

        # Create and return a regular Agent instance with guardrails configured
        return OpenAIAgent(
            name=name,
            instructions=instructions,
            input_guardrails=[hiddenlayer_input_guardrail],
            output_guardrails=[hiddenlayer_output_guardrail],
            **agent_kwargs,
        )
