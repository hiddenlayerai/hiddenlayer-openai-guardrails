import logging
import hashlib
from collections.abc import Awaitable, Callable
from typing import Any, Literal

import agents
from agents import (
    Agent as OpenAIAgent,
    GuardrailFunctionOutput,
    RunContextWrapper,
    ToolGuardrailFunctionOutput,
    ToolInputGuardrailData,
    ToolOutputGuardrailData,
    TResponseInputItem,
    output_guardrail,
    tool_input_guardrail,
    tool_output_guardrail,
)
from agents.guardrail import InputGuardrail, OutputGuardrail, input_guardrail
from hiddenlayer import AsyncHiddenLayer

from ._analysis import parse_analysis
from ._normalize import normalize_content, normalize_input_messages, safe_json_dumps, safe_parse_tool_arguments
from ._types import HiddenlayerActions, HiddenLayerParams, InputBlockedError

logger = logging.getLogger(__name__)

AnalyzeContent = Callable[
    [list[dict[str, Any]], Literal["user", "assistant"], HiddenLayerParams, AsyncHiddenLayer | None],
    Awaitable[Any],
]

_THREAD_ID_KEYS = (
    "conversation_id",
    "conversationId",
    "thread_id",
    "threadId",
    "session_id",
    "sessionId",
)
# (agent_id, thread_key) -> {scan_key: blocked}
_SCAN_DECISION_CACHE: dict[tuple[int, str], dict[str, bool]] = {}


def _extract_thread_key(context: Any) -> str:
    if context is None:
        return "default"

    if isinstance(context, dict):
        for key in _THREAD_ID_KEYS:
            value = context.get(key)
            if value is not None:
                return str(value)
        return hashlib.sha256(safe_json_dumps(context).encode("utf-8")).hexdigest()

    for key in _THREAD_ID_KEYS:
        value = getattr(context, key, None)
        if value is not None:
            return str(value)

    return hashlib.sha256(safe_json_dumps(context).encode("utf-8")).hexdigest()


def _scan_cache_key(phase_role: Literal["user", "assistant"], message_role: str, content: str) -> str:
    payload = f"{phase_role}\u241f{message_role}\u241f{content}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


async def _scan_message_once(
    *,
    agent: Any,
    context: Any,
    phase_role: Literal["user", "assistant"],
    message_role: str,
    content: Any,
    hiddenlayer_params: HiddenLayerParams,
    client: AsyncHiddenLayer | None,
    analyze_content: AnalyzeContent,
) -> bool:
    normalized_content = normalize_content(content)
    thread_key = _extract_thread_key(context)
    namespace_id = id(agent) if agent is not None else (id(context) if context is not None else id(hiddenlayer_params))
    cache_bucket = _SCAN_DECISION_CACHE.setdefault((namespace_id, thread_key), {})
    cache_key = _scan_cache_key(phase_role, message_role, normalized_content)

    cached = cache_bucket.get(cache_key)
    if cached is not None:
        return cached

    analysis = await analyze_content(
        [{"role": message_role, "content": normalized_content}],
        phase_role,
        hiddenlayer_params,
        client,
    )
    result = parse_analysis(analysis, phase_role)
    cache_bucket[cache_key] = result.block
    return result.block


def create_tool_guardrail(
    guardrail_type: str,
    hiddenlayer_params: HiddenLayerParams,
    client: AsyncHiddenLayer | None,
    *,
    analyze_content: AnalyzeContent,
) -> Callable:
    """Create a generic tool-level guardrail wrapper."""
    if guardrail_type == "input":

        @tool_input_guardrail
        async def tool_input_gr(data: ToolInputGuardrailData) -> ToolGuardrailFunctionOutput:
            parsed_arguments = safe_parse_tool_arguments(data.context.tool_arguments)
            blocked = await _scan_message_once(
                agent=data.agent,
                context=data.context.context,
                phase_role="assistant",
                message_role="assistant",
                content=parsed_arguments,
                hiddenlayer_params=hiddenlayer_params,
                client=client,
                analyze_content=analyze_content,
            )

            if blocked:
                return ToolGuardrailFunctionOutput.raise_exception(
                    output_info="Tool call was violative of policy and was blocked by Hiddenlayer"
                )

            return ToolGuardrailFunctionOutput(output_info="Guardrail check passed")

        return tool_input_gr

    @tool_output_guardrail
    async def tool_output_gr(data: ToolOutputGuardrailData) -> ToolGuardrailFunctionOutput:
        blocked = await _scan_message_once(
            agent=data.agent,
            context=data.context.context,
            phase_role="user",
            message_role="user",
            content=data.output,
            hiddenlayer_params=hiddenlayer_params,
            client=client,
            analyze_content=analyze_content,
        )

        if blocked:
            return ToolGuardrailFunctionOutput.raise_exception(
                output_info="Tool call was violative of policy and was blocked by Hiddenlayer"
            )

        return ToolGuardrailFunctionOutput(output_info="Guardrail check passed")

    return tool_output_gr


def create_input_output_guardrail(
    guardrail_type: str,
    hiddenlayer_params: HiddenLayerParams,
    client: AsyncHiddenLayer | None,
    *,
    analyze_content: AnalyzeContent,
) -> InputGuardrail | OutputGuardrail:
    """Create an input or output guardrail with HiddenLayer parameters."""
    if guardrail_type == "input":

        @input_guardrail
        async def hiddenlayer_input_guardrail(
            ctx: RunContextWrapper[None], agent: OpenAIAgent, input: str | list[TResponseInputItem]
        ) -> GuardrailFunctionOutput:
            for message in normalize_input_messages(input):
                blocked = await _scan_message_once(
                    agent=agent,
                    context=ctx.context,
                    phase_role="user",
                    message_role=str(message.get("role", "user")),
                    content=message.get("content", ""),
                    hiddenlayer_params=hiddenlayer_params,
                    client=client,
                    analyze_content=analyze_content,
                )
                if blocked:
                    return GuardrailFunctionOutput(
                        output_info="Blocked by HiddenLayer.",
                        tripwire_triggered=True,
                    )

            return GuardrailFunctionOutput(
                output_info="Nothing detected.",
                tripwire_triggered=False,
            )

        return hiddenlayer_input_guardrail

    @output_guardrail
    async def hiddenlayer_output_guardrail(
        ctx: RunContextWrapper[None], agent: OpenAIAgent, output: Any
    ) -> GuardrailFunctionOutput:
        blocked = await _scan_message_once(
            agent=agent,
            context=ctx.context,
            phase_role="assistant",
            message_role="assistant",
            content=output,
            hiddenlayer_params=hiddenlayer_params,
            client=client,
            analyze_content=analyze_content,
        )

        return GuardrailFunctionOutput(
            output_info="Blocked by HiddenLayer." if blocked else "Nothing detected.",
            tripwire_triggered=blocked,
        )

    return hiddenlayer_output_guardrail


async def scan_tool_definition(
    tool: Any,
    hiddenlayer_params: HiddenLayerParams,
    client: AsyncHiddenLayer | None,
    *,
    analyze_content: AnalyzeContent,
) -> None:
    """Scan a tool's definition for security issues at registration time."""
    tool_data = {
        "tool_name": tool.name if hasattr(tool, "name") else str(tool),
        "tool_description": getattr(tool, "description", ""),
    }

    if hasattr(tool, "params_json_schema"):
        tool_data["params_schema"] = tool.params_json_schema

    analysis = await analyze_content(
        [{"role": "user", "content": safe_json_dumps(tool_data)}],
        "user",
        hiddenlayer_params,
        client,
    )

    if analysis.evaluation and analysis.evaluation.action == HiddenlayerActions.BLOCK:
        logger.error(
            f"Tool '{tool_data['tool_name']}' blocked at registration: violates policy. "
            "Tool will not be available for use."
        )
        raise InputBlockedError(f"Tool '{tool_data['tool_name']}' registration blocked by HiddenLayer policy")


def attach_guardrail_to_tools(tools: list[Any], guardrail: Callable, guardrail_type: str) -> None:
    """Attach a guardrail to all tools in the list (idempotent)."""
    attr_name = "tool_input_guardrails" if guardrail_type == "input" else "tool_output_guardrails"

    for tool in tools:
        if not hasattr(tool, attr_name) or getattr(tool, attr_name) is None:
            setattr(tool, attr_name, [])

        existing_guardrails = getattr(tool, attr_name)
        if guardrail not in existing_guardrails:
            existing_guardrails.append(guardrail)


def wrap_mcp_tools_with_guardrails(
    agent: agents.Agent,
    tool_input_guardrail: Callable,
    tool_output_guardrail: Callable,
    hiddenlayer_params: HiddenLayerParams,
    hiddenlayer_client: AsyncHiddenLayer | None,
    *,
    analyze_content: AnalyzeContent,
    scan_tool_definition_fn: Callable[[Any, HiddenLayerParams, AsyncHiddenLayer | None], Awaitable[None]] | None = None,
) -> Any:
    """Wrap agent.get_mcp_tools() to scan and attach guardrails to discovered MCP tools."""
    if not hasattr(agent, "get_mcp_tools"):
        logger.warning(
            "Agent does not have 'get_mcp_tools' method. "
            "MCP tools will not have guardrails attached. "
            "This may indicate an OpenAI SDK version incompatibility."
        )
        return agent

    original_get_mcp_tools = agent.get_mcp_tools
    scan_cache: dict[str, bool] = {}

    def _tool_cache_key(tool: Any) -> str:
        tool_name = getattr(tool, "name", repr(tool))
        params_schema = getattr(tool, "params_json_schema", None)
        schema_blob = safe_json_dumps(params_schema) if params_schema is not None else ""
        return f"{tool_name}:{schema_blob}"

    async def get_mcp_tools_with_guardrails(run_context):
        try:
            mcp_tools = await original_get_mcp_tools(run_context)

            if not isinstance(mcp_tools, list):
                logger.warning(
                    f"Agent.get_mcp_tools returned {type(mcp_tools)}, expected list. "
                    "Signature may have changed. Returning no MCP tools (fail-closed)."
                )
                return []

            allowed_tools: list[Any] = []
            for tool in mcp_tools:
                key = _tool_cache_key(tool)
                cached_allowed = scan_cache.get(key)

                if cached_allowed is False:
                    continue

                if cached_allowed is None:
                    try:
                        if scan_tool_definition_fn is None:
                            await scan_tool_definition(
                                tool,
                                hiddenlayer_params,
                                hiddenlayer_client,
                                analyze_content=analyze_content,
                            )
                        else:
                            await scan_tool_definition_fn(tool, hiddenlayer_params, hiddenlayer_client)
                        scan_cache[key] = True
                    except InputBlockedError:
                        scan_cache[key] = False
                        continue
                    except Exception as e:
                        logger.error(f"Skipping MCP tool due to scan error: {e}")
                        scan_cache[key] = False
                        continue

                allowed_tools.append(tool)

            attach_guardrail_to_tools(allowed_tools, tool_input_guardrail, "input")
            attach_guardrail_to_tools(allowed_tools, tool_output_guardrail, "output")

            return allowed_tools
        except Exception as e:
            logger.error(f"Unexpected error in MCP tool guardrail wrapper: {e}. Returning no MCP tools (fail-closed).")
            return []

    agent.get_mcp_tools = get_mcp_tools_with_guardrails
    return agent
