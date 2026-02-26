# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Drop-in replacement for the OpenAI Agents SDK `Agent` that automatically wires HiddenLayer guardrails into agent and tool execution. Agent input/output and every tool call are sent through HiddenLayer's analyze endpoint to catch prompt-injection and policy violations.

## Commands

```bash
# Install dependencies (uses uv)
uv sync

# Run all tests
pytest tests

# Run a single test
pytest tests/test_agents_unit.py::test_normalize_content_variants -v
```

## Architecture

The library provides a single `Agent` class in `src/hiddenlayer_openai_guardrails/agents.py` that acts as a factory returning a standard `agents.Agent` with guardrails pre-configured.

**Key components:**

- `Agent.__new__` - Factory that creates an `agents.Agent` with HiddenLayer guardrails attached
- `_normalize_content(value)` - Normalizes any SDK value to a plain string for HiddenLayer analysis, preserving the text the LLM would see
- `_normalize_input_messages(input)` - Converts agent input (string or message list) into `[{role, content}]` for the interactions API
- `_analyze_content(messages, role, params)` - Sends normalized messages to HiddenLayer for analysis
- `_create_tool_guardrail(guardrail_type, hiddenlayer_params)` - Creates tool-level guardrails (input or output) that wrap tool calls
- `_create_input_output_guardrail(guardrail_type, hiddenlayer_params)` - Creates agent-level guardrails (input or output) for user/assistant messages
- `_attach_guardrail_to_tools(tools, guardrail, guardrail_type)` - Attaches guardrails to all tools in a list
- `_wrap_mcp_tools_with_guardrails(agent, ...)` - Intercepts MCP tool discovery to scan and attach guardrails; fail-closed with per-tool scan caching

**Guardrail flow:**
1. User input → `_normalize_input_messages` → `input_guardrail` → Agent processes
2. Tool call → `_normalize_content(args)` → `tool_input_guardrail` → Tool executes → `_normalize_content(output)` → `tool_output_guardrail`
3. Agent response → `_normalize_content(output)` → `output_guardrail` → Final output

All guardrails use `AsyncHiddenLayer.interactions.analyze` and enforce `BLOCK` (halt execution). Agent-level guardrails also support `REDACT` via `redact_input()`/`redact_output()`. Tool guardrails enforce block-only behavior.

## Environment Variables

- `HIDDENLAYER_CLIENT_ID` - HiddenLayer API client ID
- `HIDDENLAYER_CLIENT_SECRET` - HiddenLayer API client secret
