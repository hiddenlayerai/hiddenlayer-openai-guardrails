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
pytest tests/test_agents.py::test_hiddenlayer_guardrails_benign_with_tools -v
```

## Architecture

The library provides a single `Agent` class in `src/hiddenlayer_openai_guardrails/agents.py` that acts as a factory returning a standard `agents.Agent` with guardrails pre-configured.

**Key components:**

- `Agent.__new__` - Factory that creates an `agents.Agent` with HiddenLayer guardrails attached
- `_create_tool_guardrail(guardrail_type, hiddenlayer_params)` - Creates tool-level guardrails (input or output) that wrap tool calls
- `_create_input_output_guardrail(guardrail_type, hiddenlayer_params)` - Creates agent-level guardrails (input or output) for user/assistant messages
- `_attach_guardrail_to_tools(tools, guardrail, guardrail_type)` - Attaches guardrails to all tools in a list

**Guardrail flow:**
1. User input → `input_guardrail` → Agent processes
2. Tool call → `tool_input_guardrail` → Tool executes → `tool_output_guardrail`
3. Agent response → `output_guardrail` → Final output

All guardrails use `AsyncHiddenLayer.interactions.analyze` and handle `BLOCK` (halt execution) and `REDACT` (modify content) actions.

## Environment Variables

- `HIDDENLAYER_CLIENT_ID` - HiddenLayer API client ID
- `HIDDENLAYER_CLIENT_SECRET` - HiddenLayer API client secret
