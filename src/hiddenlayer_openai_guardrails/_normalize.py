import json
import logging
from typing import Any

from agents import TResponseInputItem
from pydantic import BaseModel

logger = logging.getLogger(__name__)


def safe_json_dumps(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        return str(value)


def normalize_content(value: Any) -> str:
    """Normalize arbitrary SDK values to a stable string payload for scanner input."""
    if isinstance(value, str):
        return value

    if value is None:
        return ""

    text_attr = getattr(value, "text", None)
    if isinstance(text_attr, str):
        return text_attr

    if isinstance(value, BaseModel):
        if isinstance(getattr(value, "text", None), str):
            return value.text
        return value.model_dump_json()

    if isinstance(value, list):
        text_parts: list[str] = []
        for part in value:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict):
                part_type = str(part.get("type", "")).lower()
                if "text" in part_type and isinstance(part.get("text"), str):
                    text_parts.append(part["text"])
                elif part_type == "text" and isinstance(part.get("content"), str):
                    text_parts.append(part["content"])
            elif hasattr(part, "text") and isinstance(getattr(part, "text"), str):
                text_parts.append(part.text)

        if text_parts:
            return "\n".join(text_parts)
        return safe_json_dumps(value)

    if isinstance(value, dict):
        return safe_json_dumps(value)

    try:
        return str(value)
    except Exception:
        return safe_json_dumps(value)


def safe_parse_tool_arguments(raw_args: str) -> Any:
    try:
        return json.loads(raw_args)
    except json.JSONDecodeError:
        logger.debug("Tool arguments were not valid JSON; scanning raw tool arguments string.")
        return raw_args


def normalize_input_messages(input_data: str | list[TResponseInputItem]) -> list[dict[str, str]]:
    if isinstance(input_data, str):
        return [{"role": "user", "content": input_data}]

    normalized_messages: list[dict[str, str]] = []
    for item in input_data:
        role: str | None = None
        content: Any = None

        if isinstance(item, dict):
            maybe_role = item.get("role")
            if isinstance(maybe_role, str):
                role = maybe_role
            if "content" in item:
                content = item.get("content")
        else:
            maybe_role = getattr(item, "role", None)
            if isinstance(maybe_role, str):
                role = maybe_role
            if hasattr(item, "content"):
                content = getattr(item, "content")

        # Only include message-shaped items. Non-message input items should not
        # be synthesized into empty user messages.
        if role is None or content is None:
            continue

        normalized_messages.append({"role": role, "content": normalize_content(content)})

    return normalized_messages

