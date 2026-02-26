import logging
from typing import Any, Literal

from hiddenlayer import AsyncHiddenLayer

from ._normalize import normalize_content
from ._types import HiddenLayerParams

logger = logging.getLogger(__name__)

_DEFAULT_HIDDENLAYER_CLIENT: AsyncHiddenLayer | None = None


def get_hiddenlayer_client(client: AsyncHiddenLayer | None = None) -> AsyncHiddenLayer:
    """Return a caller-supplied client or a lazily initialized shared client."""
    if client is not None:
        return client

    global _DEFAULT_HIDDENLAYER_CLIENT
    if _DEFAULT_HIDDENLAYER_CLIENT is None:
        _DEFAULT_HIDDENLAYER_CLIENT = AsyncHiddenLayer()
    return _DEFAULT_HIDDENLAYER_CLIENT


async def analyze_content(
    messages: list[dict[str, Any]],
    role: Literal["user", "assistant"],
    hiddenlayer_params: HiddenLayerParams,
    client: AsyncHiddenLayer | None = None,
) -> Any:
    """Single location for all HiddenLayer API calls."""
    client = get_hiddenlayer_client(client)

    metadata: dict[str, Any] = {
        "requester_id": hiddenlayer_params.requester_id,
        "model": hiddenlayer_params.model or "unknown",
        "provider": "openai",
    }

    normalized_messages = [
        {"role": str(msg.get("role", role)), "content": normalize_content(msg.get("content", ""))} for msg in messages
    ]
    message_payload = {"messages": normalized_messages}

    kwargs: dict[str, Any] = {"metadata": metadata}
    if hiddenlayer_params.project_id:
        kwargs["hl_project_id"] = hiddenlayer_params.project_id

    if role == "user":
        kwargs["input"] = message_payload
    else:
        kwargs["output"] = message_payload

    return await client.interactions.analyze(**kwargs)

