from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Any

import httpx
from hiddenlayer import AsyncHiddenLayer

from ._types import HiddenLayerParams

logger = logging.getLogger(__name__)

REQUEST_EVALUATION_PATH = "/detection/v2/request-evaluations"
RESPONSE_EVALUATION_PATH = "/detection/v2/response-evaluations"


@dataclass
class V2EvaluationResult:
    """Result from a v2 evaluation endpoint."""

    action: str | None
    payload: dict[str, Any]
    blocked: bool
    roundtrip_id: str = ""


async def evaluate(
    client: AsyncHiddenLayer,
    path: str,
    payload: dict[str, Any],
    params: HiddenLayerParams,
    *,
    roundtrip_id: str | None = None,
    session_id: str | None = None,
) -> V2EvaluationResult:
    if roundtrip_id is None:
        roundtrip_id = str(uuid.uuid4())

    headers: dict[str, str] = {
        "Hl-Roundtrip-Id": roundtrip_id,
    }
    if params.project_id:
        headers["HL-Project-Id"] = params.project_id
    if params.requester_id:
        headers["hl-requester-id"] = params.requester_id
    if session_id:
        headers["hl-runtime-session-id"] = session_id

    logger.debug("v2 %s  roundtrip_id=%s", path, roundtrip_id)

    response = await client.post(
        path,
        cast_to=httpx.Response,
        body=payload,
        options={"headers": headers},
    )

    action = response.headers.get("hl-runtime-action") or None
    body = response.json()

    return V2EvaluationResult(
        action=action,
        payload=body if isinstance(body, dict) else {},
        blocked=action is not None and action.upper() == "BLOCK",
        roundtrip_id=roundtrip_id,
    )
