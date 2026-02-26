from collections.abc import Awaitable, Callable
from typing import Any, Literal

from hiddenlayer import AsyncHiddenLayer

from ._analysis import parse_analysis
from ._normalize import normalize_content
from ._types import HiddenLayerParams, InputBlockedError, OutputBlockedError

AnalyzeContent = Callable[
    [list[dict[str, Any]], Literal["user", "assistant"], HiddenLayerParams, AsyncHiddenLayer | None],
    Awaitable[Any],
]


async def redact_content(
    content: str,
    role: Literal["user", "assistant"],
    params: HiddenLayerParams,
    client: AsyncHiddenLayer | None,
    *,
    analyze_content: AnalyzeContent,
) -> str:
    response = await analyze_content([{"role": role, "content": normalize_content(content)}], role, params, client)
    result = parse_analysis(response, role)

    if result.block:
        if role == "user":
            raise InputBlockedError("Input blocked by HiddenLayer.")
        raise OutputBlockedError("Output blocked by HiddenLayer.")

    return result.redacted_content if result.redact and result.redacted_content else content
