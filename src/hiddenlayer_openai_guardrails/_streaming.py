import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, Literal

from agents.result import RunResultStreaming
from agents.stream_events import StreamEvent
from hiddenlayer import AsyncHiddenLayer

from ._analysis import parse_analysis
from ._normalize import normalize_content
from ._types import HiddenLayerParams, OutputBlockedError

logger = logging.getLogger(__name__)

AnalyzeContent = Callable[
    [list[dict[str, Any]], Literal["user", "assistant"], HiddenLayerParams, AsyncHiddenLayer | None],
    Awaitable[Any],
]


async def safe_stream(
    streaming_result: RunResultStreaming,
    hiddenlayer_params: HiddenLayerParams,
    client: AsyncHiddenLayer | None,
    *,
    analyze_content: AnalyzeContent,
) -> AsyncIterator[StreamEvent]:
    async for event in streaming_result.stream_events():
        yield event

    output = normalize_content(streaming_result.final_output) if streaming_result.final_output else ""

    try:
        response = await analyze_content(
            [{"role": "assistant", "content": output}],
            "assistant",
            hiddenlayer_params,
            client,
        )
        result = parse_analysis(response, "assistant")
        if result.block:
            raise OutputBlockedError("Output blocked by HiddenLayer.")
    except OutputBlockedError:
        raise
    except Exception:
        logger.info("HiddenLayer scan failed during safe_stream final output scan")
