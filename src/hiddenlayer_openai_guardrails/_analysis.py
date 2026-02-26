import logging
from typing import Any, Literal

from ._types import AnalysisResult, HiddenlayerActions

logger = logging.getLogger(__name__)


def parse_analysis(response: Any, role: Literal["user", "assistant"]) -> AnalysisResult:
    """Parse HiddenLayer response into unified result."""
    try:
        action = response.evaluation.action if response.evaluation else None
        block = action == HiddenlayerActions.BLOCK
        redact = action == HiddenlayerActions.REDACT

        redacted_content = None
        if redact and response.modified_data:
            modified = response.modified_data.input if role == "user" else response.modified_data.output
            messages = modified.messages
            if messages:
                redacted_content = messages[-1].content

        return AnalysisResult(block, redact, redacted_content)
    except Exception:
        logger.warning("Failed to parse HiddenLayer analysis response. Treating as non-blocking.")
        return AnalysisResult(False, False, None)

