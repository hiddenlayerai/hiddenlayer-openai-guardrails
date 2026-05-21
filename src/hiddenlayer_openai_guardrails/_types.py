import os
from enum import Enum

from pydantic import BaseModel, Field


class HiddenLayerParams(BaseModel):
    """HiddenLayer request metadata and policy routing parameters."""

    model: str | None = None
    project_id: str | None = Field(default_factory=lambda: os.getenv("HIDDENLAYER_PROJECT_ID"))
    requester_id: str = Field(
        default_factory=lambda: os.getenv("HIDDENLAYER_REQUESTER_ID", "hiddenlayer-openai-integration")
    )
    session_id: str | None = Field(
        default=None,
        description=(
            "Per-conversation correlation ID. Forwarded as the hl-runtime-session-id "
            "header on every request/response evaluation so the HiddenLayer console "
            "can group all turns of a conversation together."
        ),
    )
    exclude_tools_from_evaluation: bool = Field(
        default=False,
        description=(
            "When True, tool definitions are stripped from the request payload "
            "sent to HiddenLayer for evaluation. The downstream OpenAI call still "
            "receives the full tools list. Useful when tool descriptions trigger "
            "false positives in injection detectors and you only want HL to "
            "evaluate user-controlled content."
        ),
    )


class HiddenlayerActions(str, Enum):
    BLOCK = "Block"
    REDACT = "Redact"


class InputBlockedError(Exception):
    """Raised when HiddenLayer blocks the input."""


class OutputBlockedError(Exception):
    """Raised when HiddenLayer blocks the output."""
