import os
from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel, Field


class HiddenLayerParams(BaseModel):
    """HiddenLayer request metadata and policy routing parameters."""

    model: str | None = None
    project_id: str | None = Field(default_factory=lambda: os.getenv("HIDDENLAYER_PROJECT_ID"))
    requester_id: str = Field(
        default_factory=lambda: os.getenv("HIDDENLAYER_REQUESTER_ID", "hiddenlayer-openai-integration")
    )


class HiddenlayerActions(str, Enum):
    BLOCK = "Block"
    REDACT = "Redact"


class InputBlockedError(Exception):
    """Raised when HiddenLayer blocks the input."""


class OutputBlockedError(Exception):
    """Raised when HiddenLayer blocks the output."""


@dataclass
class AnalysisResult:
    """Unified result from HiddenLayer analysis."""

    block: bool
    redact: bool
    redacted_content: str | None

