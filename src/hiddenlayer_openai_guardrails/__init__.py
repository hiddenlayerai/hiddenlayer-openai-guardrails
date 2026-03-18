from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("hiddenlayer-openai-guardrails")
except PackageNotFoundError:
    __version__ = "unknown"

from hiddenlayer_openai_guardrails._types import (
    InputBlockedError,
    OutputBlockedError,
)
from hiddenlayer_openai_guardrails._v2_client import V2EvaluationResult
from hiddenlayer_openai_guardrails.agents import (
    Agent,
    HiddenLayerParams,
)

__all__ = [
    "__version__",
    "Agent",
    "HiddenLayerParams",
    "InputBlockedError",
    "OutputBlockedError",
    "V2EvaluationResult",
]
