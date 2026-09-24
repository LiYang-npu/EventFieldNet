"""Shared, auditable infrastructure for the two Stage47 experiment lines."""

from .config import RunnerConfig
from .contracts import DataBundle, PreparedBatch, Stage47Model
from .probability import masked_log_softmax, masked_softmax, triangular_span_mask

__all__ = [
    "DataBundle",
    "PreparedBatch",
    "RunnerConfig",
    "Stage47Model",
    "masked_log_softmax",
    "masked_softmax",
    "triangular_span_mask",
]
