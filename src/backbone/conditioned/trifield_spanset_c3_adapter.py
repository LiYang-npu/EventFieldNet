"""Thin dense-grid/C3 adapter for the training tri-field span-set method."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import torch
from torch import Tensor, nn

try:
    from span_fields.trifield_span_set import TriFieldSequence, TriFieldSpanSetScorer
    from span_fields.trifield_span_set_objective import (
        SpanSetObjectiveResult,
        TriFieldSpanSetObjective,
    )
except ImportError:  # Standalone smoke from the local work directory.
    from trifield_span_set import TriFieldSequence, TriFieldSpanSetScorer
    from trifield_span_set_objective import (
        SpanSetObjectiveResult,
        TriFieldSpanSetObjective,
    )


@dataclass(frozen=True)
class DenseSpanSetView:
    spans_xx: Tensor
    logits: Tensor
    valid: Tensor
    grid_shape: tuple[int, int]


def dense_grid_view(
    span_logits: Tensor,
    span_valid_mask: Tensor,
    frame_valid: Optional[Tensor] = None,
) -> DenseSpanSetView:
    """Expose a start/end score grid as an ordinary set of complete spans."""

    if span_logits.ndim != 3 or span_valid_mask.shape != span_logits.shape:
        raise ValueError("dense span logits and validity must have shape [batch, L, L]")
    batch, starts, ends = span_logits.shape
    if starts != ends:
        raise ValueError("the current C3 adapter expects a square start/end grid")
    if frame_valid is None:
        lengths = torch.full(
            (batch,), starts, dtype=torch.float32, device=span_logits.device
        )
    else:
        lengths = frame_valid.bool().sum(1).clamp_min(1).float()
    start_index = torch.arange(starts, device=span_logits.device).float()
    end_index = torch.arange(ends, device=span_logits.device).float() + 1.0
    start = start_index[None, :, None] / lengths[:, None, None]
    end = end_index[None, None, :] / lengths[:, None, None]
    start, end = torch.broadcast_tensors(start, end)
    spans = torch.stack((start, end), -1).flatten(1, 2).clamp(0.0, 1.0)
    return DenseSpanSetView(
        spans_xx=spans,
        logits=span_logits.flatten(1),
        valid=span_valid_mask.bool().flatten(1),
        grid_shape=(starts, ends),
    )


class TriFieldC3SpanSetAdapter(nn.Module):
    """Apply the identical three-field scorer to a non-DETR dense span grid."""

    def __init__(
        self,
        scorer: Optional[TriFieldSpanSetScorer] = None,
        objective: Optional[TriFieldSpanSetObjective] = None,
    ) -> None:
        super().__init__()
        self.scorer = TriFieldSpanSetScorer() if scorer is None else scorer
        self.objective = (
            TriFieldSpanSetObjective(
                semantic_score_weight=1.0, quality_score_weight=0.0
            )
            if objective is None
            else objective
        )

    def forward(
        self,
        *,
        span_logits: Tensor,
        span_valid_mask: Tensor,
        fields: TriFieldSequence,
        wrong_query_fields: Optional[TriFieldSequence] = None,
    ) -> Dict[str, Any]:
        view = dense_grid_view(span_logits, span_valid_mask, fields.valid_mask)
        rank = self.scorer(
            fields,
            view.spans_xx,
            view.valid,
            wrong_query_fields=wrong_query_fields,
        )
        # C3 has a single deployed span score. Role routing remains distinct in
        # the shared scorer; the adapter only maps both calibrated residuals to
        # that backend's one score channel.
        corrected = view.logits + rank["semantic_residual"] + rank["quality_residual"]
        return {
            "span_logits": corrected.view(span_logits.shape),
            "view": view,
            "trifield_span_set": rank,
        }

    def compute_objective(
        self,
        adapted: Dict[str, Any],
        targets_xx: Sequence[Tensor],
    ) -> SpanSetObjectiveResult:
        view: DenseSpanSetView = adapted["view"]
        rank: Dict[str, Tensor] = adapted["trifield_span_set"]
        return self.objective(
            spans_xx=view.spans_xx,
            semantic_logits=adapted["span_logits"].flatten(1),
            targets_xx=targets_xx,
            candidate_valid=view.valid,
            evidence_score=rank["evidence_score"],
            wrong_query_evidence_score=rank.get("wrong_query_evidence_score"),
        )


__all__ = ["DenseSpanSetView", "TriFieldC3SpanSetAdapter", "dense_grid_view"]
