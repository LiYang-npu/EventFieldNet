"""Configurable round1 losses built on the V1 geometry contract.

The default options are equivalent to field_core:
zero-margin ordinal rank, matched-GT endpoint targets, global quality
reduction, and a linear S head. Round1 options add a positive rank margin,
independent max-G endpoint targets, per-query/near-far quality reductions,
and diagnostics. No new objective term is introduced.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor
from torch.nn import functional as F

from field_core.losses import (
    FiveLossTerms,
    GRADE_COUNT,
    GeometryTargets,
    OFFICIAL_THRESHOLDS,
    candidate_geometry,
    endpoint_bce_loss,
    official_ordinal_violation_loss,
    quality_targets,
)


NEAR_IOU_THRESHOLD = 0.30


def _batch_parts(batch: Any) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    if isinstance(batch, Mapping):
        inputs = batch.get("inputs", {})
        targets = batch.get("targets", {})
    else:
        inputs = getattr(batch, "inputs", {})
        targets = getattr(batch, "targets", {})
    if not isinstance(inputs, Mapping) or not isinstance(targets, Mapping):
        raise TypeError("batch inputs and targets must be mappings")
    return inputs, targets


def _target_spans(outputs: Any, batch: Any) -> tuple[Tensor, Tensor]:
    _, targets = _batch_parts(batch)
    spans = targets.get("gt_spans")
    mask = targets.get("gt_span_mask")
    if spans is None:
        count = int(outputs.span_logits.shape[0])
        spans = outputs.span_logits.new_zeros((count, 1, 2))
    if mask is None:
        mask = torch.zeros(spans.shape[:2], dtype=torch.bool, device=spans.device)
    spans = spans.to(device=outputs.span_logits.device, dtype=torch.float32)
    mask = mask.to(device=spans.device).bool()
    if spans.ndim != 3 or spans.shape[-1] != 2 or mask.shape != spans.shape[:2]:
        raise ValueError("gt_spans must be [B,G,2] and gt_span_mask must be [B,G]")
    return spans, mask


def _safe_zero(value: Tensor) -> Tensor:
    finite = value.float().masked_fill(~torch.isfinite(value.float()), 0.0)
    return finite.sum() * 0.0


@dataclass
class IndependentEndpointTargets:
    start: Tensor
    end: Tensor
    start_best_gt: Tensor
    end_best_gt: Tensor
    conflict: Tensor
    valid: Tensor


def independent_endpoint_targets(
    outputs: Any,
    batch: Any,
    geometry: GeometryTargets | None = None,
) -> IndependentEndpointTargets:
    """Build each endpoint target by an independent max over all valid GTs."""

    if geometry is None:
        geometry = candidate_geometry(outputs, batch)
    spans, gt_mask = _target_spans(outputs, batch)
    gt_start = torch.minimum(spans[..., 0], spans[..., 1])
    gt_end = torch.maximum(spans[..., 0], spans[..., 1])
    gt_width = (gt_end - gt_start).clamp_min(1.0e-6)
    denominator = gt_width[:, None, None, :].clamp_min(
        geometry.one_step[:, None, None, None]
    )
    start_quality = (
        1.0
        - (geometry.candidate_start[..., None] - gt_start[:, None, None, :]).abs()
        / denominator
    ).clamp(0.0, 1.0)
    end_quality = (
        1.0
        - (geometry.candidate_end[..., None] - gt_end[:, None, None, :]).abs()
        / denominator
    ).clamp(0.0, 1.0)
    gt_valid = gt_mask[:, None, None, :]
    start_values = start_quality.masked_fill(~gt_valid, -1.0)
    end_values = end_quality.masked_fill(~gt_valid, -1.0)
    start_best, start_index = start_values.max(-1)
    end_best, end_index = end_values.max(-1)
    has_gt = gt_mask.any(1)[:, None, None]
    valid = geometry.valid & has_gt
    conflict = start_index.ne(end_index) & valid
    start = (2.0 * start_best - 1.0).masked_fill(~valid, 0.0)
    end = (2.0 * end_best - 1.0).masked_fill(~valid, 0.0)
    return IndependentEndpointTargets(
        start=start.detach(),
        end=end.detach(),
        start_best_gt=start_index.detach(),
        end_best_gt=end_index.detach(),
        conflict=conflict.detach(),
        valid=valid.detach(),
    )


def _ordinal_row_margin(
    score: Tensor,
    grade: Tensor,
    margin: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Exact all-candidate margin aggregation with O(N log N + N*G) work."""

    if score.numel() == 0:
        zero = _safe_zero(score)
        return zero, {
            "eligible_pair_count": zero.detach(),
            "violating_pair_count": zero.detach(),
            "violating_pair_rate": zero.detach(),
            "weighted_signed_margin": zero.detach(),
            "score_span": zero.detach(),
        }
    score = score.float()
    grade = grade.long()
    order = torch.argsort(score.detach(), stable=True)
    ordered_score = score[order]
    ordered_grade = grade[order]
    bins = torch.arange(GRADE_COUNT, device=score.device)
    one_hot = F.one_hot(ordered_grade, num_classes=GRADE_COUNT).to(score.dtype)
    suffix_count = torch.flip(torch.cumsum(torch.flip(one_hot, (0,)), 0), (0,))
    suffix_sum = torch.flip(
        torch.cumsum(torch.flip(one_hot * ordered_score[:, None], (0,)), 0), (0,)
    )
    # A pair contributes when score_low - score_high + margin is positive.
    # Search boundaries are detached; the gathered sums retain score grads.
    boundary = ordered_score.detach() - float(margin)
    first = torch.searchsorted(ordered_score.detach(), boundary, right=False)
    gather_index = first[:, None].expand(-1, GRADE_COUNT)
    pair_count = suffix_count.gather(0, gather_index)
    pair_sum = suffix_sum.gather(0, gather_index)
    lower_grade = bins[None, :].lt(ordered_grade[:, None])
    reliability = (ordered_grade[:, None] - bins[None, :]).clamp_min(0).to(
        score.dtype
    ) / float(GRADE_COUNT - 1)
    reliability = reliability * lower_grade.to(score.dtype)
    violation = F.relu(pair_sum + (float(margin) - ordered_score[:, None]) * pair_count)
    weighted_violation = (reliability * violation).sum()

    counts = torch.bincount(grade, minlength=GRADE_COUNT).to(score.dtype)
    high, low = bins[:, None], bins[None, :]
    pair_reliability = (high - low).clamp_min(0).to(score.dtype) / float(
        GRADE_COUNT - 1
    )
    weighted_pair_count = (pair_reliability * counts[:, None] * counts[None, :]).sum()
    eligible_pair_count = (
        high.gt(low).to(score.dtype) * counts[:, None] * counts[None, :]
    ).sum()
    violating_pair_count = (lower_grade.to(score.dtype) * pair_count).sum()
    zero = _safe_zero(score)
    if not bool(weighted_pair_count.gt(0.0)):
        return zero, {
            "eligible_pair_count": eligible_pair_count.detach(),
            "violating_pair_count": violating_pair_count.detach(),
            "violating_pair_rate": zero.detach(),
            "weighted_signed_margin": zero.detach(),
            "score_span": (ordered_score.max() - ordered_score.min()).detach(),
        }
    loss = weighted_violation / weighted_pair_count
    return loss, {
        "eligible_pair_count": eligible_pair_count.detach(),
        "violating_pair_count": violating_pair_count.detach(),
        "violating_pair_rate": (
            violating_pair_count / eligible_pair_count.clamp_min(1.0)
        ).detach(),
        "weighted_signed_margin": loss.detach(),
        "score_span": (ordered_score.max() - ordered_score.min()).detach(),
    }


def _row_score_span(score: Tensor, valid: Tensor) -> Tensor:
    values = score.float().flatten(1)
    mask = valid.bool().flatten(1)
    spans = []
    for row_score, row_mask in zip(values, mask):
        selected = row_score[row_mask]
        if selected.numel():
            spans.append(selected.max() - selected.min())
    return torch.stack(spans).mean() if spans else _safe_zero(score).detach()


def official_ordinal_margin_loss(
    score: Tensor,
    iou: Tensor,
    valid: Tensor,
    margin: float = 0.0,
) -> tuple[Tensor, Mapping[str, Tensor]]:
    """Official grade ordering with an optional positive score margin.

    For margin=0 the reviewed V1 implementation is called directly, retaining
    its legal separating subgradient at score ties. Positive margins use
    detached score sorting plus suffix sums and search boundaries, never an
    NxN pair tensor.
    """

    margin = float(margin)
    if margin < 0.0:
        raise ValueError("rank margin must be non-negative")
    if margin == 0.0:
        loss, base_metrics = official_ordinal_violation_loss(score, iou, valid)
        metrics = dict(base_metrics)
        metrics["score_span"] = _row_score_span(score, valid).detach()
        metrics["margin"] = score.new_tensor(0.0).detach()
        return loss, metrics
    if score.shape != iou.shape or score.shape != valid.shape:
        raise ValueError("score, iou, and valid must have matching shapes")
    flat_score = score.float().flatten(1)
    flat_iou = iou.float().flatten(1)
    flat_valid = valid.bool().flatten(1)
    thresholds = flat_iou.new_tensor(OFFICIAL_THRESHOLDS)
    grades = flat_iou.unsqueeze(-1).ge(thresholds).sum(-1).long()
    losses: list[Tensor] = []
    diagnostics = {
        "eligible_pair_count": [],
        "violating_pair_count": [],
        "violating_pair_rate": [],
        "weighted_signed_margin": [],
        "score_span": [],
    }
    active = 0
    for row_score, row_grade, row_valid in zip(flat_score, grades, flat_valid):
        selected_score = row_score[row_valid]
        selected_grade = row_grade[row_valid]
        row_loss, row_metrics = _ordinal_row_margin(
            selected_score, selected_grade, margin
        )
        if bool(row_metrics["eligible_pair_count"].gt(0.0)):
            active += 1
            losses.append(row_loss)
        for name, value in row_metrics.items():
            diagnostics[name].append(value)
    zero = _safe_zero(score)
    loss = torch.stack(losses).mean() if losses else zero

    def mean(values: list[Tensor]) -> Tensor:
        return torch.stack(values).float().mean() if values else zero.detach()

    result = {name: mean(values).detach() for name, values in diagnostics.items()}
    result["active_query_rate"] = score.new_tensor(
        active / max(1, flat_score.shape[0]), dtype=torch.float32
    ).detach()
    result["top1_official_grade"] = zero.detach()
    result["loss"] = loss.detach()
    result["margin"] = score.new_tensor(margin).detach()
    return loss, result


def _tolerant_values(
    prediction: Tensor, target: Tensor, tolerance: float = 0.1
) -> Tensor:
    error = (prediction.float() - target.float()).abs() - float(tolerance)
    return F.relu(error).square()


def _quality_reduction(
    prediction: Tensor,
    target: Tensor,
    valid: Tensor,
    iou: Tensor,
    *,
    per_query: bool,
    stratified: bool,
) -> tuple[Tensor, Mapping[str, Tensor]]:
    """Reduce quality errors with optional query and near/far equal weighting."""

    values = _tolerant_values(prediction, target)
    valid = valid.bool()
    near = valid & iou.float().ge(NEAR_IOU_THRESHOLD)
    far = valid & ~iou.float().ge(NEAR_IOU_THRESHOLD)
    zero = _safe_zero(prediction)

    def selected_mean(mask: Tensor, source: Tensor = values) -> Tensor | None:
        selected = source[mask]
        return selected.mean() if selected.numel() else None

    near_global = selected_mean(near)
    far_global = selected_mean(far)
    if stratified and per_query:
        rows: list[Tensor] = []
        for row in range(values.shape[0]):
            parts = [
                value
                for value in (
                    selected_mean(near[row], values[row]),
                    selected_mean(far[row], values[row]),
                )
                if value is not None
            ]
            if parts:
                rows.append(torch.stack(parts).mean())
        loss = torch.stack(rows).mean() if rows else zero
    elif stratified:
        parts = [value for value in (near_global, far_global) if value is not None]
        loss = torch.stack(parts).mean() if parts else zero
    elif per_query:
        rows = []
        for row in range(values.shape[0]):
            value = selected_mean(valid[row], values[row])
            if value is not None:
                rows.append(value)
        loss = torch.stack(rows).mean() if rows else zero
    else:
        value = selected_mean(valid)
        loss = value if value is not None else zero

    return loss, {
        "near_count": near.sum().float().detach(),
        "far_count": far.sum().float().detach(),
        "near_loss": (
            near_global.detach() if near_global is not None else zero.detach()
        ),
        "far_loss": (far_global.detach() if far_global is not None else zero.detach()),
        "per_query": torch.tensor(float(per_query), device=prediction.device),
        "stratified": torch.tensor(float(stratified), device=prediction.device),
    }


def _validate_weights(weights: Mapping[str, float] | None) -> dict[str, float]:
    default = {
        "rank": 1.0,
        "evidence": 0.1,
        "support": 0.1,
        "transition": 0.1,
        "endpoint": 0.1,
    }
    if weights is None:
        return default
    result = dict(default)
    for name, value in weights.items():
        if name not in result:
            raise ValueError(f"unknown round1 loss weight {name!r}")
        numeric = float(value)
        if not torch.isfinite(torch.tensor(numeric)) or numeric < 0.0:
            raise ValueError(f"loss weight {name!r} must be finite and non-negative")
        result[name] = numeric
    return result


def compute_round1_loss_terms(
    outputs: Any,
    batch: Any,
    *,
    wrong_evidence: Tensor | None = None,
    evidence_pair_mask: Tensor | None = None,
    loss_weights: Mapping[str, float] | None = None,
    rank_margin: float = 0.0,
    transition_target_mode: str = "matched",
    quality_per_query: bool = False,
    quality_stratified: bool = False,
) -> FiveLossTerms:
    """Compute the five existing terms with round1 options."""

    if getattr(outputs, "trifield_output", None) is None:
        raise RuntimeError("outputs.trifield_output is required")
    field = outputs.trifield_output
    geometry = candidate_geometry(outputs, batch)
    valid = geometry.valid
    weights = _validate_weights(loss_weights)
    rank, rank_metrics = official_ordinal_margin_loss(
        field.score, geometry.max_iou, valid, rank_margin
    )
    support_target, matched_start, matched_end = quality_targets(geometry)
    independent = independent_endpoint_targets(outputs, batch, geometry)
    mode = str(transition_target_mode).lower()
    if mode not in {"matched", "independent_max"}:
        raise ValueError(
            "transition_target_mode must be 'matched' or 'independent_max'"
        )
    start_target = independent.start if mode == "independent_max" else matched_start
    end_target = independent.end if mode == "independent_max" else matched_end

    support, support_metrics = _quality_reduction(
        field.support,
        support_target,
        valid,
        geometry.max_iou,
        per_query=bool(quality_per_query),
        stratified=bool(quality_stratified),
    )
    start_loss, start_metrics = _quality_reduction(
        field.transition_start,
        start_target,
        valid,
        geometry.max_iou,
        per_query=bool(quality_per_query),
        stratified=bool(quality_stratified),
    )
    end_loss, end_metrics = _quality_reduction(
        field.transition_end,
        end_target,
        valid,
        geometry.max_iou,
        per_query=bool(quality_per_query),
        stratified=bool(quality_stratified),
    )
    transition = 0.5 * (start_loss + end_loss)
    endpoint = endpoint_bce_loss(outputs, batch)

    positive = valid & geometry.max_iou.ge(0.70)
    if wrong_evidence is not None:
        if wrong_evidence.shape != field.evidence.shape:
            raise ValueError("wrong_evidence must have the same shape as evidence")
        pair_mask = (
            positive
            if evidence_pair_mask is None
            else (positive & evidence_pair_mask[:, None, None].bool())
        )
        selected = (field.evidence - wrong_evidence)[pair_mask]
        evidence = (
            F.relu(0.20 - selected).mean()
            if selected.numel()
            else _safe_zero(field.raw_evidence)
        )
        pair_count = pair_mask.float().sum().detach()
    else:
        evidence = _safe_zero(field.raw_evidence)
        pair_count = field.raw_evidence.new_zeros(())

    total = (
        weights["rank"] * rank
        + weights["evidence"] * evidence
        + weights["support"] * support
        + weights["transition"] * transition
        + weights["endpoint"] * endpoint
    )
    conflict_values = independent.conflict[independent.valid]
    conflict_rate = (
        conflict_values.float().mean().detach()
        if conflict_values.numel()
        else independent.conflict.new_tensor(0.0)
    )
    metrics: dict[str, Tensor | float] = {
        "rank": rank.detach(),
        "evidence": evidence.detach(),
        "support": support.detach(),
        "transition": transition.detach(),
        "endpoint": endpoint.detach(),
        "total": total.detach(),
        "rank/eligible_pair_count": rank_metrics["eligible_pair_count"],
        "rank/violating_pair_rate": rank_metrics["violating_pair_rate"],
        "rank/score_span": rank_metrics["score_span"],
        "rank/margin": rank_metrics["margin"],
        "rank/top1_official_grade": rank_metrics["top1_official_grade"],
        "evidence/positive_pair_count": pair_count,
        "support/valid_candidate_rate": valid.float().mean().detach(),
        "transition/supervision_conflict_rate": conflict_rate,
        "transition/target_mode_independent_max": float(mode == "independent_max"),
        "geometry/multigt_gt_count": _batch_parts(batch)[1]
        .get(
            "gt_span_mask",
            valid.new_zeros((valid.shape[0], 1)),
        )
        .bool()
        .sum(1)
        .float()
        .mean()
        .detach(),
    }
    for prefix, values in (
        ("support", support_metrics),
        ("transition/start", start_metrics),
        ("transition/end", end_metrics),
    ):
        for name, value in values.items():
            metrics[f"{prefix}/{name}"] = value
    for name, value in weights.items():
        metrics[f"loss_weight/{name}"] = float(value)
    return FiveLossTerms(
        rank=rank,
        evidence=evidence,
        support=support,
        transition=transition,
        endpoint=endpoint,
        total=total,
        geometry=geometry,
        support_target=support_target,
        transition_start_target=start_target,
        transition_end_target=end_target,
        metrics=metrics,
    )


__all__ = [
    "NEAR_IOU_THRESHOLD",
    "IndependentEndpointTargets",
    "independent_endpoint_targets",
    "official_ordinal_margin_loss",
    "compute_round1_loss_terms",
]
