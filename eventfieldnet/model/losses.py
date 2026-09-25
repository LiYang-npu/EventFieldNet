"""Round31 five-term objective.

The objective has exactly five terms. rank dispatches between the reviewed
all-candidate ordinal margin and a GT-balanced KL target; the latter is a
direct replacement for the rank term and never adds a second objective.
The E/S/T/endpoint terms use deployment tensors and the unchanged geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Mapping

import torch
from torch import Tensor
from torch.nn import functional as F

from .local_evidence_loss import explicit_local_ordinal_loss

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
ROUND31_RANK_MARGIN = 0.10
ROUND31_GT_WEIGHT_THRESHOLD = 0.50
ROUND31_TOLERANCE = 0.10


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


def _pad_empty_gt(batch: Any) -> Any:
    import copy

    inputs, targets = _batch_parts(batch)
    spans = targets.get("gt_spans")
    if spans is None or spans.shape[1] != 0:
        return batch
    targets = dict(targets)
    targets["gt_spans"] = spans.new_zeros((spans.shape[0], 1, 2))
    targets["gt_span_mask"] = torch.zeros(
        (spans.shape[0], 1), dtype=torch.bool, device=spans.device
    )
    if isinstance(batch, Mapping):
        return dict(batch, targets=targets)
    replacement = copy.copy(batch)
    replacement.targets = targets
    return replacement


def _target_spans(outputs: Any, batch: Any) -> tuple[Tensor, Tensor]:
    _, targets = _batch_parts(batch)
    spans = targets.get("gt_spans")
    mask = targets.get("gt_span_mask")
    if spans is None:
        b = int(outputs.span_logits.shape[0])
        spans = outputs.span_logits.new_zeros((b, 1, 2))
    if mask is None:
        mask = torch.zeros(spans.shape[:2], dtype=torch.bool, device=spans.device)
    spans = spans.to(device=outputs.span_logits.device, dtype=torch.float32)
    mask = mask.to(device=spans.device).bool()
    if spans.ndim != 3 or spans.shape[-1] != 2 or mask.shape != spans.shape[:2]:
        raise ValueError("gt_spans must be [B,G,2] and gt_span_mask must be [B,G]")
    return spans, mask


def _safe_zero(value: Tensor) -> Tensor:
    """A finite zero with a graph edge and no NaN from invalid tensors."""

    safe = torch.nan_to_num(value.float(), nan=0.0, posinf=0.0, neginf=0.0)
    return safe.sum() * 0.0


@dataclass
class IndependentEndpointTargets:
    """Endpoint labels whose start and end each choose their own best GT."""

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
    """Compute independent max-G endpoint quality labels.

    For every candidate, each endpoint uses
    max_g [2*clamp(1-|b-boundary_g|/max(width_g, one_step),0,1)-1].
    """

    batch = _pad_empty_gt(batch)
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


def matched_endpoint_targets(outputs, batch, geometry=None):
    batch = _pad_empty_gt(batch)
    geometry = candidate_geometry(outputs, batch) if geometry is None else geometry
    den = geometry.selected_width.clamp_min(geometry.one_step[:, None, None])
    start = (
        2
        * (1 - (geometry.candidate_start - geometry.selected_start).abs() / den).clamp(
            0, 1
        )
        - 1
    ).masked_fill(~geometry.valid, 0)
    end = (
        2
        * (1 - (geometry.candidate_end - geometry.selected_end).abs() / den).clamp(0, 1)
        - 1
    ).masked_fill(~geometry.valid, 0)
    index = geometry.best_gt_index.masked_fill(~geometry.valid, -1)
    return IndependentEndpointTargets(
        start.detach(),
        end.detach(),
        index,
        index,
        torch.zeros_like(geometry.valid),
        geometry.valid,
    )


def transition_targets(outputs, batch, geometry=None, mode="independent_max"):
    if mode == "independent_max":
        return independent_endpoint_targets(outputs, batch, geometry)
    if mode == "matched_gt":
        return matched_endpoint_targets(outputs, batch, geometry)
    raise ValueError(mode)


def _ordinal_row_margin(
    score: Tensor,
    grade: Tensor,
    margin: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Aggregate all ordinal pairs in O(N log N + N*G), without N squared."""

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
    span = (ordered_score.max() - ordered_score.min()).detach()
    if not bool(weighted_pair_count.gt(0.0)):
        return zero, {
            "eligible_pair_count": eligible_pair_count.detach(),
            "violating_pair_count": violating_pair_count.detach(),
            "violating_pair_rate": zero.detach(),
            "weighted_signed_margin": zero.detach(),
            "score_span": span,
        }
    loss = weighted_violation / weighted_pair_count
    return loss, {
        "eligible_pair_count": eligible_pair_count.detach(),
        "violating_pair_count": violating_pair_count.detach(),
        "violating_pair_rate": (
            violating_pair_count / eligible_pair_count.clamp_min(1.0)
        ).detach(),
        "weighted_signed_margin": loss.detach(),
        "score_span": span,
    }


def _row_score_span(score: Tensor, valid: Tensor) -> Tensor:
    values = score.float().flatten(1)
    mask = valid.bool().flatten(1)
    rows: list[Tensor] = []
    for row_score, row_mask in zip(values, mask):
        selected = row_score[row_mask]
        if selected.numel():
            rows.append(selected.max() - selected.min())
    return torch.stack(rows).mean() if rows else _safe_zero(score).detach()


def official_ordinal_margin_loss(
    score: Tensor,
    iou: Tensor,
    valid: Tensor,
    margin: float = ROUND31_RANK_MARGIN,
) -> tuple[Tensor, Mapping[str, Tensor]]:
    """Official grade ordering with the fixed round31 positive margin."""

    margin = float(margin)
    if margin < 0.0:
        raise ValueError("rank margin must be non-negative")
    if score.shape != iou.shape or score.shape != valid.shape:
        raise ValueError("score, iou, and valid must have matching shapes")
    if margin == 0.0:
        loss, base = official_ordinal_violation_loss(score, iou, valid)
        result = dict(base)
        result["score_span"] = _row_score_span(score, valid).detach()
        result["margin"] = score.new_tensor(0.0).detach()
        return loss, result

    flat_score = score.float().flatten(1)
    flat_iou = iou.float().flatten(1)
    flat_valid = valid.bool().flatten(1)
    thresholds = flat_iou.new_tensor(OFFICIAL_THRESHOLDS)
    grades = flat_iou.unsqueeze(-1).ge(thresholds).sum(-1).long()
    losses: list[Tensor] = []
    diagnostic_lists: dict[str, list[Tensor]] = {
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
            diagnostic_lists[name].append(value)
    zero = _safe_zero(score)
    loss = torch.stack(losses).mean() if losses else zero

    def mean(values: list[Tensor]) -> Tensor:
        return torch.stack(values).float().mean() if values else zero.detach()

    metrics = {name: mean(values).detach() for name, values in diagnostic_lists.items()}
    metrics.update(
        {
            "active_query_rate": score.new_tensor(
                active / max(1, flat_score.shape[0]),
                dtype=torch.float32,
            ).detach(),
            "top1_official_grade": zero.detach(),
            "loss": loss.detach(),
            "margin": score.new_tensor(margin).detach(),
        }
    )
    return loss, metrics


def _tolerant_values(prediction: Tensor, target: Tensor) -> Tensor:
    return F.relu(
        (prediction.float() - target.float()).abs() - ROUND31_TOLERANCE
    ).square()


def _quality_reduction(
    prediction: Tensor,
    target: Tensor,
    valid: Tensor,
    iou: Tensor,
    *,
    stratified: bool,
) -> tuple[Tensor, Mapping[str, Tensor]]:
    """Global or per-query near/far equal reduction for S and each T endpoint."""

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
    if stratified:
        query_values: list[Tensor] = []
        for row in range(values.shape[0]):
            pieces = [
                item
                for item in (
                    selected_mean(near[row], values[row]),
                    selected_mean(far[row], values[row]),
                )
                if item is not None
            ]
            if pieces:
                query_values.append(torch.stack(pieces).mean())
        loss = torch.stack(query_values).mean() if query_values else zero
    else:
        selected = values[valid]
        loss = selected.mean() if selected.numel() else zero

    return loss, {
        "near_count": near.sum().float().detach(),
        "far_count": far.sum().float().detach(),
        "near_loss": (
            near_global.detach() if near_global is not None else zero.detach()
        ),
        "far_loss": (far_global.detach() if far_global is not None else zero.detach()),
        "stratified": prediction.new_tensor(float(stratified)).detach(),
        "near_query_count": near.flatten(1).any(-1).sum().float().detach(),
        "near_start_row_count": near.any(-1).sum().float().detach(),
        "far_query_count": far.flatten(1).any(-1).sum().float().detach(),
        "far_start_row_count": far.any(-1).sum().float().detach(),
    }


def _validate_weights(weights: Mapping[str, float] | None) -> dict[str, float]:
    defaults = {
        "rank": 0.5,
        "evidence": 0.1,
        "support": 0.1,
        "transition": 0.1,
        "endpoint": 0.1,
    }
    if weights is None:
        return defaults
    result = dict(defaults)
    for name, value in weights.items():
        if name not in result:
            raise ValueError(f"unknown round31 loss weight {name!r}")
        numeric = float(value)
        if not torch.isfinite(torch.tensor(numeric)) or numeric < 0.0:
            raise ValueError(f"loss weight {name!r} must be finite and non-negative")
        result[name] = numeric
    return result


def _gt_mask_for_batch(batch: Any, geometry: GeometryTargets) -> Tensor:
    _, targets = _batch_parts(batch)
    mask = targets.get("gt_span_mask")
    if mask is None:
        return torch.zeros(
            (geometry.valid.shape[0], geometry.all_iou.shape[-1]),
            dtype=torch.bool,
            device=geometry.valid.device,
        )
    return mask.to(geometry.valid.device).bool()


def _topk_restrict_valid(flat_score: Tensor, valid: Tensor, k: int) -> Tensor:
    """round40 H11: narrow `valid` to the top-K candidates per query by the
    model's OWN current score (detached -- gradient flows through the
    softmax over the retained set, not through which K were chosen).
    Queries with fewer than K valid candidates keep all of them.
    """
    b, n = flat_score.shape
    k = min(int(k), n)
    masked = flat_score.detach().masked_fill(~valid, float("-inf"))
    _, top_idx = masked.topk(k, dim=1)
    topk_mask = torch.zeros_like(valid)
    topk_mask.scatter_(1, top_idx, True)
    return valid & topk_mask


def gt_balanced_kl_rank_loss(
    score: Tensor,
    geometry: GeometryTargets,
    gt_mask: Tensor,
    rank_target_threshold: float = 0.7,
    rank_target_exponent: float = 4.0,
    *,
    return_queries: bool = False,
    length_conditional_gain: float = 0.0,
    length_ratio_per_gt: Tensor | None = None,
    rank_topk_restrict: int = 0,
    short_relief_gain: float = 0.0,
    rank_protect_gt: bool = False,
) -> tuple[Tensor, Mapping[str, Tensor]]:
    """GT-balanced KL target with O(B*N*G) memory.

    Every valid GT first gets a normalized candidate distribution from
    relu(IoU-threshold)^exponent (.7 and 4 in the fixed matrix).
    GT distributions are averaged equally in the query.
    A GT whose weights are all zero falls back to equal max-IoU ties. The
    target is detached before KL.
    """

    if score.ndim != 3 or geometry.all_iou.ndim != 4:
        raise ValueError("score must be [B,L,L] and all_iou must be [B,L,L,G]")
    b, length, width = score.shape
    if length != width or geometry.all_iou.shape[:3] != score.shape:
        raise ValueError("score and geometry candidate dimensions disagree")
    all_iou = geometry.all_iou.float().flatten(1, 2)
    valid = geometry.valid.bool().flatten(1)
    full_valid = valid.clone()
    if rank_topk_restrict:
        # round40 H11: narrow the softmax competition to the model's own
        # current top-K, mimicking the official evaluator's effective
        # candidate budget (max_candidates=30). Selection uses the
        # deployed score (detached); everything downstream (target
        # construction, KL, metrics) is otherwise unchanged.
        valid = _topk_restrict_valid(
            score.float().flatten(1), valid, rank_topk_restrict
        )
    selected_before = valid.clone()
    oracle_iou, oracle_index = all_iou.masked_fill(~full_valid[..., None], -1.0).max(1)
    if rank_protect_gt:
        add = torch.zeros_like(valid, dtype=torch.long)
        add.scatter_add_(1, oracle_index, (gt_mask.bool() & oracle_iou.ge(0.0)).long())
        valid = valid | (add.gt(0) & full_valid)
    selected_oracle = all_iou.masked_fill(~selected_before[..., None], -1.0).amax(1)
    used_oracle = all_iou.masked_fill(~valid[..., None], -1.0).amax(1)
    _n, gt_count = all_iou.shape[1], all_iou.shape[2]
    gt_mask = gt_mask.to(device=score.device).bool()
    if gt_mask.shape != (b, gt_count):
        raise ValueError(f"gt_mask shape {tuple(gt_mask.shape)} != {(b, gt_count)}")
    if rank_target_threshold not in (0.5, 0.7) or rank_target_exponent not in (
        2.0,
        4.0,
    ):
        raise ValueError("invalid round31 target parameters")
    if length_conditional_gain or short_relief_gain:
        if length_ratio_per_gt is None:
            raise ValueError(
                "length_conditional_gain/short_relief_gain requires length_ratio_per_gt"
            )
        effective_threshold = (
            rank_target_threshold - float(length_conditional_gain) * length_ratio_per_gt
        )
        if short_relief_gain:
            effective_threshold = effective_threshold - float(short_relief_gain) * (
                1.0 - length_ratio_per_gt
            )
        effective_threshold = effective_threshold.clamp_min(0.3)
        weighted = F.relu(all_iou - effective_threshold[:, None, :]).pow(
            rank_target_exponent
        )
    else:
        weighted = F.relu(all_iou - rank_target_threshold).pow(rank_target_exponent)
    weighted = weighted.masked_fill(~valid[..., None], 0.0)
    weighted = weighted.masked_fill(~gt_mask[:, None, :], 0.0)
    mass = weighted.sum(1)
    positive_mass = mass.gt(0.0)
    normalized = (
        weighted / torch.where(positive_mass, mass, torch.ones_like(mass))[:, None, :]
    )

    max_iou = all_iou.masked_fill(~valid[..., None], -1.0)
    max_per_gt = max_iou.amax(1)
    ties = valid[..., None] & all_iou.eq(max_per_gt[:, None, :]) & gt_mask[:, None, :]
    tie_count = ties.sum(1).clamp_min(1).float()
    fallback = ties.float() / tie_count[:, None, :]
    per_gt = torch.where(
        positive_mass[:, None, :],
        normalized,
        fallback,
    )
    gt_count_valid = gt_mask.sum(1).float()
    q = per_gt.sum(-1) / gt_count_valid.clamp_min(1.0)[:, None]
    q = q.detach()
    candidate_any = valid.any(1)
    active_query = gt_count_valid.gt(0.0) & candidate_any

    flat_score = score.float().flatten(1)
    # Inactive queries have a finite dummy row; their loss is excluded below.
    safe_score = flat_score.masked_fill(~valid, float("-inf"))
    safe_score = torch.where(
        candidate_any[:, None], safe_score, torch.zeros_like(safe_score)
    )
    log_p = F.log_softmax(safe_score, dim=1).masked_fill(~valid, 0.0)
    positive_q = q.gt(0.0)
    kl_each = torch.where(
        positive_q,
        q * (q.clamp_min(1.0e-12).log() - log_p),
        torch.zeros_like(q),
    ).sum(1)
    zero = _safe_zero(score)
    loss = kl_each[active_query].mean() if bool(active_query.any()) else zero

    row_spans: list[Tensor] = []
    for row_score, row_valid in zip(flat_score, valid):
        chosen = row_score[row_valid]
        if chosen.numel():
            row_spans.append(chosen.max() - chosen.min())
    score_span = torch.stack(row_spans).mean() if row_spans else zero.detach()
    fallback_gt = (~positive_mass & gt_mask).sum().float()
    total_gt = gt_mask.sum().float()
    entropy = torch.where(
        positive_q,
        -q * q.clamp_min(1.0e-12).log(),
        torch.zeros_like(q),
    ).sum(1)
    metrics = {
        "valid_query_rate": active_query.float().mean().detach(),
        "active_query_count": active_query.sum().float().detach(),
        "gt_count": total_gt.detach(),
        "fallback_gt_rate": ((fallback_gt / total_gt.clamp_min(1.0)).detach()),
        "fallback_gt_count": fallback_gt.detach(),
        "target_entropy": (
            entropy[active_query].mean().detach()
            if bool(active_query.any())
            else zero.detach()
        ),
        "score_span": score_span.detach(),
        "loss": loss.detach(),
    }
    probability = log_p.exp().masked_fill(~valid, 0.0)

    def active_mean(value):
        return (
            value[active_query].mean().detach()
            if bool(active_query.any())
            else zero.detach()
        )

    metrics.update(
        {
            "candidate_count": active_mean(valid.sum(1).float()),
            "target_max_probability": active_mean(q.max(1).values),
            "prediction_max_probability": active_mean(probability.max(1).values),
            "prediction_entropy": active_mean(-(probability * log_p).sum(1)),
            "target_support_probability": active_mean(
                (probability * positive_q).sum(1)
            ),
            "target_support_candidate_count": active_mean(positive_q.sum(1).float()),
        }
    )
    gm = gt_mask.bool()
    denom = gm.sum().clamp_min(1)
    metrics["r48_gt_protection"] = score.new_tensor(float(rank_protect_gt))
    metrics["r48_added_candidates"] = (
        (valid & ~selected_before).sum(1).float().mean().detach()
    )
    metrics["r48_oracle_iou_full"] = (oracle_iou * gm).sum().detach() / denom
    metrics["r48_oracle_iou_selected"] = (selected_oracle * gm).sum().detach() / denom
    metrics["r48_oracle_iou_used"] = (used_oracle * gm).sum().detach() / denom
    metrics["r48_recoverable_gt_missed_rate"] = (
        (oracle_iou.ge(0.7) & selected_oracle.lt(0.7) & gm).sum() / denom
    ).detach()
    metrics["target_threshold"] = score.new_tensor(rank_target_threshold).detach()
    metrics["target_exponent"] = score.new_tensor(rank_target_exponent).detach()
    max_quality = all_iou.masked_fill(~gt_mask[:, None, :], -1.0).max(-1).values
    for low, high, label in (
        (0.0, 0.5, "lt05"),
        (0.5, 0.7, "05_07"),
        (0.7, 0.75, "07_075"),
        (0.75, 0.9, "075_09"),
        (0.9, 1.000001, "ge09"),
    ):
        mask = valid & (max_quality >= low) & (max_quality < high)
        metrics["target_mass_maxiou_" + label] = active_mean((q * mask).sum(1))
        metrics["prediction_mass_maxiou_" + label] = active_mean(
            (probability * mask).sum(1)
        )
        per_mask = (
            valid[..., None] & gt_mask[:, None, :] & (all_iou >= low) & (all_iou < high)
        )
        per_mass = (per_gt.detach() * per_mask).sum(1)
        metrics["target_mass_pergt_" + label] = (
            per_mass.sum() / total_gt.clamp_min(1.0)
        ).detach()
    if return_queries:
        return loss, metrics, kl_each, active_query
    return loss, metrics


def _round31_metadata_rows(batch: Any, batch_size: int):
    from field_core.adapter import _batch_parts as base_batch_parts, _metadata_rows

    _, _, metadata = base_batch_parts(batch)
    return _metadata_rows(metadata, batch_size)


def _round31_token_valid(outputs: Any, batch: Any, geometry: GeometryTargets) -> Tensor:
    inputs, _ = _batch_parts(batch)
    padding = inputs.get("video_padding_mask")
    if isinstance(padding, Tensor):
        result = ~padding.to(device=geometry.valid.device).bool()
    else:
        result = geometry.valid.any(-1) | geometry.valid.any(-2)
    if result.shape != geometry.valid.shape[:2]:
        raise ValueError("video_padding_mask must match [B,L]")
    return result


def _round31_local_evidence_loss(
    outputs: Any,
    batch: Any,
    geometry: GeometryTargets,
    counterfactual=None,
    pair_mask=None,
):
    """Explicit-rated local ordinal loss for A, with no GT qualification."""

    field = outputs.trifield_output
    if hasattr(field, "cf_e_objective"):
        return field.cf_e_objective
    token = getattr(field, "round31_e_token", None)
    if not isinstance(token, Tensor) or token.ndim != 2:
        raise RuntimeError("A local-E loss requires field.round31_e_token [B,L]")
    token_valid = _round31_token_valid(outputs, batch, geometry)
    _, targets = _batch_parts(batch)
    ratings = targets.get("saliency_all_labels")
    if not isinstance(ratings, Tensor) or ratings.shape != token.shape:
        ratings = torch.zeros_like(token)
        rated = torch.zeros_like(token, dtype=torch.bool)
    else:
        ratings = ratings.to(device=token.device, dtype=torch.float32)
        rated = torch.zeros_like(token, dtype=torch.bool)
        for row_index, row in enumerate(_round31_metadata_rows(batch, token.shape[0])):
            if row is None:
                continue
            ids = row.get("relevant_clip_ids", ())
            if isinstance(ids, Tensor):
                ids = ids.detach().cpu().tolist()
            for clip_id in ids or ():
                if isinstance(clip_id, Tensor):
                    clip_id = clip_id.item()
                if int(clip_id) != clip_id or not 0 <= int(clip_id) < token.shape[1]:
                    raise ValueError("invalid explicitly rated clip ID")
                rated[row_index, int(clip_id)] = True
    query_metrics = {}
    if counterfactual is not None and counterfactual.get("query_relative_e", False):
        from .query_ordinal import query_relative_ordinal, explicit_query_validity
        from field_core.adapter import _batch_parts as full_parts

        metadata = full_parts(batch)[2]
        wrong = getattr(counterfactual.get("wrong_field"), "round31_e_token", None)
        paired, coverage, similarities = explicit_query_validity(
            metadata, token.shape[0], token.device, pair_mask
        )
        if wrong is None:
            coverage["missing_wrong_field_queries"] = int(paired.sum())
            if paired.any():
                raise RuntimeError(
                    "Query-relative E has valid wrong-query pairs but no wrong token field"
                )
        else:
            coverage["missing_wrong_field_queries"] = 0
        loss, detail = query_relative_ordinal(
            token, wrong, ratings, rated, token_valid, paired, margin=0.20
        )
        for key, value in detail.items():
            if isinstance(value, Tensor) and value.ndim == 0:
                query_metrics["query_relative/" + key] = value.detach()
        query_metrics.update(
            {
                "query_relative/" + key: token.new_tensor(float(value)).detach()
                for key, value in coverage.items()
            }
        )
        counterfactual["query_relative_e_probe"] = {
            "coverage": coverage,
            "jaccard_by_query": similarities,
            "wrong_valid": paired.detach(),
            "detail": detail,
        }
    else:
        loss, detail = explicit_local_ordinal_loss(
            token.float(), ratings, rated, token_valid, margin=0.20
        )
    iou = geometry.max_iou.float()
    active = (
        rated & token_valid
    )  # local tokens remain eligible when min-span excludes singletons
    iou = iou.diagonal(dim1=1, dim2=2)
    # Candidate IoU is post-hoc diagnostic only; it does not select pairs.
    candidate_bins = {}
    for low, high, name in (
        (0.0, 0.30, "lt03"),
        (0.30, 0.70, "03_07"),
        (0.70, 1.01, "ge07"),
    ):
        candidate_bins[f"rated_token_count_iou_{name}"] = (
            (active & (iou >= low) & (iou < high)).sum().float().detach()
        )
    metrics = {
        "pair_count": detail["pair_counts"].sum().float().detach(),
        "pair_weighted_margin_mean": detail["pair_margin_mean"],
        "pair_weighted_margin_met_rate": detail["pair_margin_met_rate"],
        "active_query_count": detail["active_queries"].float().detach(),
        "rated_clip_count": detail["eligible_clips"].sum().float().detach(),
        "token_valid_count": token_valid.sum().float().detach(),
        "no_pair": token.new_tensor(float(detail["pair_counts"].sum() == 0)).detach(),
        **candidate_bins,
    }
    metrics.update(query_metrics)
    return loss, metrics


def _round31_select_four(indices: list[int]) -> list[int]:
    if len(indices) <= 4:
        return indices
    positions = torch.linspace(0, len(indices) - 1, 4).round().long().tolist()
    return [indices[int(position)] for position in positions]


def _round31_edge_support_loss(
    outputs,
    batch,
    geometry,
    selector,
    epoch,
    *,
    return_queries=False,
    aux_detach_input=False,
):
    """Weak protected-donor insertion; equal GT means within equal query means."""
    f = outputs.trifield_output
    if aux_detach_input:
        h = getattr(f, "round31_h_aux", None)
        edge = getattr(f, "round31_aux_edge_score", None)
        zero_source = getattr(f, "round31_aux_support", None)
        if not all(isinstance(x, Tensor) for x in (h, edge, zero_source)):
            raise RuntimeError("R31 detached auxiliary support tensors are missing")
    else:
        h = f.round31_h
        edge = f.round31_edge_score
        zero_source = f.score
    valid = _round31_token_valid(outputs, batch, geometry).detach().cpu()
    spans, gm = _target_spans(outputs, batch)
    spans = spans.detach().cpu()
    gm = gm.detach().cpu()
    step = geometry.one_step.detach().cpu()
    rows = _round31_metadata_rows(batch, len(h))
    _, targets = _batch_parts(batch)
    ratings = targets.get("saliency_all_labels")
    counts = {
        k: 0
        for k in (
            "eligible_edge_count",
            "selected_edge_count",
            "donor_count",
            "active_gt_count",
            "active_query_count",
            "missing_qid_gt_count",
            "selected_lower_rated_edge_count",
        )
    }
    query_losses = []
    observed_gaps = []
    observed_positives = []
    observed_negatives = []
    query_values = (
        [_safe_zero(zero_source[b]) for b in range(len(h))] if return_queries else None
    )
    query_active = (
        torch.zeros(len(h), dtype=torch.bool, device=f.score.device)
        if return_queries
        else None
    )
    for b in range(len(h)):
        intervals = [
            (float(spans[b, g].min() / step[b]), float(spans[b, g].max() / step[b]))
            for g in gm[b].nonzero().flatten().tolist()
        ]
        donors = [
            i
            for i in range(h.shape[1])
            if valid[b, i]
            and all(not (i < z + 1 and i + 1 > a - 1) for a, z in intervals)
        ]
        row = rows[b] or {}
        qid = row.get("qid")
        vid = row.get("vid", "")
        rated = {
            int(i)
            for i in row.get("relevant_clip_ids", [])
            if 0 <= int(i) < h.shape[1] and valid[b, int(i)]
        }
        top = (
            max(float(ratings[b, i]) for i in rated)
            if rated and isinstance(ratings, Tensor)
            else None
        )
        gt_losses = []
        for g, (a, z) in enumerate(intervals):
            eligible = [
                i
                for i in range(h.shape[1] - 1)
                if valid[b, i] and valid[b, i + 1] and i >= a and i + 2 <= z
            ]
            counts["eligible_edge_count"] += len(eligible)
            if qid is None:
                counts["missing_qid_gt_count"] += 1
            if not eligible or not donors or qid is None:
                continue
            selected = _round31_select_four(eligible)
            key = int.from_bytes(
                hashlib.sha256(f"{qid}|{vid}|{int(epoch)}|{g}".encode()).digest()[:8],
                "big",
            )
            donor = donors[key % len(donors)]
            idx = torch.tensor(selected, device=h.device, dtype=torch.long)
            left, right = h[b, idx], h[b, idx + 1]
            replacement = h[b, donor].expand_as(left)
            # The legacy mean clean path reuses its identical deployed edge
            # evaluation to preserve the original arithmetic/gradient order.
            # All other readouts score clean and both donor orientations by
            # the same current length-two deployment dispatcher.
            positive = (
                edge[b, idx]
                if selector.support_readout == "edge_mean"
                else selector.support_score_from_pairs(left, right)
            )
            negatives = torch.stack(
                (
                    selector.support_score_from_pairs(replacement, right),
                    selector.support_score_from_pairs(left, replacement),
                ),
                dim=-1,
            )
            gt_losses.append(F.relu(0.2 - positive[:, None] + negatives).mean())
            observed_gaps.append((positive[:, None] - negatives).detach().flatten())
            observed_positives.append(positive.detach().flatten())
            observed_negatives.append(negatives.detach().flatten())
            counts["selected_edge_count"] += len(selected)
            counts["donor_count"] += 1
            counts["active_gt_count"] += 1
            if top is not None:
                counts["selected_lower_rated_edge_count"] += sum(
                    any(j in rated and float(ratings[b, j]) < top for j in (i, i + 1))
                    for i in selected
                )
        if gt_losses:
            query_loss = torch.stack(gt_losses).mean()
            query_losses.append(query_loss)
            if return_queries:
                query_values[b] = query_loss
                query_active[b] = True
    counts["active_query_count"] = len(query_losses)
    value = (
        torch.stack(query_losses).mean() if query_losses else _safe_zero(zero_source)
    )
    counts["no_pair"] = int(not query_losses)
    metrics = {k: f.score.new_tensor(float(v)).detach() for k, v in counts.items()}
    gaps = torch.cat(observed_gaps) if observed_gaps else f.score.detach().new_empty(0)
    metrics.update(
        pair_orientation_count=f.score.new_tensor(gaps.numel()).detach(),
        pair_weighted_margin_mean=gaps.mean()
        if gaps.numel()
        else f.score.detach().new_zeros(()),
        pair_weighted_margin_met_rate=(gaps >= 0.2).float().mean()
        if gaps.numel()
        else f.score.detach().new_zeros(()),
        positive_edge_mean=torch.cat(observed_positives).mean()
        if observed_positives
        else f.score.detach().new_zeros(()),
        negative_edge_mean=torch.cat(observed_negatives).mean()
        if observed_negatives
        else f.score.detach().new_zeros(()),
    )
    if return_queries:
        return value, metrics, torch.stack(query_values), query_active
    return value, metrics


def round31_rank_budget(
    original_kl,
    kl_query,
    kl_active,
    pair_query,
    pair_available,
    *,
    rank_kl_half=False,
    rank_candidate_pair=False,
):
    """Fixed 2x2 rank budget on the unchanged original-KL query denominator."""
    from .candidate_pair_loss import active_mean

    if type(rank_kl_half) is not bool or type(rank_candidate_pair) is not bool:
        raise ValueError("Round31 rank switches must be booleans")
    if not (
        kl_query.shape == kl_active.shape == pair_query.shape == pair_available.shape
    ):
        raise ValueError("Round31 rank query shapes differ")
    use_pair = kl_active & pair_available
    alpha = 0.5 if rank_kl_half else 1.0
    beta = 0.5 if rank_candidate_pair else 0.0
    # Keep the exact R28 anchor arithmetic. Reference diagnostics must not
    # create a pair contribution when beta is zero.
    effective_kl_query = (
        torch.where(use_pair, 0.5 * kl_query, kl_query) if rank_kl_half else kl_query
    )
    kl_component = (
        active_mean(effective_kl_query, kl_active) if rank_kl_half else original_kl
    )
    pair_component = active_mean(
        torch.where(use_pair, beta * pair_query, torch.zeros_like(pair_query)),
        kl_active,
    )
    if not rank_kl_half and not rank_candidate_pair:
        mixed_query, rank = kl_query, original_kl
    else:
        if rank_kl_half and rank_candidate_pair:
            mixed_query = torch.where(
                use_pair, 0.5 * kl_query + 0.5 * pair_query, kl_query
            )
        elif rank_kl_half:
            mixed_query = effective_kl_query
        else:
            mixed_query = torch.where(use_pair, kl_query + 0.5 * pair_query, kl_query)
        rank = active_mean(mixed_query, kl_active)
    pair_reference = active_mean(
        torch.where(use_pair, pair_query, torch.zeros_like(pair_query)), kl_active
    )
    k_coeff = torch.where(
        kl_active,
        torch.where(
            use_pair,
            kl_query.new_full(kl_query.shape, alpha),
            torch.ones_like(kl_query),
        ),
        torch.zeros_like(kl_query),
    )
    p_coeff = torch.where(
        use_pair,
        pair_query.new_full(pair_query.shape, beta),
        torch.zeros_like(pair_query),
    )
    return rank, dict(
        original_kl=original_kl,
        kl_query=kl_query,
        kl_active=kl_active,
        pair_query=pair_query,
        pair_available=pair_available,
        pair_active=use_pair,
        fallback_active=kl_active & ~use_pair,
        excluded=~kl_active,
        mixed_query=mixed_query,
        kl_component=kl_component,
        pair_component=pair_component,
        pair_reference=pair_reference,
        pair_reference_active_only=active_mean(pair_query, use_pair),
        alpha=alpha,
        beta=beta,
        kl_coefficients=k_coeff,
        pair_coefficients=p_coeff,
        rank_pair_enabled=rank_candidate_pair,
        rank_kl_half=rank_kl_half,
        pair_reference_only=not rank_candidate_pair,
    )


def _query_length_ratio(outputs: Any, batch: Any) -> Tensor:
    """round36 length-bias countermeasure: per-query max-GT-duration ratio.

    # round38 fix: gt_spans are already normalized to [0,1] fractions of
    # valid video length (see field_core/losses.py::candidate_geometry
    # docstring: "candidate i:j covers [i/valid_length, (j+1)/valid_length]",
    # and gt_start/gt_end are used directly against that scale with no
    # further division). The original round36 code divided duration by
    # grid_length AGAIN, shrinking every ratio by ~L (~75x) -- confirmed
    # empirically via round36 g2's real training log
    # (rank/length_ratio_mean=0.0033 held flat for 24 epochs, ~75x smaller
    # than a plausible ~0.25 duration fraction), which made H1/H4
    # numerically near-inert even at gain=2.0. duration IS the ratio.
    Returns a [B] tensor, 0 for queries with no valid GT.
    """
    spans, mask = _target_spans(outputs, batch)
    ratio = (spans[..., 1] - spans[..., 0]).clamp(0.0, 1.0)
    ratio = ratio.masked_fill(~mask, 0.0)
    return ratio.amax(dim=1) if ratio.numel() else ratio.new_zeros(spans.shape[0])


def _gt_length_ratio(outputs: Any, batch: Any, gt_count: int) -> Tensor:
    """round37 length-bias countermeasure H9: per-GT (not per-query) length
    ratio, padded/truncated to gt_count columns to align with all_iou's G
    dimension in gt_balanced_kl_rank_loss.

    round38 fix: see _query_length_ratio -- gt_spans are already [0,1]
    fractions; no grid_length division needed.
    """
    spans, mask = _target_spans(outputs, batch)
    ratio = (spans[..., 1] - spans[..., 0]).clamp(0.0, 1.0)
    ratio = ratio.masked_fill(~mask, 0.0)
    b, g = ratio.shape
    if g == gt_count:
        return ratio
    if g > gt_count:
        return ratio[:, :gt_count]
    pad = ratio.new_zeros((b, gt_count - g))
    return torch.cat([ratio, pad], dim=1)


def _duration_regression_loss(
    rank_score: Tensor, valid: Tensor, target_ratio: Tensor
) -> tuple[Tensor, Tensor]:
    """round36 length-bias countermeasure H4.

    Compares the model's own expected predicted-span-length ratio (the
    softmax-weighted mean of (j-i)/L over the model's rank-score
    distribution) against target_ratio (the query's max-GT-length ratio
    from _query_length_ratio). This gives a direct, differentiable
    corrective signal independent of the KL rank objective, using only
    already-computed quantities (no new IoU/geometry work).
    """
    b, length, _ = rank_score.shape
    idx = torch.arange(length, device=rank_score.device, dtype=rank_score.dtype)
    duration_ratio = ((idx[None, :] - idx[:, None]) / float(max(1, length - 1))).clamp(
        0.0, 1.0
    )
    flat_score = rank_score.float().flatten(1)
    flat_valid = valid.flatten(1)
    safe_score = flat_score.masked_fill(~flat_valid, float("-inf"))
    active = flat_valid.any(dim=1)
    safe_score = torch.where(active[:, None], safe_score, torch.zeros_like(safe_score))
    probability = F.softmax(safe_score, dim=1).masked_fill(~flat_valid, 0.0)
    predicted_ratio = (probability * duration_ratio.flatten()[None, :]).sum(dim=1)
    per_query = (predicted_ratio - target_ratio).square()
    zero = _safe_zero(rank_score)
    loss = per_query[active].mean() if bool(active.any()) else zero
    return loss, predicted_ratio.detach()


def compute_loss_terms(
    outputs: Any,
    batch: Any,
    *,
    wrong_evidence: Tensor | None = None,
    evidence_reduction: str = "candidate_mean",
    evidence_supervision: str = "wrong_query",
    counterfactual=None,
    evidence_pair_mask: Tensor | None = None,
    loss_weights: Mapping[str, float] | None = None,
    rank_mode: str = "gt_balanced_kl",
    rank_form: str = "dense",
    rank_competition: str = "global",
    rank_logit_scale: float = 2.0,
    rank_target_threshold: float = 0.7,
    rank_target_exponent: float = 4.0,
    rank_margin: float = ROUND31_RANK_MARGIN,
    quality_stratified: bool = True,
    transition_target_mode: str = "independent_max",
    a_local_evidence: bool = False,
    b_edge_support: bool = False,
    selector: Any = None,
    epoch: int = 0,
    support_wide_pair: bool = False,
    rank_candidate_pair: bool = False,
    rank_kl_half: bool = False,
    rank_stop_s: bool = False,
    aux_detach_input: bool = False,
    negative_selection_mode: str = "hardest",
    rank_length_reweight_gain: float = 0.0,
    duration_aux_weight: float = 0.0,
    length_conditional_gain: float = 0.0,
    carrier_desaturation_weight: float = 0.0,
    rank_topk_restrict: int = 0,
    short_relief_gain: float = 0.0,
    rank_protect_gt: bool = False,
    counterfactual_support_mix: float = 0.0,
    support_margin_scale: float = 1.0,
) -> FiveLossTerms:
    """Compute the actual round31 five-term objective."""

    if getattr(outputs, "trifield_output", None) is None:
        raise RuntimeError("outputs.trifield_output is required")
    field = outputs.trifield_output
    if type(rank_stop_s) is not bool or type(aux_detach_input) is not bool:
        raise ValueError("R31 route switches must be booleans")
    # The probe/recompute path must use the same graph route that the actual
    # forward advertised.  A missing metadata field is an integration error;
    # silently falling back to field.score would make A/B diagnostics look
    # numerically valid while measuring the wrong gradient graph.
    for name, expected in (
        ("round31_rank_stop_s", rank_stop_s),
        ("round31_aux_detach_input", aux_detach_input),
    ):
        if not hasattr(field, name):
            raise RuntimeError(f"R31 forward is missing field.{name}")
        actual = getattr(field, name)
        if type(actual) is not bool or actual != expected:
            raise RuntimeError(
                f"R31 loss route mismatch for field.{name}: "
                f"forward={actual!r}, requested={expected!r}"
            )
    rank_score = getattr(field, "round31_rank_score", None)
    if not isinstance(rank_score, Tensor) or rank_score.shape != field.score.shape:
        raise RuntimeError(
            "R31 rank score field is missing or does not match deployed F shape"
        )
    field.round31_candidate_pair_probe = None
    batch = _pad_empty_gt(batch)
    # Reuse the context labels for this exact output/batch rather than holding
    # two identical [B, L, L, G] geometry allocations during the same loss call.
    geometry = counterfactual.get("geometry") if counterfactual is not None else None
    if geometry is None:
        geometry = candidate_geometry(outputs, batch)
    valid = geometry.valid
    weights = _validate_weights(loss_weights)
    mode = str(rank_mode).lower()
    if mode != "gt_balanced_kl" or rank_form != "dense" or rank_competition != "global":
        raise ValueError("Round31 requires the fixed dense global GT-balanced KL")
    from .candidate_pair_loss import (
        select_candidate_pairs,
        pair_query_loss,
        active_mean,
        pair_metrics,
    )

    # Qualification is independent of whether the pair term is trained. In
    # particular W2 must not fall back to full KL because beta is zero.
    pair_cache = select_candidate_pairs(
        field.score,
        geometry,
        _gt_mask_for_batch(batch, geometry),
        mode=negative_selection_mode,
        epoch=epoch,
    )
    pair_probe = {
        "pairs": pair_cache,
        "support_enabled": bool(support_wide_pair),
        "rank_enabled": True,
        "rank_pair_enabled": bool(rank_candidate_pair),
        "rank_kl_half": bool(rank_kl_half),
        "rank_stop_s": bool(rank_stop_s),
        "aux_detach_input": bool(aux_detach_input),
        "selection_score": field.score,
        "rank_score": rank_score,
    }
    field.round31_candidate_pair_probe = pair_probe
    if rank_logit_scale not in (2.0, 4.0):
        raise ValueError("rank_logit_scale must be 2 or 4")
    if mode == "ordinal_margin":
        rank, rank_metrics = official_ordinal_margin_loss(
            field.score, geometry.max_iou, valid, rank_margin
        )
        rank_mode_flag = 0.0
    elif mode == "gt_balanced_kl":
        gt_mask = _gt_mask_for_batch(batch, geometry)
        length_ratio_per_gt = (
            _gt_length_ratio(outputs, batch, gt_mask.shape[1])
            if (length_conditional_gain or short_relief_gain)
            else None
        )
        original_kl, rank_metrics, kl_query, kl_active = gt_balanced_kl_rank_loss(
            rank_score.float() * rank_logit_scale,
            geometry,
            gt_mask,
            rank_target_threshold,
            rank_target_exponent,
            return_queries=True,
            length_conditional_gain=length_conditional_gain,
            length_ratio_per_gt=length_ratio_per_gt,
            rank_topk_restrict=rank_topk_restrict,
            short_relief_gain=short_relief_gain,
            rank_protect_gt=rank_protect_gt,
        )
        if rank_length_reweight_gain:
            # round36 length-bias countermeasure H1: give queries whose
            # largest GT is long a proportionally larger share of the rank
            # loss gradient. gain=0 (default) is exactly a no-op (weight
            # tensor is all-ones, kl_query unchanged, original_kl/rank_metrics
            # already computed above and NOT recomputed with this weight --
            # only the value actually used for the training loss changes).
            length_ratio = _query_length_ratio(outputs, batch)
            length_weight = 1.0 + float(rank_length_reweight_gain) * length_ratio
            kl_query = kl_query * length_weight
            rank_metrics = dict(
                rank_metrics,
                length_reweight_gain=rank_score.new_tensor(
                    float(rank_length_reweight_gain)
                ),
                length_ratio_mean=length_ratio[kl_active].mean().detach()
                if bool(kl_active.any())
                else _safe_zero(rank_score).detach(),
            )
        f_query, f_active, f_detail = pair_query_loss(
            rank_score, pair_cache, wide_only=False
        )
        rank, budget = round31_rank_budget(
            original_kl,
            kl_query,
            kl_active,
            f_query,
            f_active,
            rank_kl_half=rank_kl_half,
            rank_candidate_pair=rank_candidate_pair,
        )
        use_pair = budget["pair_active"]
        kl_component, pair_component = budget["kl_component"], budget["pair_component"]
        pair_probe["rank"] = dict(budget, detail=f_detail)
        if duration_aux_weight:
            # round36 length-bias countermeasure H4: folded into `rank`
            # (not a new FiveLossTerms field -- that dataclass's shape is
            # shared with field_core) so it rides the existing
            # loss_weights["rank"] multiplier applied by the caller.
            duration_loss, duration_pred_ratio = _duration_regression_loss(
                rank_score, valid, _query_length_ratio(outputs, batch)
            )
            rank = rank + float(duration_aux_weight) * duration_loss
            rank_metrics = dict(
                rank_metrics,
                duration_aux_loss=duration_loss.detach(),
                duration_aux_weight=rank_score.new_tensor(float(duration_aux_weight)),
                duration_pred_ratio_mean=duration_pred_ratio.mean(),
            )
        if carrier_desaturation_weight:
            # round37 length-bias countermeasure H10: hinge-squared penalty
            # on |carrier| exceeding 0.9, directly targeting the
            # probe-verified carrier_p01 pinned-at--1.0 saturation finding.
            carrier_excess = (field.carrier.float().abs() - 0.9).clamp_min(0.0)
            carrier_loss = (
                carrier_excess.square() * valid.float()
            ).sum() / valid.float().sum().clamp_min(1.0)
            rank = rank + float(carrier_desaturation_weight) * carrier_loss
            rank_metrics = dict(
                rank_metrics,
                carrier_desaturation_loss=carrier_loss.detach(),
                carrier_desaturation_weight=rank_score.new_tensor(
                    float(carrier_desaturation_weight)
                ),
            )
        rank_metrics = dict(
            rank_metrics, original_kl_loss=original_kl.detach(), loss=rank.detach()
        )
        rank_mode_flag = 1.0
    else:
        raise ValueError("rank_mode must be ordinal_margin or gt_balanced_kl")

    support_target, _, _ = quality_targets(geometry)
    independent = transition_targets(outputs, batch, geometry, transition_target_mode)
    start_target, end_target = independent.start, independent.end

    support, support_metrics = _quality_reduction(
        field.support,
        support_target,
        valid,
        geometry.max_iou,
        stratified=bool(quality_stratified),
    )
    start_loss, start_metrics = _quality_reduction(
        field.transition_start,
        start_target,
        valid,
        geometry.max_iou,
        stratified=bool(quality_stratified),
    )
    end_loss, end_metrics = _quality_reduction(
        field.transition_end,
        end_target,
        valid,
        geometry.max_iou,
        stratified=bool(quality_stratified),
    )
    transition = 0.5 * (start_loss + end_loss)
    endpoint = endpoint_bce_loss(outputs, batch)

    if evidence_reduction not in ("candidate_mean", "query_mean"):
        raise ValueError(evidence_reduction)
    if evidence_supervision not in ("wrong_query", "query_plus_background"):
        raise ValueError(evidence_supervision)
    positive = valid & geometry.max_iou.ge(0.70)
    if (not hasattr(field, "cf_e_objective") and counterfactual is not None
            and counterfactual.get("round31_bits", 0) & 1):
        from .interaction import local_e_mask

        positive = positive | local_e_mask(outputs, batch, geometry)
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
        if evidence_reduction == "query_mean":
            evidence = evidence_reduction_values(
                field.evidence, wrong_evidence, pair_mask, field.raw_evidence
            )[1]
        if evidence_supervision == "query_plus_background":
            evidence = evidence_objective_values(
                outputs,
                batch,
                geometry,
                wrong_evidence,
                pair_mask,
                evidence_supervision,
            )[0 if evidence_reduction == "candidate_mean" else 1]
        pair_count = pair_mask.float().sum().detach()
    else:
        evidence = _safe_zero(field.raw_evidence)
        pair_count = field.raw_evidence.new_zeros(())

    counterfactual_metrics = {}
    legacy_counterfactual_support = None
    if counterfactual is not None:
        from .counterfactual import apply_counterfactual_losses

        evidence, support, transition, counterfactual_metrics = (
            apply_counterfactual_losses(
                outputs,
                batch,
                geometry,
                wrong_evidence,
                evidence_pair_mask,
                evidence,
                support,
                transition,
                counterfactual,
            )
        )
        legacy_counterfactual_support = support

    semantic_metrics = {}
    if a_local_evidence:
        evidence, detail = _round31_local_evidence_loss(
            outputs, batch, geometry, counterfactual, evidence_pair_mask
        )
        semantic_metrics.update({"local_evidence/" + k: v for k, v in detail.items()})
    if b_edge_support:
        if selector is None:
            raise ValueError("B loss needs the actual deployed selector")
        if support_wide_pair:
            donor, detail, donor_query, donor_active = _round31_edge_support_loss(
                outputs,
                batch,
                geometry,
                selector,
                epoch,
                return_queries=True,
                aux_detach_input=aux_detach_input,
            )
            support_value = (
                getattr(field, "round31_aux_support", None)
                if aux_detach_input
                else field.support
            )
            if not isinstance(support_value, Tensor):
                raise RuntimeError("R31 auxiliary support value is missing")
            if not 0.0 < float(support_margin_scale) <= 1.0:
                raise ValueError("support_margin_scale outside (0,1]")
            s_cache = (
                pair_cache
                if support_margin_scale == 1.0
                else dict(
                    pair_cache, margin=pair_cache["margin"] * support_margin_scale
                )
            )
            s_query, s_active, s_detail = pair_query_loss(
                support_value, s_cache, wide_only=True
            )
            semantic_metrics["score_calibration/support_margin_scale"] = float(
                support_margin_scale
            )
            semantic_metrics["score_calibration/support_original_margin_mean"] = (
                pair_cache["margin"][s_detail["available"]].mean().detach()
                if s_detail["available"].any()
                else support_value.new_zeros(())
            )

            fallback_active = ~s_active & donor_active
            support_active = s_active | fallback_active
            mixed_query = torch.where(s_active, s_query, donor_query)
            support = active_mean(mixed_query, support_active)
            wide_component = active_mean(
                torch.where(s_active, s_query, torch.zeros_like(s_query)),
                support_active,
            )
            donor_component = active_mean(
                torch.where(
                    fallback_active, donor_query, torch.zeros_like(donor_query)
                ),
                support_active,
            )
            mix = float(counterfactual_support_mix)
            if not 0.0 <= mix <= 0.25:
                raise ValueError("counterfactual_support_mix must be in [0, 0.25]")
            use_counterfactual = bool(
                mix > 0.0
                and counterfactual is not None
                and int(counterfactual.get("bits", 0)) & 2
                and legacy_counterfactual_support is not None
            )
            wide_support = support
            if use_counterfactual:
                support = (
                    1.0 - mix
                ) * wide_support + mix * legacy_counterfactual_support
            pair_probe["support"] = dict(
                donor_reference=donor,
                donor_query=donor_query,
                donor_active=donor_active,
                pair_query=s_query,
                pair_active=s_active,
                fallback_active=fallback_active,
                active=support_active,
                detail=s_detail,
                mixed_query=mixed_query,
                wide_component=wide_component,
                donor_component=donor_component,
            )
            semantic_metrics.update(
                {
                    "candidate_support/" + k: v
                    for k, v in pair_metrics(s_cache, s_detail).items()
                }
            )
            semantic_metrics.update(
                {
                    "candidate_support/fallback_query_count": fallback_active.sum().float(),
                    "candidate_support/zero_query_count": (~support_active)
                    .sum()
                    .float(),
                    "candidate_support/wide_component": wide_component.detach(),
                    "candidate_support/donor_component": donor_component.detach(),
                    "candidate_support/final_wide_component": (
                        (1.0 - mix) * wide_support
                    ).detach(),
                    "candidate_support/final_counterfactual_component": (
                        (mix * legacy_counterfactual_support).detach()
                        if use_counterfactual
                        else support.new_zeros(())
                    ),
                    "candidate_support/counterfactual_mix": support.new_tensor(mix),
                    "candidate_support/counterfactual_active": support.new_tensor(
                        float(use_counterfactual)
                    ),
                }
            )
            # Full donor statistics are reference-only when the wide path replaces it.
            semantic_metrics.update(
                {"donor_reference/" + k: v for k, v in detail.items()}
            )
        else:
            support, detail = _round31_edge_support_loss(
                outputs, batch, geometry, selector, epoch
            )
            semantic_metrics.update({"edge_support/" + k: v for k, v in detail.items()})
    elif support_wide_pair:
        raise ValueError("wide support requires the active edge support path")
    if mode == "gt_balanced_kl":
        if mode != "gt_balanced_kl":
            raise ValueError("candidate pair rank requires KL")
        semantic_metrics.update(
            {
                "candidate_rank/" + k: v
                for k, v in pair_metrics(pair_cache, f_detail).items()
            }
        )
        semantic_metrics.update(
            {
                "candidate_rank/kl_component": kl_component.detach(),
                "candidate_rank/pair_component": pair_component.detach(),
                "candidate_rank/fallback_query_count": (kl_active & ~use_pair)
                .sum()
                .float(),
                "candidate_rank/kl_active_query_count": kl_active.sum().float(),
                "candidate_rank/excluded_query_count": (~kl_active).sum().float(),
                "candidate_rank/alpha": field.score.new_tensor(budget["alpha"]),
                "candidate_rank/beta": field.score.new_tensor(budget["beta"]),
                "candidate_rank/pair_reference": budget["pair_reference"].detach(),
                "candidate_rank/pair_reference_only": field.score.new_tensor(
                    float(not rank_candidate_pair)
                ),
            }
        )
    semantic_metrics.update(
        {
            "term_source/evidence_local_ordinal": float(a_local_evidence),
            "term_source/support_edge_insertion": float(
                b_edge_support and not support_wide_pair
            ),
            "term_source/support_wide_pair_with_donor_fallback": float(
                support_wide_pair
            ),
            "term_source/rank_candidate_pair": float(rank_candidate_pair),
            "term_source/rank_kl_half": float(rank_kl_half),
            "term_source/rank_stop_s": float(rank_stop_s),
            "term_source/aux_detach_input": float(aux_detach_input),
            "term_source/evidence_legacy_counterfactual": float(
                not a_local_evidence and counterfactual is not None
            ),
            "term_source/support_legacy_counterfactual": float(
                not b_edge_support and counterfactual is not None
            ),
            "term_source/support_counterfactual_mixed_into_final": float(
                bool(
                    counterfactual_support_mix > 0.0
                    and b_edge_support
                    and support_wide_pair
                    and counterfactual is not None
                    and int(counterfactual.get("bits", 0)) & 2
                )
            ),
            "term_source/transition_legacy_counterfactual": float(
                counterfactual is not None
            ),
        }
    )

    total = (
        weights["rank"] * rank
        + weights["evidence"] * evidence
        + weights["support"] * support
        + weights["transition"] * transition
        + weights["endpoint"] * endpoint
    )
    legacy_independent = (
        independent if transition_target_mode == "independent_max"
        else independent_endpoint_targets(outputs, batch, geometry)
    )
    conflict_values = legacy_independent.conflict[legacy_independent.valid]
    conflict_rate = (
        conflict_values.float().mean().detach()
        if conflict_values.numel()
        else independent.conflict.new_tensor(0.0)
    )
    gt_mask = _gt_mask_for_batch(batch, geometry)
    metrics: dict[str, Tensor | float] = {
        **semantic_metrics,
        "rank": rank.detach(),
        "evidence": evidence.detach(),
        "support": support.detach(),
        "transition": transition.detach(),
        "endpoint": endpoint.detach(),
        "total": total.detach(),
        "rank/mode_gt_balanced_kl": rank.new_tensor(rank_mode_flag).detach(),
        "rank/eligible_pair_count": rank_metrics.get(
            "eligible_pair_count", rank.new_zeros(())
        ),
        "rank/violating_pair_rate": rank_metrics.get(
            "violating_pair_rate", rank.new_zeros(())
        ),
        "rank/score_span": rank_metrics.get("score_span", rank.new_zeros(())),
        "rank/margin": rank.new_tensor(
            float(rank_margin) if mode == "ordinal_margin" else 0.0
        ).detach(),
        "rank/top1_official_grade": rank_metrics.get(
            "top1_official_grade", rank.new_zeros(())
        ),
        "rank/valid_query_rate": rank_metrics.get(
            "valid_query_rate", rank.new_zeros(())
        ),
        "rank/fallback_gt_rate": rank_metrics.get(
            "fallback_gt_rate", rank.new_zeros(())
        ),
        "evidence/positive_pair_count": pair_count,
        "evidence/query_mean_enabled": float(evidence_reduction == "query_mean"),
        "support/valid_candidate_rate": valid.float().mean().detach(),
        "transition/supervision_conflict_rate": conflict_rate,
        "transition/conflict_is_legacy_independent_geometry": 1.0,
        "transition/target_mode_independent_max": rank.new_tensor(
            float(transition_target_mode == "independent_max")
        ).detach(),
        "geometry/multigt_gt_count": gt_mask.sum(1).float().mean().detach(),
        "quality/stratified": rank.new_tensor(float(bool(quality_stratified))).detach(),
    }
    for prefix, values in (
        ("support", support_metrics),
        ("transition/start", start_metrics),
        ("transition/end", end_metrics),
    ):
        for name, value in values.items():
            metrics[f"{prefix}/{name}"] = value
    metrics.update(counterfactual_metrics)
    metrics.update({"rank/" + name: value for name, value in rank_metrics.items()})
    metrics["rank/logit_scale"] = rank.new_tensor(rank_logit_scale).detach()
    metrics["rank/kl_logit_span"] = rank_metrics["score_span"]
    metrics["rank/raw_score_span"] = rank_metrics["score_span"] / rank_logit_scale
    for name, value in weights.items():
        metrics[f"loss_weight/{name}"] = float(value)

    if counterfactual is not None:
        for prefix, bit in (("support/", 2), ("transition/", 4)):
            if counterfactual["bits"] & bit:
                for key in list(metrics):
                    if key.startswith(prefix):
                        metrics["legacy_reference/" + key] = metrics.pop(key)

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


compute_round31_loss_terms = compute_loss_terms


__all__ = [
    "NEAR_IOU_THRESHOLD",
    "ROUND31_RANK_MARGIN",
    "ROUND31_GT_WEIGHT_THRESHOLD",
    "IndependentEndpointTargets",
    "independent_endpoint_targets",
    "official_ordinal_margin_loss",
    "gt_balanced_kl_rank_loss",
    "compute_loss_terms",
    "compute_round31_loss_terms",
]


def rank_target_components(geometry, gt_mask):
    """Preserve frozen q_g including exact-tie fallback; return [B,N,G]."""
    iou = geometry.all_iou.float().flatten(1, 2)
    valid = geometry.valid.flatten(1)
    gt_mask = gt_mask.bool()
    weighted = (
        F.relu(iou - 0.7)
        .pow(4)
        .masked_fill(~valid[..., None], 0)
        .masked_fill(~gt_mask[:, None, :], 0)
    )
    mass = weighted.sum(1)
    positive_mass = mass.gt(0)
    normalized = (
        weighted / torch.where(positive_mass, mass, torch.ones_like(mass))[:, None, :]
    )
    ties = (
        valid[..., None]
        & iou.eq(iou.masked_fill(~valid[..., None], -1).amax(1)[:, None, :])
        & gt_mask[:, None, :]
    )
    fallback = ties.float() / ties.sum(1).clamp_min(1).float()[:, None, :]
    qg = torch.where(positive_mass[:, None, :], normalized, fallback).detach()
    return qg, valid, positive_mass


def rank_partition(score, geometry, gt_mask, competition):
    qg, valid, positive_mass = rank_target_components(geometry, gt_mask)
    P = qg.gt(0)
    all_positive = P.any(-1, keepdim=True)
    D = valid[..., None].expand_as(P)
    if competition == "gt_conditional":
        D = D & (~all_positive | P)
    elif competition != "global":
        raise ValueError(competition)
    active_gt = gt_mask.bool() & valid.any(1)[:, None]
    # P subset D and nonempty on each active GT are necessary semantic invariants.
    if bool((P & ~D).any()) or bool((active_gt & ~P.any(1)).any()):
        raise AssertionError("positive bag escaped candidate domain")
    z = score.float().flatten(1)[..., None].expand_as(qg)
    safe_D = D & active_gt[:, None, :]
    safe_P = P & active_gt[:, None, :]
    # Inactive reductions use finite dummy logits, then are excluded from query means.
    zd = torch.where(
        active_gt[:, None, :], z.masked_fill(~safe_D, -torch.inf), torch.zeros_like(z)
    )
    zp = torch.where(
        active_gt[:, None, :], z.masked_fill(~safe_P, -torch.inf), torch.zeros_like(z)
    )
    ld, lp = torch.logsumexp(zd, 1), torch.logsumexp(zp, 1)
    return qg, valid, P, D, active_gt, positive_mass, ld, lp


def configured_rank_loss(
    score,
    geometry,
    gt_mask,
    rank_target_threshold=0.7,
    rank_target_exponent=4.0,
    *,
    rank_form="dense",
    rank_competition="global",
):
    if (
        rank_target_threshold == 0.7
        and rank_target_exponent == 2.0
        and rank_form == "dense"
        and rank_competition == "global"
    ):
        return gt_balanced_kl_rank_loss(
            score, geometry, gt_mask, rank_target_threshold, rank_target_exponent
        )
    if rank_target_threshold != 0.7 or rank_target_exponent != 4.0:
        raise ValueError("Round31 fixes q threshold .7 and exponent4")
    if rank_form not in ("dense", "positive_bag"):
        raise ValueError(rank_form)
    # Fresh G0/G1 exact control: not algebraically rewritten or reweighted.
    if rank_form == "dense" and rank_competition == "global":
        return gt_balanced_kl_rank_loss(
            score, geometry, gt_mask, rank_target_threshold, rank_target_exponent
        )
    qg, valid, P, D, active, positive, ld, lp = rank_partition(
        score, geometry, gt_mask, rank_competition
    )
    z = score.float().flatten(1).masked_fill(~valid, 0)
    entropy = -(qg * qg.clamp_min(1e-30).log()).sum(1)
    ce = ld - (qg * z[..., None]).sum(1)
    per_gt = ce - entropy if rank_form == "dense" else ld - lp
    count = active.sum(1).clamp_min(1).float()
    active_query = active.any(1)

    def reduce(value):
        per_query = value.masked_fill(~active, 0).sum(1) / count
        return (
            per_query[active_query].mean()
            if bool(active_query.any())
            else _safe_zero(score)
        )

    loss = reduce(per_gt)
    metrics = {
        "score_span": _row_score_span(score, geometry.valid).detach(),
        "loss": loss.detach(),
        "active_query_count": active_query.sum().detach(),
        "gt_count": active.sum().detach(),
        "candidate_count": valid.sum(1).float().mean().detach(),
        "target_threshold": score.new_tensor(0.7),
        "target_exponent": score.new_tensor(4.0),
        "CE": reduce(ce).detach(),
        "per_GT_target_entropy": reduce(entropy).detach(),
        "positive_bag_mass": reduce((lp - ld).exp()).detach(),
        "excluded_candidate_fraction": reduce(
            (valid[..., None] & ~D).sum(1).float()
            / valid.sum(1).clamp_min(1).float()[:, None]
        ).detach(),
        "fallback_gt_rate": (
            (~positive & active).sum().float() / active.sum().clamp_min(1)
        ).detach(),
        "dense_loss_entropy_scope_perGT": score.new_tensor(float(rank_form == "dense")),
        "positive_bag_objective": score.new_tensor(float(rank_form == "positive_bag")),
        "conditional_competition": score.new_tensor(
            float(rank_competition == "gt_conditional")
        ),
    }
    return loss, metrics


def evidence_reduction_values(correct, wrong, mask, zero_reference):
    """Same eligible hinge entries, with candidate- or active-query weighting."""
    selected = (correct - wrong)[mask]
    candidate = (
        F.relu(0.20 - selected).mean()
        if selected.numel()
        else _safe_zero(zero_reference)
    )
    counts = mask.flatten(1).sum(1)
    active = counts > 0
    hinge = F.relu(0.20 - (correct - wrong)).masked_fill(~mask, 0)
    means = hinge.flatten(1).sum(1) / counts.clamp_min(1)
    query = means[active].mean() if active.any() else _safe_zero(zero_reference)
    return candidate, query, counts, means, hinge


def evidence_background_indices(geometry, spans, gt_mask):
    """Same-length background, ascending flat IDs; positive flat ID modulo pool size.

    Touching the expanded GT boundary is permitted (zero temporal overlap).
    Selection is geometry-only and never uses a score or a model feature.
    """
    b, l, _ = geometry.valid.shape
    n = l * l
    gs = (
        torch.minimum(spans[..., 0], spans[..., 1])[:, None, None, :]
        - geometry.one_step[:, None, None, None]
    )
    ge = (
        torch.maximum(spans[..., 0], spans[..., 1])[:, None, None, :]
        + geometry.one_step[:, None, None, None]
    )
    overlap = (
        (geometry.candidate_end[..., None] > gs)
        & (geometry.candidate_start[..., None] < ge)
        & gt_mask[:, None, None, :].bool()
    )
    background = geometry.valid & ~overlap.any(-1)
    ids = torch.arange(n, device=spans.device).expand(b, -1)
    widths = ids % l - ids // l + 1
    keys = (widths * n + ids).masked_fill(~background.flatten(1), (l + 2) * n)
    sorted_keys, sorted_ids = keys.sort(1)
    begin = torch.searchsorted(sorted_keys, (widths * n).contiguous())
    end = torch.searchsorted(sorted_keys, ((widths + 1) * n).contiguous())
    counts = end - begin
    positions = (begin + ids.remainder(counts.clamp_min(1))).clamp_max(n - 1)
    chosen = sorted_ids.gather(1, positions).masked_fill(counts == 0, -1)
    return chosen.reshape(b, l, l).detach(), background.detach()


def evidence_objective_values(outputs, batch, geometry, wrong, mask, supervision):
    field = outputs.trifield_output
    query_hinge = F.relu(0.20 - (field.evidence - wrong))
    ids = torch.full_like(mask, -1, dtype=torch.long)
    background_hinge = torch.zeros_like(query_hinge)
    matched = torch.zeros_like(mask)
    if supervision == "query_plus_background":
        spans, gtmask = _target_spans(outputs, batch)
        ids, _ = evidence_background_indices(geometry, spans, gtmask)
        matched = mask & ids.ge(0)
        background = (
            field.evidence.flatten(1)
            .gather(1, ids.flatten(1).clamp_min(0))
            .reshape_as(field.evidence)
        )
        background_hinge = F.relu(0.20 - (field.evidence - background))
    combined = torch.where(
        matched, 0.5 * query_hinge + 0.5 * background_hinge, query_hinge
    )
    counts = mask.flatten(1).sum(1)
    active = counts > 0
    candidate = combined[mask].mean() if mask.any() else _safe_zero(field.raw_evidence)
    means = combined.masked_fill(~mask, 0).flatten(1).sum(1) / counts.clamp_min(1)
    query = means[active].mean() if active.any() else _safe_zero(field.raw_evidence)
    return (
        candidate,
        query,
        counts,
        means,
        combined.masked_fill(~mask, 0),
        {
            "ids": ids,
            "matched": matched,
            "query_hinge": query_hinge,
            "background_hinge": background_hinge,
        },
    )
