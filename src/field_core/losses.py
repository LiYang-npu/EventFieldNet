"""The five direct objectives used by the new trifield base.

The module contains no model construction and never calls a legacy
``compute_loss`` implementation.  It only consumes the final all-candidate
score and the unchanged span grid.  Ground-truth matching is max IoU with a
stable first-index tie break; the selected ground truth is reused by S, Ts,
and Te.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor
from torch.nn import functional as F


OFFICIAL_THRESHOLDS = tuple(round(0.50 + 0.05 * i, 2) for i in range(10))
GRADE_COUNT = len(OFFICIAL_THRESHOLDS) + 1


@dataclass
class GeometryTargets:
    """Geometry labels on the model's unchanged ``[start, end]`` grid."""

    all_iou: Tensor
    max_iou: Tensor
    best_gt_index: Tensor
    selected_start: Tensor
    selected_end: Tensor
    selected_width: Tensor
    candidate_start: Tensor
    candidate_end: Tensor
    candidate_width: Tensor
    one_step: Tensor
    valid: Tensor
    has_ground_truth: Tensor


def _mapping_value(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _batch_parts(batch: Any) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    inputs = _mapping_value(batch, "inputs", {})
    targets = _mapping_value(batch, "targets", {})
    if not isinstance(inputs, Mapping) or not isinstance(targets, Mapping):
        raise TypeError("batch inputs and targets must be mappings")
    return inputs, targets


def _video_padding(outputs: Any, batch: Any) -> Tensor:
    inputs, _ = _batch_parts(batch)
    pad = inputs.get("video_padding_mask")
    if pad is None:
        token = getattr(outputs, "token_features", None)
        if token is None:
            raise KeyError(
                "video_padding_mask is absent and token_features are unavailable"
            )
        return torch.zeros(token.shape[:2], dtype=torch.bool, device=token.device)
    return pad.bool()


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
        raise ValueError(
            "gt_spans must be [B,G,2] and gt_span_mask must be [B,G], "
            f"got {tuple(spans.shape)} and {tuple(mask.shape)}"
        )
    return spans, mask


def candidate_geometry(outputs: Any, batch: Any) -> GeometryTargets:
    """Return exact candidate/GT geometry and stable max-IoU assignments.

    This mirrors the Stage55 coordinate convention: candidate ``i:j`` covers
    ``[i / valid_length, (j + 1) / valid_length]``.  ``argmax`` selects the
    first ground-truth index on ties.  All tensors here are labels or geometry
    derived from labels and are detached from model parameters.
    """

    score = outputs.span_logits
    if score.ndim != 3 or score.shape[-1] != score.shape[-2]:
        raise ValueError("span_logits must be a square [B,L,L] matrix")
    b, length, _ = score.shape
    pad = _video_padding(outputs, batch).to(device=score.device)
    if pad.shape != (b, length):
        raise ValueError(
            f"video_padding_mask shape {tuple(pad.shape)} != {(b, length)}"
        )
    spans, gt_mask = _target_spans(outputs, batch)
    spans = spans.to(device=score.device)
    gt_mask = gt_mask.to(device=score.device)

    count = (~pad).sum(1).clamp_min(1).float()
    index = torch.arange(length, device=score.device, dtype=torch.float32)
    candidate_start = index[None, :, None] / count[:, None, None]
    candidate_end = (index[None, None, :] + 1.0) / count[:, None, None]
    candidate_width = (candidate_end - candidate_start).clamp_min(1.0e-6)
    # The parent span head's minimum is two clips.  A caller can still provide
    # a different mask; this only reconstructs geometry, never membership.
    valid = outputs.span_valid_mask.bool().to(device=score.device)
    if valid.shape != (b, length, length):
        raise ValueError("span_valid_mask must match span_logits")

    gt_start = torch.minimum(spans[..., 0], spans[..., 1])
    gt_end = torch.maximum(spans[..., 0], spans[..., 1])
    gt_width = (gt_end - gt_start).clamp_min(1.0e-6)
    inter = (
        torch.minimum(candidate_end[..., None], gt_end[:, None, None, :])
        - torch.maximum(candidate_start[..., None], gt_start[:, None, None, :])
    ).clamp_min(0.0)
    union = (candidate_width[..., None] + gt_width[:, None, None, :] - inter).clamp_min(
        1.0e-6
    )
    all_iou = (inter / union).masked_fill(~gt_mask[:, None, None, :], 0.0)
    # Select only real GT slots. With no valid GT, argmax deterministically
    # returns slot zero and max IoU is clamped back to zero below.
    selection_iou = all_iou.masked_fill(~gt_mask[:, None, None, :], -1.0)
    best_index = selection_iou.argmax(dim=-1)
    max_iou = selection_iou.gather(-1, best_index[..., None]).squeeze(-1).clamp_min(0.0)
    selected_start = (
        gt_start[:, None, None, :]
        .expand(-1, length, length, -1)
        .gather(-1, best_index[..., None])
        .squeeze(-1)
    )
    selected_end = (
        gt_end[:, None, None, :]
        .expand(-1, length, length, -1)
        .gather(-1, best_index[..., None])
        .squeeze(-1)
    )
    selected_width = (
        gt_width[:, None, None, :]
        .expand(-1, length, length, -1)
        .gather(-1, best_index[..., None])
        .squeeze(-1)
    )
    has_ground_truth = gt_mask.any(1)[:, None, None].expand(-1, length, length)
    valid = valid & has_ground_truth

    # Labels must not carry accidental graph edges or be modified by the
    # candidate score's dtype (which is BF16 in the real smoke test).
    return GeometryTargets(
        all_iou=all_iou.detach(),
        max_iou=max_iou.detach().masked_fill(~valid, 0.0),
        best_gt_index=best_index.detach(),
        selected_start=selected_start.detach(),
        selected_end=selected_end.detach(),
        selected_width=selected_width.detach(),
        candidate_start=candidate_start.expand(b, -1, length).detach(),
        candidate_end=candidate_end.expand(b, length, -1).detach(),
        candidate_width=candidate_width.expand(b, -1, -1).detach(),
        one_step=(1.0 / count).detach(),
        valid=valid.detach(),
        has_ground_truth=has_ground_truth.detach(),
    )


def _zero_like(value: Tensor) -> Tensor:
    # Keep a graph edge so a skipped objective is visible in gradient probes.
    return value.float().sum() * 0.0


def official_threshold_grades(iou: Tensor, valid: Tensor) -> Tensor:
    thresholds = iou.new_tensor(OFFICIAL_THRESHOLDS)
    grades = iou.float().unsqueeze(-1).ge(thresholds).sum(-1).long()
    return grades.masked_fill(~valid.bool(), 0)


def _ordinal_row(score: Tensor, grade: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
    """Exact zero-margin all-pair ordinal violation without an N² tensor."""

    if score.numel() == 0:
        zero = score.sum() * 0.0
        return zero, {
            "eligible_pair_count": zero.detach(),
            "violating_pair_count": zero.detach(),
            "violating_pair_rate": zero.detach(),
            "weighted_signed_margin": zero.detach(),
        }
    score = score.float()
    grade = grade.long()
    order = torch.argsort(score.detach(), stable=True)
    ordered_score = score[order]
    ordered_grade = grade[order]
    one_hot = F.one_hot(ordered_grade, num_classes=GRADE_COUNT).to(ordered_score.dtype)
    suffix_count = torch.flip(torch.cumsum(torch.flip(one_hot, (0,)), 0), (0,))
    suffix_score = torch.flip(
        torch.cumsum(torch.flip(one_hot * ordered_score[:, None], (0,)), 0), (0,)
    )
    index = torch.arange(ordered_score.numel(), device=score.device)
    group_start_flag = torch.ones_like(index, dtype=torch.bool)
    if index.numel() > 1:
        group_start_flag[1:] = ordered_score[1:].ne(ordered_score[:-1])
    marker = torch.where(group_start_flag, index, index.new_zeros(()))
    group_start = torch.cummax(marker, 0).values
    lower_grade = torch.arange(GRADE_COUNT, device=score.device)[None, :].lt(
        ordered_grade[:, None]
    )
    reliability = (
        ordered_grade[:, None] - torch.arange(GRADE_COUNT, device=score.device)[None, :]
    ).clamp_min(0).to(ordered_score.dtype) / float(GRADE_COUNT - 1)
    reliability = reliability * lower_grade.to(reliability.dtype)
    pair_violation = (
        suffix_score[group_start] - ordered_score[:, None] * suffix_count[group_start]
    )
    weighted_violation = (reliability * pair_violation).sum()
    violating_pairs = (
        lower_grade.to(ordered_score.dtype) * suffix_count[group_start]
    ).sum()
    counts = torch.bincount(grade, minlength=GRADE_COUNT).to(ordered_score.dtype)
    bins = torch.arange(GRADE_COUNT, device=score.device)
    high, low = bins[:, None], bins[None, :]
    pair_reliability = (high - low).clamp_min(0).to(ordered_score.dtype) / float(
        GRADE_COUNT - 1
    )
    weighted_pair_count = (pair_reliability * counts[:, None] * counts[None, :]).sum()
    eligible_pair_count = (
        high.gt(low).to(ordered_score.dtype) * counts[:, None] * counts[None, :]
    ).sum()
    zero = score.sum() * 0.0
    if not bool(weighted_pair_count.gt(0.0)):
        return zero, {
            "eligible_pair_count": eligible_pair_count.detach(),
            "violating_pair_count": violating_pairs.detach(),
            "violating_pair_rate": zero.detach(),
            "weighted_signed_margin": zero.detach(),
        }
    grade_score_sum = torch.zeros(
        GRADE_COUNT, device=score.device, dtype=ordered_score.dtype
    ).scatter_add_(0, grade, score)
    signed_sum = (
        pair_reliability
        * (
            grade_score_sum[:, None] * counts[None, :]
            - counts[:, None] * grade_score_sum[None, :]
        )
    ).sum()
    return weighted_violation / weighted_pair_count, {
        "eligible_pair_count": eligible_pair_count.detach(),
        "violating_pair_count": violating_pairs.detach(),
        "violating_pair_rate": (
            violating_pairs / eligible_pair_count.clamp_min(1.0)
        ).detach(),
        "weighted_signed_margin": (signed_sum / weighted_pair_count).detach(),
    }


def official_ordinal_violation_loss(
    score: Tensor, iou: Tensor, valid: Tensor
) -> tuple[Tensor, Mapping[str, Tensor]]:
    """Official grade ordering over every valid candidate in every query."""

    if score.shape != iou.shape or score.shape != valid.shape:
        raise ValueError("score, iou, and valid must have matching shapes")
    flat_score, flat_iou, flat_valid = (
        score.float().flatten(1),
        iou.float().flatten(1),
        valid.bool().flatten(1),
    )
    grades = official_threshold_grades(flat_iou, flat_valid)
    losses: list[Tensor] = []
    diagnostics: dict[str, list[Tensor]] = {
        "eligible_pair_count": [],
        "violating_pair_count": [],
        "violating_pair_rate": [],
        "weighted_signed_margin": [],
    }
    top1: list[Tensor] = []
    active = 0
    for row_score, row_grade, row_valid in zip(flat_score, grades, flat_valid):
        selected_score, selected_grade = row_score[row_valid], row_grade[row_valid]
        row_loss, row_diag = _ordinal_row(selected_score, selected_grade)
        if bool(row_diag["eligible_pair_count"].gt(0.0)):
            active += 1
            losses.append(row_loss)
        for name, value in row_diag.items():
            diagnostics[name].append(value)
        if selected_score.numel():
            top1.append(
                selected_grade[selected_score.argmax()].to(selected_score.dtype)
            )
    zero = score.float().sum() * 0.0
    loss = torch.stack(losses).mean() if losses else zero

    def mean_or_zero(values: list[Tensor]) -> Tensor:
        return torch.stack(values).float().mean() if values else zero.detach()

    metrics = {
        name: mean_or_zero(values).detach() for name, values in diagnostics.items()
    }
    metrics.update(
        {
            "active_query_rate": score.new_tensor(
                active / max(1, flat_score.shape[0]), dtype=torch.float32
            ).detach(),
            "top1_official_grade": mean_or_zero(top1).detach(),
            "loss": loss.detach(),
        }
    )
    return loss, metrics


def quality_targets(geometry: GeometryTargets) -> tuple[Tensor, Tensor, Tensor]:
    """Return balanced coverage S and normalized endpoint Ts/Te targets."""

    inter = (
        torch.minimum(geometry.candidate_end, geometry.selected_end)
        - torch.maximum(geometry.candidate_start, geometry.selected_start)
    ).clamp_min(0.0)
    # candidate coordinates are normalized, so one valid temporal step is the
    # correct nonzero denominator even when a dataset supplies a zero-width GT.
    one_step = geometry.one_step[:, None, None]
    gt_width = geometry.selected_width.clamp_min(one_step)
    precision = (inter / geometry.candidate_width.clamp_min(1.0e-6)).clamp(0.0, 1.0)
    recall = (inter / gt_width).clamp(0.0, 1.0)
    support = 2.0 * torch.minimum(recall, precision) - 1.0
    denominator = gt_width
    start_quality = (
        1.0 - (geometry.candidate_start - geometry.selected_start).abs() / denominator
    ).clamp(0.0, 1.0)
    end_quality = (
        1.0 - (geometry.candidate_end - geometry.selected_end).abs() / denominator
    ).clamp(0.0, 1.0)
    return (
        support.detach().masked_fill(~geometry.valid, 0.0),
        (2.0 * start_quality - 1.0).detach().masked_fill(~geometry.valid, 0.0),
        (2.0 * end_quality - 1.0).detach().masked_fill(~geometry.valid, 0.0),
    )


def _tolerant_square(
    prediction: Tensor, target: Tensor, valid: Tensor, tolerance: float = 0.1
) -> Tensor:
    error = (prediction.float() - target.float()).abs() - float(tolerance)
    value = F.relu(error).square()
    selected = value[valid]
    return selected.mean() if selected.numel() else _zero_like(prediction)


def endpoint_gaussian_targets(
    outputs: Any, batch: Any
) -> tuple[Tensor, Tensor, Tensor]:
    """Original dense Gaussian start/end targets and nonpadding mask."""

    inputs, targets = _batch_parts(batch)
    start_logits, _end_logits = outputs.start_logits, outputs.end_logits
    pad = inputs.get("video_padding_mask")
    if pad is None:
        pad = torch.zeros(
            start_logits.shape, dtype=torch.bool, device=start_logits.device
        )
    pad = pad.bool().to(start_logits.device)
    spans = targets.get("gt_spans")
    gt_mask = targets.get("gt_span_mask")
    if spans is None:
        spans = start_logits.new_zeros((start_logits.shape[0], 1, 2))
        gt_mask = torch.zeros(
            spans.shape[:2], dtype=torch.bool, device=start_logits.device
        )
    spans = spans.to(start_logits.device, dtype=torch.float32)
    gt_mask = gt_mask.bool().to(start_logits.device)
    count = (~pad).sum(1).clamp_min(1).float()
    center = (
        torch.arange(
            start_logits.shape[-1], device=start_logits.device, dtype=torch.float32
        )[None, :]
        + 0.5
    ) / count[:, None]
    gs, ge = (
        torch.minimum(spans[..., 0], spans[..., 1]),
        torch.maximum(spans[..., 0], spans[..., 1]),
    )
    scale = count[:, None, None]
    start_target = (
        torch.exp(-0.5 * (((center[..., None] - gs[:, None]) * scale / 1.5) ** 2))
        .masked_fill(~gt_mask[:, None], 0.0)
        .amax(-1)
        .masked_fill(pad, 0.0)
    )
    end_target = (
        torch.exp(-0.5 * (((center[..., None] - ge[:, None]) * scale / 1.5) ** 2))
        .masked_fill(~gt_mask[:, None], 0.0)
        .amax(-1)
        .masked_fill(pad, 0.0)
    )
    return start_target.detach(), end_target.detach(), (~pad).detach()


def endpoint_bce_loss(outputs: Any, batch: Any) -> Tensor:
    _, targets = _batch_parts(batch)
    gt_mask = targets.get("gt_span_mask")
    if gt_mask is None or not bool(gt_mask.bool().any()):
        return _zero_like(outputs.start_logits) + _zero_like(outputs.end_logits)
    start_target, end_target, valid_token = endpoint_gaussian_targets(outputs, batch)
    if not bool(valid_token.any()):
        return _zero_like(outputs.start_logits) + _zero_like(outputs.end_logits)
    return 0.5 * (
        F.binary_cross_entropy_with_logits(
            outputs.start_logits.float()[valid_token], start_target[valid_token]
        )
        + F.binary_cross_entropy_with_logits(
            outputs.end_logits.float()[valid_token], end_target[valid_token]
        )
    )


@dataclass
class FiveLossTerms:
    rank: Tensor
    evidence: Tensor
    support: Tensor
    transition: Tensor
    endpoint: Tensor
    total: Tensor
    geometry: GeometryTargets
    support_target: Tensor
    transition_start_target: Tensor
    transition_end_target: Tensor
    metrics: Mapping[str, Tensor | float]


def compute_five_loss_terms(
    outputs: Any,
    batch: Any,
    wrong_evidence: Tensor | None = None,
    evidence_pair_mask: Tensor | None = None,
) -> FiveLossTerms:
    """Compute the frozen five-term objective on the actual deployment score."""

    if getattr(outputs, "trifield_output", None) is None:
        raise RuntimeError("outputs.trifield_output is required for the five-loss base")
    field = outputs.trifield_output
    geometry = candidate_geometry(outputs, batch)
    valid = geometry.valid
    score = field.score
    rank, rank_metrics = official_ordinal_violation_loss(score, geometry.max_iou, valid)
    support_target, ts_target, te_target = quality_targets(geometry)
    support = _tolerant_square(field.support, support_target, valid)
    transition_mask = valid
    transition = 0.5 * (
        _tolerant_square(field.transition_start, ts_target, transition_mask)
        + _tolerant_square(field.transition_end, te_target, transition_mask)
    )
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
        delta = field.evidence - wrong_evidence
        selected = delta[pair_mask]
        evidence = (
            F.relu(0.20 - selected).mean()
            if selected.numel()
            else _zero_like(field.raw_evidence)
        )
        pair_count = pair_mask.float().sum().detach()
    else:
        evidence = _zero_like(field.raw_evidence)
        pair_count = field.raw_evidence.new_zeros(())

    total = rank + 0.1 * (evidence + support + transition + endpoint)
    metrics: dict[str, Tensor | float] = {
        "rank": rank.detach(),
        "evidence": evidence.detach(),
        "support": support.detach(),
        "transition": transition.detach(),
        "endpoint": endpoint.detach(),
        "total": total.detach(),
        "rank/eligible_pair_count": rank_metrics["eligible_pair_count"],
        "rank/violating_pair_rate": rank_metrics["violating_pair_rate"],
        "rank/top1_official_grade": rank_metrics["top1_official_grade"],
        "evidence/positive_pair_count": pair_count,
        "support/valid_candidate_rate": valid.float().mean().detach(),
        "geometry/multigt_gt_count": _batch_parts(batch)[1]
        .get("gt_span_mask", valid.new_zeros((valid.shape[0], 1)))
        .bool()
        .sum(1)
        .float()
        .mean()
        .detach(),
    }
    return FiveLossTerms(
        rank=rank,
        evidence=evidence,
        support=support,
        transition=transition,
        endpoint=endpoint,
        total=total,
        geometry=geometry,
        support_target=support_target,
        transition_start_target=ts_target,
        transition_end_target=te_target,
        metrics=metrics,
    )


__all__ = [
    "OFFICIAL_THRESHOLDS",
    "GRADE_COUNT",
    "GeometryTargets",
    "FiveLossTerms",
    "candidate_geometry",
    "quality_targets",
    "endpoint_gaussian_targets",
    "endpoint_bce_loss",
    "official_threshold_grades",
    "official_ordinal_violation_loss",
    "compute_five_loss_terms",
]
