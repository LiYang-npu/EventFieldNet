"""Backend-independent complete-span calibration and ranking objective."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class SpanSetObjectiveResult:
    losses: Dict[str, Tensor]
    diagnostics: Dict[str, Tensor]
    maximum_iou: Tensor
    deployed_score: Tensor


def _as_score(values: Tensor) -> Tensor:
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[..., 0]
    if values.ndim != 2:
        raise ValueError(
            "candidate logits must have shape [batch, candidates] or [..., 1]"
        )
    return values.float()


def _pairwise_iou(spans: Tensor, targets: Tensor) -> Tensor:
    left = torch.maximum(spans[:, None, 0], targets[None, :, 0])
    right = torch.minimum(spans[:, None, 1], targets[None, :, 1])
    intersection = (right - left).clamp_min(0)
    span_width = (spans[:, 1] - spans[:, 0]).clamp_min(0)
    target_width = (targets[:, 1] - targets[:, 0]).clamp_min(0)
    union = span_width[:, None] + target_width[None, :] - intersection
    return intersection / union.clamp_min(1.0e-8)


def _maximum_iou(spans_xx: Tensor, targets_xx: Sequence[Tensor]) -> Tensor:
    rows = []
    for spans, targets in zip(spans_xx.float(), targets_xx):
        targets = targets.to(device=spans.device, dtype=torch.float32)
        if targets.ndim != 2 or targets.shape[-1] != 2 or targets.shape[0] == 0:
            raise ValueError("each example needs at least one [start, end] target")
        rows.append(_pairwise_iou(spans, targets).amax(1))
    if len(rows) != spans_xx.shape[0]:
        raise ValueError("target batch size differs from candidate batch size")
    return torch.stack(rows)


def _masked_correlation(x: Tensor, y: Tensor, valid: Tensor) -> Tensor:
    values = []
    for row_x, row_y, row_valid in zip(x, y, valid):
        row_x, row_y = row_x[row_valid], row_y[row_valid]
        if row_x.numel() < 2:
            continue
        row_x = row_x - row_x.mean()
        row_y = row_y - row_y.mean()
        denominator = row_x.square().sum().sqrt() * row_y.square().sum().sqrt()
        values.append((row_x * row_y).sum() / denominator.clamp_min(1.0e-8))
    return torch.stack(values).mean() if values else x.sum() * 0.0


def _rank(values: Tensor) -> Tensor:
    order = values.argsort()
    rank = torch.empty_like(order, dtype=torch.float32)
    rank.scatter_(0, order, torch.arange(values.numel(), device=values.device).float())
    return rank


def _masked_spearman(x: Tensor, y: Tensor, valid: Tensor) -> Tensor:
    values = []
    for row_x, row_y, row_valid in zip(x, y, valid):
        row_x, row_y = row_x[row_valid], row_y[row_valid]
        if row_x.numel() < 2:
            continue
        ones = torch.ones_like(row_x, dtype=torch.bool)[None]
        values.append(_masked_correlation(_rank(row_x)[None], _rank(row_y)[None], ones))
    return torch.stack(values).mean() if values else x.sum() * 0.0


class TriFieldSpanSetObjective(nn.Module):
    """Optimize the ordering of all complete spans emitted by any backend."""

    def __init__(
        self,
        *,
        target_temperature: float = 0.10,
        score_temperature: float = 1.0,
        semantic_score_weight: float = 0.5,
        quality_score_weight: float = 0.5,
        strict_positive_iou: float = 0.70,
        strict_negative_iou: float = 0.30,
        ranking_margin: float = 0.20,
        wrong_query_margin: float = 0.20,
    ) -> None:
        super().__init__()
        if target_temperature <= 0 or score_temperature <= 0:
            raise ValueError("temperatures must be positive")
        if semantic_score_weight <= 0:
            raise ValueError("semantic_score_weight must be positive")
        if quality_score_weight < 0:
            raise ValueError("quality_score_weight cannot be negative")
        self.target_temperature = float(target_temperature)
        self.score_temperature = float(score_temperature)
        self.semantic_score_weight = float(semantic_score_weight)
        self.quality_score_weight = float(quality_score_weight)
        self.strict_positive_iou = float(strict_positive_iou)
        self.strict_negative_iou = float(strict_negative_iou)
        self.ranking_margin = float(ranking_margin)
        self.wrong_query_margin = float(wrong_query_margin)

    def forward(
        self,
        *,
        spans_xx: Tensor,
        semantic_logits: Tensor,
        targets_xx: Sequence[Tensor],
        quality_logits: Optional[Tensor] = None,
        candidate_valid: Optional[Tensor] = None,
        evidence_score: Optional[Tensor] = None,
        wrong_query_evidence_score: Optional[Tensor] = None,
    ) -> SpanSetObjectiveResult:
        semantic = _as_score(semantic_logits)
        quality = _as_score(quality_logits) if quality_logits is not None else None
        if spans_xx.shape != (*semantic.shape, 2):
            raise ValueError("spans_xx and semantic logits shapes are inconsistent")
        valid = (
            torch.ones_like(semantic, dtype=torch.bool)
            if candidate_valid is None
            else candidate_valid.bool()
        )
        if valid.shape != semantic.shape or not valid.any(1).all():
            raise ValueError("every example must have at least one valid candidate")
        iou = (
            _maximum_iou(spans_xx.detach(), targets_xx)
            .detach()
            .masked_fill(~valid, 0.0)
        )
        deployed = self.semantic_score_weight * F.logsigmoid(semantic)
        if quality is not None:
            deployed = deployed + self.quality_score_weight * F.logsigmoid(quality)
        deployed = deployed.masked_fill(~valid, -1.0e4)

        zero = deployed[valid].sum() * 0.0
        if quality is None:
            all_quality = zero
        else:
            all_quality = F.binary_cross_entropy_with_logits(
                quality[valid], iou[valid], reduction="mean"
            )
        target_logits = (iou / self.target_temperature).masked_fill(~valid, -1.0e4)
        target_distribution = F.softmax(target_logits, 1).masked_fill(~valid, 0.0)
        predicted_log_distribution = F.log_softmax(deployed / self.score_temperature, 1)
        listwise = -(target_distribution * predicted_log_distribution).sum(1).mean()

        raw_margins = []
        for row_score, row_iou, row_valid in zip(deployed, iou, valid):
            positive = row_valid & row_iou.ge(self.strict_positive_iou)
            if not positive.any():
                positive = row_valid & row_iou.eq(row_iou[row_valid].max())
            low_negative = row_valid & row_iou.le(self.strict_negative_iou)
            near_negative = row_valid & row_iou.gt(self.strict_negative_iou) & ~positive
            negative = low_negative | near_negative
            if negative.any():
                raw_margins.append(
                    row_score[positive].max() - row_score[negative].max()
                )
        if raw_margins:
            raw_margin = torch.stack(raw_margins)
            margin_loss = F.softplus(self.ranking_margin - raw_margin).mean()
        else:
            raw_margin = deployed.new_empty(0)
            margin_loss = zero

        wrong_query_loss = zero
        wrong_query_raw = deployed.new_empty(0)
        if evidence_score is not None and wrong_query_evidence_score is not None:
            evidence = _as_score(evidence_score)
            wrong = _as_score(wrong_query_evidence_score)
            if evidence.shape != semantic.shape or wrong.shape != semantic.shape:
                raise ValueError(
                    "wrong-query evidence shapes differ from candidate scores"
                )
            query_margins = []
            for right_row, wrong_row, row_iou, row_valid in zip(
                evidence, wrong, iou, valid
            ):
                positive = row_valid & row_iou.ge(self.strict_positive_iou)
                if not positive.any():
                    positive = row_valid & row_iou.eq(row_iou[row_valid].max())
                # Same high-IoU spans under the correct and a mismatched query.
                query_margins.append(
                    right_row[positive].max() - wrong_row[positive].max()
                )
            wrong_query_raw = torch.stack(query_margins)
            wrong_query_loss = F.softplus(
                self.wrong_query_margin - wrong_query_raw
            ).mean()

        top_index = deployed.argmax(1)
        top1_iou = iou.gather(1, top_index[:, None]).squeeze(1)
        diagnostics: Dict[str, Tensor] = {
            "candidate_oracle_iou": iou.masked_fill(~valid, -1.0).max(1).values.mean(),
            "candidate_top1_iou": top1_iou.mean(),
            "candidate_top1_r07": top1_iou.ge(0.70).float().mean(),
            "candidate_score_pearson": _masked_correlation(
                deployed.detach(), iou, valid
            ),
            "candidate_score_spearman": _masked_spearman(deployed.detach(), iou, valid),
            "strict_margin": raw_margin.mean() if raw_margin.numel() else zero.detach(),
            "strict_margin_positive": (
                raw_margin.gt(0).float().mean() if raw_margin.numel() else zero.detach()
            ),
            "all_candidate_quality_loss": all_quality.detach(),
            "candidate_listwise_loss": listwise.detach(),
            "candidate_margin_loss": margin_loss.detach(),
        }
        for k in (5, 25):
            count = min(k, deployed.shape[1])
            indices = deployed.topk(count, 1).indices
            top_iou = iou.gather(1, indices).max(1).values
            diagnostics[f"candidate_top{k}_r07"] = top_iou.ge(0.70).float().mean()
        if wrong_query_raw.numel():
            diagnostics.update(
                {
                    "wrong_query_margin": wrong_query_raw.mean().detach(),
                    "wrong_query_margin_positive": wrong_query_raw.gt(0).float().mean(),
                    "wrong_query_loss": wrong_query_loss.detach(),
                }
            )
        return SpanSetObjectiveResult(
            losses={
                "loss_all_candidate_quality": all_quality,
                "loss_candidate_listwise": listwise,
                "loss_candidate_margin": margin_loss,
                "loss_wrong_query_field": wrong_query_loss,
            },
            diagnostics=diagnostics,
            maximum_iou=iou,
            deployed_score=deployed,
        )


__all__ = ["SpanSetObjectiveResult", "TriFieldSpanSetObjective"]
