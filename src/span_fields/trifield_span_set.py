"""Backend-independent Evidence/Support/Transition span-set scoring.

This module deliberately knows nothing about DETR, Hungarian matching, decoder
queries, anchors, or a particular repository.  A backend only has to expose a
set of normalized ``[start, end]`` spans and the frame-level three-field state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class TriFieldSequence:
    """Common frame-level contract consumed by the span-set scorer.

    All score tensors have shape ``[batch, frames]``. ``role_features`` is an
    optional ``[batch, frames, 3, channels]`` tensor ordered as Evidence,
    Support, Transition. ``query_identity`` is an optional candidate-independent
    ``[batch, channels]`` whole-sentence identity vector. ``valid_mask`` uses
    True for real frames.
    """

    evidence: Tensor
    support: Tensor
    transition_start: Tensor
    transition_end: Tensor
    valid_mask: Tensor
    role_features: Optional[Tensor] = None
    query_identity: Optional[Tensor] = None

    def validate(self) -> None:
        shape = self.evidence.shape
        if self.evidence.ndim != 2:
            raise ValueError("field scores must have shape [batch, frames]")
        for name, value in (
            ("support", self.support),
            ("transition_start", self.transition_start),
            ("transition_end", self.transition_end),
            ("valid_mask", self.valid_mask),
        ):
            if value.shape != shape:
                raise ValueError(f"{name} shape {tuple(value.shape)} != {tuple(shape)}")
        if self.role_features is not None:
            expected = (*shape, 3)
            if self.role_features.ndim != 4 or self.role_features.shape[:3] != expected:
                raise ValueError(
                    "role_features must have shape [batch, frames, 3, channels]"
                )
        if self.query_identity is not None:
            if (
                self.query_identity.ndim != 2
                or self.query_identity.shape[0] != shape[0]
            ):
                raise ValueError("query_identity must have shape [batch, channels]")


def _masked_standardize(values: Tensor, valid: Tensor) -> Tensor:
    weight = valid.to(values.dtype)
    count = weight.sum(1, keepdim=True).clamp_min(1.0)
    mean = (values * weight).sum(1, keepdim=True) / count
    variance = ((values - mean).square() * weight).sum(1, keepdim=True) / count
    normalized = (values - mean) / variance.sqrt().clamp_min(1.0e-4)
    return normalized.masked_fill(~valid, 0.0)


class TriFieldSpanSetScorer(nn.Module):
    """Route persistent field evidence to complete candidate spans.

    Evidence exclusively produces a semantic residual. Support and Transition
    exclusively produce a localization-quality residual. Candidate coordinates
    are detached before descriptor construction, so this module cannot become a
    hidden boundary-refinement branch.
    """

    def __init__(
        self,
        *,
        role_feature_dim: int = 512,
        boundary_temperature: float = 0.03,
        residual_bound: float = 2.0,
        role_feature_gain: float = 0.10,
    ) -> None:
        super().__init__()
        if role_feature_dim < 1:
            raise ValueError("role_feature_dim must be positive")
        if not 0.005 <= boundary_temperature <= 0.15:
            raise ValueError("boundary_temperature must be in [0.005, 0.15]")
        if not 0.1 <= residual_bound <= 4.0:
            raise ValueError("residual_bound must be in [0.1, 4.0]")
        self.role_feature_dim = int(role_feature_dim)
        self.boundary_temperature = float(boundary_temperature)
        self.residual_bound = float(residual_bound)
        self.role_feature_gain = float(role_feature_gain)

        cpu_rng_state = torch.random.get_rng_state()
        try:
            self.frame_heads = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.LayerNorm(role_feature_dim),
                        # A frame-uniform scalar bias cancels from listwise,
                        # pairwise, and correct-vs-wrong-query differences, so
                        # it is structurally unidentifiable and is omitted.
                        nn.Linear(role_feature_dim, 1, bias=False),
                    )
                    for _ in range(4)
                ]
            )
            for head in self.frame_heads:
                nn.init.normal_(head[-1].weight, std=0.01)
            self.semantic_head = nn.Linear(3, 1)
            self.quality_head = nn.Linear(7, 1)
            nn.init.zeros_(self.semantic_head.weight)
            nn.init.zeros_(self.semantic_head.bias)
            nn.init.zeros_(self.quality_head.weight)
            nn.init.zeros_(self.quality_head.bias)
        finally:
            # Adding the method must not perturb a backend's initialization RNG.
            torch.random.set_rng_state(cpu_rng_state)

    @staticmethod
    def _pool(values: Tensor, weights: Tensor) -> Tensor:
        numerator = (values[:, None, :] * weights).sum(-1)
        return numerator / weights.sum(-1).clamp_min(1.0e-5)

    def _frame_scores(self, fields: TriFieldSequence) -> tuple[Tensor, ...]:
        valid = fields.valid_mask.bool()
        evidence = _masked_standardize(fields.evidence.float(), valid)
        support = _masked_standardize(
            torch.log1p(fields.support.float().clamp_min(0)), valid
        )
        enter = _masked_standardize(fields.transition_start.float(), valid)
        leave = _masked_standardize(fields.transition_end.float(), valid)
        if fields.role_features is not None:
            role = fields.role_features.float()
            if role.shape[-1] != self.role_feature_dim:
                raise ValueError(
                    f"role feature dim {role.shape[-1]} != configured {self.role_feature_dim}"
                )
            evidence = evidence + self.role_feature_gain * self.frame_heads[0](
                role[..., 0, :]
            ).squeeze(-1)
            support = support + self.role_feature_gain * self.frame_heads[1](
                role[..., 1, :]
            ).squeeze(-1)
            enter = enter + self.role_feature_gain * self.frame_heads[2](
                role[..., 2, :]
            ).squeeze(-1)
            leave = leave + self.role_feature_gain * self.frame_heads[3](
                role[..., 2, :]
            ).squeeze(-1)
        return tuple(
            value.masked_fill(~valid, 0.0)
            for value in (evidence, support, enter, leave)
        )

    def describe(self, fields: TriFieldSequence, spans_xx: Tensor) -> Dict[str, Tensor]:
        """Build role-specific descriptors for normalized ``[start, end]`` spans."""

        fields.validate()
        if spans_xx.ndim != 3 or spans_xx.shape[-1] != 2:
            raise ValueError("spans_xx must have shape [batch, candidates, 2]")
        if spans_xx.shape[0] != fields.evidence.shape[0]:
            raise ValueError("span and field batch dimensions differ")
        spans = spans_xx.detach().float().clamp(0.0, 1.0)
        start = torch.minimum(spans[..., 0], spans[..., 1])
        end = torch.maximum(spans[..., 0], spans[..., 1])
        width = (end - start).clamp_min(1.0e-4)
        valid = fields.valid_mask.bool()
        frame_count = valid.sum(1, keepdim=True).clamp_min(1).float()
        position = (
            torch.arange(valid.shape[1], device=valid.device).float()[None, :] + 0.5
        ) / frame_count
        tau = self.boundary_temperature
        inside = torch.sigmoid((position[:, None, :] - start[..., None]) / tau)
        inside = inside * torch.sigmoid((end[..., None] - position[:, None, :]) / tau)
        inside = inside * valid[:, None, :].float()
        outside = (1.0 - inside) * valid[:, None, :].float()
        start_weight = (
            torch.exp(-0.5 * ((position[:, None, :] - start[..., None]) / tau).square())
            * valid[:, None, :].float()
        )
        end_weight = (
            torch.exp(-0.5 * ((position[:, None, :] - end[..., None]) / tau).square())
            * valid[:, None, :].float()
        )

        evidence, support, enter, leave = self._frame_scores(fields)
        evidence_inside = self._pool(evidence, inside)
        evidence_outside = self._pool(evidence, outside)
        support_inside = self._pool(support, inside)
        support_outside = self._pool(support, outside)
        transition_start = self._pool(enter, start_weight)
        transition_end = self._pool(leave, end_weight)
        return {
            "evidence_descriptor": torch.stack(
                (evidence_inside, evidence_inside - evidence_outside, width), -1
            ),
            "quality_descriptor": torch.stack(
                (
                    support_inside,
                    support_inside - support_outside,
                    transition_start,
                    transition_end,
                    torch.minimum(transition_start, transition_end),
                    (transition_start - transition_end).abs(),
                    width,
                ),
                -1,
            ),
            "evidence_score": evidence_inside,
            "support_score": support_inside,
            "transition_start_score": transition_start,
            "transition_end_score": transition_end,
        }

    def _bounded(self, values: Tensor) -> Tensor:
        return self.residual_bound * torch.tanh(values / self.residual_bound)

    def forward(
        self,
        fields: TriFieldSequence,
        spans_xx: Tensor,
        candidate_valid: Optional[Tensor] = None,
        wrong_query_fields: Optional[TriFieldSequence] = None,
    ) -> Dict[str, Tensor]:
        descriptors = self.describe(fields, spans_xx)
        semantic = self._bounded(
            self.semantic_head(descriptors["evidence_descriptor"]).squeeze(-1)
        )
        quality = self._bounded(
            self.quality_head(descriptors["quality_descriptor"]).squeeze(-1)
        )
        if candidate_valid is not None:
            if candidate_valid.shape != semantic.shape:
                raise ValueError("candidate_valid shape differs from candidate scores")
            semantic = semantic.masked_fill(~candidate_valid.bool(), 0.0)
            quality = quality.masked_fill(~candidate_valid.bool(), 0.0)
        result = dict(descriptors)
        result.update({"semantic_residual": semantic, "quality_residual": quality})
        if wrong_query_fields is not None:
            wrong = self.describe(wrong_query_fields, spans_xx)
            result["wrong_query_evidence_score"] = wrong["evidence_score"]
        return result


__all__ = ["TriFieldSequence", "TriFieldSpanSetScorer"]
