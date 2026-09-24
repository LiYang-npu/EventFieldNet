"""Small, separated E/S/T heads on the parent's unchanged candidate grid."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def span_mean(value: Tensor) -> Tensor:
    """Inclusive temporal means, retaining the parent ``[start, end]`` grid."""
    if value.ndim != 3:
        raise ValueError(f"span_mean expects [B,L,C], got {tuple(value.shape)}")
    length = value.shape[1]
    prefix = F.pad(value.cumsum(1), (0, 0, 1, 0))
    index = torch.arange(length, device=value.device)
    start, end = index[:, None], index[None, :]
    width = (end - start + 1).clamp_min(1).to(value.dtype)
    return (prefix[:, end + 1] - prefix[:, start]) / width[None, :, :, None]


def centered_bounded(raw: Tensor, valid: Tensor) -> Tensor:
    """Center raw query evidence before tanh; the mean remains differentiable."""
    mask = valid.bool()
    safe_raw = raw.float().masked_fill(~mask, 0.0)
    count = mask.sum(dim=(-1, -2), keepdim=True).clamp_min(1).to(safe_raw.dtype)
    mean = safe_raw.sum(dim=(-1, -2), keepdim=True) / count
    return torch.tanh(safe_raw - mean).masked_fill(~mask, 0.0)


class _TokenFieldEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.project = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)

    def encode(self, value: Tensor) -> Tensor:
        return F.gelu(self.project(self.norm(value.float())))

    def readout(self, value: Tensor) -> Tensor:
        return self.output(value).squeeze(-1)


@dataclass
class FieldScoreOutput:
    raw_evidence: Tensor
    raw_support: Tensor
    raw_transition_start: Tensor
    raw_transition_end: Tensor
    carrier_raw: Tensor
    carrier: Tensor
    evidence: Tensor
    support: Tensor
    transition_start: Tensor
    transition_end: Tensor
    score: Tensor
    valid: Tensor


class TriFieldScoreHeads(nn.Module):
    """Four low-rank readouts with one fixed final score equation.

    Evidence and Support use inclusive span means of their respective role
    updates plus a small explicit normalized span-width input. Transition
    reads the parent start and end inputs with distinct encoders/readouts.
    Every head is trainable and every deployed field is exactly the tensor used
    by its loss.
    """

    def __init__(self, input_dim: int = 512, hidden_dim: int = 64) -> None:
        super().__init__()
        if input_dim < 1 or hidden_dim < 4:
            raise ValueError("input_dim and hidden_dim must be positive")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.evidence = _TokenFieldEncoder(input_dim, hidden_dim)
        self.support = _TokenFieldEncoder(input_dim, hidden_dim)

        self.support_width = nn.Linear(1, 1)
        self.transition_start = _TokenFieldEncoder(input_dim, hidden_dim)
        self.transition_end = _TokenFieldEncoder(input_dim, hidden_dim)

    @staticmethod
    def _state_role_updates(state: Any) -> Tensor:
        role_updates = getattr(state, "role_updates", None)
        if role_updates is None or role_updates.ndim != 4 or role_updates.shape[2] < 3:
            raise ValueError("InputTriFieldState.role_updates must be [B,L,3,C]")
        return role_updates

    def raw_from_state(
        self,
        state: Any,
        valid: Tensor,
        carrier: Tensor,
        video_padding_mask: Tensor | None = None,
    ) -> FieldScoreOutput:
        role_updates = self._state_role_updates(state)
        if carrier.ndim != 3 or carrier.shape != valid.shape:
            raise ValueError("carrier and valid must be [B,L,L]")
        evidence_token = self.evidence.encode(role_updates[:, :, 0, :])
        support_token = self.support.encode(role_updates[:, :, 1, :])
        start_input = getattr(state, "transition_start_input", None)
        end_input = getattr(state, "transition_end_input", None)
        if (
            not isinstance(start_input, Tensor)
            or start_input.shape[:2] != role_updates.shape[:2]
        ):
            start_input = role_updates[:, :, 2, :]
        if (
            not isinstance(end_input, Tensor)
            or end_input.shape[:2] != role_updates.shape[:2]
        ):
            end_input = role_updates[:, :, 2, :]
        transition_start_token = self.transition_start.encode(start_input)
        transition_end_token = self.transition_end.encode(end_input)
        evidence_span = span_mean(evidence_token)
        support_span = span_mean(support_token)
        valid = valid.bool()
        if video_padding_mask is None:
            token_valid = valid.any(-1) | valid.any(-2)
            count = token_valid.sum(-1).clamp_min(1).float()
        else:
            token_valid = ~video_padding_mask.bool()
            count = token_valid.sum(-1).clamp_min(1).float()
        length = carrier.shape[-1]
        index = torch.arange(length, device=carrier.device, dtype=torch.float32)
        width = (index[None, None, :] - index[None, :, None] + 1.0).clamp_min(1.0)
        width = width / count[:, None, None]
        raw_evidence = self.evidence.readout(evidence_span)
        raw_support = self.support.readout(support_span) + self.support_width(
            width.unsqueeze(-1)
        ).squeeze(-1)
        raw_transition_start = self.transition_start.readout(transition_start_token)[
            :, :, None
        ].expand_as(carrier)
        raw_transition_end = self.transition_end.readout(transition_end_token)[
            :, None, :
        ].expand_as(carrier)
        carrier_raw = carrier.float()
        carrier_bound = torch.tanh(carrier_raw).masked_fill(~valid, 0.0)
        evidence = centered_bounded(raw_evidence, valid)
        support = torch.tanh(raw_support.float()).masked_fill(~valid, 0.0)
        transition_start = torch.tanh(raw_transition_start.float()).masked_fill(
            ~valid, 0.0
        )
        transition_end = torch.tanh(raw_transition_end.float()).masked_fill(~valid, 0.0)
        score = (
            carrier_bound
            + evidence
            + support
            + 0.5 * (transition_start + transition_end)
        ).masked_fill(~valid, 0.0)
        return FieldScoreOutput(
            raw_evidence=raw_evidence,
            raw_support=raw_support,
            raw_transition_start=raw_transition_start,
            raw_transition_end=raw_transition_end,
            carrier_raw=carrier_raw,
            carrier=carrier_bound,
            evidence=evidence,
            support=support,
            transition_start=transition_start,
            transition_end=transition_end,
            score=score,
            valid=valid,
        )

    def forward(
        self,
        state: Any,
        valid: Tensor,
        carrier: Tensor,
        video_padding_mask: Tensor | None = None,
    ) -> FieldScoreOutput:
        return self.raw_from_state(state, valid, carrier, video_padding_mask)


__all__ = ["FieldScoreOutput", "TriFieldScoreHeads", "centered_bounded", "span_mean"]
