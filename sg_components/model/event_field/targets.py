"""Ground-truth field construction for Event Field F0."""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor


@dataclass
class EventFieldTargets:
    """Per-timestep supervision fields."""

    evidence: Tensor
    evidence_mask: Tensor
    support: Tensor
    start_transition: Tensor
    end_transition: Tensor


def build_event_field_targets(
    gt_spans: Tensor,
    sequence_length: int,
    evidence_targets: Optional[Tensor] = None,
    padding_mask: Optional[Tensor] = None,
    gt_span_mask: Optional[Tensor] = None,
    tau: float = 0.04,
    sigma: float = 0.04,
) -> EventFieldTargets:
    """Build evidence, support, and boundary-transition targets.

    Args:
        gt_spans: Normalized spans with shape ``[B, N, 2]`` and values in
            ``[0, 1]``. Reversed endpoints are reordered and values are clamped.
        sequence_length: Number of temporal positions in the output fields.
        evidence_targets: Optional QVHighlights saliency proxy with shape
            ``[B, L]``. Finite values are clamped to ``[0, 1]`` and supervised
            only at non-padding positions. When omitted, Evidence is zero and
            its supervision mask is disabled.
        padding_mask: Optional ``[B, L]`` mask where ``True`` denotes padding.
        gt_span_mask: Optional ``[B, N]`` mask where ``True`` denotes a valid GT.
        tau: Soft-support boundary temperature.
        sigma: Gaussian transition width. The effective width is at least
            ``1 / sequence_length``.

    Multiple GT spans are merged with a pointwise maximum.
    """
    if gt_spans.ndim != 3 or gt_spans.shape[-1] != 2:
        raise ValueError("gt_spans must have shape [B, N, 2]")
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if tau <= 0 or sigma <= 0:
        raise ValueError("tau and sigma must be positive")

    batch_size, num_spans, _ = gt_spans.shape
    device = gt_spans.device
    dtype = gt_spans.dtype if gt_spans.is_floating_point() else torch.float32
    spans = gt_spans.to(dtype=dtype)

    finite_spans = torch.isfinite(spans).all(dim=-1)
    if gt_span_mask is None:
        valid_spans = finite_spans
    else:
        if gt_span_mask.shape != (batch_size, num_spans):
            raise ValueError(f"gt_span_mask must have shape {(batch_size, num_spans)}")
        valid_spans = gt_span_mask.to(device=device, dtype=torch.bool) & finite_spans

    if padding_mask is None:
        padding_mask = torch.zeros(
            batch_size, sequence_length, dtype=torch.bool, device=device
        )
    elif padding_mask.shape != (batch_size, sequence_length):
        raise ValueError(
            f"padding_mask must have shape {(batch_size, sequence_length)}"
        )
    else:
        padding_mask = padding_mask.to(device=device, dtype=torch.bool)

    evidence, evidence_mask = _prepare_evidence_targets(
        evidence_targets,
        padding_mask,
        batch_size,
        sequence_length,
        device,
        dtype,
    )

    if num_spans == 0:
        zeros = torch.zeros(batch_size, sequence_length, device=device, dtype=dtype)
        return EventFieldTargets(
            evidence,
            evidence_mask,
            zeros,
            zeros.clone(),
            zeros.clone(),
        )

    spans = torch.where(valid_spans.unsqueeze(-1), spans, torch.zeros_like(spans))
    first_endpoint = spans[..., 0].clamp(0.0, 1.0)
    second_endpoint = spans[..., 1].clamp(0.0, 1.0)
    starts = torch.minimum(first_endpoint, second_endpoint).unsqueeze(-1)
    ends = torch.maximum(first_endpoint, second_endpoint).unsqueeze(-1)
    positions = (
        (torch.arange(sequence_length, device=device, dtype=dtype) + 0.5)
        / sequence_length
    ).view(1, 1, sequence_length)

    span_validity = valid_spans.unsqueeze(-1)
    support_per_span = (
        torch.sigmoid((positions - starts) / tau)
        * torch.sigmoid((ends - positions) / tau)
        * span_validity.to(dtype)
    )
    effective_sigma = max(float(sigma), 1.0 / sequence_length)
    start_per_span = torch.exp(-0.5 * ((positions - starts) / effective_sigma).square())
    end_per_span = torch.exp(-0.5 * ((positions - ends) / effective_sigma).square())
    start_per_span = start_per_span * span_validity.to(dtype)
    end_per_span = end_per_span * span_validity.to(dtype)

    valid_positions = (~padding_mask).to(dtype)
    return EventFieldTargets(
        evidence=evidence,
        evidence_mask=evidence_mask,
        support=support_per_span.amax(dim=1) * valid_positions,
        start_transition=start_per_span.amax(dim=1) * valid_positions,
        end_transition=end_per_span.amax(dim=1) * valid_positions,
    )


def _prepare_evidence_targets(
    evidence_targets: Optional[Tensor],
    padding_mask: Tensor,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    if evidence_targets is None:
        zeros = torch.zeros(batch_size, sequence_length, device=device, dtype=dtype)
        return zeros, torch.zeros_like(padding_mask)
    if evidence_targets.shape != (batch_size, sequence_length):
        raise ValueError(
            f"evidence_targets must have shape {(batch_size, sequence_length)}"
        )

    evidence_targets = evidence_targets.to(device=device, dtype=dtype)
    evidence_mask = torch.isfinite(evidence_targets) & ~padding_mask
    evidence = torch.where(
        evidence_mask,
        evidence_targets.clamp(0.0, 1.0),
        torch.zeros_like(evidence_targets),
    )
    return evidence, evidence_mask
