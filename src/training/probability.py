"""Numerically stable masked probability and triangular-span utilities."""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor


def _broadcast_mask(mask: Tensor, values: Tensor) -> Tensor:
    mask = mask.to(device=values.device, dtype=torch.bool)
    try:
        return torch.broadcast_to(mask, values.shape)
    except RuntimeError as exc:
        raise ValueError(
            f"mask {tuple(mask.shape)} cannot broadcast to {tuple(values.shape)}"
        ) from exc


def masked_softmax(logits: Tensor, mask: Tensor, dim: int = -1) -> Tensor:
    """Softmax with exact zeros for masked entries and all-invalid rows."""
    valid = _broadcast_mask(mask, logits)
    work = logits.float()
    neg_inf = torch.tensor(float("-inf"), device=work.device, dtype=work.dtype)
    masked = torch.where(valid, work, neg_inf)
    row_max = masked.amax(dim=dim, keepdim=True)
    row_has_valid = valid.any(dim=dim, keepdim=True)
    row_max = torch.where(row_has_valid, row_max, torch.zeros_like(row_max))
    numer = torch.where(valid, torch.exp(work - row_max), torch.zeros_like(work))
    denom = numer.sum(dim=dim, keepdim=True)
    probs = torch.where(
        denom > 0,
        numer / denom.clamp_min(torch.finfo(work.dtype).tiny),
        torch.zeros_like(numer),
    )
    probs = torch.where(valid, probs, torch.zeros_like(probs))
    return probs.to(dtype=logits.dtype)


def masked_log_softmax(logits: Tensor, mask: Tensor, dim: int = -1) -> Tensor:
    """Log-softmax; all-invalid rows are exactly zero, other invalid entries -inf."""
    valid = _broadcast_mask(mask, logits)
    probs = masked_softmax(logits, valid, dim=dim).float()
    log_probs = torch.where(
        valid,
        torch.log(probs.clamp_min(torch.finfo(probs.dtype).tiny)),
        torch.full_like(probs, float("-inf")),
    )
    all_invalid = ~valid.any(dim=dim, keepdim=True)
    log_probs = torch.where(all_invalid, torch.zeros_like(log_probs), log_probs)
    return log_probs.to(dtype=logits.dtype)


def triangular_span_indices(
    length: int, device: torch.device | str | None = None
) -> Tuple[Tensor, Tensor]:
    if length < 1:
        raise ValueError("length must be positive")
    indices = torch.triu_indices(length, length, device=device)
    return indices[0], indices[1]


def triangular_span_mask(token_valid: Tensor) -> Tensor:
    """Return BxLxL mask; a span is valid iff every enclosed token is valid."""
    if token_valid.ndim != 2:
        raise ValueError("token_valid must have shape [batch, length]")
    token_valid = token_valid.to(dtype=torch.bool)
    batch, length = token_valid.shape
    starts, ends = triangular_span_indices(length, token_valid.device)
    prefix = torch.cat(
        [
            torch.zeros(batch, 1, device=token_valid.device, dtype=torch.long),
            token_valid.long().cumsum(dim=1),
        ],
        dim=1,
    )
    counts = prefix[:, ends + 1] - prefix[:, starts]
    widths = (ends - starts + 1).unsqueeze(0)
    flat_valid = counts.eq(widths)
    result = torch.zeros(
        batch, length, length, device=token_valid.device, dtype=torch.bool
    )
    result[:, starts, ends] = flat_valid
    return result


def masked_span_softmax(span_logits: Tensor, token_valid: Tensor) -> Tensor:
    if span_logits.ndim != 3 or span_logits.shape[-1] != span_logits.shape[-2]:
        raise ValueError("span_logits must have shape [batch, length, length]")
    mask = triangular_span_mask(token_valid)
    probs = masked_softmax(span_logits.flatten(1), mask.flatten(1), dim=-1).reshape_as(
        span_logits
    )
    return torch.where(mask, probs, torch.zeros_like(probs))


def assert_probability_contract(
    probs: Tensor, mask: Tensor, dim: int = -1, atol: float = 1.0e-6
) -> None:
    valid = _broadcast_mask(mask, probs)
    if not torch.isfinite(probs).all():
        raise AssertionError("probabilities contain non-finite values")
    if torch.count_nonzero(probs.masked_select(~valid)).item() != 0:
        raise AssertionError("invalid probabilities are not exact zero")
    sums = probs.sum(dim=dim)
    expected = valid.any(dim=dim).to(sums.dtype)
    error = float((sums - expected).abs().max())
    if error >= atol:
        raise AssertionError(f"probability normalization error {error} >= {atol}")
