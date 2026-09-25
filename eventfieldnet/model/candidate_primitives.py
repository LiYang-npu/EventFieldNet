"""Small, CPU-reviewable candidate mechanisms for Round31.

This module is part of the Round31 training package.
It contains the two mechanisms used by the Round31 selector:

* :func:`log_mean_exp_span` pools a bounded token field on the existing
  inclusive ``[start, end]`` candidate grid.  Its temperature is fixed at
  ``0.2`` by the current Round31 design.  It uses an O(BL^2) clipped-index ``torch.logcumsumexp`` table; it does not use a
  subtractive prefix sum and never constructs a ``B x L x L x L`` tensor.
* :class:`IndependentSProjectionEncoder` copies only the video, key, and
  value projections of an R26 ``RawInteraction``.  The copied parameters do
  not share storage with the source and the copy operation consumes no random
  numbers.  There is intentionally no readout in this encoder.

The selector imports these operations directly; ground truth is not
accepted by either public operation.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


# The design proposal fixed this value after the initial 0.25 candidate.
# Keep the value in one place so a future design revision has one explicit
# edit point rather than an accidental per-call hyperparameter.
LOG_MEAN_EXP_TEMPERATURE = 0.2


def _validate_span_inputs(
    e: Tensor, token_valid: Tensor
) -> tuple[Tensor, Tensor, int, int]:
    """Validate and sanitize the token field before any reduction.

    Invalid tokens are replaced before ``logsumexp`` is evaluated.  This is
    important when padding carries a non-finite sentinel: an invalid value
    must not enter an intermediate reduction and later turn into ``0 * NaN``
    in backward.  Non-finite values on valid tokens remain an error.
    """

    if not isinstance(e, Tensor) or e.ndim != 2:
        raise ValueError(f"e must have shape [B,L], got {getattr(e, 'shape', None)}")
    if not isinstance(token_valid, Tensor) or token_valid.shape != e.shape:
        raise ValueError(
            "token_valid must have the same [B,L] shape as e, "
            f"got {getattr(token_valid, 'shape', None)} for {tuple(e.shape)}"
        )
    batch, length = e.shape
    if batch == 0 or length == 0:
        raise ValueError("empty batch or temporal axis is not a valid span field")

    valid = token_valid.bool()
    if not bool(valid.any(dim=1).all().item()):
        raise ValueError("each batch row must contain at least one valid token")

    # Reduction is intentionally FP32 even if the caller is under BF16
    # autocast.  Mask before checking so padding sentinels are harmless.
    e32 = e.float()
    valid_values = e32.masked_select(valid)
    if not bool(torch.isfinite(valid_values).all().item()):
        raise ValueError("valid token values must be finite")
    safe = e32.masked_fill(~valid, 0.0)
    return safe, valid, batch, length


def log_mean_exp_span(
    e: Tensor,
    token_valid: Tensor,
    *,
    temperature: float = LOG_MEAN_EXP_TEMPERATURE,
) -> tuple[Tensor, Tensor]:
    """Compute fixed-temperature log-mean-exp over every valid inclusive span.

    e has shape [B,L]. token_valid has the same shape. The output score is
    FP32. A candidate is valid only when every token in its inclusive
    [start,end] interval is valid. Temperature is fixed at 0.2.

    The reduction uses a clipped-index logcumsumexp table of shape [B,L,L]
    (start, offset), then gathers the inclusive candidate grid. Every
    intermediate is finite, no subtractive prefix sum is used, and no
    B x L x L x L tensor is constructed. Clipped values are outside the
    requested candidate prefix and cannot affect any legal span. Width-one
    candidates are explicitly replaced with the source token, so singleton
    values and their gradients are exact.
    """
    try:
        tau = float(temperature)
    except (TypeError, ValueError) as exc:
        raise ValueError("temperature must be the fixed finite value 0.2") from exc
    if not math.isfinite(tau) or tau != LOG_MEAN_EXP_TEMPERATURE:
        raise ValueError(
            f"Round31 log-mean-exp uses fixed temperature {LOG_MEAN_EXP_TEMPERATURE}"
        )

    safe, valid, batch, length = _validate_span_inputs(e, token_valid)
    with torch.autocast(device_type=e.device.type, enabled=False):
        # For each start and offset, index the real token when it exists.
        # Clipping only affects offsets beyond the last real token. Such
        # offsets are never selected for an inclusive [start,end] candidate.
        starts = torch.arange(length, device=e.device)[:, None]
        offsets = torch.arange(length, device=e.device)[None, :]
        source_indices = (starts + offsets).clamp_max(length - 1)
        scaled = safe[:, source_indices] / tau
        cumulative = torch.logcumsumexp(scaled, dim=-1)
        offset_log_count = (offsets + 1).to(safe.dtype).log()
        by_start_offset = tau * (cumulative - offset_log_count)

        end_indices = torch.arange(length, device=e.device)[None, :]
        offset_grid = (end_indices - starts).clamp_min(0)
        scores = by_start_offset.gather(2, offset_grid.expand(batch, length, length))

        # Count valid tokens with integer arithmetic. This is only for the
        # legality mask; the floating-point reduction above never subtracts
        # close prefix sums.
        valid_prefix = F.pad(valid.to(torch.long).cumsum(-1), (1, 0))
        span_count = valid_prefix[:, end_indices + 1] - valid_prefix[:, starts]
        widths = end_indices - starts + 1
        span_valid = (end_indices >= starts)[None] & (span_count == widths[None])

        scores = scores.masked_fill(~span_valid, 0.0)
        eye = torch.eye(length, dtype=torch.bool, device=e.device)[None]
        scores = torch.where(eye, torch.diag_embed(safe), scores)

    return scores, span_valid


def log_mean_exp_span_scores(
    e: Tensor,
    token_valid: Tensor,
    *,
    temperature: float = LOG_MEAN_EXP_TEMPERATURE,
) -> Tensor:
    """Return only the score tensor from :func:`log_mean_exp_span`."""

    return log_mean_exp_span(e, token_valid, temperature=temperature)[0]


# A short alias is useful in exploratory code while the descriptive name is
# retained as the stable public API.
lme_span = log_mean_exp_span


class _CopiedProjection(nn.Module):
    """A bias-free linear projection cloned without random initialization."""

    def __init__(self, weight: Tensor) -> None:
        super().__init__()
        if weight.ndim != 2:
            raise ValueError(
                f"projection weight must be rank 2, got {tuple(weight.shape)}"
            )
        # clone() gives an independent Parameter and never calls an
        # initializer, so copying cannot perturb CPU or CUDA RNG state.
        self.weight = nn.Parameter(weight.detach().clone())

    def forward(self, value: Tensor) -> Tensor:
        return F.linear(value, self.weight)


def _copy_projection(source: Any, name: str) -> _CopiedProjection:
    module = getattr(source, name, None)
    if module is None or not hasattr(module, "weight"):
        raise TypeError(f"source has no projection {name!r}")
    bias = getattr(module, "bias", None)
    if bias is not None:
        raise ValueError(f"source projection {name!r} must be bias-free")
    weight = module.weight
    if not isinstance(weight, Tensor) or weight.ndim != 2:
        raise ValueError(f"source projection {name!r} has invalid weight")
    if weight.shape != (64, 512):
        raise ValueError(
            f"source projection {name!r} must be [64,512], got {tuple(weight.shape)}"
        )
    return _CopiedProjection(weight)


class IndependentSProjectionEncoder(nn.Module):
    """Independent copy of R26 ``RawInteraction``'s V/K/U encoder.

    Only ``video``, ``key``, and ``value`` are copied.  The source readout,
    transition readout, and every other source attribute are intentionally
    excluded.  The encoder keeps the source attention convention, including
    raw normalized inputs, optional projected-cosine attention, query padding
    masking, GELU on the video projection, and zeroed video padding.

    Construct with an R26 ``RawInteraction`` instance::

        s_encoder = IndependentSProjectionEncoder(raw_interaction)

    The copied projection Parameters are independent and can be optimized
    without changing the source E/T interaction.
    """

    def __init__(self, source: Any) -> None:
        super().__init__()
        self.video = _copy_projection(source, "video")
        self.key = _copy_projection(source, "key")
        self.value = _copy_projection(source, "value")
        self.projected_cosine_attention = bool(
            getattr(source, "projected_cosine_attention", False)
        )

    @classmethod
    def from_raw_interaction(cls, source: Any) -> "IndependentSProjectionEncoder":
        """Named constructor documenting the intended R26 source object."""

        return cls(source)

    def attention_logits(self, q: Tensor, k: Tensor) -> Tensor:
        if self.projected_cosine_attention:
            return F.normalize(q, dim=-1, eps=1e-6) @ F.normalize(
                k, dim=-1, eps=1e-6
            ).transpose(-1, -2)
        return q @ k.transpose(-1, -2) / math.sqrt(64)

    @staticmethod
    def _padding_or_zeros(mask: Tensor | None, reference: Tensor) -> Tensor:
        if mask is None:
            return torch.zeros(
                reference.shape[:2], dtype=torch.bool, device=reference.device
            )
        if mask.shape != reference.shape[:2]:
            raise ValueError(
                f"padding mask must have shape {tuple(reference.shape[:2])}, "
                f"got {tuple(mask.shape)}"
            )
        return mask.bool()

    def encode(
        self,
        video: Tensor,
        text: Tensor,
        video_pad: Tensor | None = None,
        text_pad: Tensor | None = None,
    ) -> Tensor:
        """Encode video tokens conditioned on text, matching R26 RawInteraction."""

        if video.ndim != 3 or text.ndim != 3:
            raise ValueError("video and text must have shape [B,L,512]")
        if (
            video.shape[0] != text.shape[0]
            or video.shape[-1] != 512
            or text.shape[-1] != 512
        ):
            raise ValueError(
                "video/text batch and feature dimensions must be B x L x 512"
            )
        pad = self._padding_or_zeros(video_pad, video)
        tpad = self._padding_or_zeros(text_pad, text)
        if bool((~tpad).sum(dim=1).eq(0).any().item()):
            raise ValueError("empty valid query token set")

        # Match RawInteraction's explicit FP32 field arithmetic even when a
        # caller invokes this method under BF16/FP16 autocast.
        with torch.autocast(device_type=video.device.type, enabled=False):
            v = F.normalize(video.float(), dim=-1, eps=1e-6)
            t = F.normalize(text.float(), dim=-1, eps=1e-6)
            q = self.video(v)
            k = self.key(t)
            u = self.value(t)
            attention = (
                self.attention_logits(q, k)
                .masked_fill(tpad[:, None, :], float("-inf"))
                .softmax(-1)
            )
            encoded = F.gelu(q) * (attention @ u)
            return encoded.masked_fill(pad[..., None], 0.0)

    forward = encode


# Explicit alias for callers that use the mechanism's role rather than its
# implementation name.
IndependentSRawEncoder = IndependentSProjectionEncoder


__all__ = [
    "LOG_MEAN_EXP_TEMPERATURE",
    "log_mean_exp_span",
    "log_mean_exp_span_scores",
    "lme_span",
    "IndependentSProjectionEncoder",
    "IndependentSRawEncoder",
]
