"""Task-specific tri-field decoder with multi-scale boundary context contrast."""

from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .framework_field_structured import EventFieldNetFieldStructured


class MultiScaleContextContrast(nn.Module):
    """Construct Evidence, Support, and Transition from contextual phase changes."""

    def __init__(
        self,
        hidden_dim: int,
        rank: int,
        scales: Sequence[int],
        residual_bound: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.scales = tuple(int(scale) for scale in scales)
        self.residual_bound = float(residual_bound)
        self.norm = nn.LayerNorm(hidden_dim)
        self.token_down = nn.Linear(hidden_dim, rank)
        self.query_down = nn.Linear(hidden_dim, rank)
        self.scale_logits = nn.Parameter(torch.zeros(len(self.scales)))
        self.evidence_up = nn.Linear(rank, hidden_dim)
        self.support_up = nn.Linear(rank, hidden_dim)
        self.start_up = nn.Linear(rank, hidden_dim)
        self.end_up = nn.Linear(rank, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.role_logits = nn.Parameter(torch.full((4,), -2.0))
        for layer in (
            self.evidence_up,
            self.support_up,
            self.start_up,
            self.end_up,
        ):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def _multi_scale_context(self, tokens: Tensor) -> Tensor:
        channels_first = tokens.transpose(1, 2)
        contexts = []
        for radius in self.scales:
            kernel = 2 * radius + 1
            contexts.append(
                F.avg_pool1d(
                    channels_first,
                    kernel_size=kernel,
                    stride=1,
                    padding=radius,
                    count_include_pad=False,
                ).transpose(1, 2)
            )
        weights = torch.softmax(self.scale_logits, dim=0)
        return sum(weight * context for weight, context in zip(weights, contexts))

    def forward(self, latent: Tensor, query_summary: Tensor):
        low_rank = self.token_down(self.norm(latent))
        query = self.query_down(query_summary)[:, None]
        alignment = low_rank * torch.tanh(query)
        context = self._multi_scale_context(low_rank)
        previous = F.pad(context[:, :-1], (0, 0, 1, 0))
        following = F.pad(context[:, 1:], (0, 0, 0, 1))
        features = (
            alignment,
            alignment + context,
            alignment + context - previous,
            alignment + context - following,
        )
        projections = (
            self.evidence_up,
            self.support_up,
            self.start_up,
            self.end_up,
        )
        outputs = []
        for index, (feature, projection) in enumerate(zip(features, projections)):
            update = projection(self.dropout(F.gelu(feature)))
            scale = self.residual_bound * torch.sigmoid(self.role_logits[index])
            outputs.append(latent + scale * torch.tanh(update))
        return tuple(outputs)


class EventFieldNetContextContrast(EventFieldNetFieldStructured):
    """Continuous tri-field model driven by inside/outside boundary contrast."""

    def __init__(
        self,
        *args,
        context_rank: int = 96,
        context_scales: Sequence[int] = (1, 3, 7),
        context_residual_bound: float = 1.0,
        context_boundary_radius: int = 4,
        use_context_quality: bool = True,
        dropout: float = 0.15,
        **kwargs,
    ) -> None:
        super().__init__(*args, dropout=dropout, **kwargs)
        self.context_boundary_radius = int(context_boundary_radius)
        self.use_context_quality = bool(use_context_quality)
        self.context_contrast = MultiScaleContextContrast(
            self.hidden_dim,
            context_rank,
            context_scales,
            context_residual_bound,
            dropout,
        )
        self.context_quality_head = nn.Sequential(
            nn.Linear(6, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
        nn.init.zeros_(self.context_quality_head[-1].weight)
        nn.init.zeros_(self.context_quality_head[-1].bias)

    def _build_field_tokens(
        self,
        latent: Tensor,
        query: Tensor,
        safe_query_mask: Tensor,
        query_summary: Tensor,
    ):
        del query, safe_query_mask
        return self.context_contrast(latent, query_summary)

    @staticmethod
    def _interval_means(values: Tensor, radius: int):
        batch, length = values.shape
        indices = torch.arange(length, device=values.device)
        starts = indices.view(length, 1)
        ends = indices.view(1, length)
        prefix = F.pad(values.cumsum(dim=1), (1, 0))
        span_lengths = (ends - starts + 1).clamp_min(1)
        inside = (prefix[:, ends + 1] - prefix[:, starts]) / span_lengths

        left_starts = (starts - radius).clamp_min(0)
        left_lengths = (starts - left_starts).clamp_min(1)
        left = (prefix[:, starts] - prefix[:, left_starts]) / left_lengths
        left = left.expand(batch, length, length)

        right_ends = (ends + 1 + radius).clamp_max(length)
        right_lengths = (right_ends - ends - 1).clamp_min(1)
        right = (prefix[:, right_ends] - prefix[:, ends + 1]) / right_lengths
        right = right.expand(batch, length, length)
        return inside, left, right

    def _quality_field(
        self,
        base_logits: Tensor,
        evidence_logits: Tensor,
        support_logits: Tensor,
        start_logits: Tensor,
        end_logits: Tensor,
        evidence_tokens: Tensor,
        support_tokens: Tensor,
        start_tokens: Tensor,
        end_tokens: Tensor,
        query_summary: Tensor,
        padding_mask: Tensor,
        valid_mask: Tensor,
    ) -> Tensor:
        quality = super()._quality_field(
            base_logits,
            evidence_logits,
            support_logits,
            start_logits,
            end_logits,
            evidence_tokens,
            support_tokens,
            start_tokens,
            end_tokens,
            query_summary,
            padding_mask,
            valid_mask,
        )
        if not self.use_context_quality:
            return quality
        evidence = torch.sigmoid(evidence_logits).masked_fill(padding_mask, 0)
        support = torch.sigmoid(support_logits).masked_fill(padding_mask, 0)
        evidence_inside, evidence_left, evidence_right = self._interval_means(
            evidence, self.context_boundary_radius
        )
        support_inside, support_left, support_right = self._interval_means(
            support, self.context_boundary_radius
        )
        length = evidence.shape[1]
        start = torch.sigmoid(start_logits)[:, :, None].expand(-1, -1, length)
        end = torch.sigmoid(end_logits)[:, None, :].expand(-1, length, -1)
        descriptors = torch.stack(
            (
                evidence_inside,
                evidence_inside - evidence_left,
                evidence_inside - evidence_right,
                support_inside - 0.5 * (support_left + support_right),
                start,
                end,
            ),
            dim=-1,
        )
        if self.detach_quality_features:
            descriptors = descriptors.detach()
        contrast = self.context_quality_head(descriptors).squeeze(-1)
        return (quality + contrast).masked_fill(~valid_mask, -1e4)
