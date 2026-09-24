"""Compact field-structured backbone for standalone EventField-Net."""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .framework import EventFieldNetOutput
from .framework_proven import EventFieldNetProven
from .framework_taskaware import SupportPyramid


@dataclass
class FieldStructuredOutput(EventFieldNetOutput):
    base_span_logits: Tensor
    quality_logits: Tensor


class LowRankEvidenceModulation(nn.Module):
    """Query-conditioned Evidence adaptation with a small low-rank residual."""

    def __init__(self, hidden_dim: int, rank: int, dropout: float) -> None:
        super().__init__()
        self.query_down = nn.Linear(hidden_dim, rank)
        self.query_up = nn.Linear(rank, 2 * hidden_dim)
        self.token_norm = nn.LayerNorm(hidden_dim)
        self.token_down = nn.Linear(hidden_dim, rank)
        self.token_up = nn.Linear(rank, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.residual_logit = nn.Parameter(torch.tensor(-2.0))
        nn.init.zeros_(self.query_up.weight)
        nn.init.zeros_(self.query_up.bias)
        nn.init.zeros_(self.token_up.weight)
        nn.init.zeros_(self.token_up.bias)

    def forward(self, tokens: Tensor, query_summary: Tensor) -> Tensor:
        gamma, beta = self.query_up(F.gelu(self.query_down(query_summary))).chunk(
            2, dim=-1
        )
        conditioned = tokens * (1.0 + 0.25 * torch.tanh(gamma[:, None]))
        conditioned = conditioned + 0.25 * beta[:, None]
        update = self.token_up(
            self.dropout(F.gelu(self.token_down(self.token_norm(conditioned))))
        )
        scale = 0.25 * torch.sigmoid(self.residual_logit)
        return tokens + scale * torch.tanh(update)


class SupportDerivedTransition(nn.Module):
    """Derive compact start/end context from Support dynamics."""

    def __init__(self, hidden_dim: int, rank: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.local = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=5,
            padding=2,
            groups=hidden_dim,
        )
        self.start_down = nn.Linear(3 * hidden_dim, rank)
        self.end_down = nn.Linear(3 * hidden_dim, rank)
        self.start_up = nn.Linear(rank, hidden_dim)
        self.end_up = nn.Linear(rank, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.start_residual_logit = nn.Parameter(torch.tensor(-2.0))
        self.end_residual_logit = nn.Parameter(torch.tensor(-2.0))
        nn.init.zeros_(self.start_up.weight)
        nn.init.zeros_(self.start_up.bias)
        nn.init.zeros_(self.end_up.weight)
        nn.init.zeros_(self.end_up.bias)

    def forward(
        self, support_tokens: Tensor, query_summary: Tensor
    ) -> tuple[Tensor, Tensor]:
        normalized = self.norm(support_tokens)
        previous = F.pad(normalized[:, :-1], (0, 0, 1, 0))
        following = F.pad(normalized[:, 1:], (0, 0, 0, 1))
        local = self.local(normalized.transpose(1, 2)).transpose(1, 2)
        query = query_summary[:, None].expand_as(normalized)
        start_input = torch.cat((normalized - previous, local, query), dim=-1)
        end_input = torch.cat((normalized - following, local, query), dim=-1)
        start_update = self.start_up(self.dropout(F.gelu(self.start_down(start_input))))
        end_update = self.end_up(self.dropout(F.gelu(self.end_down(end_input))))
        start_scale = 0.25 * torch.sigmoid(self.start_residual_logit)
        end_scale = 0.25 * torch.sigmoid(self.end_residual_logit)
        return (
            support_tokens + start_scale * torch.tanh(start_update),
            support_tokens + end_scale * torch.tanh(end_update),
        )


class SpanInteractionQualityField(nn.Module):
    """Score spans from query-conditioned boundary and interior field features."""

    def __init__(self, hidden_dim: int, rank: int, dropout: float) -> None:
        super().__init__()
        self.token_norm = nn.LayerNorm(hidden_dim)
        self.start_proj = nn.Linear(hidden_dim, rank)
        self.end_proj = nn.Linear(hidden_dim, rank)
        self.inside_proj = nn.Linear(hidden_dim, rank)
        self.query_proj = nn.Linear(hidden_dim, rank)
        self.duration_proj = nn.Sequential(
            nn.Linear(1, rank),
            nn.GELU(),
            nn.Linear(rank, rank),
        )
        self.output = nn.Sequential(
            nn.LayerNorm(rank),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rank, 1),
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(
        self,
        start_tokens: Tensor,
        end_tokens: Tensor,
        evidence_tokens: Tensor,
        support_tokens: Tensor,
        query_summary: Tensor,
        padding_mask: Tensor,
        valid_mask: Tensor,
    ) -> Tensor:
        batch, length, _ = start_tokens.shape
        indices = torch.arange(length, device=start_tokens.device)
        starts = indices.view(length, 1)
        ends = indices.view(1, length)
        span_lengths = (ends - starts + 1).clamp_min(1)
        interior_tokens = self.token_norm(0.5 * (evidence_tokens + support_tokens))
        interior = self.inside_proj(interior_tokens).masked_fill(
            padding_mask.unsqueeze(-1), 0
        )
        prefix = F.pad(interior.cumsum(dim=1), (0, 0, 1, 0))
        inside_mean = (prefix[:, ends + 1] - prefix[:, starts]) / span_lengths[
            None, :, :, None
        ]
        start = self.start_proj(self.token_norm(start_tokens))[:, :, None, :]
        end = self.end_proj(self.token_norm(end_tokens))[:, None, :, :]
        query = self.query_proj(query_summary)[:, None, None, :]
        valid_lengths = (~padding_mask).sum(dim=1).clamp_min(1)
        duration = span_lengths.to(start.dtype)[None] / valid_lengths[:, None, None].to(
            start.dtype
        )
        duration = self.duration_proj(duration.unsqueeze(-1))
        features = start + end + inside_mean + query + duration
        quality = self.output(features).squeeze(-1)
        return quality.masked_fill(~valid_mask, -1e4)


class EventFieldNetFieldStructured(EventFieldNetProven):
    """Model Evidence, Support, and Transition as coupled continuous fields."""

    def __init__(
        self,
        *args,
        field_rank: int = 64,
        use_evidence_modulation: bool = True,
        use_support_pyramid: bool = True,
        use_derived_transition: bool = True,
        use_quality_energy: bool = True,
        bounded_field_energy: bool = False,
        detach_quality_features: bool = False,
        anchor_preserving_span: bool = False,
        structured_energy_bound: float = 0.5,
        quality_energy_bound: float = 0.5,
        use_neural_quality_field: bool = False,
        quality_field_rank: int = 32,
        dropout: float = 0.15,
        **kwargs,
    ) -> None:
        super().__init__(*args, architecture="shared_full", dropout=dropout, **kwargs)
        hidden_dim = self.hidden_dim
        self.use_evidence_modulation = bool(use_evidence_modulation)
        self.use_support_pyramid = bool(use_support_pyramid)
        self.use_derived_transition = bool(use_derived_transition)
        self.use_quality_energy = bool(use_quality_energy)
        self.bounded_field_energy = bool(bounded_field_energy)
        self.detach_quality_features = bool(detach_quality_features)
        self.anchor_preserving_span = bool(anchor_preserving_span)
        self.structured_energy_bound = float(structured_energy_bound)
        self.quality_energy_bound = float(quality_energy_bound)
        self.use_neural_quality_field = bool(use_neural_quality_field)
        self.query_summary_norm = nn.LayerNorm(hidden_dim)
        self.evidence_modulation = LowRankEvidenceModulation(
            hidden_dim, field_rank, dropout
        )
        self.support_role = nn.Sequential(
            nn.Linear(hidden_dim, field_rank),
            nn.GELU(),
            nn.Linear(field_rank, hidden_dim),
        )
        self.support_pyramid = SupportPyramid(hidden_dim, dropout)
        self.support_residual_logit = nn.Parameter(torch.tensor(-2.0))
        self.transition_adapter = SupportDerivedTransition(
            hidden_dim, field_rank, dropout
        )
        self.start_derivative_scale = nn.Parameter(torch.tensor(-2.0))
        self.end_derivative_scale = nn.Parameter(torch.tensor(-2.0))
        self.quality_head = nn.Sequential(
            nn.Linear(7, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        self.quality_energy_scale = nn.Parameter(torch.tensor(-1.5))
        self.neural_quality_field = SpanInteractionQualityField(
            hidden_dim, quality_field_rank, dropout
        )
        self.structured_energy_scale = nn.Parameter(torch.tensor(-2.0))
        nn.init.zeros_(self.quality_head[-1].weight)
        nn.init.zeros_(self.quality_head[-1].bias)

    @staticmethod
    def _masked_query_summary(query: Tensor, padding_mask: Tensor) -> Tensor:
        valid = (~padding_mask).to(query.dtype)
        return (query * valid.unsqueeze(-1)).sum(dim=1) / valid.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)

    @staticmethod
    def _field_derivatives(support: Tensor) -> tuple[Tensor, Tensor]:
        previous = F.pad(support[:, :-1], (1, 0))
        following = F.pad(support[:, 1:], (0, 1))
        return F.relu(support - previous), F.relu(support - following)

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
        batch, length = evidence_logits.shape
        indices = torch.arange(length, device=evidence_logits.device)
        starts = indices.view(length, 1)
        ends = indices.view(1, length)
        span_lengths = (ends - starts + 1).clamp_min(1)
        evidence = torch.sigmoid(evidence_logits).masked_fill(padding_mask, 0)
        support = torch.sigmoid(support_logits).masked_fill(padding_mask, 0)
        evidence_prefix = F.pad(evidence.cumsum(dim=1), (1, 0))
        support_prefix = F.pad(support.cumsum(dim=1), (1, 0))
        evidence_inside = evidence_prefix[:, ends + 1] - evidence_prefix[:, starts]
        support_inside = support_prefix[:, ends + 1] - support_prefix[:, starts]
        evidence_mean = evidence_inside / span_lengths
        support_mean = support_inside / span_lengths
        evidence_coverage = (
            evidence_inside / evidence.sum(dim=1).clamp_min(1e-6)[:, None, None]
        )
        valid_lengths = (~padding_mask).sum(dim=1).clamp_min(1)
        duration = span_lengths.to(base_logits.dtype)[None] / valid_lengths[
            :, None, None
        ].to(base_logits.dtype)
        descriptors = torch.stack(
            (
                torch.sigmoid(start_logits)[:, :, None].expand(-1, -1, length),
                torch.sigmoid(end_logits)[:, None, :].expand(-1, length, -1),
                support_mean,
                evidence_mean,
                evidence_coverage,
                duration.expand(batch, -1, -1),
                torch.tanh(base_logits / 5.0),
            ),
            dim=-1,
        )
        if self.detach_quality_features:
            descriptors = descriptors.detach()
        quality = self.quality_head(descriptors).squeeze(-1)
        if self.use_neural_quality_field:
            neural_inputs = (
                evidence_tokens,
                support_tokens,
                start_tokens,
                end_tokens,
                query_summary,
            )
            if self.detach_quality_features:
                neural_inputs = tuple(item.detach() for item in neural_inputs)
            evidence_field, support_field, start_field, end_field, query_field = (
                neural_inputs
            )
            quality = quality + self.neural_quality_field(
                start_field,
                end_field,
                evidence_field,
                support_field,
                query_field,
                padding_mask,
                valid_mask,
            )
        return quality.masked_fill(~valid_mask, -1e4)

    def _build_field_tokens(
        self,
        latent: Tensor,
        query: Tensor,
        safe_query_mask: Tensor,
        query_summary: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        del query, safe_query_mask
        evidence_tokens = (
            self.evidence_modulation(latent, query_summary)
            if self.use_evidence_modulation
            else latent
        )
        if self.use_support_pyramid:
            raw_support_tokens = self.support_pyramid(
                latent, self.support_role(query_summary)
            )
            if self.bounded_field_energy:
                support_scale = 0.25 * torch.sigmoid(self.support_residual_logit)
                support_tokens = latent + support_scale * torch.tanh(
                    raw_support_tokens - latent
                )
            else:
                support_tokens = raw_support_tokens
        else:
            support_tokens = latent
        if self.use_derived_transition:
            start_tokens, end_tokens = self.transition_adapter(
                support_tokens, query_summary
            )
        else:
            start_tokens = end_tokens = latent
        return evidence_tokens, support_tokens, start_tokens, end_tokens

    def _read_field_logits(
        self,
        latent: Tensor,
        evidence_tokens: Tensor,
        support_tokens: Tensor,
        start_tokens: Tensor,
        end_tokens: Tensor,
        valid: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        del latent
        return (
            self.evidence_readout(evidence_tokens).squeeze(-1) * valid,
            self.support_readout(support_tokens).squeeze(-1) * valid,
            self.start_readout(start_tokens).squeeze(-1) * valid,
            self.end_readout(end_tokens).squeeze(-1) * valid,
        )

    def forward(
        self,
        video_features: Tensor,
        query_features: Tensor,
        video_padding_mask: Optional[Tensor] = None,
        query_padding_mask: Optional[Tensor] = None,
    ) -> FieldStructuredOutput:
        latent, query, video_padding_mask, safe_query_mask = self.stem.encode(
            video_features,
            query_features,
            video_padding_mask,
            query_padding_mask,
        )
        query_summary = self.query_summary_norm(
            self._masked_query_summary(query, safe_query_mask)
        )
        evidence_tokens, support_tokens, start_tokens, end_tokens = (
            self._build_field_tokens(latent, query, safe_query_mask, query_summary)
        )

        valid = (~video_padding_mask).to(latent.dtype)
        evidence_logits, support_logits, start_logits, end_logits = (
            self._read_field_logits(
                latent,
                evidence_tokens,
                support_tokens,
                start_tokens,
                end_tokens,
                valid,
            )
        )
        if self.use_derived_transition:
            support = torch.sigmoid(support_logits).masked_fill(video_padding_mask, 0)
            rise, fall = self._field_derivatives(support)
            start_scale = (
                0.25 * torch.sigmoid(self.start_derivative_scale)
                if self.bounded_field_energy
                else F.softplus(self.start_derivative_scale)
            )
            end_scale = (
                0.25 * torch.sigmoid(self.end_derivative_scale)
                if self.bounded_field_energy
                else F.softplus(self.end_derivative_scale)
            )
            start_logits = start_logits + start_scale * (2.0 * rise - 1.0) * valid
            end_logits = end_logits + end_scale * (2.0 * fall - 1.0) * valid

        structured_span_logits, span_valid = self._span_field(
            start_tokens,
            end_tokens,
            start_logits,
            end_logits,
            support_logits,
            video_padding_mask,
        )
        if self.anchor_preserving_span:
            anchor_start_logits = self.start_readout(latent).squeeze(-1) * valid
            anchor_end_logits = self.end_readout(latent).squeeze(-1) * valid
            anchor_support_logits = self.support_readout(latent).squeeze(-1) * valid
            anchor_span_logits, anchor_valid = self._span_field(
                latent,
                latent,
                anchor_start_logits,
                anchor_end_logits,
                anchor_support_logits,
                video_padding_mask,
            )
            span_valid = span_valid & anchor_valid
            structured_scale = self.structured_energy_bound * torch.sigmoid(
                self.structured_energy_scale
            )
            base_span_logits = anchor_span_logits + structured_scale * torch.tanh(
                structured_span_logits - anchor_span_logits
            )
        else:
            base_span_logits = structured_span_logits
        quality_logits = self._quality_field(
            base_span_logits,
            evidence_logits,
            support_logits,
            start_logits,
            end_logits,
            evidence_tokens,
            support_tokens,
            start_tokens,
            end_tokens,
            query_summary,
            video_padding_mask,
            span_valid,
        )
        span_logits = base_span_logits
        if self.use_quality_energy:
            if self.bounded_field_energy:
                quality_energy = (
                    self.quality_energy_bound
                    * torch.sigmoid(self.quality_energy_scale)
                    * torch.tanh(quality_logits)
                )
            else:
                quality_energy = F.softplus(self.quality_energy_scale) * quality_logits
            span_logits = span_logits + quality_energy
        span_logits = span_logits.masked_fill(~span_valid, -1e4)
        return FieldStructuredOutput(
            evidence_logits=evidence_logits,
            support_logits=support_logits,
            start_transition_logits=start_logits,
            end_transition_logits=end_logits,
            span_logits=span_logits,
            span_valid_mask=span_valid,
            base_span_logits=base_span_logits,
            quality_logits=quality_logits,
        )
