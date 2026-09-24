"""Standalone EventField-Net with structured field routing and 2-D spans."""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .model import EventFieldF0, EventFieldOutput


@dataclass
class EventFieldNetOutput(EventFieldOutput):
    """One-dimensional semantic fields and the joint legal-span field."""

    span_logits: Tensor
    span_valid_mask: Tensor


class CrossModalTemporalBlock(nn.Module):
    """Alternate query/video exchange, temporal attention, and local scales."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.video_norm = nn.LayerNorm(hidden_dim)
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.video_from_query = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.query_from_video = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.temporal_norm = nn.LayerNorm(hidden_dim)
        self.temporal_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.local_norm = nn.LayerNorm(hidden_dim)
        self.local_convs = nn.ModuleList(
            nn.Conv1d(
                hidden_dim,
                hidden_dim,
                kernel_size=kernel,
                padding=kernel // 2,
                groups=hidden_dim,
            )
            for kernel in (3, 7, 15)
        )
        self.scale_gate = nn.Linear(hidden_dim, len(self.local_convs))
        self.local_projection = nn.Linear(hidden_dim, hidden_dim)
        self.video_ffn_norm = nn.LayerNorm(hidden_dim)
        self.query_ffn_norm = nn.LayerNorm(hidden_dim)
        self.video_ffn = self._ffn(hidden_dim, feedforward_dim, dropout)
        self.query_ffn = self._ffn(hidden_dim, feedforward_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _ffn(hidden_dim: int, feedforward_dim: int, dropout: float) -> nn.Module:
        return nn.Sequential(
            nn.Linear(hidden_dim, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, hidden_dim),
        )

    @staticmethod
    def _masked_mean(tokens: Tensor, padding_mask: Tensor) -> Tensor:
        valid = (~padding_mask).to(tokens.dtype).unsqueeze(-1)
        return (tokens * valid).sum(1) / valid.sum(1).clamp_min(1.0)

    def forward(
        self,
        video: Tensor,
        query: Tensor,
        video_padding_mask: Tensor,
        query_padding_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        video_update, _ = self.video_from_query(
            query=self.video_norm(video),
            key=self.query_norm(query),
            value=self.query_norm(query),
            key_padding_mask=query_padding_mask,
            need_weights=False,
        )
        video = video + self.dropout(video_update)
        query_update, _ = self.query_from_video(
            query=self.query_norm(query),
            key=self.video_norm(video),
            value=self.video_norm(video),
            key_padding_mask=video_padding_mask,
            need_weights=False,
        )
        query = query + self.dropout(query_update)

        temporal = self.temporal_norm(video)
        temporal_update, _ = self.temporal_attention(
            query=temporal,
            key=temporal,
            value=temporal,
            key_padding_mask=video_padding_mask,
            need_weights=False,
        )
        video = video + self.dropout(temporal_update)

        local = self.local_norm(video).transpose(1, 2)
        scales = torch.stack(
            [convolution(local).transpose(1, 2) for convolution in self.local_convs],
            dim=2,
        )
        query_summary = self._masked_mean(query, query_padding_mask)
        scale_weights = torch.softmax(self.scale_gate(query_summary), dim=-1)
        mixed = (scales * scale_weights[:, None, :, None]).sum(dim=2)
        video = video + self.dropout(self.local_projection(mixed))
        video = video + self.dropout(self.video_ffn(self.video_ffn_norm(video)))
        query = query + self.dropout(self.query_ffn(self.query_ffn_norm(query)))
        return video, query


class FieldAdapter(nn.Module):
    """A lightweight role-specific temporal adapter."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.depthwise = nn.Conv1d(
            hidden_dim, hidden_dim, kernel_size=5, padding=2, groups=hidden_dim
        )
        self.pointwise = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )

        self.residual_log_scale = nn.Parameter(torch.tensor(-3.0))

    def forward(self, tokens: Tensor) -> Tensor:
        normalized = self.norm(tokens)
        local = self.depthwise(normalized.transpose(1, 2)).transpose(1, 2)
        update = local + self.pointwise(normalized)
        return tokens + F.softplus(self.residual_log_scale) * update


class EventFieldNet(nn.Module):
    """Estimate Evidence, Support, Transition, and a joint span field.

    ``shared`` uses a training multimodal sequence with linear field readouts.
    ``routed`` adds query-conditioned role routing and Evidence-to-Support flow.
    ``full`` additionally learns start/end compatibility in the 2-D span field.
    ``full_detach`` is the same as ``full`` but stops Evidence gradients at the
    Evidence-to-Support message, providing a controlled collaboration ablation.
    """

    VALID_ARCHITECTURES = {
        "shared",
        "shared_full",
        "routed",
        "full",
        "full_detach",
        "full_safe",
    }

    def __init__(
        self,
        video_dim: int,
        query_dim: int,
        architecture: str = "full",
        hidden_dim: int = 384,
        num_heads: int = 8,
        interaction_layers: int = 3,
        feedforward_dim: int = 1536,
        pair_dim: int = 96,
        dropout: float = 0.15,
        min_span_clips: int = 2,
    ) -> None:
        super().__init__()
        if architecture not in EventFieldNet.VALID_ARCHITECTURES:
            raise ValueError(f"unknown architecture: {architecture}")
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.architecture = architecture
        self.hidden_dim = hidden_dim
        self.min_span_clips = min_span_clips
        self.video_projection = nn.Linear(video_dim, hidden_dim)
        self.query_projection = nn.Linear(query_dim, hidden_dim)
        self.video_input_norm = nn.LayerNorm(hidden_dim)
        self.query_input_norm = nn.LayerNorm(hidden_dim)
        self.blocks = nn.ModuleList(
            CrossModalTemporalBlock(hidden_dim, num_heads, feedforward_dim, dropout)
            for _ in range(interaction_layers)
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

        self.use_routing = architecture not in {"shared", "shared_full"}
        if self.use_routing:
            self.role_tokens = nn.Parameter(torch.empty(4, hidden_dim))
            nn.init.normal_(self.role_tokens, std=0.02)
            self.role_from_query = nn.MultiheadAttention(
                hidden_dim, num_heads, dropout=dropout, batch_first=True
            )
            self.role_norm = nn.LayerNorm(hidden_dim)
            self.role_film = nn.ModuleList(
                nn.Linear(hidden_dim, 2 * hidden_dim) for _ in range(4)
            )
            for projection in self.role_film:
                nn.init.zeros_(projection.weight)
                nn.init.zeros_(projection.bias)
            self.field_adapters = nn.ModuleList(
                FieldAdapter(hidden_dim, dropout) for _ in range(4)
            )
            self.evidence_to_support = nn.Linear(hidden_dim, hidden_dim)
            self.evidence_support_gate = nn.Linear(2 * hidden_dim, hidden_dim)
            self.support_message_scale = nn.Parameter(torch.tensor(-2.0))
            self.start_change_projection = nn.Linear(2 * hidden_dim, hidden_dim)
            self.end_change_projection = nn.Linear(2 * hidden_dim, hidden_dim)
            nn.init.zeros_(self.start_change_projection.weight)
            nn.init.zeros_(self.start_change_projection.bias)
            nn.init.zeros_(self.end_change_projection.weight)
            nn.init.zeros_(self.end_change_projection.bias)

        self.evidence_readout = nn.Linear(hidden_dim, 1)
        self.support_readout = nn.Linear(hidden_dim, 1)
        self.start_readout = nn.Linear(hidden_dim, 1)
        self.end_readout = nn.Linear(hidden_dim, 1)

        self.use_pair_compatibility = architecture in {
            "shared_full",
            "full",
            "full_detach",
            "full_safe",
        }
        if self.use_pair_compatibility:
            self.start_pair = nn.Linear(hidden_dim, pair_dim)
            self.end_pair = nn.Linear(hidden_dim, pair_dim)
            self.duration_bias = nn.Sequential(
                nn.Linear(1, pair_dim),
                nn.GELU(),
                nn.Linear(pair_dim, 1),
            )
            nn.init.zeros_(self.duration_bias[-1].weight)
            nn.init.zeros_(self.duration_bias[-1].bias)
            self.pair_scale = nn.Parameter(torch.tensor(-1.0))
        self.support_span_scale = nn.Parameter(torch.tensor(0.54132485))

    @staticmethod
    def _normalize_mask(
        mask: Optional[Tensor], batch: int, length: int, device: torch.device
    ) -> Tensor:
        if mask is None:
            return torch.zeros(batch, length, dtype=torch.bool, device=device)
        if mask.shape != (batch, length):
            raise ValueError(f"padding mask must have shape {(batch, length)}")
        return mask.to(device=device, dtype=torch.bool)

    @staticmethod
    def _make_safe(tokens: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        return EventFieldF0._make_attention_safe(tokens, mask)

    def _role_route(
        self,
        encoded: Tensor,
        query: Tensor,
        query_padding_mask: Tensor,
    ) -> list[Tensor]:
        roles = self.role_tokens.unsqueeze(0).expand(encoded.shape[0], -1, -1)
        role_update, _ = self.role_from_query(
            query=roles,
            key=query,
            value=query,
            key_padding_mask=query_padding_mask,
            need_weights=False,
        )
        roles = self.role_norm(roles + role_update)
        fields = []
        for index, adapter in enumerate(self.field_adapters):
            gamma, beta = self.role_film[index](roles[:, index]).chunk(2, dim=-1)
            routed = encoded * (1.0 + 0.5 * torch.tanh(gamma[:, None]))
            routed = routed + beta[:, None]
            fields.append(adapter(routed))
        return fields

    @staticmethod
    def _directional_changes(encoded: Tensor) -> tuple[Tensor, Tensor]:
        previous = F.pad(encoded[:, :-1], (0, 0, 1, 0))
        following = F.pad(encoded[:, 1:], (0, 0, 0, 1))
        return encoded - previous, encoded - following

    def _span_field(
        self,
        start_tokens: Tensor,
        end_tokens: Tensor,
        start_logits: Tensor,
        end_logits: Tensor,
        support_logits: Tensor,
        padding_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch, length = start_logits.shape
        indices = torch.arange(length, device=start_logits.device)
        starts = indices.view(length, 1)
        ends = indices.view(1, length)
        span_lengths = ends - starts + 1
        valid_lengths = (~padding_mask).sum(dim=1)
        valid = (
            (span_lengths >= self.min_span_clips)
            .unsqueeze(0)
            .expand(batch, -1, -1)
            .clone()
        )
        valid &= starts.unsqueeze(0) < valid_lengths[:, None, None]
        valid &= ends.unsqueeze(0) < valid_lengths[:, None, None]

        span_support_logits = support_logits
        if self.architecture == "full_safe":
            span_support_logits = span_support_logits.detach()
        support = torch.sigmoid(span_support_logits).masked_fill(padding_mask, 0)
        prefix = F.pad(support.cumsum(dim=1), (1, 0))
        support_mean = (
            prefix[:, ends + 1] - prefix[:, starts]
        ) / span_lengths.clamp_min(1)
        logits = F.logsigmoid(start_logits).unsqueeze(2) + F.logsigmoid(
            end_logits
        ).unsqueeze(1)
        logits = logits + F.softplus(self.support_span_scale) * support_mean

        if self.use_pair_compatibility:
            start_pair = F.normalize(self.start_pair(start_tokens), dim=-1)
            end_pair = F.normalize(self.end_pair(end_tokens), dim=-1)
            compatibility = torch.einsum("bid,bjd->bij", start_pair, end_pair)
            duration = (
                span_lengths.to(logits.dtype)
                / valid_lengths.to(logits.dtype).clamp_min(1)[:, None, None]
            )
            duration = duration.clamp(0, 1)
            duration_bias = self.duration_bias(duration.unsqueeze(-1)).squeeze(-1)
            logits = (
                logits + F.softplus(self.pair_scale) * compatibility + duration_bias
            )
        return logits.masked_fill(~valid, -1e4), valid

    def forward(
        self,
        video_features: Tensor,
        query_features: Tensor,
        video_padding_mask: Optional[Tensor] = None,
        query_padding_mask: Optional[Tensor] = None,
    ) -> EventFieldNetOutput:
        EventFieldF0._validate_inputs(video_features, query_features)
        batch, video_length, _ = video_features.shape
        query_length = query_features.shape[1]
        video_padding_mask = self._normalize_mask(
            video_padding_mask, batch, video_length, video_features.device
        )
        query_padding_mask = self._normalize_mask(
            query_padding_mask, batch, query_length, query_features.device
        )
        video = self.video_input_norm(self.video_projection(video_features))
        query = self.query_input_norm(self.query_projection(query_features))
        query, safe_query_mask = self._make_safe(query, query_padding_mask)
        video = video + EventFieldF0._sinusoidal_encoding(
            video_length, self.hidden_dim, video.device, video.dtype
        )
        query = query + EventFieldF0._sinusoidal_encoding(
            query_length, self.hidden_dim, query.device, query.dtype
        )
        video, safe_video_mask = self._make_safe(video, video_padding_mask)
        for block in self.blocks:
            video, query = block(video, query, safe_video_mask, safe_query_mask)
        encoded = self.output_norm(video)

        if self.use_routing:
            evidence_tokens, support_tokens, start_tokens, end_tokens = (
                self._role_route(encoded, query, safe_query_mask)
            )
            evidence_message = evidence_tokens
            if self.architecture == "full_detach":
                evidence_message = evidence_message.detach()
            gate = torch.sigmoid(
                self.evidence_support_gate(
                    torch.cat((support_tokens, evidence_message), dim=-1)
                )
            )
            support_tokens = support_tokens + (
                F.softplus(self.support_message_scale)
                * gate
                * self.evidence_to_support(evidence_message)
            )
            enter_change, exit_change = self._directional_changes(encoded)
            start_tokens = start_tokens + self.start_change_projection(
                torch.cat((start_tokens, enter_change), dim=-1)
            )
            end_tokens = end_tokens + self.end_change_projection(
                torch.cat((end_tokens, exit_change), dim=-1)
            )
        else:
            evidence_tokens = support_tokens = start_tokens = end_tokens = encoded

        valid = (~video_padding_mask).to(encoded.dtype)
        evidence_logits = self.evidence_readout(evidence_tokens).squeeze(-1) * valid
        support_logits = self.support_readout(support_tokens).squeeze(-1) * valid
        start_logits = self.start_readout(start_tokens).squeeze(-1) * valid
        end_logits = self.end_readout(end_tokens).squeeze(-1) * valid
        span_logits, span_valid = self._span_field(
            start_tokens,
            end_tokens,
            start_logits,
            end_logits,
            support_logits,
            video_padding_mask,
        )
        return EventFieldNetOutput(
            evidence_logits=evidence_logits,
            support_logits=support_logits,
            start_transition_logits=start_logits,
            end_transition_logits=end_logits,
            span_logits=span_logits,
            span_valid_mask=span_valid,
        )
