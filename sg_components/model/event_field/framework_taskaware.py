"""Task-aware backbone for Evidence, Support, and Transition fields."""

from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .framework import EventFieldNetOutput
from .framework_proven import EventFieldNetProven


class QueryFieldAdapter(nn.Module):
    """Field-specific query interaction with an identity initialization."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.role_film = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        nn.init.zeros_(self.role_film.weight)
        nn.init.zeros_(self.role_film.bias)
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)
        self.residual_logit = nn.Parameter(torch.tensor(-3.0))

    def forward(
        self,
        tokens: Tensor,
        query: Tensor,
        query_padding_mask: Tensor,
        role: Tensor,
    ) -> Tensor:
        context, _ = self.cross_attention(
            query=tokens,
            key=query,
            value=query,
            key_padding_mask=query_padding_mask,
            need_weights=False,
        )
        gamma, beta = self.role_film(role).chunk(2, dim=-1)
        conditioned = self.norm(tokens + context)
        conditioned = conditioned * (1 + 0.5 * torch.tanh(gamma[:, None]))
        conditioned = conditioned + beta[:, None]
        scale = 0.25 * torch.sigmoid(self.residual_logit)
        return tokens + scale * self.output(conditioned)


class SupportPyramid(nn.Module):
    """Query-conditioned receptive fields for event-interior occupancy."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.convs = nn.ModuleList(
            nn.Conv1d(
                hidden_dim,
                hidden_dim,
                kernel_size=kernel,
                padding=kernel // 2,
                groups=hidden_dim,
            )
            for kernel in (3, 7, 15)
        )
        self.scale_gate = nn.Linear(hidden_dim, len(self.convs))
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)
        self.residual_logit = nn.Parameter(torch.tensor(-3.0))

    def forward(self, tokens: Tensor, role: Tensor) -> Tensor:
        normalized = self.norm(tokens).transpose(1, 2)
        scales = torch.stack(
            [convolution(normalized).transpose(1, 2) for convolution in self.convs],
            dim=2,
        )
        weights = torch.softmax(self.scale_gate(role), dim=-1)
        mixed = (scales * weights[:, None, :, None]).sum(dim=2)
        scale = 0.25 * torch.sigmoid(self.residual_logit)
        return tokens + scale * self.output(mixed)


class TransitionDynamics(nn.Module):
    """Directional temporal changes for event entry and exit."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.start_projection = self._make_projection(hidden_dim, dropout)
        self.end_projection = self._make_projection(hidden_dim, dropout)
        self.start_residual_logit = nn.Parameter(torch.tensor(-3.0))
        self.end_residual_logit = nn.Parameter(torch.tensor(-3.0))

    @staticmethod
    def _make_projection(hidden_dim: int, dropout: float) -> nn.Sequential:
        module = nn.Sequential(
            nn.LayerNorm(3 * hidden_dim),
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        nn.init.zeros_(module[-1].weight)
        nn.init.zeros_(module[-1].bias)
        return module

    def forward(
        self,
        tokens: Tensor,
        start_role: Tensor,
        end_role: Tensor,
    ) -> tuple[Tensor, Tensor]:
        previous = F.pad(tokens[:, :-1], (0, 0, 1, 0))
        following = F.pad(tokens[:, 1:], (0, 0, 0, 1))
        enter = tokens - previous
        exit_ = tokens - following
        start_input = torch.cat(
            (tokens, enter, start_role[:, None].expand_as(tokens)), dim=-1
        )
        end_input = torch.cat(
            (tokens, exit_, end_role[:, None].expand_as(tokens)), dim=-1
        )
        start_scale = 0.25 * torch.sigmoid(self.start_residual_logit)
        end_scale = 0.25 * torch.sigmoid(self.end_residual_logit)
        return (
            tokens + start_scale * self.start_projection(start_input),
            tokens + end_scale * self.end_projection(end_input),
        )


class EventFieldNetTaskAware(EventFieldNetProven):
    """Strong shared semantics followed by task-aware field interactions."""

    VALID_ARCHITECTURES = {"task_query", "task_temporal", "task_full"}

    def __init__(
        self,
        *args,
        architecture: str = "task_full",
        num_heads: int = 8,
        dropout: float = 0.15,
        **kwargs,
    ) -> None:
        if architecture not in self.VALID_ARCHITECTURES:
            raise ValueError(f"unknown task-aware architecture: {architecture}")
        self.task_architecture = architecture
        super().__init__(
            *args,
            architecture="shared",
            num_heads=num_heads,
            dropout=dropout,
            **kwargs,
        )
        hidden_dim = self.hidden_dim
        self.role_tokens = nn.Parameter(torch.empty(4, hidden_dim))
        nn.init.normal_(self.role_tokens, std=0.02)
        self.role_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.role_norm = nn.LayerNorm(hidden_dim)
        self.query_adapters = nn.ModuleList(
            QueryFieldAdapter(hidden_dim, num_heads, dropout) for _ in range(4)
        )
        self.support_pyramid = SupportPyramid(hidden_dim, dropout)
        self.transition_dynamics = TransitionDynamics(hidden_dim, dropout)
        self.task_readouts = nn.ModuleList(nn.Linear(hidden_dim, 1) for _ in range(4))
        for readout in self.task_readouts:
            nn.init.zeros_(readout.weight)
            nn.init.zeros_(readout.bias)

    def _query_roles(self, query: Tensor, query_padding_mask: Tensor) -> Tensor:
        roles = self.role_tokens.unsqueeze(0).expand(query.shape[0], -1, -1)
        update, _ = self.role_attention(
            query=roles,
            key=query,
            value=query,
            key_padding_mask=query_padding_mask,
            need_weights=False,
        )
        return self.role_norm(roles + update)

    @staticmethod
    def _task_logit(
        base_readout: nn.Linear,
        task_readout: nn.Linear,
        shared: Tensor,
        specialized: Tensor,
        valid: Tensor,
    ) -> Tensor:
        base = base_readout(shared).squeeze(-1)
        residual = task_readout(specialized).squeeze(-1)
        return (base + residual) * valid

    def forward(
        self,
        video_features: Tensor,
        query_features: Tensor,
        video_padding_mask: Optional[Tensor] = None,
        query_padding_mask: Optional[Tensor] = None,
    ) -> EventFieldNetOutput:
        latent, query, video_padding_mask, safe_query_mask = self.stem.encode(
            video_features,
            query_features,
            video_padding_mask,
            query_padding_mask,
        )
        roles = self._query_roles(query, safe_query_mask)
        fields = [latent, latent, latent, latent]

        if self.task_architecture in {"task_query", "task_full"}:
            fields = [
                adapter(latent, query, safe_query_mask, roles[:, index])
                for index, adapter in enumerate(self.query_adapters)
            ]
        if self.task_architecture in {"task_temporal", "task_full"}:
            fields[1] = self.support_pyramid(fields[1], roles[:, 1])
            fields[2], fields[3] = self.transition_dynamics(
                fields[2], roles[:, 2], roles[:, 3]
            )

        valid = (~video_padding_mask).to(latent.dtype)
        base_readouts = (
            self.evidence_readout,
            self.support_readout,
            self.start_readout,
            self.end_readout,
        )
        logits = [
            self._task_logit(base, task, latent, field, valid)
            for base, task, field in zip(base_readouts, self.task_readouts, fields)
        ]
        evidence_logits, support_logits, start_logits, end_logits = logits
        span_logits, span_valid = self._span_field(
            fields[2],
            fields[3],
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
