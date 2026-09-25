"""Boundary-preserving contextual EventField decoder.

The proven Stage14 field branch remains the anchor.  This module only adds a
controlled contextual residual whose temporal messages are attenuated when
their path crosses a learned event boundary.
"""

from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from sg_components.model.event_field.framework_context_contrast import (
    EventFieldNetContextContrast,
)


class BoundaryPreservingContext(nn.Module):
    """Query-conditioned multi-scale context with soft boundary barriers."""

    def __init__(
        self,
        hidden_dim: int,
        rank: int,
        scales: Sequence[int],
        residual_bound: float,
        use_boundary_gate: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        self.scales = tuple(int(scale) for scale in scales)
        self.residual_bound = float(residual_bound)
        self.use_boundary_gate = bool(use_boundary_gate)
        self.norm = nn.LayerNorm(hidden_dim)
        self.token_down = nn.Linear(hidden_dim, rank)
        self.query_down = nn.Linear(hidden_dim, rank)
        self.boundary_head = nn.Sequential(
            nn.Linear(3 * rank, rank),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rank, 1),
        )
        self.boundary_strength = nn.Parameter(torch.tensor(-2.0))
        self.scale_logits = nn.Parameter(torch.zeros(len(self.scales)))
        self.role_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(rank, rank),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(rank, hidden_dim),
                )
                for _ in range(4)
            ]
        )
        self.role_scales = nn.Parameter(torch.full((4,), -2.0))
        for head in self.role_heads:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    @staticmethod
    def _shift(tokens: Tensor, offset: int) -> tuple[Tensor, Tensor]:
        batch, length, channels = tokens.shape
        if offset == 0:
            return tokens, torch.ones(
                batch, length, dtype=tokens.dtype, device=tokens.device
            )
        distance = abs(offset)
        zeros = torch.zeros(
            batch, distance, channels, dtype=tokens.dtype, device=tokens.device
        )
        valid = torch.ones(
            batch, length - distance, dtype=tokens.dtype, device=tokens.device
        )
        invalid = torch.zeros(batch, distance, dtype=tokens.dtype, device=tokens.device)
        if offset > 0:
            return torch.cat((tokens[:, offset:], zeros), dim=1), torch.cat(
                (valid, invalid), dim=1
            )
        return torch.cat((zeros, tokens[:, :offset]), dim=1), torch.cat(
            (invalid, valid), dim=1
        )

    def _barrier_context(
        self, tokens: Tensor, boundary: Tensor, radius: int, strength: Tensor
    ) -> Tensor:
        weighted = torch.zeros_like(tokens)
        normalizer = torch.zeros(
            tokens.shape[0],
            tokens.shape[1],
            1,
            dtype=tokens.dtype,
            device=tokens.device,
        )
        for offset in range(-radius, radius + 1):
            shifted, valid = self._shift(tokens, offset)
            gate = torch.ones_like(valid)
            if offset != 0 and self.use_boundary_gate:
                for step in range(abs(offset)):
                    edge_offset = step if offset > 0 else offset + step
                    edge, edge_valid = self._shift(boundary.unsqueeze(-1), edge_offset)
                    gate = gate * torch.exp(-strength * edge.squeeze(-1))
                    gate = gate * edge_valid
            weighted = weighted + shifted * (gate * valid).unsqueeze(-1)
            normalizer = normalizer + (gate * valid).unsqueeze(-1)
        return weighted / normalizer.clamp_min(1e-6)

    def forward(
        self, latent: Tensor, query_summary: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        low_rank = self.token_down(self.norm(latent))
        query = self.query_down(query_summary)[:, None]
        aligned = low_rank * torch.tanh(query)
        previous = F.pad(low_rank[:, :-1], (0, 0, 1, 0))
        following = F.pad(low_rank[:, 1:], (0, 0, 0, 1))
        boundary_logits = self.boundary_head(
            torch.cat((low_rank - previous, low_rank - following, aligned), dim=-1)
        ).squeeze(-1)
        boundary = torch.sigmoid(boundary_logits)
        strength = 1.5 * torch.sigmoid(self.boundary_strength)
        scale_weights = torch.softmax(self.scale_logits, dim=0)
        role_features = [torch.zeros_like(low_rank) for _ in range(4)]
        for scale_weight, radius in zip(scale_weights, self.scales):
            context = self._barrier_context(low_rank, boundary, radius, strength)
            previous_context = F.pad(context[:, :-1], (0, 0, 1, 0))
            next_context = F.pad(context[:, 1:], (0, 0, 0, 1))
            features = (
                aligned + (low_rank - context),
                aligned + context,
                aligned + (low_rank - context) + context - previous_context,
                aligned + (low_rank - context) + context - next_context,
            )
            for role, feature in enumerate(features):
                role_features[role] = role_features[role] + scale_weight * feature
        outputs = []
        for role, (feature, head) in enumerate(zip(role_features, self.role_heads)):
            update = head(feature)
            scale = self.residual_bound * torch.sigmoid(self.role_scales[role])
            outputs.append(latent + scale * torch.tanh(update))
        return tuple(outputs)


class EventFieldNetBoundaryPreserving(EventFieldNetContextContrast):
    """Stage14-compatible context with an optional learned boundary barrier."""

    def __init__(
        self,
        *args,
        boundary_rank: int = 96,
        boundary_scales: Sequence[int] = (1, 3, 7),
        boundary_residual_bound: float = 1.0,
        use_boundary_gate: bool = True,
        dropout: float = 0.15,
        **kwargs,
    ) -> None:
        super().__init__(*args, dropout=dropout, **kwargs)
        self.boundary_preserving = BoundaryPreservingContext(
            self.hidden_dim,
            boundary_rank,
            boundary_scales,
            boundary_residual_bound,
            use_boundary_gate,
            dropout,
        )

    def _build_field_tokens(
        self,
        latent: Tensor,
        query: Tensor,
        safe_query_mask: Tensor,
        query_summary: Tensor,
    ):
        del query, safe_query_mask
        return self.boundary_preserving(latent, query_summary)
