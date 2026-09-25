"""Controlled cross-modal interaction variants for continuous Event Field."""

from typing import Optional

import torch
from torch import Tensor, nn

from .model import EventFieldF0, EventFieldOutput


class EventFieldInteraction(EventFieldF0):
    """Event Field with composable bidirectional, role, and scale interaction."""

    VALID_VARIANTS = {"i1", "i2", "i3", "i4", "i5"}

    def __init__(
        self,
        video_dim: int,
        query_dim: int,
        variant: str,
        hidden_dim: int = 256,
        num_heads: int = 8,
        temporal_layers: int = 2,
        feedforward_dim: int = 1024,
        dropout: float = 0.1,
        num_roles: int = 4,
    ) -> None:
        if variant not in self.VALID_VARIANTS:
            raise ValueError(f"unknown interaction variant: {variant}")
        super().__init__(
            video_dim=video_dim,
            query_dim=query_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            temporal_layers=temporal_layers,
            feedforward_dim=feedforward_dim,
            dropout=dropout,
        )
        self.variant = variant
        self.use_bidirectional = variant in {"i1", "i4", "i5"}
        self.use_roles = variant in {"i2", "i4", "i5"}
        self.use_multiscale = variant in {"i3", "i5"}

        if self.use_bidirectional:
            self.video_to_query = nn.MultiheadAttention(
                hidden_dim, num_heads, dropout=dropout, batch_first=True
            )
            self.query_feedback = nn.MultiheadAttention(
                hidden_dim, num_heads, dropout=dropout, batch_first=True
            )
            self.query_update_norm = nn.LayerNorm(hidden_dim)
            self.feedback_gate = nn.Linear(2 * hidden_dim, hidden_dim)

        if self.use_roles:
            self.role_tokens = nn.Parameter(torch.empty(num_roles, hidden_dim))
            nn.init.normal_(self.role_tokens, std=0.02)
            self.role_from_query = nn.MultiheadAttention(
                hidden_dim, num_heads, dropout=dropout, batch_first=True
            )
            self.video_from_roles = nn.MultiheadAttention(
                hidden_dim, num_heads, dropout=dropout, batch_first=True
            )
            self.role_norm = nn.LayerNorm(hidden_dim)
            self.role_gate = nn.Linear(2 * hidden_dim, hidden_dim)

        if self.use_multiscale:
            self.scale_convs = nn.ModuleList(
                nn.Conv1d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=kernel,
                    padding=kernel // 2,
                    groups=hidden_dim,
                )
                for kernel in (3, 5, 9)
            )
            self.scale_projection = nn.Linear(hidden_dim, hidden_dim)
            self.scale_gate = nn.Linear(hidden_dim, len(self.scale_convs))
            self.scale_norm = nn.LayerNorm(hidden_dim)

        self.latent_norm = nn.LayerNorm(hidden_dim)
        self.latent_projection = nn.Linear(hidden_dim, hidden_dim)
        self.latent_activation = nn.GELU()
        self.latent_temporal = nn.Conv1d(
            hidden_dim, hidden_dim, kernel_size=3, padding=1
        )

    @staticmethod
    def _masked_query_mean(query: Tensor, padding_mask: Tensor) -> Tensor:
        valid = (~padding_mask).to(query.dtype).unsqueeze(-1)
        return (query * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)

    def _apply_bidirectional(
        self,
        video: Tensor,
        query: Tensor,
        conditioned: Tensor,
        video_padding_mask: Tensor,
        query_padding_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        query_update, _ = self.video_to_query(
            query=query,
            key=video,
            value=video,
            key_padding_mask=video_padding_mask,
            need_weights=False,
        )
        updated_query = self.query_update_norm(query + query_update)
        feedback, _ = self.query_feedback(
            query=conditioned,
            key=updated_query,
            value=updated_query,
            key_padding_mask=query_padding_mask,
            need_weights=False,
        )
        gate = torch.sigmoid(
            self.feedback_gate(torch.cat((conditioned, feedback), dim=-1))
        )
        return self.cross_attention_norm(conditioned + gate * feedback), updated_query

    def _apply_roles(
        self,
        conditioned: Tensor,
        query: Tensor,
        query_padding_mask: Tensor,
    ) -> Tensor:
        roles = self.role_tokens.unsqueeze(0).expand(query.shape[0], -1, -1)
        role_update, _ = self.role_from_query(
            query=roles,
            key=query,
            value=query,
            key_padding_mask=query_padding_mask,
            need_weights=False,
        )
        roles = self.role_norm(roles + role_update)
        role_context, _ = self.video_from_roles(
            query=conditioned,
            key=roles,
            value=roles,
            need_weights=False,
        )
        gate = torch.sigmoid(
            self.role_gate(torch.cat((conditioned, role_context), dim=-1))
        )
        return self.cross_attention_norm(conditioned + gate * role_context)

    def _apply_multiscale(
        self,
        conditioned: Tensor,
        query: Tensor,
        query_padding_mask: Tensor,
    ) -> Tensor:
        sequence = conditioned.transpose(1, 2)
        scales = torch.stack(
            [convolution(sequence).transpose(1, 2) for convolution in self.scale_convs],
            dim=2,
        )
        query_summary = self._masked_query_mean(query, query_padding_mask)
        weights = torch.softmax(self.scale_gate(query_summary), dim=-1)
        mixed = (scales * weights[:, None, :, None]).sum(dim=2)
        return self.scale_norm(conditioned + self.scale_projection(mixed))

    def forward(
        self,
        video_features: Tensor,
        query_features: Tensor,
        video_padding_mask: Optional[Tensor] = None,
        query_padding_mask: Optional[Tensor] = None,
    ) -> EventFieldOutput:
        self._validate_inputs(video_features, query_features)
        batch_size, video_length, _ = video_features.shape
        query_length = query_features.shape[1]
        video_padding_mask = self._normalize_mask(
            video_padding_mask,
            batch_size,
            video_length,
            video_features.device,
        )
        query_padding_mask = self._normalize_mask(
            query_padding_mask,
            batch_size,
            query_length,
            query_features.device,
        )
        video = self.video_projection(video_features)
        query = self.query_projection(query_features)
        query, safe_query_mask = self._make_attention_safe(query, query_padding_mask)
        conditioned_update, _ = self.cross_attention(
            query=video,
            key=query,
            value=query,
            key_padding_mask=safe_query_mask,
            need_weights=False,
        )
        conditioned = self.cross_attention_norm(video + conditioned_update)

        if self.use_bidirectional:
            conditioned, query = self._apply_bidirectional(
                video,
                query,
                conditioned,
                video_padding_mask,
                safe_query_mask,
            )
        if self.use_roles:
            conditioned = self._apply_roles(conditioned, query, safe_query_mask)
        if self.use_multiscale:
            conditioned = self._apply_multiscale(conditioned, query, safe_query_mask)

        conditioned = conditioned + self._sinusoidal_encoding(
            video_length,
            conditioned.shape[-1],
            conditioned.device,
            conditioned.dtype,
        )
        conditioned, safe_video_mask = self._make_attention_safe(
            conditioned, video_padding_mask
        )
        encoded = self.temporal_encoder(
            conditioned,
            src_key_padding_mask=safe_video_mask,
        )
        latent_update = self.latent_projection(self.latent_norm(encoded))
        latent_update = self.latent_activation(latent_update)
        latent_update = self.latent_temporal(latent_update.transpose(1, 2))
        latent = encoded + latent_update.transpose(1, 2)
        valid = (~video_padding_mask).to(dtype=latent.dtype)
        return EventFieldOutput(
            evidence_logits=self._readout(self.evidence_readout, latent, valid),
            support_logits=self._readout(self.support_readout, latent, valid),
            start_transition_logits=self._readout(
                self.start_transition_readout, latent, valid
            ),
            end_transition_logits=self._readout(
                self.end_transition_readout, latent, valid
            ),
        )
