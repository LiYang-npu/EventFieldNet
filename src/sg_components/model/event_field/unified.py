"""Unified latent Event Field model used by the F1 comparison."""

from typing import Optional

from torch import Tensor, nn

from .model import EventFieldF0, EventFieldOutput


class EventFieldF1(EventFieldF0):
    """F0 conditioner followed by a shared temporal latent field."""

    def __init__(
        self,
        video_dim: int,
        query_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 8,
        temporal_layers: int = 2,
        feedforward_dim: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(
            video_dim=video_dim,
            query_dim=query_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            temporal_layers=temporal_layers,
            feedforward_dim=feedforward_dim,
            dropout=dropout,
        )
        self.latent_norm = nn.LayerNorm(hidden_dim)
        self.latent_projection = nn.Linear(hidden_dim, hidden_dim)
        self.latent_activation = nn.GELU()
        self.latent_temporal = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            padding=1,
        )

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
        video_tokens = self.video_projection(video_features)
        query_tokens = self.query_projection(query_features)
        query_tokens, safe_query_mask = self._make_attention_safe(
            query_tokens,
            query_padding_mask,
        )
        conditioned, _ = self.cross_attention(
            query=video_tokens,
            key=query_tokens,
            value=query_tokens,
            key_padding_mask=safe_query_mask,
            need_weights=False,
        )
        conditioned = self.cross_attention_norm(video_tokens + conditioned)
        conditioned = conditioned + self._sinusoidal_encoding(
            video_length,
            conditioned.shape[-1],
            conditioned.device,
            conditioned.dtype,
        )
        conditioned, safe_video_mask = self._make_attention_safe(
            conditioned,
            video_padding_mask,
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
                self.start_transition_readout,
                latent,
                valid,
            ),
            end_transition_logits=self._readout(
                self.end_transition_readout,
                latent,
                valid,
            ),
        )
