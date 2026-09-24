"""Query-conditioned temporal Event Field F0 model."""

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn


@dataclass
class EventFieldOutput:
    """Per-timestep logits produced by Event Field F0."""

    evidence_logits: Tensor
    support_logits: Tensor
    start_transition_logits: Tensor
    end_transition_logits: Tensor


class EventFieldF0(nn.Module):
    """Standalone query-conditioned event field over a video sequence.

    The module only consumes generic video/query features. It does not read
    SG-DETR saliency, SGCA, proposal, or detector outputs.
    """

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
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.video_projection = nn.Linear(video_dim, hidden_dim)
        self.query_projection = nn.Linear(query_dim, hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attention_norm = nn.LayerNorm(hidden_dim)
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            temporal_layer,
            num_layers=temporal_layers,
            norm=nn.LayerNorm(hidden_dim),
        )
        self.evidence_readout = nn.Linear(hidden_dim, 1)
        self.support_readout = nn.Linear(hidden_dim, 1)
        self.start_transition_readout = nn.Linear(hidden_dim, 1)
        self.end_transition_readout = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        video_features: Tensor,
        query_features: Tensor,
        video_padding_mask: Optional[Tensor] = None,
        query_padding_mask: Optional[Tensor] = None,
    ) -> EventFieldOutput:
        """Compute four event-field logits.

        Padding masks follow the PyTorch convention: ``True`` denotes padding.
        Returned logits have shape ``[batch, video_length]``. Logits at padded
        video positions are set to zero so the public output remains finite.
        """
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
            query_tokens, query_padding_mask
        )

        conditioned_tokens, _ = self.cross_attention(
            query=video_tokens,
            key=query_tokens,
            value=query_tokens,
            key_padding_mask=safe_query_mask,
            need_weights=False,
        )
        conditioned_tokens = self.cross_attention_norm(
            video_tokens + conditioned_tokens
        )
        conditioned_tokens = conditioned_tokens + self._sinusoidal_encoding(
            video_length,
            conditioned_tokens.shape[-1],
            conditioned_tokens.device,
            conditioned_tokens.dtype,
        )

        conditioned_tokens, safe_video_mask = self._make_attention_safe(
            conditioned_tokens,
            video_padding_mask,
        )
        encoded_tokens = self.temporal_encoder(
            conditioned_tokens,
            src_key_padding_mask=safe_video_mask,
        )
        valid_positions = (~video_padding_mask).to(dtype=encoded_tokens.dtype)

        return EventFieldOutput(
            evidence_logits=self._readout(
                self.evidence_readout, encoded_tokens, valid_positions
            ),
            support_logits=self._readout(
                self.support_readout, encoded_tokens, valid_positions
            ),
            start_transition_logits=self._readout(
                self.start_transition_readout,
                encoded_tokens,
                valid_positions,
            ),
            end_transition_logits=self._readout(
                self.end_transition_readout,
                encoded_tokens,
                valid_positions,
            ),
        )

    @staticmethod
    def _validate_inputs(video_features: Tensor, query_features: Tensor) -> None:
        if video_features.ndim != 3 or query_features.ndim != 3:
            raise ValueError(
                "video_features and query_features must have shape [B, L, D]"
            )
        if video_features.shape[0] != query_features.shape[0]:
            raise ValueError(
                "video_features and query_features must have the same batch size"
            )
        if video_features.shape[1] == 0 or query_features.shape[1] == 0:
            raise ValueError("video and query sequences must be non-empty")

    @staticmethod
    def _normalize_mask(
        mask: Optional[Tensor],
        batch_size: int,
        sequence_length: int,
        device: torch.device,
    ) -> Tensor:
        if mask is None:
            return torch.zeros(
                batch_size, sequence_length, dtype=torch.bool, device=device
            )
        if mask.shape != (batch_size, sequence_length):
            raise ValueError(
                f"padding mask must have shape {(batch_size, sequence_length)}"
            )
        return mask.to(device=device, dtype=torch.bool)

    @staticmethod
    def _make_attention_safe(
        tokens: Tensor, padding_mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Avoid all-masked attention rows, which otherwise produce NaNs."""
        all_masked = padding_mask.all(dim=1)
        if not torch.any(all_masked):
            return tokens, padding_mask
        safe_tokens = tokens.clone()
        safe_mask = padding_mask.clone()
        safe_tokens[all_masked, 0] = 0
        safe_mask[all_masked, 0] = False
        return safe_tokens, safe_mask

    @staticmethod
    def _sinusoidal_encoding(
        length: int,
        hidden_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        positions = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(
            1
        )
        frequencies = torch.exp(
            torch.arange(0, hidden_dim, 2, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / hidden_dim),
        )
        encoding = torch.zeros(length, hidden_dim, device=device, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        encoding[:, 1::2] = torch.cos(
            positions * frequencies[: encoding[:, 1::2].shape[1]]
        )
        return encoding.to(dtype=dtype).unsqueeze(0)

    @staticmethod
    def _readout(readout: nn.Linear, tokens: Tensor, valid_positions: Tensor) -> Tensor:
        return readout(tokens).squeeze(-1) * valid_positions
