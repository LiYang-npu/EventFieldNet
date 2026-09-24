"""EventField-Net using the validated I5 interaction as its internal stem."""

from typing import Optional

import torch
from torch import Tensor

from .framework import EventFieldNet, EventFieldNetOutput
from .interaction import EventFieldInteraction


class ProvenInteractionStem(EventFieldInteraction):
    """Expose the latent sequence of the standalone I5 interaction encoder."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, variant="i5", **kwargs)

    def encode(
        self,
        video_features: Tensor,
        query_features: Tensor,
        video_padding_mask: Optional[Tensor] = None,
        query_padding_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
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
        conditioned, query = self._apply_bidirectional(
            video,
            query,
            conditioned,
            video_padding_mask,
            safe_query_mask,
        )
        conditioned = self._apply_roles(conditioned, query, safe_query_mask)
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
        return latent, query, video_padding_mask, safe_query_mask


class EventFieldNetProven(EventFieldNet):
    """Strong standalone EventField-Net with structured field specialization."""

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
        super().__init__(
            video_dim=video_dim,
            query_dim=query_dim,
            architecture=architecture,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            interaction_layers=interaction_layers,
            feedforward_dim=feedforward_dim,
            pair_dim=pair_dim,
            dropout=dropout,
            min_span_clips=min_span_clips,
        )
        del self.video_projection
        del self.query_projection
        del self.video_input_norm
        del self.query_input_norm
        del self.blocks
        del self.output_norm
        self.stem = ProvenInteractionStem(
            video_dim=video_dim,
            query_dim=query_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            temporal_layers=4,
            feedforward_dim=feedforward_dim,
            dropout=dropout,
        )

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
        if self.use_routing:
            evidence_tokens, support_tokens, start_tokens, end_tokens = (
                self._role_route(latent, query, safe_query_mask)
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
                torch.nn.functional.softplus(self.support_message_scale)
                * gate
                * self.evidence_to_support(evidence_message)
            )
            enter_change, exit_change = self._directional_changes(latent)
            start_tokens = start_tokens + self.start_change_projection(
                torch.cat((start_tokens, enter_change), dim=-1)
            )
            end_tokens = end_tokens + self.end_change_projection(
                torch.cat((end_tokens, exit_change), dim=-1)
            )
        else:
            evidence_tokens = support_tokens = start_tokens = end_tokens = latent

        valid = (~video_padding_mask).to(latent.dtype)
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
