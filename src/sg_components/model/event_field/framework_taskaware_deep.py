"""Deep task-aware backbone with field streams inside temporal encoding."""

from typing import Optional

import torch
from torch import Tensor, nn

from .framework import EventFieldNetOutput
from .framework_proven import EventFieldNetProven
from .framework_taskaware import QueryFieldAdapter, SupportPyramid, TransitionDynamics


class EventFieldNetTaskAwareDeep(EventFieldNetProven):
    """Split E/S/T streams between shared temporal encoder layers."""

    VALID_ARCHITECTURES = {"deep_query", "deep_temporal", "deep_full"}

    def __init__(
        self,
        *args,
        architecture: str = "deep_full",
        num_heads: int = 8,
        dropout: float = 0.15,
        **kwargs,
    ) -> None:
        if architecture not in self.VALID_ARCHITECTURES:
            raise ValueError(f"unknown deep task-aware architecture: {architecture}")
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
        branch_layers = len(self.stem.temporal_encoder.layers) // 2
        self.query_adapters = nn.ModuleList(
            nn.ModuleList(
                QueryFieldAdapter(hidden_dim, num_heads, dropout) for _ in range(4)
            )
            for _ in range(branch_layers)
        )
        self.support_pyramid = nn.ModuleList(
            SupportPyramid(hidden_dim, dropout) for _ in range(branch_layers)
        )
        self.transition_dynamics = nn.ModuleList(
            TransitionDynamics(hidden_dim, dropout) for _ in range(branch_layers)
        )

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

    def _shared_input(
        self,
        video_features: Tensor,
        query_features: Tensor,
        video_padding_mask: Optional[Tensor],
        query_padding_mask: Optional[Tensor],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        self.stem._validate_inputs(video_features, query_features)
        batch, video_length, _ = video_features.shape
        query_length = query_features.shape[1]
        video_padding_mask = self.stem._normalize_mask(
            video_padding_mask, batch, video_length, video_features.device
        )
        query_padding_mask = self.stem._normalize_mask(
            query_padding_mask, batch, query_length, query_features.device
        )
        video = self.stem.video_projection(video_features)
        query = self.stem.query_projection(query_features)
        query, safe_query_mask = self.stem._make_attention_safe(
            query, query_padding_mask
        )
        update, _ = self.stem.cross_attention(
            query=video,
            key=query,
            value=query,
            key_padding_mask=safe_query_mask,
            need_weights=False,
        )
        conditioned = self.stem.cross_attention_norm(video + update)
        conditioned, query = self.stem._apply_bidirectional(
            video,
            query,
            conditioned,
            video_padding_mask,
            safe_query_mask,
        )
        conditioned = self.stem._apply_roles(conditioned, query, safe_query_mask)
        conditioned = self.stem._apply_multiscale(conditioned, query, safe_query_mask)
        conditioned = conditioned + self.stem._sinusoidal_encoding(
            video_length,
            conditioned.shape[-1],
            conditioned.device,
            conditioned.dtype,
        )
        conditioned, safe_video_mask = self.stem._make_attention_safe(
            conditioned, video_padding_mask
        )
        return conditioned, query, video_padding_mask, safe_query_mask, safe_video_mask

    def _latent_refine(self, encoded: Tensor) -> Tensor:
        update = self.stem.latent_projection(self.stem.latent_norm(encoded))
        update = self.stem.latent_activation(update)
        update = self.stem.latent_temporal(update.transpose(1, 2)).transpose(1, 2)
        return encoded + update

    def forward(
        self,
        video_features: Tensor,
        query_features: Tensor,
        video_padding_mask: Optional[Tensor] = None,
        query_padding_mask: Optional[Tensor] = None,
    ) -> EventFieldNetOutput:
        shared, query, video_padding_mask, safe_query_mask, safe_video_mask = (
            self._shared_input(
                video_features,
                query_features,
                video_padding_mask,
                query_padding_mask,
            )
        )
        layers = self.stem.temporal_encoder.layers
        split = len(layers) // 2
        for layer in layers[:split]:
            shared = layer(shared, src_key_padding_mask=safe_video_mask)

        fields = [shared, shared, shared, shared]
        roles = self._query_roles(query, safe_query_mask)
        for index, layer in enumerate(layers[split:]):
            fields = [
                layer(field, src_key_padding_mask=safe_video_mask) for field in fields
            ]
            if self.task_architecture in {"deep_query", "deep_full"}:
                fields = [
                    adapter(field, query, safe_query_mask, roles[:, role_index])
                    for role_index, (adapter, field) in enumerate(
                        zip(self.query_adapters[index], fields)
                    )
                ]
            if self.task_architecture in {"deep_temporal", "deep_full"}:
                fields[1] = self.support_pyramid[index](fields[1], roles[:, 1])
                fields[2], fields[3] = self.transition_dynamics[index](
                    fields[2], roles[:, 2], roles[:, 3]
                )

        if self.stem.temporal_encoder.norm is not None:
            fields = [self.stem.temporal_encoder.norm(field) for field in fields]

        fields = [self._latent_refine(field) for field in fields]
        valid = (~video_padding_mask).to(fields[0].dtype)
        evidence_logits = self.evidence_readout(fields[0]).squeeze(-1) * valid
        support_logits = self.support_readout(fields[1]).squeeze(-1) * valid
        start_logits = self.start_readout(fields[2]).squeeze(-1) * valid
        end_logits = self.end_readout(fields[3]).squeeze(-1) * valid
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
