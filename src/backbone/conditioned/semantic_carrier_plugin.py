"""SG-inspired supervised semantic carrier for scratch C0 causal tests."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from training.config import RunnerConfig
from training.contracts import DataBundle, LossResult
from training.probability import masked_softmax
from sg_components.dataset.collate import move_inputs_to_device

from backbone.model import STAGE32_KWARGS

from .extensions import build_identity_extension
from .plugin import build_repository_data as build_base_repository_data
from .scratch_plugin import ScratchE2EModel, initialize_scratch_e2e


MODES = ("aux_only", "span_mean_gate")


class QuerySemanticCarrier(nn.Module):
    """Dedicated query-conditioned clip relevance head, matching SG's cosine form."""

    def __init__(self, hidden_dim: int = 384, rank: int = 128) -> None:
        super().__init__()
        self.video = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, rank)
        )
        self.query = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, rank)
        )
        self.log_temperature = nn.Parameter(torch.tensor(3.0))
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, tokens: Tensor, query: Tensor, padding_mask: Tensor) -> Tensor:
        video = F.normalize(self.video(tokens).float(), p=2, dim=-1)
        text = F.normalize(self.query(query).float(), p=2, dim=-1)
        scale = self.log_temperature.clamp(0.0, 4.0).exp()
        logits = (video * text[:, None]).sum(-1) * scale + self.bias
        return logits.masked_fill(padding_mask, 0.0)


def _span_mean(values: Tensor) -> Tensor:
    _, length = values.shape
    prefix = F.pad(values.cumsum(1), (1, 0))
    index = torch.arange(length, device=values.device)
    starts = index[:, None]
    ends = index[None, :]
    width = ends - starts + 1
    return (prefix[:, ends + 1] - prefix[:, starts]) / width.clamp_min(1)[None]


class SemanticCarrierScratchModel(ScratchE2EModel):
    """C0 plus explicit clip relevance supervision and optional span use."""

    def __init__(
        self,
        *args: Any,
        semantic_mode: str = "aux_only",
        semantic_rank: int = 128,
        saliency_loss_weight: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if semantic_mode not in MODES:
            raise ValueError(f"semantic_mode must be one of {MODES}")
        self.semantic_mode = semantic_mode
        self.semantic_carrier = QuerySemanticCarrier(384, semantic_rank)
        self.semantic_span_gate = nn.Parameter(torch.zeros(()))
        self.saliency_loss_weight = float(saliency_loss_weight)
        if self.saliency_loss_weight <= 0.0:
            raise ValueError("saliency_loss_weight must be positive")

    def forward(
        self,
        inputs: Any,
        query_features: Optional[Tensor] = None,
        video_padding_mask: Optional[Tensor] = None,
        query_padding_mask: Optional[Tensor] = None,
    ):
        output = super().forward(
            inputs,
            query_features,
            video_padding_mask,
            query_padding_mask,
        )
        pad = (
            self._video_pad(inputs, output)
            if video_padding_mask is None
            else video_padding_mask.bool()
        )
        evidence = self.semantic_carrier(
            output.token_features, output.query_features, pad
        )
        values = {"evidence_logits": evidence.to(output.evidence_logits.dtype)}
        if self.semantic_mode == "span_mean_gate":
            clip_score = torch.sigmoid(evidence)
            global_mean = (clip_score * (~pad).to(clip_score.dtype)).sum(1) / (
                ~pad
            ).sum(1).clamp_min(1)
            semantic_span = _span_mean(clip_score) - global_mean[:, None, None]
            valid_values = semantic_span.masked_fill(~output.span_valid_mask, 0.0)
            count = output.span_valid_mask.flatten(1).sum(1).clamp_min(1)
            scale = (
                (valid_values.square().flatten(1).sum(1) / count)
                .sqrt()
                .clamp_min(1.0e-3)
            )
            semantic_span = semantic_span / scale[:, None, None]
            delta = 2.0 * torch.tanh(self.semantic_span_gate) * semantic_span
            logits = output.span_logits + delta.masked_fill(
                ~output.span_valid_mask, 0.0
            )
            probs = masked_softmax(
                logits.flatten(1).float(), output.span_valid_mask.flatten(1)
            ).reshape_as(logits)
            values.update(
                span_logits=logits, span_probs=probs.to(output.span_probs.dtype)
            )
        return replace(output, **values)

    @staticmethod
    def _saliency_loss(
        evidence: Tensor, batch: Any
    ) -> tuple[Tensor, Mapping[str, Tensor]]:
        pad = batch.inputs["video_padding_mask"].bool()
        valid = ~pad
        required = ("saliency_all_labels", "saliency_pos_labels", "saliency_neg_labels")
        missing = [name for name in required if name not in batch.targets]
        if missing:
            raise RuntimeError(
                f"semantic supervision missing from prepared batch: {missing}"
            )
        labels = batch.targets["saliency_all_labels"][:, : evidence.shape[1]].clamp(
            0, 1
        )
        bce = F.binary_cross_entropy_with_logits(evidence[valid], labels[valid])
        pos_index = batch.targets["saliency_pos_labels"].clamp_max(
            evidence.shape[1] - 1
        )
        neg_index = batch.targets["saliency_neg_labels"].clamp_max(
            evidence.shape[1] - 1
        )
        score = torch.sigmoid(evidence)
        positive = score.gather(1, pos_index)
        negative = score.gather(1, neg_index)
        margin = F.relu(0.15 + negative - positive).mean() * 2.0
        return bce + margin, {
            "saliency_bce": bce.detach(),
            "saliency_margin": margin.detach(),
            "saliency_pair_gap": (positive - negative).mean().detach(),
            "saliency_pair_accuracy": (positive > negative).float().mean().detach(),
        }

    def compute_loss(
        self, outputs, batch: Any, teacher_outputs: Any, epoch: int
    ) -> LossResult:
        base = super().compute_loss(outputs, batch, teacher_outputs, epoch)
        saliency, metrics = self._saliency_loss(outputs.evidence_logits.float(), batch)
        combined = base.loss + self.saliency_loss_weight * saliency
        result = dict(base.metrics)
        result.update(metrics)
        result["saliency_loss"] = saliency.detach()
        result["saliency_loss_weight"] = self.saliency_loss_weight
        result["semantic_span_gate"] = torch.tanh(self.semantic_span_gate).detach()
        return LossResult(combined, result)

    def diagnostics(self, outputs, batch: Any) -> Mapping[str, Tensor | float]:
        result = dict(super().diagnostics(outputs, batch))
        result["semantic_span_gate"] = torch.tanh(self.semantic_span_gate)
        result["semantic_temperature"] = self.semantic_carrier.log_temperature.exp()
        result["semantic_evidence_std"] = outputs.evidence_logits.float().std()
        return result

    def experiment_contract(self) -> Mapping[str, Any]:
        result = dict(super().experiment_contract())
        result.update(
            {
                "semantic_carrier": "dedicated_query_conditioned_cosine_head",
                "semantic_mode": self.semantic_mode,
                "saliency_supervision": "SG_local_BCE_plus_positive_negative_margin",
                "saliency_loss_weight": self.saliency_loss_weight,
                "span_integration": self.semantic_mode == "span_mean_gate",
                "semantic_span_coordinates": "unchanged_original_grid",
                "old_prediction_score_fusion": False,
            }
        )
        return result


def build_semantic_carrier_model(
    config: Optional[RunnerConfig],
    semantic_mode: str = "aux_only",
    semantic_rank: int = 128,
    saliency_loss_weight: float = 1.0,
    extension_lr: float = 1.0e-4,
    scratch_total_epochs: int = 260,
    scratch_residual_gate: float = 0.05,
    **kwargs: Any,
) -> SemanticCarrierScratchModel:
    del config
    model = SemanticCarrierScratchModel(
        extension=build_identity_extension(),
        extension_lr=extension_lr,
        scratch_total_epochs=scratch_total_epochs,
        semantic_mode=semantic_mode,
        semantic_rank=semantic_rank,
        saliency_loss_weight=saliency_loss_weight,
        **STAGE32_KWARGS,
        **kwargs,
    )
    model.initialization_audit = initialize_scratch_e2e(
        model, residual_gate=scratch_residual_gate
    )
    with torch.no_grad():
        model.semantic_carrier.log_temperature.fill_(3.0)
        model.semantic_carrier.bias.zero_()
        model.semantic_span_gate.zero_()
    model.initialization_audit["semantic_carrier"] = {
        "head": "cosine_projection",
        "temperature_log_init": 3.0,
        "bias_init": 0.0,
        "span_gate_init": 0.0,
    }
    return model


def build_repository_data_with_saliency(config: RunnerConfig, root: str) -> DataBundle:
    base = build_base_repository_data(config, root=root)

    def prepare(raw: Any, device: torch.device):
        batch = base.prepare_batch(raw, device)
        _, raw_batch = raw
        _, labels = move_inputs_to_device(raw_batch, device)
        if labels is None:
            raise RuntimeError("repository labels missing")
        targets = dict(batch.targets)
        for key in (
            "saliency_pos_labels",
            "saliency_neg_labels",
            "saliency_all_labels",
            "relevant_clips",
        ):
            if key in labels:
                targets[key] = labels[key]
        return replace(batch, targets=targets)

    metadata = dict(base.metadata)
    metadata["semantic_supervision_preserved"] = True
    return DataBundle(
        train_loader=base.train_loader,
        val_loader=base.val_loader,
        prepare_batch=prepare,
        metadata=metadata,
    )


__all__ = [
    "MODES",
    "QuerySemanticCarrier",
    "SemanticCarrierScratchModel",
    "build_repository_data_with_saliency",
    "build_semantic_carrier_model",
]
