"""Direct high-IoU complete-span ranking for Token-Evidence E/S/T.

This objective acts on the model's official final span logits.  It adds no
parameters, never moves coordinates, and is not a post-hoc reranker.  The
negative set contains currently high-scoring complete spans whose IoU is
already plausible (0.50 <= IoU < 0.95), matching the observed wider/narrower
confusions without imposing a fixed correction direction.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import torch
from torch import Tensor
from torch.nn import functional as F

from backbone.model import STAGE32_KWARGS
from training.config import RunnerConfig
from training.contracts import LossResult

from .scratch_plugin import initialize_scratch_e2e
from .trifield_input_conditioner_plugin import (
    _freeze_boundary_gate_only_branch,
    build_repository_data_with_saliency,
)
from .trifield_semantic_preserve_plugin import SemanticPreservedInputModel


class FinalHardRankInputModel(SemanticPreservedInputModel):
    """Train the official final logits against score-hard high-IoU spans."""

    def __init__(
        self,
        *args: Any,
        final_hard_rank_weight: float = 0.10,
        final_hard_rank_margin: float = 0.20,
        final_hard_rank_topk: int = 8,
        final_hard_rank_start_epoch: int = 0,
        final_hard_rank_ramp_epochs: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.final_hard_rank_weight = float(final_hard_rank_weight)
        self.final_hard_rank_margin = float(final_hard_rank_margin)
        self.final_hard_rank_topk = int(final_hard_rank_topk)
        self.final_hard_rank_start_epoch = int(final_hard_rank_start_epoch)
        self.final_hard_rank_ramp_epochs = int(final_hard_rank_ramp_epochs)
        if not 0.0 < self.final_hard_rank_weight <= 0.25:
            raise ValueError("final_hard_rank_weight must be in (0, 0.25]")
        if not 0.0 < self.final_hard_rank_margin <= 1.0:
            raise ValueError("final_hard_rank_margin must be in (0, 1]")
        if not 1 <= self.final_hard_rank_topk <= 32:
            raise ValueError("final_hard_rank_topk must be in [1, 32]")
        if self.final_hard_rank_start_epoch < 0:
            raise ValueError("final_hard_rank_start_epoch must be non-negative")
        if self.final_hard_rank_ramp_epochs < 0:
            raise ValueError("final_hard_rank_ramp_epochs must be non-negative")

    def _hard_rank_schedule(self, epoch: int) -> float:
        if self.final_hard_rank_start_epoch == 0:
            return 1.0
        if int(epoch) <= self.final_hard_rank_start_epoch:
            return 0.0
        if self.final_hard_rank_ramp_epochs == 0:
            return 1.0
        return min(
            1.0,
            float(int(epoch) - self.final_hard_rank_start_epoch)
            / float(self.final_hard_rank_ramp_epochs),
        )

    def _final_hard_rank(
        self, score: Tensor, iou: Tensor, valid: Tensor
    ) -> tuple[Tensor, Mapping[str, Tensor]]:
        flat_score = score.float().flatten(1)
        flat_iou = iou.float().flatten(1)
        flat_valid = valid.bool().flatten(1)
        positive_index = flat_iou.masked_fill(~flat_valid, -1.0).argmax(1, keepdim=True)
        positive_score = flat_score.gather(1, positive_index).squeeze(1)
        negative = flat_valid & flat_iou.ge(0.50) & flat_iou.lt(0.95)
        negative = negative.scatter(1, positive_index, False)
        count = negative.sum(1)
        topk = min(self.final_hard_rank_topk, flat_score.shape[1])
        negative_score, negative_index = flat_score.masked_fill(~negative, -1.0e4).topk(
            topk, dim=1
        )
        negative_iou = flat_iou.gather(1, negative_index)
        selected = torch.arange(topk, device=score.device)[None, :] < count[:, None]
        zero = score.float().sum() * 0.0
        if not bool(selected.any()):
            metrics = {
                "loss": zero.detach(),
                "pair_margin": zero.detach(),
                "hardest_margin": zero.detach(),
                "positive_rate": zero.detach(),
                "rows_with_negatives": zero.detach(),
                "selected_pairs_per_row": zero.detach(),
            }
            return zero, metrics

        pair_margin = (positive_score[:, None] - negative_score)[selected]
        selected_iou = negative_iou[selected]
        # Give the most confusable near-correct spans more weight while keeping
        # every selected high-IoU negative active in the objective.
        pair_weight = 0.5 + ((selected_iou - 0.50) / 0.45).clamp(0.0, 1.0)
        loss = (
            F.softplus(self.final_hard_rank_margin - pair_margin) * pair_weight
        ).sum() / pair_weight.sum().clamp_min(1.0e-6)
        has = count.gt(0)
        hardest_margin = positive_score[has] - negative_score[has, 0]
        metrics = {
            "loss": loss.detach(),
            "pair_margin": pair_margin.mean().detach(),
            "hardest_margin": hardest_margin.mean().detach(),
            "positive_rate": pair_margin.gt(0.0).float().mean().detach(),
            "rows_with_negatives": has.float().mean().detach(),
            "selected_pairs_per_row": selected.sum(1).float().mean().detach(),
        }
        return loss, metrics

    def compute_loss(
        self, outputs: Any, batch: Any, teacher_outputs: Any, epoch: int
    ) -> LossResult:
        base = super().compute_loss(outputs, batch, teacher_outputs, epoch)
        rank_loss, rank_metrics = self._final_hard_rank(
            outputs.span_logits,
            self._iou(outputs, batch),
            outputs.span_valid_mask,
        )
        metrics = dict(base.metrics)
        schedule = self._hard_rank_schedule(epoch)
        effective_weight = self.final_hard_rank_weight * schedule
        metrics.update(
            {f"final_hardrank/{name}": value for name, value in rank_metrics.items()}
        )
        metrics.update(
            {
                "final_hardrank/weight": self.final_hard_rank_weight,
                "final_hardrank/schedule": schedule,
                "final_hardrank/effective_weight": effective_weight,
                "final_hardrank/required_margin": self.final_hard_rank_margin,
                "final_hardrank/topk": float(self.final_hard_rank_topk),
            }
        )
        return LossResult(
            base.loss + effective_weight * rank_loss,
            metrics,
        )

    def diagnostics(self, outputs: Any, batch: Any) -> Mapping[str, Tensor | float]:
        result = dict(super().diagnostics(outputs, batch))
        _, metrics = self._final_hard_rank(
            outputs.span_logits,
            self._iou(outputs, batch),
            outputs.span_valid_mask,
        )
        result.update(
            {f"final_hardrank/{name}": value for name, value in metrics.items()}
        )
        return result

    def experiment_contract(self) -> Mapping[str, Any]:
        contract = dict(super().experiment_contract())
        contract.update(
            {
                "direct_final_logit_hard_ranking": True,
                "hard_negative_iou_range": [0.50, 0.95],
                "hard_negative_selection": "topk_by_current_official_final_logit",
                "final_hard_rank_weight": self.final_hard_rank_weight,
                "final_hard_rank_margin": self.final_hard_rank_margin,
                "final_hard_rank_topk": self.final_hard_rank_topk,
                "final_hard_rank_start_epoch": self.final_hard_rank_start_epoch,
                "final_hard_rank_ramp_epochs": self.final_hard_rank_ramp_epochs,
                "new_ranking_parameters": 0,
                "validation_score_fusion": False,
                "posthoc_reranking": False,
                "coordinate_movement": False,
            }
        )
        return contract


def build_final_hardrank_trifield_model(
    config: Optional[RunnerConfig],
    semantic_preservation_weight: float = 0.0,
    semantic_preservation_decay_start_epoch: int = 0,
    semantic_preservation_decay_epochs: int = 0,
    final_hard_rank_weight: float = 0.10,
    final_hard_rank_margin: float = 0.20,
    final_hard_rank_topk: int = 8,
    final_hard_rank_start_epoch: int = 0,
    final_hard_rank_ramp_epochs: int = 0,
    input_field_rank: int = 64,
    transition_barrier: bool = True,
    initial_field_gate: float = 0.05,
    conditioner_output_gain: float = 0.10,
    max_update_ratio: float = 0.05,
    support_hops: int = 1,
    transition_input_mode: str = "feature_delta",
    evidence_input_mode: str = "token_attention_product",
    support_input_mode: str = "diffusion_residual",
    extension_lr: float = 1.0e-4,
    scratch_total_epochs: int = 420,
    scratch_residual_gate: float = 0.05,
    **kwargs: Any,
) -> FinalHardRankInputModel:
    del config
    if evidence_input_mode != "token_attention_product":
        raise ValueError("final hard ranking requires token-selective Evidence")
    inherited = dict(STAGE32_KWARGS)
    inherited.update(kwargs)
    inherited["use_boundary_gate"] = False
    model = FinalHardRankInputModel(
        extension_lr=extension_lr,
        scratch_total_epochs=scratch_total_epochs,
        semantic_preservation_weight=semantic_preservation_weight,
        semantic_preservation_decay_start_epoch=(
            semantic_preservation_decay_start_epoch
        ),
        semantic_preservation_decay_epochs=semantic_preservation_decay_epochs,
        final_hard_rank_weight=final_hard_rank_weight,
        final_hard_rank_margin=final_hard_rank_margin,
        final_hard_rank_topk=final_hard_rank_topk,
        final_hard_rank_start_epoch=final_hard_rank_start_epoch,
        final_hard_rank_ramp_epochs=final_hard_rank_ramp_epochs,
        input_field_rank=input_field_rank,
        transition_barrier=transition_barrier,
        initial_field_gate=initial_field_gate,
        conditioner_output_gain=conditioner_output_gain,
        max_update_ratio=max_update_ratio,
        support_hops=support_hops,
        transition_input_mode=transition_input_mode,
        evidence_input_mode=evidence_input_mode,
        support_input_mode=support_input_mode,
        **inherited,
    )
    frozen = _freeze_boundary_gate_only_branch(model)
    conditioner = model._modules.pop("input_conditioner")
    try:
        model.initialization_audit = initialize_scratch_e2e(
            model, residual_gate=scratch_residual_gate
        )
    finally:
        model.add_module("input_conditioner", conditioner)
    model.initialization_audit["final_hardrank"] = {
        "checkpoint": None,
        "boundary_gate": False,
        "inactive_boundary_parameters": frozen,
        "semantic_preservation_weight": semantic_preservation_weight,
        "semantic_preservation_decay_start_epoch": (
            semantic_preservation_decay_start_epoch
        ),
        "semantic_preservation_decay_epochs": semantic_preservation_decay_epochs,
        "rank_weight": final_hard_rank_weight,
        "required_margin": final_hard_rank_margin,
        "topk": final_hard_rank_topk,
        "start_epoch": final_hard_rank_start_epoch,
        "ramp_epochs": final_hard_rank_ramp_epochs,
        "new_trainable_parameters": 0,
        "source_prediction_outputs_read": False,
        "shared_initialization_rng_isolation": True,
    }
    return model


__all__ = [
    "FinalHardRankInputModel",
    "build_final_hardrank_trifield_model",
    "build_repository_data_with_saliency",
]
