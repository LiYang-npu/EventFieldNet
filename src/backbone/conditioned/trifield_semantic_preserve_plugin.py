"""Token-Evidence tri-field input with parameter-free semantic preservation.

The auxiliary target is the raw CLIP query-token agreement profile already
used by the Evidence field.  It is not a teacher prediction and it never moves
coordinates or changes validation decoding.  The loss only discourages the
end-to-end latent carrier from forgetting the input semantic profile.
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
    InputTriFieldState,
    TriFieldInputConditionedModel,
    _freeze_boundary_gate_only_branch,
    build_repository_data_with_saliency,
)


def _masked_profile(value: Tensor, padding: Tensor) -> Tensor:
    valid = (~padding).to(value.dtype)
    count = valid.sum(1).clamp_min(1.0)
    mean = (value * valid).sum(1) / count
    variance = ((value - mean[:, None]).square() * valid).sum(1) / count
    profile = (value - mean[:, None]) / variance.clamp_min(1.0e-6).sqrt()[:, None]
    return (3.0 * torch.tanh(profile / 3.0)).masked_fill(padding, 0.0)


def _profile_correlation(left: Tensor, right: Tensor, padding: Tensor) -> Tensor:
    valid = (~padding).to(left.dtype)
    count = valid.sum(1).clamp_min(1.0)
    left_mean = (left * valid).sum(1) / count
    right_mean = (right * valid).sum(1) / count
    left_center = (left - left_mean[:, None]) * valid
    right_center = (right - right_mean[:, None]) * valid
    covariance = (left_center * right_center).sum(1) / count
    left_std = (left_center.square().sum(1) / count).clamp_min(1.0e-6).sqrt()
    right_std = (right_center.square().sum(1) / count).clamp_min(1.0e-6).sqrt()
    return (covariance / (left_std * right_std)).mean()


class SemanticPreservedInputModel(TriFieldInputConditionedModel):
    def __init__(
        self,
        *args: Any,
        semantic_preservation_weight: float = 0.10,
        semantic_preservation_decay_start_epoch: int = 0,
        semantic_preservation_decay_epochs: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.semantic_preservation_weight = float(semantic_preservation_weight)
        if not 0.0 <= self.semantic_preservation_weight <= 0.25:
            raise ValueError("semantic_preservation_weight must be in [0, 0.25]")
        self.semantic_preservation_decay_start_epoch = int(
            semantic_preservation_decay_start_epoch
        )
        self.semantic_preservation_decay_epochs = int(
            semantic_preservation_decay_epochs
        )
        if self.semantic_preservation_decay_start_epoch < 0:
            raise ValueError(
                "semantic_preservation_decay_start_epoch must be non-negative"
            )
        if self.semantic_preservation_decay_epochs < 0:
            raise ValueError("semantic_preservation_decay_epochs must be non-negative")

    def _semantic_schedule(self, epoch: int) -> float:
        if self.semantic_preservation_decay_start_epoch == 0:
            return 1.0
        if int(epoch) <= self.semantic_preservation_decay_start_epoch:
            return 1.0
        if self.semantic_preservation_decay_epochs == 0:
            return 0.0
        return max(
            0.0,
            1.0
            - float(int(epoch) - self.semantic_preservation_decay_start_epoch)
            / float(self.semantic_preservation_decay_epochs),
        )

    def _semantic_preservation(self, outputs: Any, batch: Any) -> tuple[Tensor, Tensor]:
        state = outputs.extension_output.state if outputs.extension_output else None
        if not isinstance(state, InputTriFieldState):
            raise RuntimeError(
                "input tri-field state missing before semantic preservation"
            )
        padding = batch.inputs["video_padding_mask"].bool()
        raw_profile = _masked_profile(state.evidence_score.float().detach(), padding)
        latent = F.normalize(outputs.token_features.float(), dim=-1, eps=1.0e-6)
        query = F.normalize(outputs.query_features.float(), dim=-1, eps=1.0e-6)
        latent_score = (latent * query[:, None]).sum(-1).masked_fill(padding, 0.0)
        latent_profile = _masked_profile(latent_score, padding)
        loss = F.smooth_l1_loss(latent_profile[~padding], raw_profile[~padding])
        correlation = _profile_correlation(
            latent_score, state.evidence_score.float(), padding
        )
        return loss, correlation.detach()

    def compute_loss(
        self, outputs: Any, batch: Any, teacher_outputs: Any, epoch: int
    ) -> LossResult:
        base = super().compute_loss(outputs, batch, teacher_outputs, epoch)
        preservation, correlation = self._semantic_preservation(outputs, batch)
        schedule = self._semantic_schedule(epoch)
        effective_weight = self.semantic_preservation_weight * schedule
        metrics = dict(base.metrics)
        metrics.update(
            {
                "semantic_preserve/loss": preservation.detach(),
                "semantic_preserve/weight": self.semantic_preservation_weight,
                "semantic_preserve/schedule": schedule,
                "semantic_preserve/effective_weight": effective_weight,
                "semantic_preserve/raw_latent_profile_corr": correlation,
            }
        )
        return LossResult(
            base.loss + effective_weight * preservation,
            metrics,
        )

    def diagnostics(self, outputs: Any, batch: Any) -> Mapping[str, Tensor | float]:
        result = dict(super().diagnostics(outputs, batch))
        preservation, correlation = self._semantic_preservation(outputs, batch)
        result.update(
            {
                "semantic_preserve/loss": preservation.detach(),
                "semantic_preserve/raw_latent_profile_corr": correlation,
            }
        )
        return result

    def experiment_contract(self) -> Mapping[str, Any]:
        contract = dict(super().experiment_contract())
        contract.update(
            {
                "semantic_preservation": self.semantic_preservation_weight > 0.0,
                "semantic_preservation_target": (
                    "parameter_free_raw_query_token_agreement_profile"
                ),
                "semantic_preservation_weight": self.semantic_preservation_weight,
                "semantic_preservation_decay_start_epoch": (
                    self.semantic_preservation_decay_start_epoch
                ),
                "semantic_preservation_decay_epochs": (
                    self.semantic_preservation_decay_epochs
                ),
                "teacher_checkpoint": False,
                "teacher_prediction": False,
                "validation_score_fusion": False,
            }
        )
        return contract


def build_semantic_preserved_trifield_model(
    config: Optional[RunnerConfig],
    semantic_preservation_weight: float = 0.10,
    semantic_preservation_decay_start_epoch: int = 0,
    semantic_preservation_decay_epochs: int = 0,
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
) -> SemanticPreservedInputModel:
    del config
    if evidence_input_mode != "token_attention_product":
        raise ValueError("semantic preservation requires token-selective Evidence")
    inherited = dict(STAGE32_KWARGS)
    inherited.update(kwargs)
    inherited["use_boundary_gate"] = False
    model = SemanticPreservedInputModel(
        extension_lr=extension_lr,
        scratch_total_epochs=scratch_total_epochs,
        semantic_preservation_weight=semantic_preservation_weight,
        semantic_preservation_decay_start_epoch=(
            semantic_preservation_decay_start_epoch
        ),
        semantic_preservation_decay_epochs=semantic_preservation_decay_epochs,
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
    model.initialization_audit["semantic_preservation"] = {
        "checkpoint": None,
        "boundary_gate": False,
        "inactive_boundary_parameters": frozen,
        "target": "parameter_free_raw_query_token_agreement_profile",
        "weight": semantic_preservation_weight,
        "decay_start_epoch": semantic_preservation_decay_start_epoch,
        "decay_epochs": semantic_preservation_decay_epochs,
        "source_prediction_outputs_read": False,
        "shared_initialization_rng_isolation": True,
    }
    return model


__all__ = [
    "SemanticPreservedInputModel",
    "build_repository_data_with_saliency",
    "build_semantic_preserved_trifield_model",
]
