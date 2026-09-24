"""Checkpoint-free, budget-matched end-to-end C3 training entrypoint."""

from __future__ import annotations

from typing import Any, Mapping, Optional

import torch
from torch import nn
from torch.nn import functional as F

from training.config import RunnerConfig
from training.contracts import LossResult
from training.probability import masked_softmax
from backbone.model import STAGE32_KWARGS

from .model import C3ExperimentModel, C3ExperimentOutput
from .plugin import build_extension


class ScratchE2EModel(C3ExperimentModel):
    """C0 full path trained once from random initialization.

    All graph-active parameters remain trainable for the complete run. Loss
    weights change smoothly, but no module is frozen, reloaded, or reset.
    """

    def __init__(
        self, *args: Any, scratch_total_epochs: int = 420, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.scratch_total_epochs = int(scratch_total_epochs)
        if self.scratch_total_epochs < 1:
            raise ValueError("scratch_total_epochs must be positive")

    def _curriculum_weights(self, epoch: int) -> tuple[float, float, float, float]:
        progress = max(
            0.0, min(1.0, float(epoch) / max(1.0, 0.70 * self.scratch_total_epochs))
        )
        listwise_weight = 0.25 + 0.75 * progress
        anchor_weight = 0.20 - 0.15 * progress
        endpoint_weight = 0.30 - 0.15 * progress
        return listwise_weight, anchor_weight, endpoint_weight, progress

    def compute_loss(
        self,
        outputs: C3ExperimentOutput,
        batch: Any,
        teacher_outputs: Any,
        epoch: int,
    ) -> LossResult:
        if teacher_outputs is not None:
            raise AssertionError("scratch_e2e forbids checkpoint/KD prediction fusion")
        if outputs.extension_output is None:
            raise RuntimeError("C3 extension output missing before loss")
        extension_loss = self.extension.compute_loss(
            outputs.extension_output, outputs, batch, int(epoch)
        )
        if extension_loss is not None:
            raise RuntimeError("scratch_e2e C0 requires the identity extension")

        iou = self._iou(outputs, batch)
        valid = outputs.span_valid_mask
        target = masked_softmax(
            (iou / self.c5_temperature).flatten(1), valid.flatten(1)
        ).reshape_as(iou)
        listwise = (
            -(target * torch.log(outputs.span_probs.clamp_min(1.0e-8)))
            .flatten(1)
            .sum(1)
            .mean()
        )
        inherited_probs = masked_softmax(
            outputs.inherited_span_logits.flatten(1).float(), valid.flatten(1)
        ).reshape_as(outputs.inherited_span_logits)
        anchor_listwise = (
            -(target * torch.log(inherited_probs.clamp_min(1.0e-8)))
            .flatten(1)
            .sum(1)
            .mean()
        )

        pad = batch.inputs["video_padding_mask"].bool()
        count = (~pad).sum(1).clamp_min(1)
        center = (
            torch.arange(
                outputs.start_logits.shape[-1], device=iou.device, dtype=iou.dtype
            )[None, :]
            + 0.5
        ) / count.to(iou.dtype)[:, None]
        spans = batch.targets["gt_spans"].to(iou.dtype)
        mask = batch.targets["gt_span_mask"].bool()
        start = torch.minimum(spans[..., 0], spans[..., 1])
        end = torch.maximum(spans[..., 0], spans[..., 1])
        scale = count[:, None, None]
        start_target = (
            torch.exp(
                -0.5 * (((center[..., None] - start[:, None]) * scale / 1.5) ** 2)
            )
            .masked_fill(~mask[:, None], 0.0)
            .amax(-1)
            .masked_fill(pad, 0.0)
        )
        end_target = (
            torch.exp(-0.5 * (((center[..., None] - end[:, None]) * scale / 1.5) ** 2))
            .masked_fill(~mask[:, None], 0.0)
            .amax(-1)
            .masked_fill(pad, 0.0)
        )
        endpoint = 0.5 * (
            F.binary_cross_entropy_with_logits(
                outputs.start_logits[~pad], start_target[~pad]
            )
            + F.binary_cross_entropy_with_logits(
                outputs.end_logits[~pad], end_target[~pad]
            )
        )

        listwise_weight, anchor_weight, endpoint_weight, progress = (
            self._curriculum_weights(epoch)
        )
        loss = (
            listwise_weight * listwise
            + anchor_weight * anchor_listwise
            + endpoint_weight * endpoint
        )
        return LossResult(
            loss,
            {
                "listwise": listwise.detach(),
                "endpoint": endpoint.detach(),
                "anchor_listwise": anchor_listwise.detach(),
                "curriculum_progress": progress,
                "listwise_weight": listwise_weight,
                "anchor_weight": anchor_weight,
                "endpoint_weight": endpoint_weight,
                "epoch": float(epoch),
            },
        )

    def experiment_contract(self) -> Mapping[str, Any]:
        contract = dict(super().experiment_contract())
        contract.update(
            {
                "base": "scratch_e2e_random_initialization",
                "initialization": "scratch_e2e_random_initialization",
                "checkpoint_inheritance": False,
                "single_optimizer_history": True,
                "loss_curriculum": "continuous_0_to_70_percent_then_fixed",
            }
        )
        return contract


def initialize_scratch_e2e(
    model: ScratchE2EModel,
    residual_gate: float = 0.05,
) -> dict[str, Any]:
    if not 0.0 < residual_gate < 1.0:
        raise ValueError("residual_gate must be in (0, 1)")
    initialized = 0
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, nn.MultiheadAttention):
                nn.init.xavier_uniform_(module.in_proj_weight)
                initialized += 1
                if module.in_proj_bias is not None:
                    nn.init.zeros_(module.in_proj_bias)
                if module.bias_k is not None:
                    nn.init.zeros_(module.bias_k)
                if module.bias_v is not None:
                    nn.init.zeros_(module.bias_v)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                initialized += 1
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                initialized += 1
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                    initialized += 1
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.trunc_normal_(module.weight, std=0.02)
                initialized += 1

        for name, parameter in model.named_parameters():
            if name.endswith("role_tokens"):
                nn.init.trunc_normal_(parameter, std=0.02)
            elif name in {
                "shared_head.start.weight",
                "shared_head.end.weight",
                "shared_head.score.3.weight",
            }:
                nn.init.xavier_uniform_(parameter, gain=0.1)
            elif name.startswith("boundary_preserving.role_heads.") and name.endswith(
                ".3.weight"
            ):
                nn.init.normal_(parameter, mean=0.0, std=1.0e-3)
            elif name.startswith("boundary_preserving.role_heads.") and name.endswith(
                ".3.bias"
            ):
                nn.init.zeros_(parameter)

        for block in model.extra_blocks:
            block.gate.fill_(float(residual_gate))

    return {
        "checkpoint": None,
        "source_epoch": None,
        "source_mode": None,
        "source_selection": "scratch_e2e_random_initialization",
        "strict_base_load": False,
        "initialized_module_tensors": initialized,
        "initialization_scheme": {
            "linear_attention": "xavier_uniform",
            "convolution": "kaiming_normal",
            "layer_norm": "weight_one_bias_zero",
            "role_tokens": "truncated_normal_std_0.02",
            "span_output_gain": 0.1,
            "boundary_output_std": 1.0e-3,
            "extra_block_raw_gate": float(residual_gate),
            "extra_block_effective_gate": float(
                torch.tanh(torch.tensor(residual_gate))
            ),
        },
        "source_prediction_outputs_read": False,
        "prediction_fusion": False,
        "coordinate_movement": False,
        "extension_contract": dict(model.extension.contract()),
    }


def build_scratch_e2e_model(
    config: Optional[RunnerConfig],
    extension: Mapping[str, Any]
    | str = "backbone.conditioned.extensions:build_identity_extension",
    extension_lr: float = 1.0e-4,
    scratch_total_epochs: int = 420,
    scratch_residual_gate: float = 0.05,
    **kwargs: Any,
) -> ScratchE2EModel:
    del config
    model = ScratchE2EModel(
        extension=build_extension(extension),
        extension_lr=extension_lr,
        scratch_total_epochs=scratch_total_epochs,
        **STAGE32_KWARGS,
        **kwargs,
    )
    model.initialization_audit = initialize_scratch_e2e(
        model, residual_gate=scratch_residual_gate
    )
    return model


__all__ = ["ScratchE2EModel", "build_scratch_e2e_model", "initialize_scratch_e2e"]
