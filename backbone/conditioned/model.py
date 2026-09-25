"""Trainable span backbone with optional additive span-score extensions."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Mapping, Optional

import torch
from torch import Tensor, nn

from training.contracts import LossResult, ParameterGroup
from training.probability import masked_softmax
from backbone.model import SpanBackboneModel, SpanBackboneOutput

from .contracts import (
    C3Context,
    C3ExtensionModule,
    C3ExtensionOutput,
    validate_extension,
)
from .extensions import IdentityExtension


TRAINABILITY_MODES = ("legacy_c3", "full_path")

# Parameters in inherited branches that C3 bypasses have grad=None on a real
# backward pass. They stay frozen because they are outside the C3 graph, not
# because full_path is a partial fine-tuning mode. Smoke tests guard this list.
FULL_PATH_INACTIVE_PREFIXES = (
    "stem.evidence_readout.",
    "stem.support_readout.",
    "stem.start_transition_readout.",
    "stem.end_transition_readout.",
    "evidence_modulation.",
    "support_role.",
    "support_pyramid.",
    "transition_adapter.",
    "neural_quality_field.",
    "context_contrast.",
    "matched_head.",
)
FULL_PATH_INACTIVE_EXACT = frozenset(("support_residual_logit",))


def is_full_path_inactive_parameter(name: str) -> bool:
    return name in FULL_PATH_INACTIVE_EXACT or name.startswith(
        FULL_PATH_INACTIVE_PREFIXES
    )


@dataclass
class ConditionedSpanOutput(SpanBackboneOutput):
    """Span-backbone outputs with the optional additive extension result."""
    extension_output: Optional[C3ExtensionOutput] = None


class ConditionedSpanModel(SpanBackboneModel):
    """Trainable span scorer with an optional additive span-grid extension.
    
    The extension operates on the existing temporal coordinates and can provide
    additional scores and losses without receiving ground-truth input in forward."""

    def __init__(
        self,
        *,
        extension: Optional[C3ExtensionModule] = None,
        extension_lr: float = 1.0e-4,
        trainability: str = "full_path",
        **kwargs: Any,
    ) -> None:
        kwargs.pop("mode", None)
        super().__init__(mode="C3_identity", **kwargs)
        if trainability not in TRAINABILITY_MODES:
            raise ValueError(
                f"trainability must be one of {TRAINABILITY_MODES}, got {trainability!r}"
            )
        self.trainability = trainability
        self._apply_trainability()
        self.extension = IdentityExtension() if extension is None else extension
        if not isinstance(self.extension, nn.Module):
            raise TypeError("C3 extension must be torch.nn.Module")
        validate_extension(self.extension)
        self.extension_lr = float(extension_lr)
        if self.extension_lr < 0.0:
            raise ValueError("extension_lr must be non-negative")

    def _apply_trainability(self) -> None:
        if self.trainability == "legacy_c3":
            return
        # The old C3 constructor froze several inherited modules and detached
        # quality descriptors. full_path keeps the same forward values while
        # restoring gradient flow through every loss-connected C3 component.
        self.detach_quality_features = False
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(not is_full_path_inactive_parameter(name))

    def trainability_audit(self) -> Mapping[str, Any]:
        trainable = [name for name, p in self.named_parameters() if p.requires_grad]
        frozen = [name for name, p in self.named_parameters() if not p.requires_grad]
        return {
            "mode": self.trainability,
            "end_to_end": self.trainability == "full_path",
            "detach_quality_features": bool(self.detach_quality_features),
            "trainable_tensors": len(trainable),
            "frozen_tensors": len(frozen),
            "frozen_names": frozen,
        }

    @staticmethod
    def _base_fields(output: SpanBackboneOutput) -> dict[str, Any]:
        return {
            field.name: getattr(output, field.name) for field in fields(SpanBackboneOutput)
        }

    @staticmethod
    def _video_pad(inputs: Any, output: SpanBackboneOutput) -> Tensor:
        if isinstance(inputs, Mapping) and inputs.get("video_padding_mask") is not None:
            return inputs["video_padding_mask"].bool()
        return torch.zeros(
            output.token_features.shape[:2],
            dtype=torch.bool,
            device=output.token_features.device,
        )

    def forward(
        self,
        inputs: Any,
        query_features: Optional[Tensor] = None,
        video_padding_mask: Optional[Tensor] = None,
        query_padding_mask: Optional[Tensor] = None,
    ) -> ConditionedSpanOutput:
        base = super().forward(
            inputs,
            query_features,
            video_padding_mask,
            query_padding_mask,
        )
        pad = (
            video_padding_mask.bool()
            if video_padding_mask is not None
            else self._video_pad(inputs, base)
        )
        context = C3Context(
            token_features=base.token_features,
            query_features=base.query_features,
            video_padding_mask=pad,
            span_valid_mask=base.span_valid_mask,
            base_span_logits=base.span_logits,
            base_span_probs=base.span_probs,
            start_logits=base.start_logits,
            end_logits=base.end_logits,
        )
        extension_output = self.extension(context)
        if not isinstance(extension_output, C3ExtensionOutput):
            raise TypeError("C3 extension forward must return C3ExtensionOutput")
        values = self._base_fields(base)
        if extension_output.logit_delta is not None:
            delta = extension_output.logit_delta
            if delta.shape != base.span_logits.shape:
                raise ValueError(
                    f"extension logit_delta shape {tuple(delta.shape)} != {tuple(base.span_logits.shape)}"
                )
            if not torch.isfinite(delta[base.span_valid_mask]).all():
                raise FloatingPointError("non-finite C3 extension logit delta")
            logits = base.span_logits + delta.masked_fill(~base.span_valid_mask, 0.0)
            probs = masked_softmax(
                logits.flatten(1).float(), base.span_valid_mask.flatten(1)
            ).reshape_as(logits)
            values["span_logits"] = logits
            values["span_probs"] = probs.to(base.span_probs.dtype)
        return ConditionedSpanOutput(**values, extension_output=extension_output)

    def compute_loss(
        self,
        outputs: ConditionedSpanOutput,
        batch: Any,
        teacher_outputs: Any,
        epoch: int,
    ) -> LossResult:
        base = super().compute_loss(outputs, batch, teacher_outputs, epoch)
        if outputs.extension_output is None:
            raise RuntimeError("C3 extension output missing before loss")
        extra = self.extension.compute_loss(
            outputs.extension_output, outputs, batch, int(epoch)
        )
        if extra is None:
            return base
        if not isinstance(extra, LossResult):
            raise TypeError("C3 extension compute_loss must return LossResult or None")
        if extra.loss.ndim != 0 or not torch.isfinite(extra.loss):
            raise FloatingPointError("non-finite or non-scalar C3 extension loss")
        metrics = dict(base.metrics)
        metrics.update(
            {f"extension/{name}": value for name, value in extra.metrics.items()}
        )
        metrics["extension/loss"] = extra.loss.detach()
        return LossResult(base.loss + extra.loss, metrics)

    def diagnostics(
        self, outputs: ConditionedSpanOutput, batch: Any
    ) -> Mapping[str, Tensor | float]:
        result = dict(super().diagnostics(outputs, batch))
        if outputs.extension_output is not None:
            result.update(
                {
                    f"extension/{name}": value
                    for name, value in outputs.extension_output.diagnostics.items()
                }
            )
        return result

    def set_epoch(self, epoch: int, training: bool) -> Mapping[str, Any]:
        result = dict(super().set_epoch(epoch, training))
        extension_phase = self.extension.set_epoch(int(epoch), bool(training)) or {}
        result.update(
            {f"extension/{name}": value for name, value in extension_phase.items()}
        )
        result["extension/api_version"] = self.extension.api_version
        return result

    def parameter_groups(self) -> list[ParameterGroup]:
        extension_ids = {id(parameter) for parameter in self.extension.parameters()}
        groups: list[ParameterGroup] = []
        for group in super().parameter_groups():
            params = [
                parameter
                for parameter in group.params
                if id(parameter) not in extension_ids
            ]
            if params:
                groups.append(
                    ParameterGroup(
                        name=group.name,
                        params=params,
                        lr=group.lr,
                        weight_decay=group.weight_decay,
                    )
                )
        extension_params = [
            parameter
            for parameter in self.extension.parameters()
            if parameter.requires_grad
        ]
        if extension_params:
            groups.append(
                ParameterGroup(
                    name="c3_extension",
                    params=extension_params,
                    lr=self.extension_lr,
                )
            )
        grouped = {id(parameter) for group in groups for parameter in group.params}
        missing = [
            name
            for name, parameter in self.named_parameters()
            if parameter.requires_grad and id(parameter) not in grouped
        ]
        if missing:
            raise RuntimeError(f"C3 extension parameter group omission: {missing}")
        return groups

    def experiment_contract(self) -> Mapping[str, Any]:
        audit = getattr(self, "initialization_audit", {})
        source = audit.get("source_selection") if isinstance(audit, Mapping) else None
        random_initialization = source == "random_initialization"
        stage14_initialization = source == "stage14_best_official_validation"
        return {
            "base": (
                "random_initialization"
                if random_initialization
                else (
                    "Stage14_local_best_val"
                    if stage14_initialization
                    else "Stage55_C3_identity_best_val"
                )
            ),
            "base_mode": "C3_identity",
            "initialization": source,
            "extension": dict(self.extension.contract()),
            "trainability": self.trainability,
            "end_to_end": self.trainability == "full_path",
            "detach_quality_features": bool(self.detach_quality_features),
            "original_coordinates": True,
            "source_prediction_outputs_read": False,
            "fixed_final_and_independent_best": True,
        }


__all__ = [
    'ConditionedSpanModel',
    'ConditionedSpanOutput',
    "TRAINABILITY_MODES",
    "FULL_PATH_INACTIVE_PREFIXES",
    "FULL_PATH_INACTIVE_EXACT",
    "is_full_path_inactive_parameter",
]
