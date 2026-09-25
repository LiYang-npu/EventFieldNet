"""Round1 adapter around the reviewed reusable parent model."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from field_core.adapter import (
    DEFAULT_PARENT_FACTORY,
    LossResult,
    TriFieldBaseModel,
    _resolve_target,
)

from .losses import compute_round1_loss_terms
from .selector import Round1FieldScoreHeads


_DEFAULT_WEIGHTS = {
    "rank": 1.0,
    "evidence": 0.1,
    "support": 0.1,
    "transition": 0.1,
    "endpoint": 0.1,
}


# Runtime owns the LR-only R6 change. The model variant remains the exact
# control objective there, while this table makes all eight slots explicit.
ROUND1_VARIANTS: dict[str, dict[str, Any]] = {
    "r0_control": {
        "rank_margin": 0.0,
        "transition_target_mode": "matched",
        "support_mode": "linear",
        "quality_per_query": False,
        "quality_stratified": False,
        "optimizer_lr_scale": 1.0,
    },
    "r1_rank_margin": {
        "rank_margin": 0.1,
        "transition_target_mode": "matched",
        "support_mode": "linear",
        "quality_per_query": False,
        "quality_stratified": False,
        "optimizer_lr_scale": 1.0,
    },
    "r2_consistent_t": {
        "rank_margin": 0.0,
        "transition_target_mode": "independent_max",
        "support_mode": "linear",
        "quality_per_query": False,
        "quality_stratified": False,
        "optimizer_lr_scale": 1.0,
    },
    "r3_nonlinear_s": {
        "rank_margin": 0.0,
        "transition_target_mode": "matched",
        "support_mode": "nonlinear",
        "quality_per_query": False,
        "quality_stratified": False,
        "optimizer_lr_scale": 1.0,
    },
    "r4_query_mean": {
        "rank_margin": 0.0,
        "transition_target_mode": "matched",
        "support_mode": "linear",
        "quality_per_query": True,
        "quality_stratified": False,
        "optimizer_lr_scale": 1.0,
    },
    "r5_stratified_quality": {
        "rank_margin": 0.0,
        "transition_target_mode": "matched",
        "support_mode": "linear",
        "quality_per_query": True,
        "quality_stratified": True,
        "optimizer_lr_scale": 1.0,
    },
    "r6_half_lr": {
        "rank_margin": 0.0,
        "transition_target_mode": "matched",
        "support_mode": "linear",
        "quality_per_query": False,
        "quality_stratified": False,
        "optimizer_lr_scale": 0.5,
    },
    "r7_combined": {
        "rank_margin": 0.1,
        "transition_target_mode": "independent_max",
        "support_mode": "nonlinear",
        "quality_per_query": False,
        "quality_stratified": False,
        "optimizer_lr_scale": 1.0,
    },
}
ROUND1_VARIANTS.update(
    {
        "control": ROUND1_VARIANTS["r0_control"],
        "rank_margin": ROUND1_VARIANTS["r1_rank_margin"],
        "consistent_t": ROUND1_VARIANTS["r2_consistent_t"],
        "nonlinear_s": ROUND1_VARIANTS["r3_nonlinear_s"],
        "query_mean": ROUND1_VARIANTS["r4_query_mean"],
        "stratified_quality": ROUND1_VARIANTS["r5_stratified_quality"],
        "half_lr": ROUND1_VARIANTS["r6_half_lr"],
        "combined": ROUND1_VARIANTS["r7_combined"],
    }
)


def _normalise_variant(value: str | None) -> str:
    value = "r0_control" if value in (None, "") else str(value).lower()
    aliases = {
        "r0": "r0_control",
        "r1": "r1_rank_margin",
        "r2": "r2_consistent_t",
        "r3": "r3_nonlinear_s",
        "r4": "r4_query_mean",
        "r5": "r5_stratified_quality",
        "r6": "r6_half_lr",
        "r7": "r7_combined",
    }
    return aliases.get(value, value)


def _validated_weights(values: Mapping[str, float] | None) -> dict[str, float]:
    result = dict(_DEFAULT_WEIGHTS)
    if values is None:
        return result
    for name, value in values.items():
        if name not in result:
            raise ValueError(f"unknown round1 loss weight {name!r}")
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < 0.0:
            raise ValueError(f"loss weight {name!r} must be finite and non-negative")
        result[name] = numeric
    return result


@dataclass(frozen=True)
class Round1Options:
    variant: str = "r0_control"
    rank_margin: float = 0.0
    transition_target_mode: str = "matched"
    support_mode: str = "linear"
    quality_per_query: bool = False
    quality_stratified: bool = False
    optimizer_lr_scale: float = 1.0
    loss_weights: Mapping[str, float] = field(
        default_factory=lambda: dict(_DEFAULT_WEIGHTS)
    )

    @classmethod
    def from_values(
        cls,
        *,
        variant: str | None = None,
        rank_margin: float | None = None,
        transition_target_mode: str | None = None,
        support_mode: str | None = None,
        quality_per_query: bool | None = None,
        quality_stratified: bool | None = None,
        quality_reduction: str | None = None,
        optimizer_lr_scale: float | None = None,
        loss_weights: Mapping[str, float] | None = None,
    ) -> "Round1Options":
        key = _normalise_variant(variant)
        if key not in ROUND1_VARIANTS:
            raise ValueError(f"unknown round1 variant {variant!r}")
        values = dict(ROUND1_VARIANTS[key])
        if quality_reduction is not None:
            reduction = str(quality_reduction).lower()
            reductions = {
                "global": (False, False),
                "per_query": (True, False),
                "stratified": (False, True),
                "per_query_stratified": (True, True),
            }
            if reduction not in reductions:
                raise ValueError(
                    "quality_reduction must be global, per_query, stratified, "
                    "or per_query_stratified"
                )
            values["quality_per_query"], values["quality_stratified"] = reductions[
                reduction
            ]
        for name, value in (
            ("rank_margin", rank_margin),
            ("transition_target_mode", transition_target_mode),
            ("support_mode", support_mode),
            ("quality_per_query", quality_per_query),
            ("quality_stratified", quality_stratified),
            ("optimizer_lr_scale", optimizer_lr_scale),
        ):
            if value is not None:
                values[name] = value
        margin = float(values["rank_margin"])
        if margin < 0.0 or not math.isfinite(margin):
            raise ValueError("rank_margin must be finite and non-negative")
        target_mode = str(values["transition_target_mode"]).lower()
        if target_mode not in {"matched", "independent_max"}:
            raise ValueError(
                "transition_target_mode must be matched or independent_max"
            )
        support_mode_value = str(values["support_mode"]).lower()
        if support_mode_value not in {"linear", "nonlinear"}:
            raise ValueError("support_mode must be linear or nonlinear")
        lr_scale = float(values["optimizer_lr_scale"])
        if lr_scale <= 0.0 or not math.isfinite(lr_scale):
            raise ValueError("optimizer_lr_scale must be finite and positive")
        return cls(
            variant=key,
            rank_margin=margin,
            transition_target_mode=target_mode,
            support_mode=support_mode_value,
            quality_per_query=bool(values["quality_per_query"]),
            quality_stratified=bool(values["quality_stratified"]),
            optimizer_lr_scale=lr_scale,
            loss_weights=_validated_weights(loss_weights),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "rank_margin": self.rank_margin,
            "transition_target_mode": self.transition_target_mode,
            "support_mode": self.support_mode,
            "quality_per_query": self.quality_per_query,
            "quality_stratified": self.quality_stratified,
            "optimizer_lr_scale": self.optimizer_lr_scale,
            "loss_weights": dict(self.loss_weights),
        }


class TriFieldRound1Model(TriFieldBaseModel):
    """V1-compatible wrapper with the round1 objective switches."""

    def __init__(
        self,
        parent_model: nn.Module,
        *,
        options: Round1Options | None = None,
        selector_hidden_dim: int = 64,
        selector_lr: float = 1.0e-4,
        parent_factory: str = DEFAULT_PARENT_FACTORY,
    ) -> None:
        options = options or Round1Options.from_values()
        super().__init__(
            parent_model,
            selector_hidden_dim=selector_hidden_dim,
            selector_lr=selector_lr,
            parent_factory=parent_factory,
        )
        self.options = options
        # V1 super() has already constructed the canonical selector once.
        # Construct the round1 facade inside a forked RNG and copy that state:
        # R0 then has identical selector weights and identical global RNG
        # trajectory to V1, while nonlinear S gets isolated extra parameters.
        canonical_selector = self.selector
        with torch.random.fork_rng(devices=[]):
            round1_selector = Round1FieldScoreHeads(
                input_dim=512,
                hidden_dim=selector_hidden_dim,
                support_mode=options.support_mode,
            )
        round1_selector.load_state_dict(
            canonical_selector.state_dict(),
            strict=False,
        )
        if (
            options.support_mode == "nonlinear"
            and round1_selector.support_nonlinear is not None
        ):
            with torch.no_grad():
                round1_selector.support_nonlinear[-1].weight.zero_()
                round1_selector.support_nonlinear[-1].bias.zero_()
        self.selector = round1_selector
        self.loss_weights = dict(options.loss_weights)
        self.initialization_audit.update(
            {
                "new_base": "field_primitives",
                "round1_variant": options.variant,
                "round1_options": options.as_dict(),
                "loss_weights": dict(self.loss_weights),
            }
        )

    def compute_loss(
        self,
        outputs: Any,
        batch: Any,
        teacher_outputs: Any,
        epoch: int,
    ) -> LossResult:
        if teacher_outputs is not None:
            raise AssertionError(
                "field_primitives is scratch-only and forbids teacher/KD fusion"
            )
        wrong_evidence, pair = self._wrong_query_evidence(batch, outputs)
        terms = compute_round1_loss_terms(
            outputs,
            batch,
            wrong_evidence=wrong_evidence,
            evidence_pair_mask=pair,
            loss_weights=self.loss_weights,
            rank_margin=self.options.rank_margin,
            transition_target_mode=self.options.transition_target_mode,
            quality_per_query=self.options.quality_per_query,
            quality_stratified=self.options.quality_stratified,
        )
        metrics = dict(terms.metrics)
        metrics.update(
            {
                "epoch": float(epoch),
                "round1/optimizer_lr_scale": self.options.optimizer_lr_scale,
                "evidence/pair_rate": self._last_evidence_pairs["pair_rate"],
                "evidence/no_pair_rate": self._last_evidence_pairs["no_pair_rate"],
            }
        )
        return LossResult(terms.total, metrics)

    def diagnostics(self, outputs: Any, batch: Any) -> Mapping[str, Tensor | float]:
        result = dict(super().diagnostics(outputs, batch))
        result.update(
            {
                "round1/rank_margin": self.options.rank_margin,
                "round1/quality_per_query": float(self.options.quality_per_query),
                "round1/quality_stratified": float(self.options.quality_stratified),
                "round1/optimizer_lr_scale": self.options.optimizer_lr_scale,
            }
        )
        return result

    def set_epoch(self, epoch: int, training: bool) -> Mapping[str, Any]:
        setter = getattr(self.parent_model, "set_epoch", None)
        result = dict(setter(epoch, training)) if callable(setter) else {}
        result.update(
            {
                "base": "field_primitives",
                "variant": self.options.variant,
                "epoch": int(epoch),
                "training": bool(training),
            }
        )
        return result

    def experiment_contract(self) -> Mapping[str, Any]:
        result = dict(super().experiment_contract())
        result.update(
            {
                "base": "field_primitives",
                "round1_variant": self.options.variant,
                "round1_options": self.options.as_dict(),
                "final_score": "tanh(carrier_raw)+E+S+0.5*(Ts+Te)",
                "rank": (
                    "official all-candidate ordinal with positive margin "
                    f"{self.options.rank_margin:g}"
                ),
                "support": (
                    "V1 balanced coverage target with "
                    + (
                        "nonlinear post-aggregation residual"
                        if self.options.support_mode == "nonlinear"
                        else "linear readout"
                    )
                ),
                "transition": (
                    "independent max-G endpoint targets"
                    if self.options.transition_target_mode == "independent_max"
                    else "V1 matched-G endpoint targets"
                ),
                "quality_reduction": {
                    "per_query_mean": self.options.quality_per_query,
                    "near_far_equal": self.options.quality_stratified,
                    "near_iou_threshold": 0.3,
                },
                "loss_weights": dict(self.loss_weights),
                "optimizer_lr_scale": self.options.optimizer_lr_scale,
            }
        )
        return result


def build_trifield_round1_model(
    config: Any = None,
    *,
    parent_factory: str = DEFAULT_PARENT_FACTORY,
    selector_hidden_dim: int = 64,
    selector_lr: float = 1.0e-4,
    variant: str | None = None,
    rank_margin: float | None = None,
    transition_target_mode: str | None = None,
    support_mode: str | None = None,
    quality_per_query: bool | None = None,
    quality_stratified: bool | None = None,
    quality_reduction: str | None = None,
    optimizer_lr_scale: float | None = None,
    loss_weights: Mapping[str, float] | None = None,
    **kwargs: Any,
) -> TriFieldRound1Model:
    """Build the reusable parent plus the selected round1 model objective."""

    options = Round1Options.from_values(
        variant=variant,
        rank_margin=rank_margin,
        transition_target_mode=transition_target_mode,
        support_mode=support_mode,
        quality_per_query=quality_per_query,
        quality_stratified=quality_stratified,
        quality_reduction=quality_reduction,
        optimizer_lr_scale=optimizer_lr_scale,
        loss_weights=loss_weights,
    )
    factory = (
        _resolve_target(parent_factory)
        if isinstance(parent_factory, str)
        else parent_factory
    )
    parent_model = factory(config, **kwargs)
    return TriFieldRound1Model(
        parent_model,
        options=options,
        selector_hidden_dim=selector_hidden_dim,
        selector_lr=selector_lr,
        parent_factory=parent_factory,
    )


build_model = build_trifield_round1_model
Round1BaseModel = TriFieldRound1Model


__all__ = [
    "DEFAULT_PARENT_FACTORY",
    "ROUND1_VARIANTS",
    "Round1Options",
    "Round1BaseModel",
    "TriFieldRound1Model",
    "build_model",
    "build_trifield_round1_model",
]
