"""Three-field model adapter with query-conditioned scoring and losses."""

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

from .losses import compute_loss_terms
from .selector import ThreeFieldScoreHeads


_DEFAULT_WEIGHTS = {
    "rank": 0.5,
    "evidence": 0.1,
    "support": 0.1,
    "transition": 0.1,
    "endpoint": 0.1,
}

ROUND31_FIXED_CONTROL_BITS = 0b111
ROUND31_RANK_MARGIN = 0.1

# Round31 factor matrix.  R28/U4 field construction is fixed in every cell;
# only the three inexpensive rank/support factors vary.  The legacy fields
# remain serialized for parent/runtime compatibility and are pinned below.
ROUND31_READOUTS = ("edge_mean",)

# Four candidate-S readouts crossed with the two retained rank backgrounds.
# X0..X3 are the R29 W1/K control family; X4..X7 are the R29 W5/K+.5P
# family.  Wide S, RMS/joint-residual S, and all legacy U4 controls are fixed.
ROUND31_VARIANTS = {
    f"y{i}": {
        "min_span_clips": 1,
        "rank_mode": "gt_balanced_kl",
        "quality_stratified": True,
        "rank_logit_scale": 2.0,
        "rank_target_threshold": 0.7,
        "rank_target_exponent": 4.0,
        "rank_weight": 0.5,
        "carrier_mode": "tanh",
        "linear_calibration": False,
        "rank_form": "dense",
        "rank_competition": "global",
        "pairwise_interactions": False,
        "transition_context_mode": "none",
        "transition_target_mode": "independent_max",
        "projected_cosine_attention": False,
        "local_top_quarter_e": False,
        "difference_before_interaction_t": False,
        "a_local_evidence": True,
        "e_readout_scale": "fixed_init",
        "query_relative_e": False,
        "e_span_lme": False,
        "independent_s_projection": True,
        "b_edge_support": True,
        "c_latent_composition": False,
        "support_deployment": "in_score",
        "support_input": "rms",
        "support_pair_mode": "joint_residual",
        "support_wide_pair": True,
        "support_length_center": False,
        "rank_kl_half": False,
        "rank_candidate_pair": bool(i >= 4),
        "support_readout": "edge_mean",
        "support_candidate_chunk_size": 128,
        "rank_stop_s": bool(i & 1),
        "aux_detach_input": bool(i & 2),
    }
    for i in range(8)
}

# round32 extension: y8 is the same fixed cell as y4 (rank_stop_s=False,
# aux_detach_input=False, rank_candidate_pair=True) with the three trifield
# losses (evidence/support/transition) given more weight relative to rank,
# to test whether the field losses carry more of the optimization signal.
# This is a new, explicitly declared, separately named configuration -- it
# does not reinterpret or override what "y4" means anywhere in this table.
ROUND31_VARIANTS["y8"] = dict(
    ROUND31_VARIANTS["y4"],
    rank_weight=0.4,
    evidence_weight=0.15,
    support_weight=0.15,
    transition_weight=0.15,
    endpoint_weight=0.1,
)

# round32 extension: y9 is also y4's cell, but only boosts transition -- the
# one field whose Round31 loss has no "native" override path (it only ever
# goes through the counterfactual replacement pairs; see
# agent_workspace/claude/work/trifield_probe_alignment.md). This isolates
# whether transition specifically is under-weighted, as a contrast to y8's
# "boost all three equally" hypothesis.
ROUND31_VARIANTS["y9"] = dict(
    ROUND31_VARIANTS["y4"],
    rank_weight=0.4,
    evidence_weight=0.1,
    support_weight=0.1,
    transition_weight=0.2,
    endpoint_weight=0.1,
)

# round32 extension: y10-y16 structural ablation group (MR57 push)
# Each is y4's exact cell (rank_stop_s=False, aux_detach_input=False,
# rank_candidate_pair=True) with exactly one dormant-but-implemented
# mechanism flag flipped on. y10-y14 flags are NOT wired to vary across
# y0..y9; all are hard-fixed False in ROUND31_VARIANTS and validated as
# fixed in ThreeFieldOptions.from_values, so (like y8/y9) a new named
# variant is required rather than editing y4 in place. y15/y16 introduce
# two brand-new fixed fields (interaction_depth, interaction_feedforward)
# not present in y0..y9 at all; every existing variant gets the neutral
# defaults (depth=1, feedforward=False) added below so from_values's
# fixed-name check has a value to compare against for y0..y9 too, with
# zero behavior change for them (see model/interaction.py).
ROUND31_VARIANTS["y10"] = dict(ROUND31_VARIANTS["y4"], linear_calibration=True)
ROUND31_VARIANTS["y11"] = dict(ROUND31_VARIANTS["y4"], pairwise_interactions=True)
ROUND31_VARIANTS["y12"] = dict(ROUND31_VARIANTS["y4"], e_span_lme=True)
ROUND31_VARIANTS["y13"] = dict(ROUND31_VARIANTS["y4"], projected_cosine_attention=True)
ROUND31_VARIANTS["y14"] = dict(
    ROUND31_VARIANTS["y4"], difference_before_interaction_t=True
)
for _key in list(ROUND31_VARIANTS):
    ROUND31_VARIANTS[_key].setdefault("interaction_depth", 1)
    ROUND31_VARIANTS[_key].setdefault("interaction_feedforward", False)
    ROUND31_VARIANTS[_key].setdefault("length_calibration", False)
    ROUND31_VARIANTS[_key].setdefault("interaction_dropout", 0.0)
    ROUND31_VARIANTS[_key].setdefault("edge_head_dropout", 0.0)
ROUND31_VARIANTS["y15"] = dict(ROUND31_VARIANTS["y4"], interaction_depth=2)
ROUND31_VARIANTS["y16"] = dict(ROUND31_VARIANTS["y4"], interaction_feedforward=True)
# round36 extension: y17 is y4's cell + length_calibration=True (H3,
# duration-conditioned score calibration; see model/selector.py::
# length_residual). Requires the round36 apply_residual wiring fix.
ROUND31_VARIANTS["y17"] = dict(ROUND31_VARIANTS["y4"], length_calibration=True)
# round39 extension: y18-y27 are y4's cell + regularization dials
# (interaction_dropout / edge_head_dropout) and, for y25-y27, capacity
# increases (interaction_depth / interaction_feedforward) PAIRED with
# dropout this time -- round35's y15/y16 tested capacity alone with zero
# dropout anywhere in the extension and saw no improvement; round39
# tests whether capacity increases need regularization to help.
ROUND31_VARIANTS["y18"] = dict(ROUND31_VARIANTS["y4"], interaction_dropout=0.1)
ROUND31_VARIANTS["y19"] = dict(ROUND31_VARIANTS["y4"], interaction_dropout=0.2)
ROUND31_VARIANTS["y20"] = dict(ROUND31_VARIANTS["y4"], interaction_dropout=0.3)
ROUND31_VARIANTS["y21"] = dict(ROUND31_VARIANTS["y4"], interaction_dropout=0.5)
ROUND31_VARIANTS["y22"] = dict(ROUND31_VARIANTS["y4"], edge_head_dropout=0.1)
ROUND31_VARIANTS["y23"] = dict(ROUND31_VARIANTS["y4"], edge_head_dropout=0.3)
ROUND31_VARIANTS["y24"] = dict(
    ROUND31_VARIANTS["y4"], interaction_dropout=0.2, edge_head_dropout=0.2
)
ROUND31_VARIANTS["y25"] = dict(
    ROUND31_VARIANTS["y4"], interaction_depth=2, interaction_dropout=0.2
)
ROUND31_VARIANTS["y26"] = dict(
    ROUND31_VARIANTS["y4"], interaction_feedforward=True, interaction_dropout=0.2
)
ROUND31_VARIANTS["y27"] = dict(
    ROUND31_VARIANTS["y4"],
    interaction_depth=2,
    interaction_feedforward=True,
    interaction_dropout=0.2,
)
ROUND31_VARIANTS["y31"] = dict(
    ROUND31_VARIANTS["y4"], interaction_depth=2, interaction_dropout=0.5
)
ROUND31_VARIANTS["y32"] = dict(
    ROUND31_VARIANTS["y4"], interaction_feedforward=True, interaction_dropout=0.5
)
ROUND31_VARIANTS["y33"] = dict(
    ROUND31_VARIANTS["y4"],
    interaction_depth=2,
    interaction_feedforward=True,
    interaction_dropout=0.5,
)
_VARIANT_ALIASES: dict[str, str] = {}


def _normalise_variant(value: str | None) -> str:
    key = "y0" if value in (None, "") else str(value).lower()
    return _VARIANT_ALIASES.get(key, key)


def _validated_weights(values: Mapping[str, float] | None) -> dict[str, float]:
    result = dict(_DEFAULT_WEIGHTS)
    if values is None:
        return result
    for name, value in values.items():
        if name not in result:
            raise ValueError(f"unknown round31 loss weight {name!r}")
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < 0.0:
            raise ValueError(f"loss weight {name!r} must be finite and non-negative")
        result[name] = numeric
    return result


@dataclass(frozen=True)
class ThreeFieldOptions:
    """Validated configuration for candidate scoring and field supervision."""
    transition_context_mode: str = "none"
    linear_calibration: bool = False
    pairwise_interactions: bool = False
    carrier_mode: str = "tanh"
    variant: str = "y0"
    min_span_clips: int = 1
    rank_form: str = "dense"
    rank_competition: str = "global"
    rank_mode: str = "gt_balanced_kl"
    rank_kl_half: bool = False
    rank_stop_s: bool = False
    aux_detach_input: bool = False
    rank_margin: float = ROUND31_RANK_MARGIN
    quality_stratified: bool = True
    rank_logit_scale: float = 2.0
    rank_target_threshold: float = 0.7
    rank_target_exponent: float = 4.0
    transition_target_mode: str = "independent_max"
    support_mode: str = "nonlinear"
    projected_cosine_attention: bool = False
    local_top_quarter_e: bool = False
    difference_before_interaction_t: bool = False
    a_local_evidence: bool = True
    e_readout_scale: str = "fixed_init"
    query_relative_e: bool = False
    e_span_lme: bool = False
    independent_s_projection: bool = False
    b_edge_support: bool = True
    c_latent_composition: bool = False
    support_deployment: str = "in_score"
    support_input: str = "rms"
    support_pair_mode: str = "joint_residual"
    support_readout: str = "edge_mean"
    support_candidate_chunk_size: int = 128
    support_wide_pair: bool = True
    support_length_center: bool = False
    rank_candidate_pair: bool = False
    interaction_depth: int = 1
    interaction_feedforward: bool = False
    length_calibration: bool = False
    interaction_dropout: float = 0.0
    edge_head_dropout: float = 0.0
    fixed_control_bits: int = ROUND31_FIXED_CONTROL_BITS
    loss_weights: Mapping[str, float] = field(
        default_factory=lambda: dict(_DEFAULT_WEIGHTS)
    )

    @property
    def mechanism_bits(self) -> int:
        # Existing local counterfactual mechanism mask remains unchanged.
        return 1 if self.a_local_evidence else 0

    @property
    def support_deployment_zero(self) -> bool:
        return self.support_deployment == "zero"

    @property
    def support_input_rms(self) -> bool:
        return self.support_input == "rms"

    @property
    def support_pair_joint(self) -> bool:
        return self.support_pair_mode == "joint_residual"

    @property
    def factor_bits(self) -> int:
        """Serialized Y route bits: A=rank-stop, B=aux-detach, C=P."""
        return (
            (1 if self.rank_stop_s else 0)
            | (2 if self.aux_detach_input else 0)
            | (4 if self.rank_candidate_pair else 0)
        )

    @property
    def route_bits(self) -> int:
        return self.factor_bits

    @property
    def alpha(self) -> float:
        """Effective coefficient of the original KL rank objective."""
        return 0.5 if self.rank_kl_half else 1.0

    @property
    def beta(self) -> float:
        """Effective coefficient of the candidate-pair rank objective."""
        return 0.5 if self.rank_candidate_pair else 0.0

    @property
    def rank_alpha(self) -> float:
        return self.alpha

    @property
    def rank_beta(self) -> float:
        return self.beta

    @property
    def region_gain(self):
        return 1.0

    @property
    def transition_gain(self):
        return 1.0

    @property
    def support_context(self):
        return True

    @property
    def support_dispersion(self):
        return True

    @property
    def evidence_pool(self):
        return False

    @property
    def evidence_context(self):
        return False

    @property
    def evidence_reduction(self):
        return "candidate_mean"

    @property
    def evidence_supervision(self):
        return "wrong_query"

    @classmethod
    def from_values(
        cls,
        *,
        carrier_mode: str | None = None,
        linear_calibration: bool | None = None,
        pairwise_interactions: bool | None = None,
        variant: str | None = None,
        transition_target_mode: str | None = None,
        transition_context_mode: str | None = None,
        support_mode: str = "nonlinear",
        min_span_clips: int | None = None,
        rank_mode: str | None = None,
        rank_kl_half: bool | None = None,
        rank_stop_s: bool | None = None,
        aux_detach_input: bool | None = None,
        evidence_reduction: str | None = None,
        evidence_supervision: str | None = None,
        rank_form: str | None = None,
        rank_competition: str | None = None,
        rank_logit_scale: float | None = None,
        rank_target_threshold: float | None = None,
        rank_target_exponent: float | None = None,
        rank_margin: float | None = None,
        quality_stratified: bool | None = None,
        projected_cosine_attention: bool | None = None,
        local_top_quarter_e: bool | None = None,
        difference_before_interaction_t: bool | None = None,
        a_local_evidence: bool | None = None,
        e_readout_scale: str | None = None,
        query_relative_e: bool | None = None,
        e_span_lme: bool | None = None,
        independent_s_projection: bool | None = None,
        b_edge_support: bool | None = None,
        c_latent_composition: bool | None = None,
        support_deployment: str | None = None,
        support_input: str | None = None,
        support_pair_mode: str | None = None,
        support_readout: str | None = None,
        support_candidate_chunk_size: int | None = None,
        support_wide_pair: bool | None = None,
        support_length_center: bool | None = None,
        rank_candidate_pair: bool | None = None,
        interaction_depth: int | None = None,
        interaction_feedforward: bool | None = None,
        length_calibration: bool | None = None,
        interaction_dropout: float | None = None,
        edge_head_dropout: float | None = None,
        fixed_control_bits: int | None = None,
        loss_weights: Mapping[str, float] | None = None,
    ) -> 'ThreeFieldOptions':
        if support_mode != "nonlinear":
            raise ValueError("round31 fixes nonlinear support")
        key = _normalise_variant(variant)
        if key not in ROUND31_VARIANTS:
            raise ValueError(f"unknown round31 variant {variant!r}")
        if evidence_reduction not in (None, "candidate_mean"):
            raise ValueError("round31 fixes evidence_reduction=candidate_mean")
        if evidence_supervision not in (None, "wrong_query"):
            raise ValueError("round31 fixes evidence_supervision=wrong_query")
        values = dict(ROUND31_VARIANTS[key])
        for name, value in (
            ("carrier_mode", carrier_mode),
            ("linear_calibration", linear_calibration),
            ("pairwise_interactions", pairwise_interactions),
            ("transition_context_mode", transition_context_mode),
            ("transition_target_mode", transition_target_mode),
            ("min_span_clips", min_span_clips),
            ("rank_mode", rank_mode),
            ("rank_kl_half", rank_kl_half),
            ("rank_stop_s", rank_stop_s),
            ("aux_detach_input", aux_detach_input),
            ("rank_form", rank_form),
            ("rank_competition", rank_competition),
            ("rank_logit_scale", rank_logit_scale),
            ("rank_target_threshold", rank_target_threshold),
            ("rank_target_exponent", rank_target_exponent),
            ("rank_margin", rank_margin),
            ("quality_stratified", quality_stratified),
            ("projected_cosine_attention", projected_cosine_attention),
            ("local_top_quarter_e", local_top_quarter_e),
            ("difference_before_interaction_t", difference_before_interaction_t),
            ("a_local_evidence", a_local_evidence),
            ("e_readout_scale", e_readout_scale),
            ("query_relative_e", query_relative_e),
            ("e_span_lme", e_span_lme),
            ("independent_s_projection", independent_s_projection),
            ("b_edge_support", b_edge_support),
            ("c_latent_composition", c_latent_composition),
            ("support_deployment", support_deployment),
            ("support_input", support_input),
            ("support_pair_mode", support_pair_mode),
            ("support_readout", support_readout),
            ("support_candidate_chunk_size", support_candidate_chunk_size),
            ("support_wide_pair", support_wide_pair),
            ("support_length_center", support_length_center),
            ("rank_candidate_pair", rank_candidate_pair),
            ("interaction_depth", interaction_depth),
            ("interaction_feedforward", interaction_feedforward),
            ("length_calibration", length_calibration),
            ("interaction_dropout", interaction_dropout),
            ("edge_head_dropout", edge_head_dropout),
        ):
            if value is not None:
                values[name] = value
        if (
            fixed_control_bits is not None
            and int(fixed_control_bits) != ROUND31_FIXED_CONTROL_BITS
        ):
            raise ValueError("round31 fixes legacy control bits at 7")
        if int(values["min_span_clips"]) not in (1, 2):
            raise ValueError("round31 min_span_clips must be 1 or 2")
        if str(values["support_input"]).lower() != "rms":
            raise ValueError("round31 fixes support_input=rms")
        if str(values["e_readout_scale"]).lower() not in {
            "raw",
            "fixed_init",
            "query_rms",
            "token_rms",
        }:
            raise ValueError(
                "e_readout_scale must be raw, fixed_init, query_rms, or token_rms"
            )
        if str(values["support_pair_mode"]).lower() != "joint_residual":
            raise ValueError("round31 fixes support_pair_mode=joint_residual")
        if str(values["support_readout"]).lower() not in ROUND31_READOUTS:
            raise ValueError(
                "support_readout must be one of the frozen Round31 readouts"
            )
        if int(values["support_candidate_chunk_size"]) != 128:
            raise ValueError("round31 fixes support_candidate_chunk_size=128")
        if str(values["rank_mode"]).lower() not in {"ordinal_margin", "gt_balanced_kl"}:
            raise ValueError("rank_mode must be ordinal_margin or gt_balanced_kl")
        margin = float(values.get("rank_margin", ROUND31_RANK_MARGIN))
        if not math.isfinite(margin) or abs(margin - ROUND31_RANK_MARGIN) > 1e-12:
            raise ValueError(f"round31 fixes rank_margin at {ROUND31_RANK_MARGIN:g}")
        expected = ROUND31_VARIANTS[key]
        fixed_names = (
            "rank_form",
            "rank_competition",
            "linear_calibration",
            "pairwise_interactions",
            "transition_context_mode",
            "transition_target_mode",
            "carrier_mode",
            "min_span_clips",
            "rank_mode",
            "quality_stratified",
            "rank_logit_scale",
            "rank_target_threshold",
            "rank_target_exponent",
            "projected_cosine_attention",
            "local_top_quarter_e",
            "difference_before_interaction_t",
            "a_local_evidence",
            "query_relative_e",
            "e_span_lme",
            "independent_s_projection",
            "e_readout_scale",
            "b_edge_support",
            "c_latent_composition",
            "support_deployment",
            "support_input",
            "support_pair_mode",
            "support_readout",
            "support_candidate_chunk_size",
            "interaction_depth",
            "interaction_feedforward",
            "length_calibration",
            "interaction_dropout",
            "edge_head_dropout",
        )
        for name in fixed_names:
            if values[name] != expected[name]:
                raise ValueError(f"{key} fixes {name}={expected[name]}")
        # The three factor kwargs are variant-controlled too.  Accept an
        # explicit value only when it agrees with the encoded v0..v7 cell;
        # otherwise a config could silently select a different experiment.
        for name in (
            "support_wide_pair",
            "support_length_center",
            "rank_kl_half",
            "rank_candidate_pair",
            "rank_stop_s",
            "aux_detach_input",
        ):
            # Do not coerce strings/integers to bool: a malformed config must
            # not silently select a different route cell.
            if (
                type(values[name]) is not type(expected[name])
                or values[name] != expected[name]
            ):
                raise ValueError(f"{key} fixes {name}={expected[name]!r}")
        # round32 extension: a variant may optionally override evidence/
        # support/transition/endpoint weights too (via *_weight keys), not
        # just rank. Variants that do not set these keys (y0..y7) fall back
        # to _DEFAULT_WEIGHTS exactly as before -- no behavior change for them.
        expected_weights = dict(
            _DEFAULT_WEIGHTS,
            rank=expected["rank_weight"],
            evidence=expected.get("evidence_weight", _DEFAULT_WEIGHTS["evidence"]),
            support=expected.get("support_weight", _DEFAULT_WEIGHTS["support"]),
            transition=expected.get(
                "transition_weight", _DEFAULT_WEIGHTS["transition"]
            ),
            endpoint=expected.get("endpoint_weight", _DEFAULT_WEIGHTS["endpoint"]),
        )
        resolved_weights = (
            expected_weights
            if loss_weights is None
            else _validated_weights(loss_weights)
        )
        if resolved_weights != expected_weights:
            raise ValueError(f"{key} fixes loss weights at {expected_weights}")
        return cls(
            carrier_mode=values["carrier_mode"],
            linear_calibration=bool(values["linear_calibration"]),
            pairwise_interactions=bool(values["pairwise_interactions"]),
            variant=key,
            min_span_clips=int(values["min_span_clips"]),
            rank_mode=str(values["rank_mode"]).lower(),
            rank_kl_half=bool(values["rank_kl_half"]),
            rank_stop_s=bool(values["rank_stop_s"]),
            aux_detach_input=bool(values["aux_detach_input"]),
            rank_form=values["rank_form"],
            rank_competition=values["rank_competition"],
            rank_logit_scale=float(values["rank_logit_scale"]),
            rank_target_threshold=float(values["rank_target_threshold"]),
            rank_target_exponent=float(values["rank_target_exponent"]),
            rank_margin=ROUND31_RANK_MARGIN,
            quality_stratified=bool(values["quality_stratified"]),
            transition_target_mode=values["transition_target_mode"],
            transition_context_mode=values["transition_context_mode"],
            support_mode="nonlinear",
            projected_cosine_attention=bool(values["projected_cosine_attention"]),
            local_top_quarter_e=bool(values["local_top_quarter_e"]),
            difference_before_interaction_t=bool(
                values["difference_before_interaction_t"]
            ),
            a_local_evidence=bool(values["a_local_evidence"]),
            e_readout_scale=str(values["e_readout_scale"]).lower(),
            query_relative_e=bool(values["query_relative_e"]),
            e_span_lme=bool(values["e_span_lme"]),
            independent_s_projection=bool(values["independent_s_projection"]),
            b_edge_support=bool(values["b_edge_support"]),
            c_latent_composition=bool(values["c_latent_composition"]),
            support_deployment=str(values["support_deployment"]),
            support_input=str(values["support_input"]),
            support_pair_mode=str(values["support_pair_mode"]),
            support_readout=str(values["support_readout"]).lower(),
            support_candidate_chunk_size=int(values["support_candidate_chunk_size"]),
            support_wide_pair=bool(values["support_wide_pair"]),
            support_length_center=bool(values["support_length_center"]),
            rank_candidate_pair=bool(values["rank_candidate_pair"]),
            interaction_depth=int(values["interaction_depth"]),
            interaction_feedforward=bool(values["interaction_feedforward"]),
            length_calibration=bool(values["length_calibration"]),
            interaction_dropout=float(values["interaction_dropout"]),
            edge_head_dropout=float(values["edge_head_dropout"]),
            fixed_control_bits=ROUND31_FIXED_CONTROL_BITS,
            loss_weights=resolved_weights,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "carrier_mode": self.carrier_mode,
            "linear_calibration": self.linear_calibration,
            "pairwise_interactions": self.pairwise_interactions,
            "transition_context_mode": self.transition_context_mode,
            "variant": self.variant,
            "min_span_clips": self.min_span_clips,
            "rank_mode": self.rank_mode,
            "rank_kl_half": self.rank_kl_half,
            "rank_stop_s": self.rank_stop_s,
            "aux_detach_input": self.aux_detach_input,
            "evidence_reduction": self.evidence_reduction,
            "evidence_supervision": self.evidence_supervision,
            "rank_form": self.rank_form,
            "rank_competition": self.rank_competition,
            "rank_logit_scale": self.rank_logit_scale,
            "rank_target_threshold": self.rank_target_threshold,
            "rank_target_exponent": self.rank_target_exponent,
            "rank_margin": self.rank_margin,
            "quality_stratified": self.quality_stratified,
            "transition_target_mode": self.transition_target_mode,
            "support_mode": self.support_mode,
            "projected_cosine_attention": self.projected_cosine_attention,
            "local_top_quarter_e": self.local_top_quarter_e,
            "difference_before_interaction_t": self.difference_before_interaction_t,
            "a_local_evidence": self.a_local_evidence,
            "e_readout_scale": self.e_readout_scale,
            "query_relative_e": self.query_relative_e,
            "e_span_lme": self.e_span_lme,
            "independent_s_projection": self.independent_s_projection,
            "b_edge_support": self.b_edge_support,
            "c_latent_composition": self.c_latent_composition,
            "support_deployment": self.support_deployment,
            "support_input": self.support_input,
            "support_pair_mode": self.support_pair_mode,
            "support_readout": self.support_readout,
            "readout_index": ROUND31_READOUTS.index(self.support_readout),
            "rank_background": "K+.5P" if self.rank_candidate_pair else "K",
            "support_candidate_chunk_size": self.support_candidate_chunk_size,
            "support_wide_pair": self.support_wide_pair,
            "support_length_center": self.support_length_center,
            "rank_candidate_pair": self.rank_candidate_pair,
            "alpha": self.alpha,
            "beta": self.beta,
            "factor_bits": self.factor_bits,
            "route_bits": self.route_bits,
            "support_deployment_zero": self.support_deployment_zero,
            "support_input_rms": self.support_input_rms,
            "support_pair_joint": self.support_pair_joint,
            "fixed_control_bits": self.fixed_control_bits,
            "mechanism_bits": self.mechanism_bits,
            "loss_weights": dict(self.loss_weights),
        }


def _parent_min_span(parent_model: nn.Module, expected: int) -> None:
    shared_head = getattr(parent_model, "shared_head", None)
    if shared_head is None or not hasattr(shared_head, "min_span_clips"):
        raise RuntimeError(
            "round31 requires parent.shared_head.min_span_clips as the active "
            "candidate-membership control"
        )
    shared_head.min_span_clips = int(expected)
    actual = int(getattr(shared_head, "min_span_clips"))
    if actual != int(expected):
        raise RuntimeError(
            f"parent shared_head.min_span_clips readback {actual} != {expected}"
        )


class ThreeFieldModel(TriFieldBaseModel):
    """Carrier model with evidence, support, transition scores and their losses."""

    def __init__(
        self,
        parent_model: nn.Module,
        *,
        options: ThreeFieldOptions | None = None,
        selector_hidden_dim: int = 64,
        selector_lr: float = 1.0e-4,
        parent_factory: str = DEFAULT_PARENT_FACTORY,
    ) -> None:
        options = options or ThreeFieldOptions.from_values()
        _parent_min_span(parent_model, options.min_span_clips)
        super().__init__(
            parent_model,
            selector_hidden_dim=selector_hidden_dim,
            selector_lr=selector_lr,
            parent_factory=parent_factory,
        )
        _parent_min_span(self.parent_model, options.min_span_clips)
        self.options = options
        canonical_selector = self.selector
        with torch.random.fork_rng(devices=[]):
            round31_selector = ThreeFieldScoreHeads(
                input_dim=512,
                hidden_dim=selector_hidden_dim,
                support_mode="nonlinear",
                carrier_mode=options.carrier_mode,
                region_gain=options.region_gain,
                transition_gain=options.transition_gain,
                support_context=options.support_context,
                support_dispersion=options.support_dispersion,
                evidence_pool=options.evidence_pool,
                evidence_context=options.evidence_context,
                linear_calibration=options.linear_calibration,
                pairwise_interactions=options.pairwise_interactions,
                length_calibration=options.length_calibration,
                interaction_dropout=options.interaction_dropout,
                edge_head_dropout=options.edge_head_dropout,
                transition_context_mode=options.transition_context_mode,
                learned_e=True,
                learned_t=True,
                projected_cosine_attention=options.projected_cosine_attention,
                local_top_quarter_e=options.local_top_quarter_e,
                difference_before_interaction_t=options.difference_before_interaction_t,
                a_local_evidence=options.a_local_evidence,
                e_readout_scale=options.e_readout_scale,
                e_span_lme=options.e_span_lme,
                independent_s_projection=options.independent_s_projection,
                b_edge_support=options.b_edge_support,
                c_latent_composition=options.c_latent_composition,
                support_deployment=options.support_deployment,
                support_input=options.support_input,
                support_pair_mode=options.support_pair_mode,
                support_readout=options.support_readout,
                support_candidate_chunk_size=options.support_candidate_chunk_size,
                support_length_center=options.support_length_center,
                rank_stop_s=options.rank_stop_s,
                aux_detach_input=options.aux_detach_input,
                interaction_depth=options.interaction_depth,
                interaction_feedforward=options.interaction_feedforward,
            )
        round31_selector.load_state_dict(
            canonical_selector.state_dict(),
            strict=False,
        )
        if round31_selector.support_nonlinear is None:
            raise RuntimeError("round31 nonlinear support head was not constructed")
        with torch.no_grad():
            round31_selector.support_nonlinear[-1].weight.zero_()
            round31_selector.support_nonlinear[-1].bias.zero_()
        self.selector = round31_selector
        self.loss_weights = dict(options.loss_weights)
        # round34 extension: which eligible candidate becomes the rank/support
        # negative (see model/candidate_pair_loss.py::NEGATIVE_SELECTION_MODES).
        # A plain mutable attribute, same pattern as self.loss_weights above --
        # not part of the construction-time ROUND31_VARIANTS lock, opt-in via
        # config, "hardest" reproduces the original always-argmax behavior.
        self.negative_selection_mode = "hardest"
        # round36 extension: length-bias countermeasure H1 (see
        # model/losses.py::_query_length_ratio). Plain mutable attribute,
        # same pattern as negative_selection_mode above -- 0.0 reproduces
        # the original unweighted rank loss exactly.
        self.rank_length_reweight_gain = 0.0
        # round36 extension: length-bias countermeasure H4 (see
        # model/losses.py::_duration_regression_loss). 0.0 adds exactly zero.
        self.duration_aux_weight = 0.0
        # round37 extension: length-bias countermeasure H9 (see
        # model/losses.py::_gt_length_ratio). 0.0 leaves every GT's
        # effective rank-target threshold exactly at rank_target_threshold.
        self.length_conditional_gain = 0.0
        # round37 extension: length-bias countermeasure H10 (see
        # model/losses.py, carrier_desaturation block). 0.0 adds exactly zero.
        self.carrier_desaturation_weight = 0.0
        # round40 extension: H11, top-K restricted rank competition
        # (see model/losses.py::_topk_restrict_valid). 0 = current dense-
        # grid behavior, exact no-op.
        self.rank_topk_restrict = 0
        # round42 extension: H13, symmetric IoU-threshold relief for
        # SHORT spans (see model/losses.py, short_relief_gain block).
        # 0.0 adds exactly zero.
        self.short_relief_gain = 0.0
        # round45 extension: H14, relaxed shift-negative fallback for
        # width=1 support-field counterfactual pairs (see
        # model/local_counterfactual.py). False = current behavior,
        # exact no-op.
        self.support_shift_relaxed_fallback = False
        # R47: the old counterfactual support term is otherwise superseded by
        # the deployed wide-pair support term.  Keep zero as an exact no-op;
        # nonzero values are explicitly logged by the final-loss probe.
        self.counterfactual_support_mix = 0.0
        self._last_wrong_query_field = None
        self.initialization_audit.update(
            {
                "new_base": "trifield_round31",
                "round31_variant": options.variant,
                "round31_options": options.as_dict(),
                "loss_weights": dict(self.loss_weights),
            }
        )

    def _attach_raw_inputs(self, state, inputs):
        if self.options.fixed_control_bits & 6:
            key = getattr(self, "counterfactual_video_key", "src_vid")
            if key not in inputs:
                raise RuntimeError(
                    "Round31 learned field requires actual raw video input"
                )
            state.round31_raw_inputs = (
                inputs[key][..., :512],
                inputs["src_txt"],
                inputs["video_padding_mask"],
                inputs.get("query_padding_mask"),
            )

    def forward(
        self,
        inputs,
        query_features=None,
        video_padding_mask=None,
        query_padding_mask=None,
    ):
        if not self.options.fixed_control_bits & 6:
            return super().forward(
                inputs, query_features, video_padding_mask, query_padding_mask
            )
        if not isinstance(inputs, Mapping):
            raise TypeError("Round31 learned fields require explicit raw input mapping")
        from field_core.adapter import _masked_softmax

        output = self.parent_model(
            inputs,
            query_features=query_features,
            video_padding_mask=video_padding_mask,
            query_padding_mask=query_padding_mask,
        )
        state = output.extension_output.state
        self._attach_raw_inputs(state, inputs)
        valid = output.span_valid_mask.bool()
        field = self.selector(
            state,
            valid,
            output.span_logits,
            video_padding_mask=inputs.get("video_padding_mask"),
        )
        output.span_logits = field.score.masked_fill(~valid, -1.0e4)
        output.span_probs = _masked_softmax(output.span_logits, valid)
        output.trifield_output = field
        self._last_inputs, self._last_field_output = inputs, field
        return output

    def _wrong_query_evidence(
        self, batch: Any, outputs: Any
    ) -> tuple[Tensor | None, Tensor]:
        """Re-run the actual parent conditioner/backbone for a real wrong query."""

        from field_core.adapter import _batch_parts, _valid_wrong_query_pairs

        self._last_wrong_query_field = None
        inputs, _, metadata = _batch_parts(batch)
        source_text = inputs.get("src_txt")
        if not isinstance(source_text, Tensor):
            self._last_evidence_pairs = {
                "pair_count": 0,
                "row_count": int(outputs.span_logits.shape[0]),
                "pair_rate": 0.0,
                "no_pair_rate": 1.0,
            }
            return None, torch.zeros(
                outputs.span_logits.shape[0],
                dtype=torch.bool,
                device=outputs.span_logits.device,
            )
        permutation_cpu, pair_cpu = _valid_wrong_query_pairs(
            metadata, source_text.shape[0]
        )
        permutation, pair = (
            permutation_cpu.to(source_text.device),
            pair_cpu.to(source_text.device),
        )
        pair_count = int(pair.sum().item())
        self._last_evidence_pairs = {
            "pair_count": pair_count,
            "row_count": int(pair.numel()),
            "pair_rate": float(pair.float().mean().item()) if pair.numel() else 0.0,
            "no_pair_rate": float((~pair).float().mean().item())
            if pair.numel()
            else 1.0,
        }
        if pair_count == 0:
            return None, pair
        wrong_inputs = dict(inputs)
        wrong_inputs["src_txt"] = source_text.index_select(0, permutation)
        query_pad = inputs.get("query_padding_mask")
        if isinstance(query_pad, Tensor):
            wrong_inputs["query_padding_mask"] = query_pad.index_select(0, permutation)
        wrong_output = self.parent_model(wrong_inputs)
        wrong_extension = getattr(wrong_output, "extension_output", None)
        wrong_state = getattr(wrong_extension, "state", None)
        if wrong_state is None:
            raise RuntimeError(
                "wrong-query parent forward did not expose InputTriFieldState"
            )
        self._attach_raw_inputs(wrong_state, wrong_inputs)
        wrong_field = self.selector.raw_from_state(
            wrong_state,
            outputs.span_valid_mask.bool(),
            wrong_output.span_logits,
            video_padding_mask=inputs.get("video_padding_mask"),
        )
        self._last_wrong_query_field = wrong_field
        return wrong_field.evidence, pair

    def compute_loss(
        self,
        outputs: Any,
        batch: Any,
        teacher_outputs: Any,
        epoch: int,
    ) -> LossResult:
        if teacher_outputs is not None:
            raise AssertionError(
                "trifield_round31 is scratch-only and forbids teacher/KD fusion"
            )
        wrong_evidence, pair = self._wrong_query_evidence(batch, outputs)
        from .counterfactual import build_counterfactual_context

        context = build_counterfactual_context(self, outputs, batch, epoch, pair)
        context["query_relative_e"] = self.options.query_relative_e
        terms = compute_loss_terms(
            outputs,
            batch,
            wrong_evidence=wrong_evidence,
            evidence_reduction=self.options.evidence_reduction,
            evidence_supervision=self.options.evidence_supervision,
            evidence_pair_mask=pair,
            counterfactual=context,
            loss_weights=self.loss_weights,
            rank_mode=self.options.rank_mode,
            rank_form=self.options.rank_form,
            rank_competition=self.options.rank_competition,
            rank_logit_scale=self.options.rank_logit_scale,
            rank_target_threshold=self.options.rank_target_threshold,
            rank_target_exponent=self.options.rank_target_exponent,
            rank_margin=self.options.rank_margin,
            quality_stratified=self.options.quality_stratified,
            transition_target_mode=self.options.transition_target_mode,
            a_local_evidence=self.options.a_local_evidence,
            b_edge_support=self.options.b_edge_support,
            support_wide_pair=self.options.support_wide_pair,
            rank_candidate_pair=self.options.rank_candidate_pair,
            rank_kl_half=self.options.rank_kl_half,
            rank_stop_s=self.options.rank_stop_s,
            aux_detach_input=self.options.aux_detach_input,
            selector=self.selector,
            epoch=epoch,
            negative_selection_mode=self.negative_selection_mode,
            rank_length_reweight_gain=self.rank_length_reweight_gain,
            duration_aux_weight=self.duration_aux_weight,
            length_conditional_gain=self.length_conditional_gain,
            carrier_desaturation_weight=self.carrier_desaturation_weight,
            rank_topk_restrict=self.rank_topk_restrict,
            short_relief_gain=self.short_relief_gain,
            counterfactual_support_mix=self.counterfactual_support_mix,
            rank_protect_gt=getattr(self, "r48_protect_gt", False),
            support_margin_scale=getattr(self, "r48_support_margin_scale", 1.0),
        )
        if hasattr(self, "r50_adjust_terms"):
            terms = self.r50_adjust_terms(terms, outputs, batch, epoch)
        metrics = dict(terms.metrics)
        if hasattr(self, "r48_loss_probe"):
            metrics.update(self.r48_loss_probe(terms, epoch))
        metrics.update(
            {
                "epoch": float(epoch),
                "round31/min_span_clips": float(self.options.min_span_clips),
                "round31/rank_mode_gt_balanced_kl": float(
                    self.options.rank_mode == "gt_balanced_kl"
                ),
                "round31/quality_stratified": float(self.options.quality_stratified),
                "round31/support_deployment_zero": float(
                    self.options.support_deployment_zero
                ),
                "round31/support_input_rms": float(self.options.support_input_rms),
                "round31/support_wide_pair": float(self.options.support_wide_pair),
                "round31/support_length_center": float(
                    self.options.support_length_center
                ),
                "round31/rank_candidate_pair": float(self.options.rank_candidate_pair),
                "round31/rank_kl_half": float(self.options.rank_kl_half),
                "round31/rank_stop_s": float(self.options.rank_stop_s),
                "round31/aux_detach_input": float(self.options.aux_detach_input),
                "round31/rank_alpha": float(self.options.alpha),
                "round31/rank_beta": float(self.options.beta),
                "round31/factor_bits": float(self.options.factor_bits),
                "evidence/pair_rate": float(self._last_evidence_pairs["pair_rate"]),
                "evidence/no_pair_rate": float(
                    self._last_evidence_pairs["no_pair_rate"]
                ),
            }
        )
        return LossResult(terms.total, metrics)

    def diagnostics(self, outputs: Any, batch: Any) -> Mapping[str, Tensor | float]:
        result = dict(super().diagnostics(outputs, batch))
        result.update(
            {
                "round31/min_span_clips": float(self.options.min_span_clips),
                "round31/rank_mode_gt_balanced_kl": float(
                    self.options.rank_mode == "gt_balanced_kl"
                ),
                "round31/quality_stratified": float(self.options.quality_stratified),
                "round31/support_deployment_zero": float(
                    self.options.support_deployment_zero
                ),
                "round31/support_input_rms": float(self.options.support_input_rms),
                "round31/support_wide_pair": float(self.options.support_wide_pair),
                "round31/support_length_center": float(
                    self.options.support_length_center
                ),
                "round31/rank_candidate_pair": float(self.options.rank_candidate_pair),
                "round31/rank_kl_half": float(self.options.rank_kl_half),
                "round31/rank_stop_s": float(self.options.rank_stop_s),
                "round31/aux_detach_input": float(self.options.aux_detach_input),
                "round31/rank_alpha": float(self.options.alpha),
                "round31/rank_beta": float(self.options.beta),
                "round31/factor_bits": float(self.options.factor_bits),
            }
        )
        field = getattr(outputs, "trifield_output", None)
        if field is not None:
            route_rank = getattr(field, "round31_rank_score", None)
            route_aux = getattr(field, "round31_aux_support", None)
            feature_e = getattr(field, "round31_z_e", None)
            feature_s = getattr(field, "round31_z_s", None)
            feature_aux = getattr(field, "round31_h_aux", None)
            raw_inputs = getattr(field, "round31_raw_inputs", None)
            raw_video = (
                raw_inputs[0]
                if isinstance(raw_inputs, (tuple, list)) and len(raw_inputs) >= 1
                else None
            )
            raw_text = (
                raw_inputs[1]
                if isinstance(raw_inputs, (tuple, list)) and len(raw_inputs) >= 2
                else None
            )
            raw_available = isinstance(raw_video, torch.Tensor) and isinstance(
                raw_text, torch.Tensor
            )
            raw_all_false = (
                (not raw_video.requires_grad) and (not raw_text.requires_grad)
                if raw_available
                else None
            )
            result.update(
                {
                    "round31/rank_score_requires_grad": float(
                        bool(
                            isinstance(route_rank, torch.Tensor)
                            and route_rank.requires_grad
                        )
                    ),
                    "round31/feature_z_e_requires_grad": float(
                        bool(
                            isinstance(feature_e, torch.Tensor)
                            and feature_e.requires_grad
                        )
                    ),
                    "round31/feature_z_s_requires_grad": float(
                        bool(
                            isinstance(feature_s, torch.Tensor)
                            and feature_s.requires_grad
                        )
                    ),
                    "round31/feature_h_aux_requires_grad": float(
                        bool(
                            isinstance(feature_aux, torch.Tensor)
                            and feature_aux.requires_grad
                        )
                    ),
                    "round31/aux_support_requires_grad": float(
                        bool(
                            isinstance(route_aux, torch.Tensor)
                            and route_aux.requires_grad
                        )
                    ),
                    "round31/raw_inputs_available": float(raw_available),
                    "round31/raw_inputs_verified": float(
                        raw_available and raw_all_false is True
                    ),
                    "round31/raw_inputs_all_requires_grad_false": (
                        float(raw_all_false) if raw_all_false is not None else -1.0
                    ),
                    "round31/raw_src_vid_requires_grad": (
                        float(raw_video.requires_grad)
                        if isinstance(raw_video, torch.Tensor)
                        else -1.0
                    ),
                    "round31/raw_src_txt_requires_grad": (
                        float(raw_text.requires_grad)
                        if isinstance(raw_text, torch.Tensor)
                        else -1.0
                    ),
                    "round31/route_rank_stop_s": float(self.options.rank_stop_s),
                    "round31/route_aux_detach_input": float(
                        self.options.aux_detach_input
                    ),
                    # Epoch metrics are scalar reductions; full source labels live
                    # in round31_route and raw_inputs in the structured probe.
                    "round31/route_rank_score_source_matches": float(
                        getattr(field, "round31_rank_score_source", None)
                        == (
                            "same_F_arithmetic_with_detached_S"
                            if self.options.rank_stop_s
                            else "deployed_F"
                        )
                    ),
                }
            )
        return result

    def set_epoch(self, epoch: int, training: bool) -> Mapping[str, Any]:
        setter = getattr(self.parent_model, "set_epoch", None)
        result = dict(setter(epoch, training)) if callable(setter) else {}
        result.update(
            {
                "base": "trifield_round31",
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
                "base": "trifield_round31",
                "round31_variant": self.options.variant,
                "round31_route_bits": self.options.route_bits,
                "round31_rank_stop_s": self.options.rank_stop_s,
                "round31_aux_detach_input": self.options.aux_detach_input,
                "round31_options": self.options.as_dict(),
                "candidate_membership_min_span_clips": self.options.min_span_clips,
                "candidate_membership_ownership": (
                    "parent.shared_head.min_span_clips; coordinates unchanged"
                ),
                "final_score": "carrier+deployed_E+deployed_S+0.5*(Ts+Te); FP32 score",
                "carrier_bound": "range(-1,1); tanh(carrier_raw)",
                "rank": (
                    "official all-candidate ordinal positive margin 0.1"
                    if self.options.rank_mode == "ordinal_margin"
                    else (
                        self.options.rank_form
                        + "/"
                        + self.options.rank_competition
                        + ": fixed per-GT q from relu(IoU-.7)^4 with original tie fallback; rank alpha="
                        + str(self.options.alpha)
                        + ", pair beta="
                        + str(self.options.beta)
                    )
                ),
                "rank_mode": self.options.rank_mode,
                "rank_form": self.options.rank_form,
                "rank_competition": self.options.rank_competition,
                "support": (
                    "R30 selected candidate readout on token RMS features; "
                    "same joint-residual head for deployment, wide-S and "
                    "protected-donor length-two fallback"
                ),
                "transition": (
                    "Round17 D6 position/query-opposite supervision; optional learned raw transition residual"
                ),
                "quality_reduction": {
                    "near_far_equal_per_query": self.options.quality_stratified,
                    "near_iou_threshold": 0.3,
                    "empty_strata_dropped_without_reweighting_other_queries": True,
                },
                "loss_weights": dict(self.loss_weights),
                "factors": {
                    "support_readout": self.options.support_readout,
                    "rank_background": "K+.5P"
                    if self.options.rank_candidate_pair
                    else "K",
                    "legacy_compatibility_bits": self.options.factor_bits,
                    "support_wide_pair_fixed": True,
                    "rank_kl_half_fixed": False,
                    "rank_candidate_pair": self.options.rank_candidate_pair,
                    "rank_stop_s": self.options.rank_stop_s,
                    "aux_detach_input": self.options.aux_detach_input,
                    "route_bits": self.options.route_bits,
                    "legacy_u4_query_relative_e": self.options.query_relative_e,
                    "legacy_u4_e_span_lme": self.options.e_span_lme,
                    "legacy_u4_independent_s_projection": self.options.independent_s_projection,
                    "legacy_u4_E_local": self.options.a_local_evidence,
                    "support_length_center_fixed_off": self.options.support_length_center
                    is False,
                    "rank_alpha": self.options.alpha,
                    "rank_beta": self.options.beta,
                    "e_readout_scale": self.options.e_readout_scale,
                    "scheduler_factors": "external runtime configuration; no scheduler bit reaches model construction",
                },
                "fixed_control_bits": 7,
                "fixed_matrix": "y0..y7; rank-stop-S × auxiliary-LS-input-detach × K or K+.5P; edge_mean/wide-S/full-KL fixed; unchanged local E/T/carrier, independent S, five losses and external H12/W3",
                "e_readout_scale": self.options.e_readout_scale,
                "support_deployment": self.options.support_deployment,
                "support_input": self.options.support_input,
                "support_pair_mode": self.options.support_pair_mode,
                "support_readout": self.options.support_readout,
                "readout_index": ROUND31_READOUTS.index(self.options.support_readout),
                "rank_background": "K+.5P" if self.options.rank_candidate_pair else "K",
                "support_candidate_chunk_size": self.options.support_candidate_chunk_size,
                "support_wide_pair": self.options.support_wide_pair,
                "support_length_center": self.options.support_length_center,
                "rank_kl_half": self.options.rank_kl_half,
                "rank_candidate_pair": self.options.rank_candidate_pair,
                "rank_alpha": self.options.alpha,
                "rank_beta": self.options.beta,
                "score_bounds": {
                    "carrier": [-1.0, 1.0],
                    "evidence": [-1.0, 1.0],
                    "support": [-1.0, 1.0],
                    "transition_half_sum": [-1.0, 1.0],
                    "final_score": [-4.0, 4.0],
                },
                "region_gain": self.options.region_gain,
                "transition_gain": self.options.transition_gain,
                "evidence_reduction": self.options.evidence_reduction,
                "evidence_supervision": self.options.evidence_supervision,
                "evidence_pool": False,
                "evidence_context": self.options.evidence_context,
                "support_context": self.options.support_context,
                "support_dispersion": self.options.support_dispersion,
                "probes": {
                    "losses": [
                        "rank",
                        "evidence",
                        "support",
                        "transition",
                        "endpoint",
                    ],
                    "fields": ["carrier", "evidence", "support", "transition"],
                    "structures": [
                        "candidate membership",
                        "score identity",
                        "conditioner ablation",
                        "field ablation",
                    ],
                    "round31": [
                        "actual configured T target consistency",
                        "short-GT candidate/full/top10/top30 coverage",
                        "duplicate rate",
                        "field saturation",
                        "support pair decomposition",
                        "joint residual cancellation",
                        "single-endpoint zero identity and gradients",
                        "candidate pair selection, wide-pair/donor fallback counts",
                        "same-length support centering and signed length strata",
                        "candidate-rank KL/pair decomposition",
                        "actual score/support bounds",
                    ],
                },
            }
        )
        return result


def build_trifield_round31_model(
    config: Any = None,
    *,
    parent_factory: str = DEFAULT_PARENT_FACTORY,
    selector_hidden_dim: int = 64,
    selector_lr: float = 1.0e-4,
    carrier_mode: str | None = None,
    linear_calibration: bool | None = None,
    pairwise_interactions: bool | None = None,
    variant: str | None = None,
    transition_target_mode: str | None = None,
    transition_context_mode: str | None = None,
    support_mode: str = "nonlinear",
    min_span_clips: int | None = None,
    rank_mode: str | None = None,
    rank_kl_half: bool | None = None,
    rank_stop_s: bool | None = None,
    aux_detach_input: bool | None = None,
    evidence_reduction: str | None = None,
    evidence_supervision: str | None = None,
    rank_form: str | None = None,
    rank_competition: str | None = None,
    rank_logit_scale: float | None = None,
    rank_target_threshold: float | None = None,
    rank_target_exponent: float | None = None,
    quality_stratified: bool | None = None,
    rank_margin: float | None = None,
    a_local_evidence: bool | None = None,
    e_readout_scale: str | None = None,
    query_relative_e: bool | None = None,
    e_span_lme: bool | None = None,
    independent_s_projection: bool | None = None,
    b_edge_support: bool | None = None,
    c_latent_composition: bool | None = None,
    support_deployment: str | None = None,
    support_input: str | None = None,
    support_pair_mode: str | None = None,
    support_readout: str | None = None,
    support_candidate_chunk_size: int | None = None,
    support_wide_pair: bool | None = None,
    support_length_center: bool | None = None,
    rank_candidate_pair: bool | None = None,
    fixed_control_bits: int | None = None,
    loss_weights: Mapping[str, float] | None = None,
    counterfactual_video_key: str | None = None,
    counterfactual_visual_channels: int | None = None,
    counterfactual_support_mix: float = 0.0,
    **kwargs: Any,
) -> ThreeFieldModel:
    """Build the reusable parent plus one of the fixed round31 variants."""

    forbidden_budget_overrides = {"alpha", "beta", "rank_alpha", "rank_beta"}
    illegal = sorted(forbidden_budget_overrides.intersection(kwargs))
    if illegal:
        raise ValueError(
            "round31 derives alpha/beta from variant bits; free overrides are forbidden: "
            + ", ".join(illegal)
        )

    options = ThreeFieldOptions.from_values(
        carrier_mode=carrier_mode,
        linear_calibration=linear_calibration,
        pairwise_interactions=pairwise_interactions,
        variant=variant,
        transition_target_mode=transition_target_mode,
        transition_context_mode=transition_context_mode,
        support_mode=support_mode,
        min_span_clips=min_span_clips,
        rank_mode=rank_mode,
        rank_kl_half=rank_kl_half,
        rank_stop_s=rank_stop_s,
        aux_detach_input=aux_detach_input,
        evidence_reduction=evidence_reduction,
        evidence_supervision=evidence_supervision,
        rank_form=rank_form,
        rank_competition=rank_competition,
        rank_logit_scale=rank_logit_scale,
        rank_target_threshold=rank_target_threshold,
        rank_target_exponent=rank_target_exponent,
        rank_margin=rank_margin,
        quality_stratified=quality_stratified,
        a_local_evidence=a_local_evidence,
        e_readout_scale=e_readout_scale,
        query_relative_e=query_relative_e,
        e_span_lme=e_span_lme,
        independent_s_projection=independent_s_projection,
        b_edge_support=b_edge_support,
        c_latent_composition=c_latent_composition,
        support_deployment=support_deployment,
        support_input=support_input,
        support_pair_mode=support_pair_mode,
        support_readout=support_readout,
        support_candidate_chunk_size=support_candidate_chunk_size,
        support_wide_pair=support_wide_pair,
        support_length_center=support_length_center,
        rank_candidate_pair=rank_candidate_pair,
        fixed_control_bits=fixed_control_bits,
        loss_weights=loss_weights,
    )
    factory = (
        _resolve_target(parent_factory)
        if isinstance(parent_factory, str)
        else parent_factory
    )
    parent_model = factory(config, **kwargs)
    model = ThreeFieldModel(
        parent_model,
        options=options,
        selector_hidden_dim=selector_hidden_dim,
        selector_lr=selector_lr,
        parent_factory=(
            parent_factory
            if isinstance(parent_factory, str)
            else getattr(factory, "__module__", str(factory))
        ),
    )

    model.counterfactual_video_key = counterfactual_video_key
    model.counterfactual_visual_channels = counterfactual_visual_channels
    model.counterfactual_support_mix = float(counterfactual_support_mix)
    return model


build_model = build_trifield_round31_model
FieldModel = ThreeFieldModel


__all__ = [
    "DEFAULT_PARENT_FACTORY",
    "ROUND31_VARIANTS",
    'ThreeFieldOptions',
    'FieldModel',
    'ThreeFieldModel',
    "build_model",
    "build_trifield_round31_model",
]
