"""EventFieldNet: full three-field scoring and rank-only support-gradient routing."""

import dataclasses
import torch
from . import local_support
from .model import losses
from .length_objective import durations
from .gradient_routing import local_parameters
from . import coverage_loss as s_helpers

MODES = ("hinge_legacy_route",)
NEUTRAL_OBJECTIVE = frozenset(MODES)
HINGE_OBJECTIVE = frozenset(MODES)
LEGACY_PROJECTION_ROUTE = frozenset(MODES)
LATE_PARENT = frozenset()


def route_parameters(model):
    named = local_parameters(model)
    if model.r68_mode in LEGACY_PROJECTION_ROUTE:
        named = named + [
            ("selector.s_projection." + n, p)
            for n, p in model.selector.s_projection.named_parameters()
            if p.requires_grad
        ]
    if getattr(model, "followup_spec", {}).get("edge_route", False):
        named += [
            ("selector.edge_head." + n, p)
            for n, p in model.selector.edge_head.named_parameters()
            if p.requires_grad
        ]
    if len({id(p) for _, p in named}) != len(named):
        raise RuntimeError("Duplicate R68 routed parameter")
    return named


def rank_gradient_correction(model, rank):
    """Same K04 formula, applied once after every final rank substitution."""
    named = route_parameters(model)
    correction = rank.new_zeros(())
    if not torch.is_grad_enabled() or not rank.requires_grad:
        return correction, {"gradient_queries_active": rank.new_zeros(())}
    with torch.autocast(device_type=rank.device.type, enabled=False):
        grads = torch.autograd.grad(
            rank, [p for _, p in named], retain_graph=True, allow_unused=True
        )
    norm2 = rank.new_zeros((), dtype=torch.float32)
    for (_, parameter), gradient in zip(named, grads):
        if gradient is not None:
            detached = gradient.detach().float()
            correction = (
                correction
                + ((parameter - parameter.detach()).float() * (-0.9 * detached)).sum()
            )
            norm2 = norm2 + detached.square().sum()
    return correction, {
        "gradient_queries_active": rank.new_ones(()),
        "unscaled_rank_routed_norm": norm2.sqrt(),
        "declared_rank_routed_scale": rank.new_tensor(0.1),
        "zero_value_error": correction.detach().abs(),
    }


class EventFieldNet(local_support.Model):
    def r68_parent_lr_multiplier(self, epoch):
        return 0.5 if self.r68_mode in LATE_PARENT and int(epoch) >= 11 else 1.0

    def experiment_contract(self):
        c = dict(super().experiment_contract())
        c.update(
            r68_mode=self.r68_mode,
            r68_reference="R67 K04; exact R66 J07 structure, local rank parameter gradient x0.1",
            r68_fixed_budget=24,
            r68_model_factory_checkpoint_loads=0,
            r68_inference_uses_gt=False,
            r68_no_new_parameters=True,
            r68_shared_initialization_rng_unchanged=True,
            r68_training_precision="fp32",
            r68_official_evaluation_precision="fp32; no TF32",
            r68_rank_gradient_order="final KL/quality/pair/length rank assembled, then one local/projection correction",
            r68_local_rank_gradient_multiplier=0.1,
            r68_s_projection_rank_gradient_multiplier=0.1
            if self.r68_mode in LEGACY_PROJECTION_ROUTE
            else 1.0,
            r68_edge_head_rank_gradient_multiplier=0.1
            if getattr(self, "followup_spec", {}).get("edge_route", False)
            else 1.0,
            r68_route_scope="local token gate and local readout"
            + (
                "; selector.s_projection and selector.edge_head"
                if self.r68_mode in LEGACY_PROJECTION_ROUTE
                else ""
            ),
            r68_support_objective=(
                "hinge_truncation_and_neutral_expansion"
                if self.r68_mode in HINGE_OBJECTIVE
                else "symmetric_coverage_gap_and_neutral_expansion"
                if self.r68_mode in NEUTRAL_OBJECTIVE
                else "D07_coverage_pair"
            ),
            r68_short_balanced_kl=self.r68_mode == "short_balanced_kl",
            r68_short_weight="1 + valid short GT fraction; active-query normalization; unknown duration not short",
            r68_late_parent_enabled=self.r68_late_parent_enabled,
            r68_late_parent_start_epoch=11,
            r68_parent_lr_multiplier_after_start=0.5
            if self.r68_late_parent_enabled
            else 1.0,
            r68_optimizer_policy_scope="initial model groups unchanged; actual e11 parent-only scheduler multiplier owned and audited by runtime",
        )
        c["long_s_spec"] = self.long_s_spec
        c["followup_spec"] = self.followup_spec
        return c

    def r50_adjust_terms(self, terms, outputs, batch, epoch):
        base = super().r50_adjust_terms(terms, outputs, batch, epoch)
        field = outputs.trifield_output
        rank, support, total = base.rank, base.support, base.total
        metrics, diag = dict(base.metrics), {}
        parts = dict(
            base_reference_rank=rank,
            reference_support=support,
            base_reference_total=total,
        )
        spans = gt_mask = None
        if self.r68_mode in NEUTRAL_OBJECTIVE or self.r68_mode == "short_balanced_kl":
            spans, gt_mask = losses._target_spans(outputs, batch)
        if self.r68_mode in NEUTRAL_OBJECTIVE:
            field.long_s_spec = self.long_s_spec
            field.long_s_durations = durations(batch, spans)
            support, details, payload = s_helpers.coverage_gap_objective(
                field,
                base.geometry,
                spans,
                gt_mask,
                hinge_truncation=self.r68_mode in HINGE_OBJECTIVE,
                return_payload=True,
            )
            total = total + self.loss_weights["support"] * (support - base.support)
            parts["actual_coverage_objective"] = support
            field.r68_coverage_payload = payload
            diag.update({"coverage/" + k: v for k, v in details.items()})
            diag["reference_support_loss"] = base.support.detach()
            for key in list(metrics):
                if key.startswith("support_geometry/coverage/"):
                    metrics["legacy_reference/" + key] = metrics.pop(key)
        parts.update(
            reference_rank=rank,
            reference_total=total,
            final_unrouted_rank=rank,
            final_unrouted_total=total,
        )
        correction, route = rank_gradient_correction(self, rank)
        rank = rank + correction
        total = total + self.loss_weights["rank"] * correction
        parts["rank_gradient_correction"] = correction
        diag.update({"route/" + k: v for k, v in route.items()})
        diag.update(
            epoch=total.new_tensor(float(epoch)),
            declared_parent_lr_multiplier=total.new_tensor(
                self.r68_parent_lr_multiplier(epoch)
            ),
        )
        field.r68_loss_parts = parts
        self.r68_last_training = {
            k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in diag.items()
        }
        metrics.update(
            {"model_factory/" + k: v for k, v in self.r68_last_training.items()}
        )
        metrics.update(
            rank=rank.detach(), support=support.detach(), total=total.detach()
        )
        result = dataclasses.replace(
            base, rank=rank, support=support, total=total, metrics=metrics
        )
        self.r50_last_terms = (
            result if getattr(self, "r50_capture_terms", False) else None
        )
        return result


def build_model(
    config, *, r68_spec=None, long_s_spec=None, followup_spec=None, **kwargs
):
    spec = dict(r68_spec or {})
    if spec != {"mode": "hinge_legacy_route"}:
        raise ValueError("This release implements only the fixed V00 recipe")
    if any(key in kwargs for key in ("r65_spec", "r66_spec", "r67_spec")):
        raise ValueError("R68 supplies its exact R66 J07 reference internally")
    precision = (
        config.get("precision")
        if isinstance(config, dict)
        else getattr(config, "precision", None)
    )
    if precision != "fp32":
        raise ValueError("All R68 training and official inference must use strict FP32")
    model = local_support.build_model(
        config, r66_spec={"mode": "uniform_null_s"}, **kwargs
    )
    model.__class__ = EventFieldNet
    repair = dict(long_s_spec or {"upper": False, "long_weight": False})
    if set(repair) != {"upper", "long_weight"} or any(
        type(v) is not bool for v in repair.values()
    ):
        raise ValueError("Explicit boolean long S repair specification required")
    followup = dict(followup_spec or {"edge_route": False, "long_kl": False})
    if set(followup) != {"edge_route", "long_kl"} or any(
        type(v) is not bool for v in followup.values()
    ):
        raise ValueError("Explicit followup booleans required")
    if followup != {"edge_route": True, "long_kl": False} or any(repair.values()):
        raise ValueError(
            "V00 requires edge rank routing, without auxiliary repair or Long KL"
        )
    model.followup_spec = followup
    model.long_s_spec = repair
    model.r68_mode = spec["mode"]
    model.r68_last_training = {}
    model.r68_late_parent_enabled = spec["mode"] in LATE_PARENT
    model.r68_late_parent_start_epoch = 11
    return model


# Compatibility name for inherited diagnostic helpers.
Model = EventFieldNet
