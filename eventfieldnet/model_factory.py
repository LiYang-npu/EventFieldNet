"""Full E/S/T scoring, neutral support coverage, and rank-gradient routing."""

import dataclasses
import torch

from . import local_support
from .model import losses
from .length_objective import durations
from .gradient_routing import local_parameters
from . import coverage_loss as s_helpers


def route_parameters(model):
    named = local_parameters(model)
    named = named + [
        ("selector.s_projection." + name, parameter)
        for name, parameter in model.selector.s_projection.named_parameters()
        if parameter.requires_grad
    ]
    named += [
        ("selector.edge_head." + name, parameter)
        for name, parameter in model.selector.edge_head.named_parameters()
        if parameter.requires_grad
    ]
    if len({id(parameter) for _, parameter in named}) != len(named):
        raise RuntimeError("Duplicate routed parameter")
    return named


def rank_gradient_correction(model, rank):
    """Scale only selected ranking gradients by 0.1, preserving the loss value."""
    named = route_parameters(model)
    correction = rank.new_zeros(())
    if not torch.is_grad_enabled() or not rank.requires_grad:
        return correction, {"gradient_queries_active": rank.new_zeros(())}
    with torch.autocast(device_type=rank.device.type, enabled=False):
        grads = torch.autograd.grad(
            rank, [parameter for _, parameter in named],
            retain_graph=True, allow_unused=True,
        )
    norm2 = rank.new_zeros((), dtype=torch.float32)
    for (_, parameter), gradient in zip(named, grads):
        if gradient is not None:
            detached = gradient.detach().float()
            correction = correction + (
                (parameter - parameter.detach()).float() * (-0.9 * detached)
            ).sum()
            norm2 = norm2 + detached.square().sum()
    return correction, {
        "gradient_queries_active": rank.new_ones(()),
        "unscaled_rank_routed_norm": norm2.sqrt(),
        "declared_rank_routed_scale": rank.new_tensor(0.1),
        "zero_value_error": correction.detach().abs(),
    }


class EventFieldNet(local_support.Model):
    def r68_parent_lr_multiplier(self, epoch):
        return 1.0

    def experiment_contract(self):
        contract = dict(super().experiment_contract())
        contract.update(
            r68_mode="hinge_legacy_route", r68_fixed_budget=24,
            r68_model_factory_checkpoint_loads=0,
            r68_inference_uses_gt=False, r68_no_new_parameters=True,
            r68_shared_initialization_rng_unchanged=True,
            r68_training_precision="fp32",
            r68_official_evaluation_precision="fp32; no TF32",
            r68_rank_gradient_order="final rank assembled, then one correction",
            r68_local_rank_gradient_multiplier=0.1,
            r68_s_projection_rank_gradient_multiplier=0.1,
            r68_edge_head_rank_gradient_multiplier=0.1,
            r68_route_scope="local support, support projection, support edge head",
            r68_support_objective="hinge_truncation_and_neutral_expansion",
            r68_short_balanced_kl=False,
            r68_late_parent_enabled=False,
            r68_late_parent_start_epoch=11,
            r68_parent_lr_multiplier_after_start=1.0,
            r68_optimizer_policy_scope="unchanged model parameter groups",
            long_s_spec=self.long_s_spec, followup_spec=self.followup_spec,
        )
        return contract

    def r50_adjust_terms(self, terms, outputs, batch, epoch):
        base = super().r50_adjust_terms(terms, outputs, batch, epoch)
        field = outputs.trifield_output
        rank, support, total = base.rank, base.support, base.total
        metrics, diag = dict(base.metrics), {}
        parts = dict(
            base_reference_rank=rank, reference_support=support,
            base_reference_total=total,
        )
        spans, gt_mask = losses._target_spans(outputs, batch)
        field.long_s_spec = self.long_s_spec
        field.long_s_durations = durations(batch, spans)
        support, details, payload = s_helpers.coverage_gap_objective(
            field, base.geometry, spans, gt_mask,
            hinge_truncation=True, return_payload=True,
        )
        total = total + self.loss_weights["support"] * (support - base.support)
        parts["actual_coverage_objective"] = support
        field.r68_coverage_payload = payload
        diag.update({"coverage/" + key: value for key, value in details.items()})
        diag["reference_support_loss"] = base.support.detach()
        for key in list(metrics):
            if key.startswith("support_geometry/coverage/"):
                metrics["legacy_reference/" + key] = metrics.pop(key)
        parts.update(
            reference_rank=rank, reference_total=total,
            final_unrouted_rank=rank, final_unrouted_total=total,
        )
        correction, route = rank_gradient_correction(self, rank)
        rank = rank + correction
        total = total + self.loss_weights["rank"] * correction
        parts["rank_gradient_correction"] = correction
        diag.update({"route/" + key: value for key, value in route.items()})
        diag.update(
            epoch=total.new_tensor(float(epoch)),
            declared_parent_lr_multiplier=total.new_tensor(
                self.r68_parent_lr_multiplier(epoch)
            ),
        )
        field.r68_loss_parts = parts
        self.r68_last_training = {
            key: value.detach() if isinstance(value, torch.Tensor) else value
            for key, value in diag.items()
        }
        metrics.update({
            "model_factory/" + key: value
            for key, value in self.r68_last_training.items()
        })
        metrics.update(rank=rank.detach(), support=support.detach(), total=total.detach())
        result = dataclasses.replace(
            base, rank=rank, support=support, total=total, metrics=metrics
        )
        self.r50_last_terms = result if getattr(self, "r50_capture_terms", False) else None
        return result


def build_model(config, *, r68_spec=None, long_s_spec=None, followup_spec=None, **kwargs):
    if r68_spec != {"mode": "hinge_legacy_route"}:
        raise ValueError("This release implements the fixed support objective")
    if any(key in kwargs for key in ("r65_spec", "r66_spec", "r67_spec")):
        raise ValueError("The internal scoring configuration is fixed")
    precision = config.get("precision") if isinstance(config, dict) else config.precision
    if precision != "fp32":
        raise ValueError("Training and official inference require strict FP32")
    model = local_support.build_model(
        config, r66_spec={"mode": "uniform_null_s"}, **kwargs
    )
    model.__class__ = EventFieldNet
    repair = dict(long_s_spec or {"upper": False, "long_weight": False})
    followup = dict(followup_spec or {"edge_route": False, "long_kl": False})
    if (
        repair != {"upper": False, "long_weight": False}
        or followup != {"edge_route": True, "long_kl": False}
        or any(type(value) is not bool for value in [*repair.values(), *followup.values()])
    ):
        raise ValueError("The fixed model requires edge routing and unweighted coverage")
    model.followup_spec = followup
    model.long_s_spec = repair
    model.r68_mode = "hinge_legacy_route"
    model.r68_last_training = {}
    model.r68_late_parent_enabled = False
    model.r68_late_parent_start_epoch = 11
    return model


Model = EventFieldNet
