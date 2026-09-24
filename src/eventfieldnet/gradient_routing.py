"""R67: controlled retention and local-S mechanisms on the R66 J07 model.

All eight arms train from scratch; validation never controls a loss, route,
learning rate, or stopping rule. The one routing arm scales only the local
S parameter derivative of rank, leaving its forward value and all other
parameter derivatives intact.
"""

import dataclasses
import torch
from . import local_support
from .model import losses
from .local_support_helpers import coverage_gap_objective
from .routing_helpers import prepare_coverage_residual

MODES = (
    "j07_control",
    "j07_bf16",
    "mid_decay",
    "slow_parent",
    "local_rank_tenth",
    "coverage_only",
    "neutral_objective",
    "coverage_neutral",
)
COVERAGE_READOUT = frozenset(("coverage_only", "coverage_neutral"))
NEUTRAL_OBJECTIVE = frozenset(("neutral_objective", "coverage_neutral"))


def local_parameters(model):
    return [
        ("selector.r59_local_support.weight", model.selector.r59_local_support.weight),
        *[
            ("selector.r66_centered_s_readout." + n, p)
            for n, p in model.selector.r66_centered_s_readout.named_parameters()
        ],
    ]


def rank_gradient_correction(model, rank):
    """Fixed first-order rank gradient multiplier; no extra fitted loss.

    The derivative query explicitly disables autocast, matching ordinary
    backward. Detached coefficients prevent higher-order and outside-route
    gradients. A parameter minus its detached copy gives exact zero value.
    """
    named = local_parameters(model)
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
        "unscaled_rank_local_norm": norm2.sqrt(),
        "declared_rank_local_scale": rank.new_tensor(0.1),
        "zero_value_error": correction.detach().abs(),
    }


class Model(local_support.Model):
    def parameter_groups(self):
        groups = super().parameter_groups()
        if self.r67_mode != "slow_parent":
            return groups
        parent = {id(p) for p in self.parent_model.parameters() if p.requires_grad}
        changed = []
        for group in groups:
            pp = [p for p in group.params if id(p) in parent]
            other = [p for p in group.params if id(p) not in parent]
            if pp:
                changed.append(
                    dataclasses.replace(
                        group,
                        params=pp,
                        lr=0.5 * group.lr,
                        name=group.name if not other else group.name + "_r67_parent",
                    )
                )
            if other:
                changed.append(dataclasses.replace(group, params=other))
        return changed

    def experiment_contract(self):
        result = dict(super().experiment_contract())
        result.update(
            r67_mode=self.r67_mode,
            r67_model_factory_checkpoint_loads=0,
            r67_inference_uses_gt=False,
            r67_shared_initialization_rng_unchanged=True,
            r67_reference="R66 J07 uniform_null_s; shared parent/EST/gate/calibration initialization",
            r67_parent_lr_multiplier=0.5 if self.r67_mode == "slow_parent" else 1.0,
            r67_local_rank_parameter_gradient_multiplier=0.1
            if self.r67_mode == "local_rank_tenth"
            else 1.0,
            r67_gradient_route="local token gate and local readout parameters only; auxiliary S and all other routes unchanged",
            r67_support_objective="coverage_gap_and_pure_expansion_neutrality"
            if self.r67_mode in NEUTRAL_OBJECTIVE
            else "D07_coverage_pair",
            r67_support_readout=type(self.selector.r66_centered_s_readout).__name__,
            r67_readout_parameter_count=sum(
                p.numel() for p in self.selector.r66_centered_s_readout.parameters()
            ),
            r67_fixed_budget=24,
            r67_official_evaluation_precision="strict fp32; no TF32",
            r67_no_exact_gpu_determinism_claim=True,
        )
        return result

    def r50_adjust_terms(self, terms, outputs, batch, epoch):
        base = super().r50_adjust_terms(terms, outputs, batch, epoch)
        field = outputs.trifield_output
        rank, support, total = base.rank, base.support, base.total
        metrics, diagnostic = dict(base.metrics), {}
        parts = dict(
            reference_rank=rank, reference_support=support, reference_total=total
        )
        if self.r67_mode in NEUTRAL_OBJECTIVE:
            spans, gt_mask = losses._target_spans(outputs, batch)
            support, details = coverage_gap_objective(
                field, base.geometry, spans, gt_mask
            )
            total = total + self.loss_weights["support"] * (support - base.support)
            parts["actual_coverage_objective"] = support
            diagnostic.update(
                {"coverage/" + key: value for key, value in details.items()}
            )
            diagnostic["reference_support_loss"] = base.support.detach()
            for key in list(metrics):
                if key.startswith("support_geometry/coverage/"):
                    metrics["legacy_reference/" + key] = metrics.pop(key)
        if self.r67_mode == "local_rank_tenth":
            correction, route = rank_gradient_correction(self, rank)
            rank = rank + correction
            total = total + self.loss_weights["rank"] * correction
            parts["rank_gradient_correction"] = correction
            diagnostic.update({"route/" + key: value for key, value in route.items()})
        field.r67_loss_parts = parts
        self.r67_last_training = {
            k: v.detach() if isinstance(v, torch.Tensor) else v
            for k, v in diagnostic.items()
        }
        metrics.update(
            {"gradient_routing/" + k: v for k, v in self.r67_last_training.items()}
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


def build_model(config, *, r67_spec=None, **kwargs):
    spec = dict(r67_spec or {})
    if set(spec) != {"mode"} or spec["mode"] not in MODES:
        raise ValueError("R67 requires exactly one declared mode")
    if "r66_spec" in kwargs or "r65_spec" in kwargs:
        raise ValueError("R67 supplies its exact R66 J07 reference internally")
    model = local_support.build_model(
        config, r66_spec={"mode": "uniform_null_s"}, **kwargs
    )
    model.__class__ = Model
    model.r67_mode = spec["mode"]
    model.r67_last_training = {}
    if spec["mode"] in COVERAGE_READOUT:
        prepare_coverage_residual(model.selector)
    return model
