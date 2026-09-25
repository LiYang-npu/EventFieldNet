"""Cold-start calibration, preserving the deployed three-field score."""

import dataclasses
from unittest.mock import patch
import torch
from torch import nn
from torch.nn import functional as F
from . import candidate_pools, query_objective, joint_training
from .model import losses


def quality_weights(geo, gm, pools, balanced=False):
    y = geo.max_iou.flatten(1).float()
    union = torch.stack(list(pools.values())).any(0)
    if not balanced:
        return sum(
            m.float() / m.sum(1).clamp_min(1)[:, None] / len(pools)
            for m in pools.values()
        )
    qs = geo.all_iou.flatten(1, 2).float().masked_fill(~gm[:, None, :], -1)
    owner = qs.argmax(-1)
    pos = union & (y >= 0.5)
    neg = union & ~pos
    assign = (
        (owner[..., None] == torch.arange(gm.shape[1], device=y.device))
        & pos[..., None]
        & gm[:, None, :]
    )
    mass = assign.float() / assign.sum(1).clamp_min(1)[:, None, :]
    active = assign.any(1) & gm
    positive = (mass * active[:, None, :]).sum(-1) / active.sum(1).clamp_min(1)[:, None]
    negative = neg.float() / neg.sum(1).clamp_min(1)[:, None]
    weights = 0.5 * positive + 0.5 * negative
    return weights / weights.sum(1).clamp_min(1e-9)[:, None]


def coefficients(mode, epoch):
    if mode == "kl":
        return 1.0, 0.0
    if mode in ["kl_plus_quality", "kl_plus_balanced"]:
        return 1.0, 0.25
    if mode == "curriculum":
        alpha = max(0.0, min(1.0, (float(epoch) - 3.0) / 5.0))
        return 1.0 - alpha, alpha
    return 0.0, 1.0


class Model(joint_training.Model):
    def experiment_contract(self):
        c = super().experiment_contract()
        c.update(
            initialization="scratch_random_initialization",
            checkpoint_inheritance=False,
            optimizer_state_inherited=False,
            single_optimizer_history=True,
            end_to_end=True,
            end_to_end_scope="fixed_offline_features_to_final_prediction",
            raw_video_text_encoders_trainable=False,
            parent_weights_frozen=False,
            r58_model_factory_checkpoint_loads=0,
            r58_spec=self.r58_spec,
            r58_bias_in_deployment=False,
            r58_bias_prior_source="fixed_train_panel_only",
            r58_bias_init=self.r58_bias_init,
        )
        return c

    def parameter_groups(self):
        # Parent coverage assertion only concerns parent+selector; append one scalar here.
        groups = super().parameter_groups()
        if hasattr(self.selector, "r58_calibration_bias"):
            from training.contracts import ParameterGroup

            bias = self.selector.r58_calibration_bias
            groups = [
                dataclasses.replace(g, params=[p for p in g.params if p is not bias])
                for g in groups
            ]
            groups = [g for g in groups if g.params]
            groups.append(ParameterGroup("r58_calibration", [bias], 1e-3, 0.0))
        return groups

    def r50_adjust_terms(self, terms, outputs, batch, epoch):
        base = super().r50_adjust_terms(terms, outputs, batch, epoch)
        f = outputs.trifield_output
        g = base.geometry
        _, gm = losses._target_spans(outputs, batch)
        pools = candidate_pools.candidate_pools(f, g, gm, True)
        union = torch.stack(list(pools.values())).any(0)
        weights = quality_weights(
            g, gm, pools, self.r58_spec["mode"] in ["balanced", "kl_plus_balanced"]
        )
        y = g.max_iou.flatten(1).float()
        z = 2 * f.score.flatten(1).float()
        bias = getattr(self.selector, "r58_calibration_bias", z.new_tensor(0.0))
        calibrated = z + bias
        q = (
            F.binary_cross_entropy_with_logits(calibrated, y, reduction="none")
            * weights
        ).sum(1)
        length = 1 + self.rank_length_reweight_gain * losses._query_length_ratio(
            outputs, batch
        )
        bce = query_objective.query_mean(q * length, gm.any(1))
        restricted = dataclasses.replace(g, valid=union.reshape_as(g.valid))
        _, _, kq, active = losses.gt_balanced_kl_rank_loss(
            2 * f.score.float(), restricted, gm, return_queries=True
        )
        kl = query_objective.query_mean(kq * length, active)
        ck, cq = coefficients(self.r58_spec["mode"], epoch)
        rank = base.rank + (ck - 1.0) * kl + cq * bce
        values = {
            k: getattr(base, k)
            for k in ["evidence", "support", "transition", "endpoint"]
        }
        values["rank"] = rank
        total = sum(self.loss_weights[k] * v for k, v in values.items())
        v = f.carrier.detach().float()[g.valid]
        metrics = dict(base.metrics)
        diag = {
            "kl_coefficient": ck,
            "quality_coefficient": cq,
            "quality_bce": bce.detach(),
            "mixed_kl": kl.detach(),
            "weighted_target_mean": query_objective.query_mean(
                (weights * y).sum(1), gm.any(1)
            ).detach(),
            "weighted_probability_mean": query_objective.query_mean(
                (weights * calibrated.detach().sigmoid()).sum(1), gm.any(1)
            ).detach(),
            "weighted_brier": query_objective.query_mean(
                (weights * (calibrated.detach().sigmoid() - y).square()).sum(1),
                gm.any(1),
            ).detach(),
            "calibration_bias": bias.detach(),
            "carrier_signed_mean": v.mean(),
            "carrier_negative_saturation": (v < -0.95).float().mean(),
            "carrier_positive_saturation": (v > 0.95).float().mean(),
            "carrier_tanh_derivative_mean": (1 - v.square()).mean(),
            "bias_analytic_gradient": self.loss_weights["rank"]
            * cq
            * query_objective.query_mean(
                ((calibrated.detach().sigmoid() - y) * weights).sum(1) * length,
                gm.any(1),
            ).detach(),
        }
        metrics.update({"field_objective/" + k: v for k, v in diag.items()})
        metrics.update({k: v.detach() for k, v in values.items()})
        metrics["total"] = total.detach()
        result = dataclasses.replace(base, **values, total=total, metrics=metrics)
        self.r50_last_terms = (
            result if getattr(self, "r50_capture_terms", False) else None
        )
        return result


def build_model(config, *, r58_spec=None, r58_bias_init=0.0, **kwargs):
    spec = dict(r58_spec or {"mode": "kl"})
    assert spec["mode"] in [
        "kl",
        "source",
        "balanced",
        "curriculum",
        "kl_plus_quality",
        "kl_plus_balanced",
    ]
    assert not getattr(config, "base_checkpoint", None) and not getattr(
        config, "resume", False
    )
    assert kwargs.get("trainability") == "full_path" and not kwargs.get(
        "r54_spec", {}
    ).get("freeze_parent")
    assert not kwargs.get("r51_init_checkpoint") and not kwargs.get("r54_checkpoint")
    assert not kwargs.get("r51_spec", {}), (
        "R57 owns the quality substitution exactly once"
    )

    def forbidden_load(*args, **kw):
        raise RuntimeError("R57 hidden checkpoint load")

    with patch("torch.load", side_effect=forbidden_load):
        model = joint_training.build_model(config, **kwargs)
    model.__class__ = Model
    model.r58_spec = spec
    model.r58_bias_init = float(r58_bias_init)
    if spec.get("bias"):
        model.selector.r58_calibration_bias = nn.Parameter(
            torch.tensor(float(r58_bias_init))
        )
    return model
