"""R65: paired seed robustness mechanisms on the exact D07 reference.

All labels enter losses only. Controls retain the original computation;
treatments replace one actual objective/readout or route, never just a metric.
"""

import dataclasses
import torch
from . import candidate_pools, query_objective, support_geometry
from .model import losses
from . import carrier_helpers as carrier_helpers
from . import support_helpers as s_helpers

MODES = (
    "control",
    "calibration_bias",
    "centered_carrier",
    "score_independent_kl",
    "soft_kl",
    "direct_completeness_s",
    "direct_quality_s",
    "protect_s_gradient",
)


def mixed_kl(model, outputs, batch, geometry, gt_mask, pool, exponent=4.0):
    restricted = dataclasses.replace(geometry, valid=pool.reshape_as(geometry.valid))
    _, _, query, active = losses.gt_balanced_kl_rank_loss(
        2.0 * outputs.trifield_output.score.float(),
        restricted,
        gt_mask,
        rank_target_threshold=0.7,
        rank_target_exponent=exponent,
        return_queries=True,
    )
    length = 1.0 + model.rank_length_reweight_gain * losses._query_length_ratio(
        outputs, batch
    )
    return query_objective.query_mean(query * length, active)


def protect_s_encoder(model, rank, support, total):
    """Project only conflicting weighted rank gradient on the S encoder.

    A zero-valued surrogate adds the first-order correction to the actual
    backward. Other modules/loss paths remain intact; no second-order graph.
    """
    params = [p for p in model.selector.s_projection.parameters() if p.requires_grad]

    def grad(value):
        return (
            torch.autograd.grad(value, params, retain_graph=True, allow_unused=True)
            if value.requires_grad
            else [None] * len(params)
        )

    gr = grad(model.loss_weights["rank"] * rank)
    gs = grad(model.loss_weights["support"] * support)
    zero = total.new_zeros((), dtype=torch.float32)
    norm_r = sum((g.detach().float().square().sum() for g in gr if g is not None), zero)
    norm_s = sum((g.detach().float().square().sum() for g in gs if g is not None), zero)
    dot = sum(
        (
            (a.detach().float() * b.detach().float()).sum()
            for a, b in zip(gr, gs)
            if a is not None and b is not None
        ),
        zero,
    )
    coefficient = torch.where(
        norm_s > 0.0, dot.clamp_max(0.0) / norm_s.clamp_min(1e-30), zero
    )
    correction = total.new_zeros(())
    correction_norm = zero
    for p, g in zip(params, gs):
        if g is not None:
            change = -coefficient * g.detach().float()
            correction = correction + ((p - p.detach()).float() * change).sum()
            correction_norm = correction_norm + change.square().sum()
    stats = {
        "route_rank_norm": norm_r.sqrt(),
        "route_aux_norm": norm_s.sqrt(),
        "route_dot": dot,
        "route_cosine": dot / (norm_r * norm_s).sqrt().clamp_min(1e-30),
        "route_active": ((dot < 0.0) & (norm_s > 0.0)).float(),
        "route_correction_norm": correction_norm.sqrt(),
        "route_post_dot": dot - coefficient * norm_s,
        "route_numeric_change": correction.detach().abs(),
    }
    return total + correction, stats


class Selector(support_geometry.Selector):
    def raw_from_state(self, state, valid, carrier, video_padding_mask=None):
        original = carrier
        stats = {}
        if self.r65_mode == "centered_carrier":
            carrier, stats = carrier_helpers.carrier_input(
                carrier, valid, "centered_rms"
            )
        field = super().raw_from_state(
            state, valid, carrier, video_padding_mask=video_padding_mask
        )
        field.r65_carrier_raw = original
        field.r65_carrier_stats = stats
        if self.r65_mode in ("direct_completeness_s", "direct_quality_s"):
            field = s_helpers.apply_support(self, field, valid, video_padding_mask)
        return field


class Model(support_geometry.Model):
    def experiment_contract(self):
        result = dict(super().experiment_contract())
        result.update(
            r65_mode=self.r65_mode,
            r65_inference_uses_gt=False,
            r65_model_factory_checkpoint_loads=0,
            r65_shared_initialization_rng_unchanged=True,
            r65_score_independent_pool="stratified+gt_neighborhood; KL only",
            r65_kl_exponent=2.0 if self.r65_mode == "soft_kl" else 4.0,
            r65_gradient_scope="S encoder only; conflicting rank projection"
            if self.r65_mode == "protect_s_gradient"
            else "original full path",
            r65_replaced_edge_head=self.r65_mode
            in ("direct_completeness_s", "direct_quality_s"),
            r65_anchor_source="carrier+E+T without old S; detached; GT-free; audit-independent"
            if self.r65_mode in ("direct_completeness_s", "direct_quality_s")
            else result.get("r59_anchor_source"),
        )
        if self.r65_mode == "calibration_bias":
            result["r58_bias_prior_source"] = (
                "zero initialized learned train-only intercept"
            )
        return result

    def r50_adjust_terms(self, terms, outputs, batch, epoch):
        base = super().r50_adjust_terms(terms, outputs, batch, epoch)
        field = outputs.trifield_output
        metrics = dict(base.metrics)
        rank, support = base.rank, base.support
        total = base.total
        diag = {}
        if self.r65_mode in ("score_independent_kl", "soft_kl"):
            _, gt_mask = losses._target_spans(outputs, batch)
            pools = candidate_pools.candidate_pools(field, base.geometry, gt_mask, True)
            old_pool = torch.stack(list(pools.values())).any(0)
            pool = (
                pools["stratified"] | pools["gt_neighborhood"]
                if self.r65_mode == "score_independent_kl"
                else old_pool
            )
            old = mixed_kl(self, outputs, batch, base.geometry, gt_mask, old_pool)
            new = mixed_kl(
                self,
                outputs,
                batch,
                base.geometry,
                gt_mask,
                pool,
                2.0 if self.r65_mode == "soft_kl" else 4.0,
            )
            # Exactly R50 mixed KL is substituted; BCE/pair/length terms survive.
            rank = base.rank + (new - old)
            total = total + self.loss_weights["rank"] * (new - old)
            diag.update(
                old_mixed_kl=old.detach(),
                new_mixed_kl=new.detach(),
                kl_pool_count=pool.sum(1).float().mean(),
                old_kl_pool_count=old_pool.sum(1).float().mean(),
                kl_actual_change=(new - old).detach(),
            )
            for name in ("field_objective/mixed_kl", "candidate_pools/mixed_kl"):
                if name in metrics:
                    metrics["legacy_reference/" + name] = metrics.pop(name)
            metrics["quality_calibration/actual_mixed_kl"] = new.detach()
        if self.r65_mode == "direct_quality_s":
            support, quality_stats = s_helpers.direct_quality_objective(
                field, base.geometry
            )
            total = total + self.loss_weights["support"] * (support - base.support)
            diag.update({"quality/" + k: v for k, v in quality_stats.items()})
            for name in list(metrics):
                if name.startswith("support_geometry/coverage/"):
                    metrics["legacy_reference/" + name] = metrics.pop(name)
            diag["legacy_coverage_loss"] = base.support.detach()
        if self.r65_mode == "protect_s_gradient" and torch.is_grad_enabled():
            total, route = protect_s_encoder(self, rank, support, total)
            diag.update(route)
        self.r65_last_training = {
            "mode": self.r65_mode,
            **{k: v.detach() for k, v in diag.items()},
        }
        metrics.update(
            {"quality_calibration/" + k: v.detach() for k, v in diag.items()}
        )
        metrics.update(
            {
                "quality_calibration/carrier/" + k: v.float().mean().detach()
                for k, v in field.r65_carrier_stats.items()
            }
        )
        if hasattr(field, "r65_s_stats"):
            metrics.update(
                {
                    "quality_calibration/s/" + k: v.detach()
                    for k, v in field.r65_s_stats.items()
                }
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


def build_model(config, *, r65_spec=None, **kwargs):
    spec = dict(r65_spec or {})
    if set(spec) != {"mode"} or spec["mode"] not in MODES:
        raise ValueError("R65 requires exactly one known mode")
    if kwargs.get("r59_spec") != {"mode": "local_s_coverage"}:
        raise ValueError("R65 reference must be D07 local_s_coverage")
    model = support_geometry.build_model(config, **kwargs)
    model.__class__ = Model
    model.r65_mode = spec["mode"]
    model.r65_last_training = {}
    model.selector.__class__ = Selector
    model.selector.r65_mode = spec["mode"]
    if spec["mode"] == "calibration_bias":
        carrier_helpers.install_quality_bias(model, 0.0)
    elif spec["mode"] in ("direct_completeness_s", "direct_quality_s"):
        s_helpers.prepare_selector(
            model.selector,
            "content_completeness"
            if spec["mode"] == "direct_completeness_s"
            else "direct_quality",
        )
    return model
