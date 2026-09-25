"""Train-only quality calibration and the fixed mixed-KL helper."""

import dataclasses

from . import carrier_helpers, query_objective, support_geometry
from .model import losses


def mixed_kl(model, outputs, batch, geometry, gt_mask, pool, exponent=4.0):
    restricted = dataclasses.replace(geometry, valid=pool.reshape_as(geometry.valid))
    _, _, query, active = losses.gt_balanced_kl_rank_loss(
        2.0 * outputs.trifield_output.score.float(), restricted, gt_mask,
        rank_target_threshold=0.7, rank_target_exponent=exponent,
        return_queries=True,
    )
    length = 1.0 + model.rank_length_reweight_gain * losses._query_length_ratio(
        outputs, batch
    )
    return query_objective.query_mean(query * length, active)


class Selector(support_geometry.Selector):
    def raw_from_state(self, state, valid, carrier, video_padding_mask=None):
        original = carrier
        field = super().raw_from_state(
            state, valid, carrier, video_padding_mask=video_padding_mask
        )
        field.r65_carrier_raw = original
        field.r65_carrier_stats = {}
        return field


class Model(support_geometry.Model):
    def experiment_contract(self):
        result = dict(super().experiment_contract())
        result.update(
            r65_mode="calibration_bias", r65_inference_uses_gt=False,
            r65_model_factory_checkpoint_loads=0,
            r65_shared_initialization_rng_unchanged=True,
            r65_score_independent_pool="stratified+gt_neighborhood; KL only",
            r65_kl_exponent=4.0, r65_gradient_scope="original full path",
            r65_replaced_edge_head=False,
            r65_anchor_source=result.get("r59_anchor_source"),
            r58_bias_prior_source="zero initialized learned train-only intercept",
        )
        return result

    def r50_adjust_terms(self, terms, outputs, batch, epoch):
        base = super().r50_adjust_terms(terms, outputs, batch, epoch)
        field = outputs.trifield_output
        metrics = dict(base.metrics)
        rank, support, total = base.rank, base.support, base.total
        self.r65_last_training = {"mode": "calibration_bias"}
        metrics.update({
            "quality_calibration/carrier/" + key: value.float().mean().detach()
            for key, value in field.r65_carrier_stats.items()
        })
        metrics.update(rank=rank.detach(), support=support.detach(), total=total.detach())
        result = dataclasses.replace(
            base, rank=rank, support=support, total=total, metrics=metrics
        )
        self.r50_last_terms = result if getattr(self, "r50_capture_terms", False) else None
        return result


def build_model(config, *, r65_spec=None, **kwargs):
    if r65_spec != {"mode": "calibration_bias"}:
        raise ValueError("This model requires train-only quality calibration")
    if kwargs.get("r59_spec") != {"mode": "local_s_coverage"}:
        raise ValueError("This model requires local support coverage")
    model = support_geometry.build_model(config, **kwargs)
    model.__class__ = Model
    model.r65_mode = "calibration_bias"
    model.r65_last_training = {}
    model.selector.__class__ = Selector
    model.selector.r65_mode = "calibration_bias"
    carrier_helpers.install_quality_bias(model, 0.0)
    return model
