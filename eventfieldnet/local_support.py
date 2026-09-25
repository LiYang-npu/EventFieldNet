"""Centered carrier scores, softened KL targets, and local support readout."""

import dataclasses
import torch

from . import candidate_pools, quality_calibration
from .model import losses
from .carrier_helpers import carrier_input
from .precision import configure_precision, snapshot_precision
from . import local_support_helpers as s_helpers


class Selector(quality_calibration.Selector):
    def raw_from_state(self, state, valid, carrier, video_padding_mask=None):
        original = carrier
        carrier, stats = carrier_input(carrier, valid, "centered_rms")
        field = super().raw_from_state(
            state, valid, carrier, video_padding_mask=video_padding_mask
        )
        field.r65_carrier_raw = original
        field.r65_carrier_stats = stats
        return s_helpers.apply_centered_residual(self, field, valid)


class Model(quality_calibration.Model):
    def experiment_contract(self):
        result = dict(super().experiment_contract())
        if "r65_kl_exponent" in result:
            result["r65_reference_kl_exponent"] = result.pop("r65_kl_exponent")
        result.update(
            r66_mode="uniform_null_s", r66_model_factory_checkpoint_loads=0,
            r66_inference_uses_gt=False, r66_kl_exponent=2.0,
            r66_centered_carrier=True, r66_support_objective="D07_coverage_pair",
            r66_support_readout="old_edge_plus_centered_local_residual",
            r66_backend_policy=snapshot_precision(),
            r66_factory_precision_policy=self.r66_precision_receipt,
            r66_official_evaluation_precision="fp32; no TF32",
            r66_no_exact_gpu_determinism_claim=True,
            r66_shared_initialization_rng_unchanged=True,
        )
        return result

    def r50_adjust_terms(self, terms, outputs, batch, epoch):
        base = super().r50_adjust_terms(terms, outputs, batch, epoch)
        field = outputs.trifield_output
        rank, support, total = base.rank, base.support, base.total
        metrics, diag = dict(base.metrics), {}
        field.r66_loss_parts = dict(
            reference_rank=base.rank, reference_support=base.support,
            reference_total=base.total,
        )
        _, gt_mask = losses._target_spans(outputs, batch)
        pools = candidate_pools.candidate_pools(field, base.geometry, gt_mask, True)
        union = torch.stack(list(pools.values())).any(0)
        old = quality_calibration.mixed_kl(
            self, outputs, batch, base.geometry, gt_mask, union, 4.0
        )
        new = quality_calibration.mixed_kl(
            self, outputs, batch, base.geometry, gt_mask, union, 2.0
        )
        rank = rank + (new - old)
        total = total + self.loss_weights["rank"] * (new - old)
        field.r66_loss_parts.update(old_mixed_kl=old, actual_mixed_kl=new)
        diag.update(
            old_mixed_kl=old.detach(), actual_mixed_kl=new.detach(),
            mixed_kl_change=(new - old).detach(),
            kl_pool_count=union.sum(1).float().mean(),
        )
        for key in ("field_objective/mixed_kl", "candidate_pools/mixed_kl"):
            if key in metrics:
                metrics["legacy_reference/" + key] = metrics.pop(key)
        if hasattr(field, "r66_s_stats"):
            diag.update({"local_s/" + key: value for key, value in field.r66_s_stats.items()})
        self.r66_last_training = {
            key: value.detach() if isinstance(value, torch.Tensor) else value
            for key, value in diag.items()
        }
        metrics.update({"local_support/" + key: value for key, value in self.r66_last_training.items()})
        metrics.update(rank=rank.detach(), support=support.detach(), total=total.detach())
        result = dataclasses.replace(
            base, rank=rank, support=support, total=total, metrics=metrics
        )
        self.r50_last_terms = result if getattr(self, "r50_capture_terms", False) else None
        return result


def build_model(config, *, r66_spec=None, **kwargs):
    if r66_spec != {"mode": "uniform_null_s"}:
        raise ValueError("This model requires centered local support")
    if "r65_spec" in kwargs:
        raise ValueError("The quality calibration mode is fixed internally")
    precision = getattr(config, "precision", None)
    if precision is None and isinstance(config, dict):
        precision = config.get("precision")
    receipt = configure_precision(precision)
    model = quality_calibration.build_model(
        config, r65_spec={"mode": "calibration_bias"}, **kwargs
    )
    model.__class__ = Model
    model.r66_mode = "uniform_null_s"
    model.r66_precision_receipt = receipt
    model.r66_last_training = {}
    model.selector.__class__ = Selector
    model.selector.r66_mode = "uniform_null_s"
    s_helpers.prepare_centered_residual(model.selector)
    current = snapshot_precision()
    assert not current["matmul_allow_tf32"] and not current["cudnn_allow_tf32"]
    assert current["float32_matmul_precision"] == "highest"
    return model
