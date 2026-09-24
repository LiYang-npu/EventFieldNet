"""R66 paired precision, calibrated ranking and bounded local-S hypotheses.

All arms retain the exact H01 train-only scalar calibration intercept. The
official evaluator uses FP32 for every arm; training autocast is a runtime
choice. No historical/smoke checkpoint is loaded by this factory.
"""

import dataclasses
import torch
from . import candidate_pools, quality_calibration
from .model import losses
from .carrier_helpers import carrier_input
from .precision import configure_precision, snapshot_precision
from . import local_support_helpers as s_helpers

MODES = (
    "calibration_bf16",
    "calibration_fp32",
    "calibration_softkl",
    "calibration_center",
    "calibration_center_softkl",
    "calibration_center_softkl_early",
    "coverage_consistent_s",
    "uniform_null_s",
)
CENTER = frozenset(MODES[3:])
SOFT_KL = frozenset((MODES[2], *MODES[4:]))


class Selector(quality_calibration.Selector):
    def raw_from_state(self, state, valid, carrier, video_padding_mask=None):
        original = carrier
        stats = {}
        if self.r66_mode in CENTER:
            carrier, stats = carrier_input(carrier, valid, "centered_rms")
        field = super().raw_from_state(
            state, valid, carrier, video_padding_mask=video_padding_mask
        )
        field.r65_carrier_raw = original
        field.r65_carrier_stats = stats
        if self.r66_mode == "uniform_null_s":
            field = s_helpers.apply_centered_residual(self, field, valid)
        return field


class Model(quality_calibration.Model):
    def experiment_contract(self):
        result = dict(super().experiment_contract())
        if "r65_kl_exponent" in result:
            result["r65_reference_kl_exponent"] = result.pop("r65_kl_exponent")
        result.update(
            r66_mode=self.r66_mode,
            r66_model_factory_checkpoint_loads=0,
            r66_inference_uses_gt=False,
            r66_kl_exponent=2.0 if self.r66_mode in SOFT_KL else 4.0,
            r66_centered_carrier=self.r66_mode in CENTER,
            r66_support_objective="coverage_gap_and_pure_expansion_neutrality"
            if self.r66_mode == "coverage_consistent_s"
            else "D07_coverage_pair",
            r66_support_readout="old_edge_plus_centered_local_residual"
            if self.r66_mode == "uniform_null_s"
            else "D07_local_residual",
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
        field.r66_loss_parts = {
            "reference_rank": base.rank,
            "reference_support": base.support,
            "reference_total": base.total,
        }
        if self.r66_mode in SOFT_KL:
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
                old_mixed_kl=old.detach(),
                actual_mixed_kl=new.detach(),
                mixed_kl_change=(new - old).detach(),
                kl_pool_count=union.sum(1).float().mean(),
            )
            for key in ("field_objective/mixed_kl", "candidate_pools/mixed_kl"):
                if key in metrics:
                    metrics["legacy_reference/" + key] = metrics.pop(key)
        if self.r66_mode == "coverage_consistent_s":
            spans, gt_mask = losses._target_spans(outputs, batch)
            support, details = s_helpers.coverage_gap_objective(
                field, base.geometry, spans, gt_mask
            )
            total = total + self.loss_weights["support"] * (support - base.support)
            field.r66_loss_parts["actual_coverage_objective"] = support
            diag["reference_coverage_loss"] = base.support.detach()
            diag.update(
                {
                    "coverage/" + key: value.detach()
                    if isinstance(value, torch.Tensor)
                    else value
                    for key, value in details.items()
                }
            )
            for key in list(metrics):
                if key.startswith("support_geometry/coverage/"):
                    metrics["legacy_reference/" + key] = metrics.pop(key)
        if hasattr(field, "r66_s_stats"):
            diag.update(
                {"local_s/" + key: value for key, value in field.r66_s_stats.items()}
            )
        self.r66_last_training = {
            key: value.detach() if isinstance(value, torch.Tensor) else value
            for key, value in diag.items()
        }
        metrics.update(
            {
                "local_support/" + key: value
                for key, value in self.r66_last_training.items()
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


def build_model(config, *, r66_spec=None, **kwargs):
    spec = dict(r66_spec or {})
    if set(spec) != {"mode"} or spec["mode"] not in MODES:
        raise ValueError("R66 requires exactly one declared mode")
    if "r65_spec" in kwargs:
        raise ValueError("R66 supplies its exact H01 reference internally")
    precision = getattr(config, "precision", None)
    if precision is None and isinstance(config, dict):
        precision = config.get("precision")
    receipt = configure_precision(precision)
    model = quality_calibration.build_model(
        config, r65_spec={"mode": "calibration_bias"}, **kwargs
    )
    model.__class__ = Model
    model.r66_mode = spec["mode"]
    model.r66_precision_receipt = receipt
    model.r66_last_training = {}
    model.selector.__class__ = Selector
    model.selector.r66_mode = spec["mode"]
    if spec["mode"] == "uniform_null_s":
        s_helpers.prepare_centered_residual(model.selector)
    current = snapshot_precision()
    assert not current["matmul_allow_tf32"] and not current["cudnn_allow_tf32"]
    assert current["float32_matmul_precision"] == "highest"
    return model
