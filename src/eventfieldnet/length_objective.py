"""S coverage, duration bias and optimization diagnostics; GT only in losses."""

import dataclasses
import torch
from torch.nn import functional as F
from field_core.adapter import _batch_parts, _metadata_rows
from . import support_geometry
from .model.losses import _target_spans

MODES = (
    "control",
    "half_s",
    "width_center",
    "quality_calibration",
    "nested_balance",
    "utility",
    "aux_s_encoder_route",
    "early_decay",
)


def durations(batch, like):
    rows = _metadata_rows(_batch_parts(batch)[2], batch.batch_size)
    return like.new_tensor([float((x or {}).get("duration") or 0.0) for x in rows])


def mean(value, mask):
    return (value * mask).sum() / mask.sum().clamp_min(1)


def center_width(s, valid):
    b, l, _ = s.shape
    ix = torch.arange(l, device=s.device)
    widths = (ix[None, :] - ix[:, None]).clamp_min(0).flatten()[None].expand(b, -1)
    vf = valid.flatten(1).float()
    sf = s.float().flatten(1)
    sums = sf.new_zeros(b, l).scatter_add(1, widths, sf * vf)
    counts = sf.new_zeros(b, l).scatter_add(1, widths, vf)
    mu = (sums / counts.clamp_min(1)).gather(1, widths)
    return (0.5 * (sf - mu) * vf).reshape_as(s), mu.reshape_as(s)


def balanced_gt(value, mask, seconds):
    # Equal occupied duration buckets within a query, then equal active queries.
    bucket = (
        torch.stack((seconds <= 10, (seconds > 10) & (seconds <= 30), seconds > 30), -1)
        & mask[..., None]
    )
    counts = bucket.sum(1)
    values = (value[..., None] * bucket).sum(1) / counts.clamp_min(1)
    active = counts > 0
    perq = (values * active).sum(1) / active.sum(1).clamp_min(1)
    return mean(perq, active.any(1)), {
        label: mean(value, bucket[..., k])
        for k, label in enumerate(["short", "middle", "long"])
    }


def nested_objective(field, geo, spans, gm, batch, utility=False):
    b, l, _ = field.support.shape
    g = gm.shape[1]
    valid = geo.valid.flatten(1)
    iou = geo.all_iou.float().flatten(1, 2)
    best = iou.masked_fill(~valid[..., None], -1).argmax(1)
    pquality = iou.gather(1, best[:, None]).squeeze(1)
    a, z = best // l, best % l
    step = ((z - a + 1) // 4).clamp_min(1)
    na = torch.stack((a + step, a, a - step, a), -1)
    nz = torch.stack((z, z - step, z, z + step), -1)
    bounds = (na >= 0) & (nz < l) & (na <= nz)
    ids = na.clamp(0, l - 1) * l + nz.clamp(0, l - 1)
    flat = ids.flatten(1)
    nq = geo.max_iou.float().flatten(1).gather(1, flat).reshape(b, g, 4)
    delta = pquality[..., None] - nq
    mask = (
        bounds
        & gm[..., None]
        & valid.gather(1, flat).reshape(b, g, 4)
        & (pquality[..., None] >= 0.7)
        & (delta >= 0.1)
    )
    s = field.support.float().flatten(1)
    sgap = s.gather(1, best)[..., None] - s.gather(1, flat).reshape(b, g, 4)
    # Counterfactual reference uses no S and is detached: E/T/carrier cannot
    # satisfy this auxiliary objective in place of S.
    base = (
        (
            field.carrier.float()
            + field.evidence.float()
            + 0.5 * (field.transition_start.float() + field.transition_end.float())
        )
        .detach()
        .flatten(1)
    )
    bgap = base.gather(1, best)[..., None] - base.gather(1, flat).reshape(b, g, 4)
    margin = 0.2 * delta
    gap = sgap + bgap if utility else sgap
    hinge = F.relu(margin - gap)
    pergt = (hinge * mask).sum(-1) / mask.sum(-1).clamp_min(1)
    seconds = (spans[..., 1] - spans[..., 0]) * durations(batch, spans)[:, None]
    active = mask.any(-1) & gm
    value, parts = balanced_gt(pergt, active, seconds)
    stats = {
        "pairs": mask.sum(),
        "truncation_pairs": mask[..., :2].sum(),
        "expansion_pairs": mask[..., 2:].sum(),
        "loss": value,
        "s_gap": mean(sgap, mask),
        "base_gap": mean(bgap, mask),
        "baseline_violation": mean((bgap < margin).float(), mask),
        "with_s_violation": mean((bgap + sgap < margin).float(), mask),
        "quality_delta": mean(delta, mask),
        "active_gt": active.sum(),
    }
    for k, label in enumerate(["short", "middle", "long"]):
        bucket = (
            (seconds <= 10)
            if k == 0
            else ((seconds > 10) & (seconds <= 30))
            if k == 1
            else (seconds > 30)
        )
        m = mask & bucket[..., None]
        stats.update(
            {
                label + "_loss": parts[label],
                label + "_pairs": m.sum(),
                label + "_s_gap": mean(sgap, m),
                label + "_utility": mean(
                    ((bgap < margin).float() - (bgap + sgap < margin).float()), m
                ),
            }
        )
    return value, stats


def calibration_objective(field, geo, batch):
    valid = geo.valid
    target = geo.max_iou.detach().float()
    pred = (field.support.float() + 1) * 0.5
    error = (pred - target).square()
    sec = (geo.candidate_end - geo.candidate_start) * durations(batch, target)[
        :, None, None
    ]
    lengths = [sec <= 10, (sec > 10) & (sec <= 30), sec > 30]
    qualities = [target < 0.3, (target >= 0.3) & (target < 0.7), target >= 0.7]
    values = []
    active = []
    stats = {}
    for li, lmask in enumerate(lengths):
        stats[["short", "middle", "long"][li] + "_brier"] = mean(error, valid & lmask)
        for qi, qmask in enumerate(qualities):
            mask = valid & lmask & qmask
            count = mask.sum((1, 2))
            values.append((error * mask).sum((1, 2)) / count.clamp_min(1))
            active.append(count > 0)
    values = torch.stack(values, 1)
    active = torch.stack(active, 1)
    result = mean((values * active).sum(1) / active.sum(1).clamp_min(1), active.any(1))
    stats.update(loss=result, active_cells=active.sum(), brier=mean(error, valid))
    return result, stats


class Selector(support_geometry.Selector):
    def raw_from_state(self, *args, **kwargs):
        field = super().raw_from_state(*args, **kwargs)
        valid = args[1] if len(args) > 1 else kwargs["valid"]
        original = field.support
        if self.r63_mode == "half_s":
            field.support = 0.5 * original
            field.r59_old_support = 0.5 * field.r59_old_support
            field.r59_support_details["delta"] = (
                0.5 * field.r59_support_details["delta"]
            )
        elif self.r63_mode == "width_center":
            field.support, mu = center_width(original, valid)
            field.r59_old_support, _ = center_width(field.r59_old_support, valid)
            field.r59_support_details["delta"] = field.support - field.r59_old_support
        field.r63_original_support = original
        field.r63_delta = field.support - original
        field.score = self.compose_score(field)
        field.round31_rank_score = field.score
        field.round31_score_with_support = field.score
        field.round31_score_without_support = self.compose_score(
            field, {"support": torch.zeros_like(field.support)}
        )
        field.round31_deployed_support = field.support
        return field


class Model(support_geometry.Model):
    def experiment_contract(self):
        x = super().experiment_contract()
        x.update(
            r63_mode=self.r63_mode,
            r63_inference_uses_gt=False,
            r63_no_new_parameters=True,
            r63_aux_s_encoder_route=self.r63_mode == "aux_s_encoder_route",
        )
        return x

    def r50_adjust_terms(self, terms, outputs, batch, epoch):
        base = super().r50_adjust_terms(terms, outputs, batch, epoch)
        metrics = dict(base.metrics)
        field = outputs.trifield_output
        metrics["length_objective/s_transform_abs"] = mean(
            field.r63_delta.abs(), base.geometry.valid
        ).detach()
        support = base.support
        diag = {}
        if self.r63_mode == "quality_calibration":
            support, diag = calibration_objective(field, base.geometry, batch)
        elif self.r63_mode in ["nested_balance", "utility"]:
            spans, gm = _target_spans(outputs, batch)
            support, diag = nested_objective(
                field, base.geometry, spans, gm, batch, self.r63_mode == "utility"
            )
        if self.r63_mode in ["quality_calibration", "nested_balance", "utility"]:
            for k in list(metrics):
                if k.startswith("support_geometry/coverage/"):
                    metrics["legacy_reference/" + k] = metrics.pop(k)
            metrics["length_objective/old_support_loss"] = base.support.detach()
        total = base.total + self.loss_weights["support"] * (support - base.support)
        correction = total.new_zeros(())
        if self.r63_mode == "aux_s_encoder_route" and torch.is_grad_enabled():
            parameters = [
                p for p in self.selector.s_projection.parameters() if p.requires_grad
            ]
            weighted = self.loss_weights["support"] * support
            grads = (
                torch.autograd.grad(
                    weighted, parameters, retain_graph=True, allow_unused=True
                )
                if weighted.requires_grad
                else [None] * len(parameters)
            )
            norm = total.new_zeros(())
            connected = 0
            for p, g in zip(parameters, grads):
                if g is not None:
                    correction = correction - ((p - p.detach()) * g.detach()).sum()
                    norm = norm + g.detach().float().square().sum()
                    connected += 1
            total = total + correction
            metrics["length_objective/route_raw_s_encoder_aux_norm"] = norm.sqrt()
            metrics["length_objective/route_connected"] = float(connected)
        metrics.update({"length_objective/" + k: v.detach() for k, v in diag.items()})
        metrics.update(support=support.detach(), total=total.detach())
        metrics["length_objective/route_numeric_change"] = correction.detach().abs()
        result = dataclasses.replace(
            base, support=support, total=total, metrics=metrics
        )
        self.r50_last_terms = (
            result if getattr(self, "r50_capture_terms", False) else None
        )
        return result


def build_model(config, *, r63_spec=None, **kwargs):
    spec = dict(r63_spec or {})
    assert set(spec) == {"mode"} and spec["mode"] in MODES
    assert kwargs.get("r59_spec") == {"mode": "local_s_coverage"}
    m = support_geometry.build_model(config, **kwargs)
    m.__class__ = Model
    m.r63_mode = spec["mode"]
    m.selector.__class__ = Selector
    m.selector.r63_mode = spec["mode"]
    return m
