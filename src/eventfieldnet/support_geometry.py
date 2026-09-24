"""R59 bounded mechanisms on the exact R58 KL + .25 quality control.

This is a model wrapper, not a launcher. All forward anchors use detached,
pre-R59 base-F; neither targets nor post-ablation scores select anchors.
Parent freezing starts at epoch 9 using leaf-gradient clearing, preserving
upstream gradients, dropout mode, and registered optimizer membership.
"""

import dataclasses
import weakref

import torch
from torch import nn
from torch.nn import functional as F

from . import candidate_pools, query_objective, support_objective, field_objective
from .model import losses


MODES = (
    "control",
    "freeze_parent",
    "t_only",
    "quality_nms10",
    "precise_pair",
    "shared_e",
    "local_s",
    "local_s_coverage",
)


def masked_mean(value, mask):
    mask = mask.bool()
    return (value * mask).sum() / mask.sum().clamp_min(1)


@torch.no_grad()
def select_anchors(base_score, valid, k=8, nms_threshold=0.5):
    """Greedy, stable-first-index NMS on the complete pre-treatment grid."""
    batch, length, _ = base_score.shape
    flat = base_score.detach().float().flatten(1)
    valid = valid.bool().flatten(1)
    idx = torch.arange(length * length, device=flat.device)
    start = (idx // length).float()
    end = (idx % length + 1).float()
    alive = valid.clone()
    picks, masks, tie_counts = [], [], []
    for _ in range(min(k, flat.shape[1])):
        active = alive.any(1)
        pick = flat.masked_fill(~alive, -torch.inf).argmax(1)
        picks.append(pick)
        masks.append(active)
        chosen_score = flat.gather(1, pick[:, None])
        tie_counts.append(((flat == chosen_score) & alive).sum(1))
        inter = (
            torch.minimum(end[None], end[pick, None])
            - torch.maximum(start[None], start[pick, None])
        ).clamp_min(0)
        union = (end - start)[None] + (end[pick] - start[pick])[:, None] - inter
        overlap = inter / union.clamp_min(1e-8)
        alive &= (overlap <= nms_threshold) & active[:, None]
    ids = torch.stack(picks, 1)
    active = torch.stack(masks, 1)
    a, b = start[ids], end[ids]
    inter = (
        torch.minimum(end[None, :, None], b[:, None])
        - torch.maximum(start[None, :, None], a[:, None])
    ).clamp_min(0)
    union = (end - start)[None, :, None] + (b - a)[:, None] - inter
    iou = (inter / union.clamp_min(1e-8)).masked_fill(
        ~(valid[..., None] & active[:, None]), 0.0
    )
    best_iou, assignment = iou.max(-1)
    return {
        "ids": ids,
        "active": active,
        "tie_counts": torch.stack(tie_counts, 1),
        "iou": iou,
        "assignment": assignment,
        "best_iou": best_iou,
    }


def shared_evidence(evidence, valid, anchors):
    flat = evidence.float().flatten(1)
    weights = anchors["iou"].pow(4) * (anchors["iou"] >= 0.5)
    den = weights.sum(-1, keepdim=True)
    weights = weights / den.clamp_min(1e-12)
    anchor_values = flat.gather(1, anchors["ids"])
    shared = (weights * anchor_values[:, None]).sum(-1)
    shared = torch.where(den[..., 0] > 0, shared, flat)
    after = (0.5 * flat + 0.5 * shared).reshape_as(evidence).masked_fill(~valid, 0.0)
    stats = {
        "matched_fraction": masked_mean((den[..., 0] > 0).float(), valid.flatten(1)),
        "delta_abs": masked_mean((after - evidence).abs(), valid),
    }
    return after, shared.reshape_as(evidence), stats


class LocalSupport(nn.Module):
    """Local content coverage/density, centered on the exact uniform gate.

    The 64-vector is zero-initialized without consuming the RNG. Subtracting
    one valid token's gate logit makes any spatially uniform gate exactly
    zero before exp, so uniform-gate no-op does not rely on FP tolerance.
    R is a local anchor region; B is its outer ring, never the whole video.
    """

    def __init__(self, dim=64):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, h, support, valid, token_valid, anchors):
        batch, length, dim = h.shape
        with torch.autocast(device_type=h.device.type, enabled=False):
            z = F.layer_norm(h.float(), (dim,))
            logits = F.linear(z, self.weight.float()[None]).squeeze(-1)
            first = token_valid.long().argmax(1, keepdim=True)
            logits = (logits - logits.gather(1, first)).clamp(-8.0, 8.0)
            mass = logits.exp() * token_valid.float()
            mass_prefix = F.pad(mass.cumsum(1), (1, 0))
            count_prefix = F.pad(token_valid.float().cumsum(1), (1, 0))
            idx = torch.arange(length * length, device=h.device)
            start = (idx // length)[None].expand(batch, -1)
            end = torch.maximum(idx % length + 1, idx // length + 1)[None].expand(
                batch, -1
            )
            ids = anchors["ids"].gather(1, anchors["assignment"])
            a, b = ids // length, ids % length + 1
            width = (b - a).clamp_min(1)
            radius = (width // 4).clamp_min(1)
            outer_radius = torch.maximum(width // 2, radius + 1)
            rs, re = (a - radius).clamp_min(0), (b + radius).clamp_max(length)
            os, oe = (
                (a - outer_radius).clamp_min(0),
                (b + outer_radius).clamp_max(length),
            )
            xs = torch.maximum(start, rs)
            xe = torch.maximum(xs, torch.minimum(end, re))

            def integral(prefix, left, right):
                return prefix.gather(1, right) - prefix.gather(1, left)

            reference_mass = integral(mass_prefix, rs, re)
            reference_count = integral(count_prefix, rs, re)
            inside_mass = integral(mass_prefix, xs, xe)
            inside_count = integral(count_prefix, xs, xe)
            window_mass = integral(mass_prefix, start, end)
            window_count = integral(count_prefix, start, end)
            background_mass = integral(mass_prefix, os, rs) + integral(
                mass_prefix, re, oe
            )
            background_count = integral(count_prefix, os, rs) + integral(
                count_prefix, re, oe
            )
            coverage = inside_mass / reference_mass.clamp_min(1e-8)
            window_density = window_mass / window_count.clamp_min(1)
            background_density = background_mass / background_count.clamp_min(1)
            purity = window_density / (window_density + background_density).clamp_min(
                1e-8
            )
            quality = coverage * purity
            uniform = 0.5 * (inside_count / reference_count.clamp_min(1))
            eligible = (
                valid.flatten(1)
                & (anchors["best_iou"] >= 0.3)
                & (background_count > 0)
                & (reference_count > 0)
                & (window_count > 0)
            )
            residual = (quality - uniform).masked_fill(~eligible, 0.0)
            old = support.float().flatten(1)
            delta = 0.5 * (1.0 - old.abs()).clamp_min(0.0) * residual
            after = (old + delta).reshape_as(support).masked_fill(~valid, 0.0)
            details = {
                "coverage": coverage.reshape_as(support),
                "density_ratio": purity.reshape_as(support),
                "quality": quality.reshape_as(support),
                "uniform": uniform.reshape_as(support),
                "eligible": eligible.reshape_as(valid),
                "delta": delta.reshape_as(support),
                "token_mass": mass,
            }
            stats = {
                "eligible_fraction": masked_mean(eligible.float(), valid.flatten(1)),
                "delta_abs": masked_mean(delta.abs(), valid.flatten(1)),
                "delta_signed": masked_mean(delta, valid.flatten(1)),
                "coverage": masked_mean(coverage, eligible),
                "density_ratio": masked_mean(purity, eligible),
                "quality": masked_mean(quality, eligible),
                "uniform_quality": masked_mean(uniform, eligible),
                "gate_weight_norm": self.weight.float().norm(),
                "gate_logit_abs": masked_mean(logits.abs(), token_valid),
            }
            return after, details, stats


def precise_pair_loss(score, geometry, gt_mask):
    """Replace old pair with precise full-grid positives vs current raw30."""
    flat = score.float().flatten(1)
    valid = geometry.valid.bool().flatten(1)
    all_iou = geometry.all_iou.float().flatten(1, 2)
    max_iou = geometry.max_iou.float().flatten(1)
    best = all_iou.masked_fill(~valid[..., None], -1.0).argmax(1)
    positive_quality = all_iou.gather(1, best[:, None]).squeeze(1)
    positive_score = flat.gather(1, best)
    raw = candidate_pools.top_mask(flat, valid, 30)
    bad = (
        raw[..., None]
        & gt_mask[:, None]
        & (positive_quality[:, None] >= 0.90)
        & (max_iou[..., None] <= positive_quality[:, None] - 0.02)
    )
    expanded_score = flat[..., None].expand_as(all_iou)
    ids = (
        expanded_score.detach()
        .masked_fill(~bad, -torch.inf)
        .topk(min(8, flat.shape[1]), dim=1)
        .indices
    )
    mask = bad.gather(1, ids)
    negative_score = expanded_score.gather(1, ids)
    negative_quality = max_iou[..., None].expand_as(all_iou).gather(1, ids)
    gap = positive_score[:, None] - negative_score
    margin = 0.5 * (positive_quality[:, None] - negative_quality).clamp(0.02, 0.60)
    hinge = F.relu(margin - gap)
    per_gt = (hinge * mask).sum(1) / mask.sum(1).clamp_min(1)
    active_gt = mask.any(1) & gt_mask
    per_query = (per_gt * active_gt).sum(1) / active_gt.sum(1).clamp_min(1)
    value = query_objective.query_mean(per_query, active_gt.any(1))
    stats = {
        "loss": value,
        "pairs": mask.sum(),
        "active_queries": active_gt.any(1).sum(),
        "active_gt_fraction": masked_mean(active_gt.float(), gt_mask),
        "grid_oracle90": masked_mean((positive_quality >= 0.90).float(), gt_mask),
        "grid_oracle95": masked_mean((positive_quality >= 0.95).float(), gt_mask),
        "positive_in_raw30": masked_mean(raw.gather(1, best).float(), gt_mask),
        "positive_gap": masked_mean(gap, mask),
        "margin": masked_mean(margin, mask),
        "violation_fraction": masked_mean((hinge > 0).float(), mask),
    }
    return value, stats


def coverage_pair_loss(support, geometry, spans, gt_mask):
    """Pure, nested truncations; no loss on a pure outward expansion.

    Positive and negative purity are measured against the same GT. Other-GT
    near-positive negatives are excluded. Integer truncation is one quarter
    of the chosen positive's width (minimum one clip), with all final
    coverage/purity tests using the actual normalized candidate coordinates.
    """
    batch, length, _ = support.shape
    count_gt = gt_mask.shape[1]
    valid = geometry.valid.bool().flatten(1)
    iou = geometry.all_iou.float().flatten(1, 2)
    best = iou.masked_fill(~valid[..., None], -1.0).argmax(1)
    start, end = best // length, best % length
    step = ((end - start + 1) // 4).clamp_min(1)
    na = torch.stack((start + step, start), -1)
    nb = torch.stack((end, end - step), -1)
    inbounds = (na >= 0) & (nb < length) & (na <= nb)
    neg_ids = na.clamp(0, length - 1) * length + nb.clamp(0, length - 1)
    flat_neg = neg_ids.flatten(1)
    cs = geometry.candidate_start.float().flatten(1)
    ce = geometry.candidate_end.float().flatten(1)
    ps, pe = cs.gather(1, best), ce.gather(1, best)
    ns = cs.gather(1, flat_neg).reshape(batch, count_gt, 2)
    ne = ce.gather(1, flat_neg).reshape(batch, count_gt, 2)
    gs, ge = spans[..., 0].float(), spans[..., 1].float()
    gt_width = (ge - gs).clamp_min(1e-8)
    pi = (torch.minimum(pe, ge) - torch.maximum(ps, gs)).clamp_min(0.0)
    ni = (
        torch.minimum(ne, ge[..., None]) - torch.maximum(ns, gs[..., None])
    ).clamp_min(0.0)
    pc = pi / gt_width
    nc = ni / gt_width[..., None]
    pp = pi / (pe - ps).clamp_min(1e-8)
    npurity = ni / (ne - ns).clamp_min(1e-8)
    delta = pc[..., None] - nc
    neg_all_iou = iou.gather(1, flat_neg[..., None].expand(-1, -1, count_gt)).reshape(
        batch, count_gt, 2, count_gt
    )
    gidx = torch.arange(count_gt, device=support.device)
    other_gt = (gidx[None, :, None, None] != gidx[None, None, None, :]) & gt_mask[
        :, None, None
    ]
    other_hit = ((neg_all_iou >= 0.7) & other_gt).any(-1)
    mask = (
        inbounds
        & gt_mask[..., None]
        & valid.gather(1, best)[..., None]
        & valid.gather(1, flat_neg).reshape(batch, count_gt, 2)
        & (pp[..., None] >= 0.9)
        & (npurity >= 0.9)
        & (pc[..., None] >= 0.9)
        & (delta >= 0.15)
        & ~other_hit
    )
    flat_support = support.float().flatten(1)
    positive = flat_support.gather(1, best)[..., None]
    negative = flat_support.gather(1, flat_neg).reshape(batch, count_gt, 2)
    gap = positive - negative
    margin = 0.2 * delta
    hinge = F.relu(margin - gap)
    per_gt = (hinge * mask).sum(-1) / mask.sum(-1).clamp_min(1)
    active_gt = mask.any(-1) & gt_mask
    per_query = (per_gt * active_gt).sum(1) / active_gt.sum(1).clamp_min(1)
    value = query_objective.query_mean(per_query, active_gt.any(1))
    stats = {
        "loss": value,
        "pairs": mask.sum(),
        "active_queries": active_gt.any(1).sum(),
        "active_gt_fraction": masked_mean(active_gt.float(), gt_mask),
        "other_gt_exclusions": (other_hit & inbounds & gt_mask[..., None]).sum(),
        "coverage_delta": masked_mean(delta, mask),
        "actual_gap": masked_mean(gap, mask),
        "violation_fraction": masked_mean((hinge > 0).float(), mask),
        "positive_gap_fraction": masked_mean((gap > 0).float(), mask),
    }
    return value, stats


class Selector(support_objective.Selector):
    def raw_from_state(self, state, valid, carrier, video_padding_mask=None):
        field = super().raw_from_state(
            state, valid, carrier, video_padding_mask=video_padding_mask
        )
        # Bypass possible audit compose_score overrides when selecting anchors.
        # Factory rejects non-additive modes, so this is exactly untreated F.
        base_f = (
            field.carrier.float()
            + field.evidence.float()
            + field.support.float()
            + 0.5 * (field.transition_start.float() + field.transition_end.float())
        ).masked_fill(~valid, 0.0)
        anchors = select_anchors(base_f, valid)
        field.r59_base_score = base_f
        field.r59_anchors = anchors
        field.r59_old_evidence = field.evidence
        field.r59_old_support = field.support
        stats = {
            "anchor_count": anchors["active"].sum(1).float().mean(),
            "anchor_tie_fraction": masked_mean(
                (anchors["tie_counts"] > 1).float(), anchors["active"]
            ),
            "anchor_tie_count": masked_mean(
                anchors["tie_counts"].float(), anchors["active"]
            ),
            "anchor_best_iou": masked_mean(anchors["best_iou"], valid.flatten(1)),
        }
        if self.r59_mode == "shared_e":
            field.evidence, field.r59_shared_evidence, diag = shared_evidence(
                field.evidence, valid, anchors
            )
            stats.update({"e/" + k: v for k, v in diag.items()})
        else:
            token_valid = self._token_valid(valid, video_padding_mask)
            field.support, field.r59_support_details, diag = self.r59_local_support(
                field.round31_z_s, field.support, valid, token_valid, anchors
            )
            stats.update({"s/" + k: v for k, v in diag.items()})
        field.score = self.compose_score(field)
        field.round31_rank_score = field.score
        field.round31_score_with_support = field.score
        field.round31_score_without_support = self.compose_score(
            field, {"support": torch.zeros_like(field.support)}
        )
        field.round31_deployed_support = field.support
        # Original token/edge auxiliary caches intentionally remain reference
        # views. Mode local_s_coverage explicitly supervises field.support.
        field.r59_stats = {k: v.detach() for k, v in stats.items()}
        return field


def install_parent_freeze_hooks(model):
    """Clear applied leaf grads while preserving all upstream derivatives."""
    owner = weakref.ref(model)
    handles = []

    def clear_if_frozen(parameter):
        current = owner()
        if current is not None and current.r59_parent_frozen:
            parameter.grad = None

    for parameter in model.parent_model.parameters():
        if parameter.requires_grad:
            if not hasattr(parameter, "register_post_accumulate_grad_hook"):
                raise RuntimeError(
                    "R59 parent freeze requires PyTorch post-accumulate leaf hooks"
                )
            handles.append(
                parameter.register_post_accumulate_grad_hook(clear_if_frozen)
            )
    if not handles:
        raise RuntimeError("R59 freeze treatment found no trainable parent parameters")
    model.r59_freeze_hook_handles = handles


class Model(field_objective.Model):
    def _sync_freeze_epoch(self, epoch):
        self.r59_epoch = int(epoch)
        active = (
            self.r59_mode == "freeze_parent" and int(epoch) >= self.r59_freeze_epoch
        )
        if active and not self.r59_parent_frozen:
            for parameter in self.parent_model.parameters():
                parameter.grad = None
        self.r59_parent_frozen = active

    def set_epoch(self, epoch, training):
        result = dict(super().set_epoch(epoch, training))
        self._sync_freeze_epoch(epoch)
        result.update(
            r59_mode=self.r59_mode,
            r59_parent_updates_disabled=self.r59_parent_frozen,
            r59_parent_requires_grad_unchanged=True,
        )
        return result

    def compute_loss(self, outputs, batch, teacher_outputs, epoch):
        # Also correct for standalone contract/probe calls without set_epoch.
        self._sync_freeze_epoch(epoch)
        return super().compute_loss(outputs, batch, teacher_outputs, epoch)

    def parameter_groups(self):
        groups = super().parameter_groups()
        ids = [id(p) for group in groups for p in group.params]
        expected = {id(p) for p in self.parameters() if p.requires_grad}
        if len(ids) != len(set(ids)) or set(ids) != expected:
            raise RuntimeError(
                "R59 optimizer membership must uniquely cover all registered trainable parameters"
            )
        return groups

    def experiment_contract(self):
        result = dict(super().experiment_contract())
        result.update(
            r59_spec=dict(self.r59_spec),
            r59_mode=self.r59_mode,
            r59_model_factory_checkpoint_loads=0,
            r59_anchor_source="detached additive base-F before R59; no GT; audit-independent",
            r59_anchor_count=8,
            r59_inference_uses_gt=False,
            r59_freeze_epoch=9 if self.r59_mode == "freeze_parent" else None,
            r59_freeze_method="post_accumulate parent leaf grad=None; upstream graph intact",
            r59_parent_raw_gradients_preserved=True,
            r59_parent_updates_disabled=self.r59_parent_frozen,
            parent_weights_frozen=self.r59_parent_frozen,
            end_to_end=not self.r59_parent_frozen,
            r59_legacy_length_kl_preserved=True,
            r59_support_uniform_gate_noop=True,
            r59_t_new_head_lr_multiplier=1,
        )
        return result

    def r50_adjust_terms(self, terms, outputs, batch, epoch):
        base = super().r50_adjust_terms(terms, outputs, batch, epoch)
        mode = self.r59_mode
        field = outputs.trifield_output
        metrics = dict(base.metrics)
        metrics["support_geometry/parent_updates_disabled"] = float(
            self.r59_parent_frozen
        )
        metrics.update(
            {
                "support_geometry/structure/" + k: v
                for k, v in getattr(field, "r59_stats", {}).items()
            }
        )
        if mode not in ("quality_nms10", "precise_pair", "local_s_coverage"):
            result = dataclasses.replace(base, metrics=metrics)
            self.r50_last_terms = (
                result if getattr(self, "r50_capture_terms", False) else None
            )
            return result
        values = {
            k: getattr(base, k)
            for k in ("rank", "evidence", "support", "transition", "endpoint")
        }
        spans, gt_mask = losses._target_spans(outputs, batch)
        geometry = base.geometry
        if mode == "quality_nms10":
            pools = candidate_pools.candidate_pools(field, geometry, gt_mask, True)
            old_weights = field_objective.quality_weights(
                geometry, gt_mask, pools, False
            )
            quality_pools = dict(pools)
            quality_pools["nms"] = candidate_pools.nms_mask(
                field.score.float().flatten(1),
                geometry.valid.flatten(1),
                field.score.shape[1],
                k=10,
            )
            new_weights = field_objective.quality_weights(
                geometry, gt_mask, quality_pools, False
            )
            target = geometry.max_iou.float().flatten(1)
            logits = 2.0 * field.score.float().flatten(1)
            bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
            length = 1.0 + self.rank_length_reweight_gain * losses._query_length_ratio(
                outputs, batch
            )
            old = query_objective.query_mean(
                (bce * old_weights).sum(1) * length, gt_mask.any(1)
            )
            new = query_objective.query_mean(
                (bce * new_weights).sum(1) * length, gt_mask.any(1)
            )
            values["rank"] = base.rank + 0.25 * (new - old)
            # Old R58 quality telemetry no longer describes the deployed loss.
            for key in (
                "quality_bce",
                "weighted_target_mean",
                "weighted_probability_mean",
                "weighted_brier",
                "bias_analytic_gradient",
            ):
                name = "field_objective/" + key
                if name in metrics:
                    metrics["legacy_reference/" + name] = metrics.pop(name)
            diag = {
                "old_quality_bce": old,
                "actual_quality_bce": new,
                "weighted_target_mean": query_objective.query_mean(
                    (target * new_weights).sum(1), gt_mask.any(1)
                ),
                "weighted_probability_mean": query_objective.query_mean(
                    (logits.sigmoid() * new_weights).sum(1), gt_mask.any(1)
                ),
                "weighted_brier": query_objective.query_mean(
                    ((logits.sigmoid() - target).square() * new_weights).sum(1),
                    gt_mask.any(1),
                ),
            }
            for name, mask in quality_pools.items():
                diag["pool/" + name + "_count"] = mask.sum(1).float().mean()
            metrics.update(
                {"support_geometry/quality/" + k: v.detach() for k, v in diag.items()}
            )
        elif mode == "precise_pair":
            old_pair = field.round31_candidate_pair_probe["rank"]["pair_component"]
            new_pair, diag = precise_pair_loss(field.score, geometry, gt_mask)
            values["rank"] = base.rank - old_pair + 0.5 * new_pair
            for key in list(metrics):
                if key.startswith("candidate_rank/"):
                    metrics["legacy_reference/" + key] = metrics.pop(key)
            metrics["support_geometry/pair/old_component"] = old_pair.detach()
            metrics["support_geometry/pair/actual_component"] = (
                0.5 * new_pair
            ).detach()
            metrics.update(
                {"support_geometry/pair/" + k: v.detach() for k, v in diag.items()}
            )
        else:
            new_support, diag = coverage_pair_loss(
                field.support, geometry, spans, gt_mask
            )
            values["support"] = new_support
            for key in list(metrics):
                if key.startswith(
                    (
                        "support/",
                        "edge_support/",
                        "candidate_support/",
                        "joint_training/support/",
                    )
                ):
                    metrics["legacy_reference/" + key] = metrics.pop(key)
            metrics["support_geometry/coverage/old_support_loss"] = (
                base.support.detach()
            )
            metrics.update(
                {"support_geometry/coverage/" + k: v.detach() for k, v in diag.items()}
            )
        total = sum(self.loss_weights[k] * value for k, value in values.items())
        metrics.update({k: value.detach() for k, value in values.items()})
        metrics["total"] = total.detach()
        metrics["support_geometry/recomposition_error"] = (
            (total - sum(self.loss_weights[k] * value for k, value in values.items()))
            .abs()
            .detach()
        )
        result = dataclasses.replace(base, **values, total=total, metrics=metrics)
        self.r50_last_terms = (
            result if getattr(self, "r50_capture_terms", False) else None
        )
        return result


def build_model(config, *, r59_spec=None, **kwargs):
    spec = dict(r59_spec or {"mode": "control"})
    if set(spec) != {"mode"} or spec["mode"] not in MODES:
        raise ValueError("R59 expects exactly r59_spec={'mode': one of MODES}")
    mode = spec["mode"]
    expected_quality = {"mode": "kl_plus_quality", "bias": False}
    incoming_quality = dict(kwargs.pop("r58_spec", expected_quality))
    if (
        incoming_quality != expected_quality
        or float(kwargs.pop("r58_bias_init", 0.0)) != 0.0
    ):
        raise ValueError(
            "R59 control requires the exact unbiased R58 KL + .25 source-quality"
        )
    for name in ("r51_spec", "r53_spec", "r54_spec"):
        if kwargs.get(name):
            raise ValueError(
                "R59 owns all treatments; caller must leave " + name + " empty"
            )
        kwargs[name] = {}
    if mode == "t_only":
        kwargs["r53_spec"] = {"boundary": True}
    if kwargs.get("rank_stop_s", False) or kwargs.get("aux_detach_input", False):
        raise ValueError("R59 requires original full-path rank and auxiliary routing")
    model = field_objective.build_model(
        config, r58_spec=expected_quality, r58_bias_init=0.0, **kwargs
    )
    model.__class__ = Model
    model.r59_spec = spec
    model.r59_mode = mode
    model.r59_epoch = 0
    model.r59_freeze_epoch = 9
    model.r59_parent_frozen = False
    if mode == "freeze_parent":
        install_parent_freeze_hooks(model)
    if mode in ("shared_e", "local_s", "local_s_coverage"):
        selector = model.selector
        if (
            getattr(selector, "region_gain", 1.0) != 1.0
            or getattr(selector, "transition_gain", 1.0) != 1.0
            or getattr(selector, "linear_calibration", False)
            or getattr(selector, "pairwise_interactions", False)
            or getattr(selector, "length_calibration", False)
            or getattr(selector, "support_deployment", "in_score") != "in_score"
        ):
            raise ValueError("R59 anchors require the documented additive base-F")
        selector.__class__ = Selector
        selector.r59_mode = mode
        if mode in ("local_s", "local_s_coverage"):
            selector.r59_local_support = LocalSupport(64)
    return model
