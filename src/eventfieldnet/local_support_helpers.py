"""Separate R66 S objective and uniform-null local residual treatments.

GT is used exclusively in coverage_gap_objective. The structural arm keeps
the inherited edge-S, anchors and token gate; it replaces only its residual.
"""

import torch
from torch import nn
from torch.nn import functional as F


def mean(value, mask):
    return (value * mask).sum() / mask.sum().clamp_min(1)


class CenteredSupportReadout(nn.Module):
    def __init__(self):
        super().__init__()
        # Nonzero deterministic quality coefficient activates the zero-init
        # token gate immediately, without consuming initialization RNG.
        self.weight = nn.Parameter(torch.tensor([0.0, 0.0, 2.0], dtype=torch.float32))

    def forward(self, details):
        with torch.autocast(device_type=details["coverage"].device.type, enabled=False):
            coverage = details["coverage"].float()
            uniform = details["uniform"].float()
            x = torch.stack(
                (
                    coverage - 2.0 * uniform,
                    details["density_ratio"].float() - 0.5,
                    details["quality"].float() - uniform,
                ),
                -1,
            )
            z = F.linear(x, self.weight.float()[None]).squeeze(-1)
            return torch.tanh(z).masked_fill(~details["eligible"], 0.0), x


def prepare_centered_residual(selector):
    if not hasattr(selector, "r59_local_support"):
        raise ValueError("R66 residual requires the D07 registered token gate")
    if hasattr(selector, "r66_centered_s_readout"):
        raise ValueError("R66 residual already installed")
    selector.r66_centered_s_readout = CenteredSupportReadout()


def apply_centered_residual(selector, field, valid):
    """Apply after D07 forward; retain pre-residual anchors and original S."""
    old = field.r59_old_support
    details = field.r59_support_details
    local, features = selector.r66_centered_s_readout(details)
    oldf = old.float()
    delta = ((1.0 - oldf.abs()).clamp_min(0.0) * local).masked_fill(~valid, 0.0)
    field.support = (oldf + delta).masked_fill(~valid, 0.0)
    field.r66_s_features = features
    field.r66_s_local = local
    field.r66_s_legacy_support = old
    field.r66_s_stats = {
        k: v.detach()
        for k, v in {
            "residual_abs": mean(delta.abs(), valid),
            "residual_signed": mean(delta, valid),
            "legacy_abs": mean(oldf.abs(), valid),
            "deployed_abs": mean(field.support.abs(), valid),
            "eligible_fraction": mean(details["eligible"].float(), valid),
            "readout_weight_norm": selector.r66_centered_s_readout.weight.float().norm(),
            "centered_feature_abs": mean(features.abs().mean(-1), details["eligible"]),
            "deployed_saturation": mean((field.support.abs() > 0.95).float(), valid),
        }.items()
    }
    field.r59_support_details = dict(details, delta=delta)
    field.r59_stats = dict(
        field.r59_stats,
        **{
            "s/delta_abs": mean(delta.abs(), valid).detach(),
            "s/delta_signed": mean(delta, valid).detach(),
        },
    )
    field.score = selector.compose_score(field)
    field.round31_rank_score = field.score
    field.round31_score_with_support = field.score
    field.round31_score_without_support = selector.compose_score(
        field, {"support": torch.zeros_like(field.support)}
    )
    field.round31_deployed_support = field.support
    return field


def coverage_gap_objective(field, geometry, spans, gt_mask, report_gt_mask=None):
    """One coverage-gap Huber loss, with neutral pure outward expansions.

    Equal available direction families per GT, equal active GTs per query,
    then equal active queries. Candidate selection and masks are GT-only
    loss construction; inference does not call this function. An optional
    report mask limits the evaluated GTs for offline diagnostics only; all
    other-GT exclusions still use the original complete gt_mask.
    """
    support = field.support.float()
    batch, length, _ = support.shape
    count_gt = gt_mask.shape[1]
    if report_gt_mask is not None and report_gt_mask.shape != gt_mask.shape:
        raise ValueError("report GT mask must match complete GT mask shape")
    if count_gt == 0:
        zero = support.sum() * 0.0
        names = [
            "loss",
            "active_queries",
            "active_gt",
            "pairs",
            "other_gt_exclusions",
            "expansion_new_other_gt_exclusions",
            "active_families_per_gt",
            "truncation/positive_gap_fraction",
            "expansion/nonneutral_fraction",
        ]
        names += [
            family + "/" + key
            for family in ("truncation", "expansion")
            for key in (
                "pairs",
                "loss",
                "s_gap",
                "target_gap",
                "absolute_error",
                "coverage_gap",
                "coverage_gap_abs",
            )
        ]
        return zero, {name: zero.detach() for name in names}
    valid = geometry.valid.bool().flatten(1)
    all_iou = geometry.all_iou.detach().float().flatten(1, 2)
    best = all_iou.masked_fill(~valid[..., None], -1.0).argmax(1)
    start, end = best // length, best % length
    step = ((end - start + 1) // 4).clamp_min(1)
    ns = torch.stack((start + step, start, start - step, start), -1)
    ne = torch.stack((end, end - step, end, end + step), -1)
    bounds = (ns >= 0) & (ne < length) & (ns <= ne)
    negative = ns.clamp(0, length - 1) * length + ne.clamp(0, length - 1)
    flat = negative.flatten(1)
    cs = geometry.candidate_start.detach().float().flatten(1)
    ce = geometry.candidate_end.detach().float().flatten(1)
    ps, pe = cs.gather(1, best), ce.gather(1, best)
    a = cs.gather(1, flat).reshape(batch, count_gt, 4)
    z = ce.gather(1, flat).reshape(batch, count_gt, 4)
    gs, ge = spans.detach().float().unbind(-1)
    width = (ge - gs).clamp_min(1e-8)
    pi = (torch.minimum(pe, ge) - torch.maximum(ps, gs)).clamp_min(0.0)
    ni = (torch.minimum(z, ge[..., None]) - torch.maximum(a, gs[..., None])).clamp_min(
        0.0
    )
    pc, nc = pi / width, ni / width[..., None]
    pp = pi / (pe - ps).clamp_min(1e-8)
    npurity = ni / (z - a).clamp_min(1e-8)
    difference = pc[..., None] - nc
    eligible = (
        bounds
        & gt_mask[..., None]
        & valid.gather(1, best)[..., None]
        & valid.gather(1, flat).reshape_as(negative)
        & (negative != best[..., None])
        & (pp[..., None] >= 0.9)
        & (pc[..., None] >= 0.9)
    )

    gidx = torch.arange(count_gt, device=support.device)
    other = (gidx[None, :, None, None] != gidx[None, None, None, :]) & gt_mask[
        :, None, None, :
    ]
    neg_iou = all_iou.gather(1, flat[..., None].expand(-1, -1, count_gt)).reshape(
        batch, count_gt, 4, count_gt
    )
    other_hit = ((neg_iou >= 0.7) & other).any(-1)
    # A wide expansion can newly cover another event while its IoU with that
    # other event stays low. Exclude newly acquired other-GT content too.
    pi_other = (
        torch.minimum(pe[:, :, None], ge[:, None, :])
        - torch.maximum(ps[:, :, None], gs[:, None, :])
    ).clamp_min(0.0)
    ni_other = (
        torch.minimum(z[..., None], ge[:, None, None, :])
        - torch.maximum(a[..., None], gs[:, None, None, :])
    ).clamp_min(0.0)
    new_other = ((ni_other > pi_other[:, :, None, :] + 1e-7) & other).any(-1)
    if report_gt_mask is not None:
        eligible = eligible & report_gt_mask.bool()[..., None]
    trunc = (
        eligible[..., :2]
        & (npurity[..., :2] >= 0.9)
        & (difference[..., :2] >= 0.15)
        & ~other_hit[..., :2]
    )
    expand = (
        eligible[..., 2:]
        & (difference[..., 2:].abs() <= 0.02)
        & ~other_hit[..., 2:]
        & ~new_other[..., 2:]
    )
    mask = torch.cat((trunc, expand), -1)
    target = torch.cat(
        (
            0.2 * difference[..., :2].clamp_min(0.0),
            torch.zeros_like(difference[..., 2:]),
        ),
        -1,
    )
    flat_s = support.flatten(1)
    gap = flat_s.gather(1, best)[..., None] - flat_s.gather(1, flat).reshape_as(target)
    error = F.smooth_l1_loss(gap, target, reduction="none", beta=0.05)
    families = error.reshape(batch, count_gt, 2, 2)
    family_mask = mask.reshape(batch, count_gt, 2, 2)
    family_count = family_mask.sum(-1)
    family_loss = (families * family_mask).sum(-1) / family_count.clamp_min(1)
    active_family = family_count > 0
    per_gt = (family_loss * active_family).sum(-1) / active_family.sum(-1).clamp_min(1)
    active_gt = active_family.any(-1) & gt_mask
    per_query = (per_gt * active_gt).sum(-1) / active_gt.sum(-1).clamp_min(1)
    loss = mean(per_query, active_gt.any(-1))
    stats = {
        "loss": loss,
        "pairs": mask.sum(),
        "active_gt": active_gt.sum(),
        "active_queries": active_gt.any(-1).sum(),
        "other_gt_exclusions": (other_hit & eligible).sum(),
        "expansion_new_other_gt_exclusions": (
            new_other[..., 2:] & eligible[..., 2:]
        ).sum(),
        "active_families_per_gt": mean(active_family.sum(-1).float(), active_gt),
    }
    for label, sl in [("truncation", slice(0, 2)), ("expansion", slice(2, 4))]:
        m = mask[..., sl]
        stats.update(
            {
                label + "/pairs": m.sum(),
                label + "/loss": mean(error[..., sl], m),
                label + "/s_gap": mean(gap[..., sl], m),
                label + "/target_gap": mean(target[..., sl], m),
                label + "/coverage_gap": mean(difference[..., sl], m),
                label + "/coverage_gap_abs": mean(difference[..., sl].abs(), m),
                label + "/absolute_error": mean(
                    (gap[..., sl] - target[..., sl]).abs(), m
                ),
            }
        )
    stats["truncation/positive_gap_fraction"] = mean((gap[..., :2] > 0).float(), trunc)
    stats["expansion/nonneutral_fraction"] = mean(
        (gap[..., 2:].abs() > 0.02).float(), expand
    )
    return loss, {k: v.detach() for k, v in stats.items()}
