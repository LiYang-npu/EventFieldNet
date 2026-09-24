"""R68 exact K06 pairs with optional one-sided truncation, plus train-query KL weights.

The default coverage arithmetic and all masks/reductions are copied from the
sealed R66 helper. Only hinge_truncation changes its first two error terms.
No inference function accepts GT and no new trainable parameter is introduced.
"""

import torch
from torch.nn import functional as F


def mean(value, mask):
    return (value * mask).sum() / mask.sum().clamp_min(1)


def short_query_weights(spans, gt_mask, durations_seconds):
    """Detached training-GT short fractions, with unknown durations excluded.

    Unknown-duration but otherwise valid GTs stay in the denominator. Known
    positive duration and positive finite span width are required to be short.
    The optional BxG duration shape also supports isolated mathematical tests.
    """
    spans = spans.detach().float()
    duration = durations_seconds.detach().float()
    if duration.ndim == 1:
        duration = duration[:, None].expand_as(gt_mask)
    if duration.shape != gt_mask.shape or spans.shape[:-1] != gt_mask.shape:
        raise ValueError(
            "R68 short weights require BxG spans/mask and B or BxG duration"
        )
    width = spans[..., 1] - spans[..., 0]
    valid = gt_mask.bool() & torch.isfinite(spans).all(-1) & (width > 0.0)
    known = torch.isfinite(duration) & (duration > 0.0)
    seconds = torch.where(valid & known, width * duration, torch.zeros_like(width))
    short = valid & known & (seconds <= 10.0)
    count = valid.sum(-1)
    short_count = short.sum(-1)
    fraction = short_count.float() / count.clamp_min(1)
    weights = 1.0 + fraction
    stats = dict(
        valid_gt=count.sum(),
        short_gt=short_count.sum(),
        unknown_duration_gt=(valid & ~known).sum(),
        queries_with_valid_gt=(count > 0).sum(),
        queries_with_short_gt=(short_count > 0).sum(),
        weight_min=weights.min() if weights.numel() else spans.new_tensor(1.0),
        weight_max=weights.max() if weights.numel() else spans.new_tensor(1.0),
    )
    payload = dict(
        valid_gt_mask=valid,
        short_mask=short,
        known_duration_mask=known,
        short_fraction=fraction,
        query_weight=weights,
        seconds=seconds,
    )
    return weights, {k: v.detach() for k, v in stats.items()}, payload


def long_query_weights(spans, gt_mask, durations_seconds):
    """Detached training-GT long fractions, with unknown durations excluded.

    Unknown-duration but otherwise valid GTs stay in the denominator. Known
    positive duration and positive finite span width are required to be long.
    The optional BxG duration shape also supports isolated mathematical tests.
    """
    spans = spans.detach().float()
    duration = durations_seconds.detach().float()
    if duration.ndim == 1:
        duration = duration[:, None].expand_as(gt_mask)
    if duration.shape != gt_mask.shape or spans.shape[:-1] != gt_mask.shape:
        raise ValueError(
            "R68 long weights require BxG spans/mask and B or BxG duration"
        )
    width = spans[..., 1] - spans[..., 0]
    valid = gt_mask.bool() & torch.isfinite(spans).all(-1) & (width > 0.0)
    known = torch.isfinite(duration) & (duration > 0.0)
    seconds = torch.where(valid & known, width * duration, torch.zeros_like(width))
    long = valid & known & (seconds > 30.0)
    count = valid.sum(-1)
    long_count = long.sum(-1)
    fraction = long_count.float() / count.clamp_min(1)
    weights = 1.0 + fraction
    stats = dict(
        valid_gt=count.sum(),
        long_gt=long_count.sum(),
        unknown_duration_gt=(valid & ~known).sum(),
        queries_with_valid_gt=(count > 0).sum(),
        queries_with_long_gt=(long_count > 0).sum(),
        weight_min=weights.min() if weights.numel() else spans.new_tensor(1.0),
        weight_max=weights.max() if weights.numel() else spans.new_tensor(1.0),
    )
    payload = dict(
        valid_gt_mask=valid,
        long_mask=long,
        known_duration_mask=known,
        long_fraction=fraction,
        query_weight=weights,
        seconds=seconds,
    )
    return weights, {k: v.detach() for k, v in stats.items()}, payload


def weighted_query_kl(query, length, active, weights):
    """Only active queries normalize the actual KL; length remains unchanged."""
    if not (query.shape == length.shape == active.shape == weights.shape):
        raise ValueError("R68 query KL inputs must share a one-dimensional shape")
    selected = active.bool()
    values = query * length
    if not selected.any():
        return values.sum() * 0.0
    w = weights.detach()[selected]
    # Exactly preserve the old arithmetic when all active weights coincide.
    if bool((w == w[0]).all()):
        return values[selected].mean()
    return (values[selected] * w).sum() / w.sum()


def coverage_gap_objective(
    field,
    geometry,
    spans,
    gt_mask,
    report_gt_mask=None,
    *,
    hinge_truncation=False,
    return_payload=False,
):
    """One coverage-gap Huber loss, with neutral pure outward expansions.

    Equal available direction families per GT, equal active GTs per query,
    then equal active queries. Candidate selection and masks are GT-only
    loss construction; inference does not call this function. An optional
    report mask limits the evaluated GTs for offline diagnostics only; all
    other-GT exclusions still use the original complete gt_mask.
    """
    repair = getattr(field, "long_s_spec", {"upper": False, "long_weight": False})
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
        stats = {name: zero.detach() for name in names}
        stats.update(
            {
                "truncation/over_margin_fraction": zero.detach(),
                "truncation/violation_fraction": zero.detach(),
                "hinge_truncation": zero.new_tensor(float(hinge_truncation)),
            }
        )
        payload = {
            "gap": support.new_empty((batch, 0, 4)),
            "target": support.new_empty((batch, 0, 4)),
            "mask": torch.zeros((batch, 0, 4), dtype=torch.bool, device=support.device),
            "truncation_mask": torch.zeros(
                (batch, 0, 2), dtype=torch.bool, device=support.device
            ),
            "expansion_mask": torch.zeros(
                (batch, 0, 2), dtype=torch.bool, device=support.device
            ),
            "error": support.new_empty((batch, 0, 4)),
            "gt_mask": gt_mask,
            "best_indices": torch.empty(
                (batch, 0), dtype=torch.long, device=support.device
            ),
            "negative_indices": torch.empty(
                (batch, 0, 4), dtype=torch.long, device=support.device
            ),
        }
        return (zero, stats, payload) if return_payload else (zero, stats)
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
    if hinge_truncation:
        error = torch.cat((F.relu(target[..., :2] - gap[..., :2]), error[..., 2:]), -1)
    upper_error = F.relu(gap[..., :2] - target[..., :2] - 0.05)
    if repair["upper"] and hinge_truncation:
        error = torch.cat((error[..., :2] + 0.25 * upper_error, error[..., 2:]), -1)
    families = error.reshape(batch, count_gt, 2, 2)
    family_mask = mask.reshape(batch, count_gt, 2, 2)
    family_count = family_mask.sum(-1)
    family_loss = (families * family_mask).sum(-1) / family_count.clamp_min(1)
    active_family = family_count > 0
    per_gt = (family_loss * active_family).sum(-1) / active_family.sum(-1).clamp_min(1)
    active_gt = active_family.any(-1) & gt_mask
    dur = getattr(field, "long_s_durations", spans.new_zeros(batch)).detach()[:, None]
    long_gt = (
        gt_mask
        & torch.isfinite(dur)
        & (dur > 0)
        & ((spans[..., 1] - spans[..., 0]) * dur > 30.0)
    )
    gt_weights = (
        1.0 + long_gt.float() if repair["long_weight"] else torch.ones_like(per_gt)
    )
    if repair["long_weight"]:
        # Normalize only among active GTs; preserve all-long query scale.
        per_query = (per_gt * active_gt * gt_weights).sum(-1) / (
            active_gt * gt_weights
        ).sum(-1).clamp_min(1.0)
        # Also balance active queries: queries containing long GT receive up to 2x.
        qweight = 1.0 + (long_gt * active_gt).sum(-1) / active_gt.sum(-1).clamp_min(1)
    else:
        per_query = (per_gt * active_gt).sum(-1) / active_gt.sum(-1).clamp_min(1)
        qweight = torch.ones_like(per_query)
    loss = mean(per_query, active_gt.any(-1))
    if repair["long_weight"]:
        active_q = active_gt.any(-1)
        loss = (per_query * qweight * active_q).sum() / (
            qweight * active_q
        ).sum().clamp_min(1.0)
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
    stats["truncation/over_margin_fraction"] = mean(
        (gap[..., :2] > target[..., :2]).float(), trunc
    )
    stats["truncation/violation_fraction"] = mean(
        (gap[..., :2] < target[..., :2]).float(), trunc
    )
    stats["repair/upper_enabled"] = loss.new_tensor(float(repair["upper"]))
    stats["repair/long_weight_enabled"] = loss.new_tensor(float(repair["long_weight"]))
    stats["repair/active_long_gt"] = (long_gt & active_gt).sum()
    stats["repair/over_upper_pairs"] = ((upper_error > 0) & trunc).sum()
    stats["repair/upper_penalty"] = mean(upper_error, trunc)
    stats["repair/query_weight_mean"] = mean(qweight, active_gt.any(-1))
    stats["hinge_truncation"] = loss.new_tensor(float(hinge_truncation))
    detached = {k: v.detach() for k, v in stats.items()}
    payload = {
        "gap": gap,
        "target": target,
        "mask": mask,
        "truncation_mask": trunc,
        "expansion_mask": expand,
        "error": error,
        "gt_mask": gt_mask,
        "best_indices": best.detach(),
        "negative_indices": negative.detach(),
        "repair": repair,
        "gt_weights": gt_weights.detach(),
        "query_weights": qweight.detach(),
        "active_gt": active_gt,
        "per_gt": per_gt,
    }
    return (loss, detached, payload) if return_payload else (loss, detached)
