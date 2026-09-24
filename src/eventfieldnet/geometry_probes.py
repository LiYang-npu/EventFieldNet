"""No-extra-forward diagnostics; panel IoU/recall are explicitly not official AP."""

import torch
from field_core.adapter import _batch_parts, _metadata_rows
from .model.losses import _target_spans
from .candidate_pools import candidate_pools


def f(x):
    return float(x.detach()) if isinstance(x, torch.Tensor) else float(x)


@torch.no_grad()
def field_geometry_panel(model, outputs, batch, terms):
    field = outputs.trifield_output
    geo = terms.geometry
    spans, gm = _target_spans(outputs, batch)
    rows = _metadata_rows(_batch_parts(batch)[2], batch.batch_size)
    durations = spans.new_tensor(
        [float((row or {}).get("duration") or 0.0) for row in rows]
    )
    seconds = (spans[..., 1] - spans[..., 0]).clamp_min(0) * durations[:, None]
    known = gm & (durations[:, None] > 0)
    buckets = {
        "all": gm,
        "short_0_10s": known & (seconds <= 10),
        "middle_10_30s": known & (seconds > 10) & (seconds <= 30),
        "long_over30s": known & (seconds > 30),
    }
    valid = geo.valid.flatten(1)
    all_iou = geo.all_iou.flatten(1, 2).float()

    def top(score):
        return score.flatten(1).masked_fill(~valid, -torch.inf).argmax(1)

    def chosen(ids):
        return all_iou.gather(1, ids[:, None, None].expand(-1, 1, gm.shape[1])).squeeze(
            1
        )

    base = chosen(top(field.score))
    removed = {}
    for name, attrs in {
        "E": ["evidence"],
        "S": ["support"],
        "T": ["transition_start", "transition_end"],
    }.items():
        score = model.selector.compose_score(
            field, {attr: torch.zeros_like(getattr(field, attr)) for attr in attrs}
        )
        removed[name] = chosen(top(score))
    pools = candidate_pools(field, geo, gm, True)
    maxima = {"full_grid": all_iou.masked_fill(~valid[..., None], -1).amax(1)}
    for key in ["raw", "nms"]:
        maxima[key + "30"] = all_iou.masked_fill(~pools[key][..., None], -1).amax(1)
    result = {
        "scope": "fixed-panel per-GT top1 IoU and pool recall, not official mAP; duration unavailable GT excluded from length buckets",
        "duration_known_gt_count": int(known.sum()),
        "valid_gt_count": int(gm.sum()),
        "buckets": {},
    }
    for name, mask in buckets.items():
        n = int(mask.sum())
        row = {"gt_count": n}
        if n:
            row["full_top1_iou"] = f(base[mask].mean())
            row["full_minus_deleted_top1_iou"] = {
                k: f((base - v)[mask].mean()) for k, v in removed.items()
            }
            row["recall"] = {
                k: {str(t): f((v[mask] >= t).float().mean()) for t in [0.7, 0.9, 0.95]}
                for k, v in maxima.items()
            }
        result["buckets"][name] = row
    if hasattr(field, "r59_anchors"):
        a = field.r59_anchors
        anchor_iou = all_iou.gather(1, a["ids"][..., None].expand(-1, -1, gm.shape[1]))
        near = (anchor_iou >= 0.3) & gm[:, None, :]
        result["anchors"] = {
            "scope": "GT only diagnoses learned GT-free anchors; never used in forward selection",
            "active": int(a["active"].sum()),
            "matching_two_or_more_gt": int(((near.sum(-1) >= 2) & a["active"]).sum()),
            "anchor_gt_recall07": f(
                (
                    (anchor_iou.masked_fill(~a["active"][..., None], -1).amax(1) >= 0.7)
                    & gm
                ).sum()
                / gm.sum().clamp_min(1)
            ),
        }
    if hasattr(field, "r59_support_details"):
        # Candidate-wide eligibility can conceal inactive GT-aligned candidates.
        best_ids = all_iou.masked_fill(~valid[..., None], -1).argmax(1)
        details = field.r59_support_details
        eligible = details["eligible"].flatten(1).gather(1, best_ids)
        delta = details["delta"].flatten(1).gather(1, best_ids)
        score_without_residual = model.selector.compose_score(
            field, {"support": field.r59_old_support}
        )
        without_residual = chosen(top(score_without_residual))
        result["local_s_residual"] = {
            "scope": "best-IoU grid candidate eligibility by GT; residual-only deletion holds GT-free anchors fixed; not whole-S deletion or AP",
            "buckets": {},
        }
        for name, mask in buckets.items():
            n = int(mask.sum())
            row = {"gt_count": n, "eligible_gt": int((mask & eligible).sum())}
            if n:
                row.update(
                    eligible_fraction=f(eligible[mask].float().mean()),
                    best_gt_candidate_abs_delta=f(delta[mask].abs().mean()),
                    full_minus_residual_deleted_top1_iou=f(
                        (base - without_residual)[mask].mean()
                    ),
                )
            result["local_s_residual"]["buckets"][name] = row
    return result
