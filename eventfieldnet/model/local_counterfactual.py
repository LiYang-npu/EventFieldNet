"""Round31 local interventions; labels only select existing candidate IDs."""

import hashlib
import torch
from torch.nn import functional as F
from field_core.adapter import _batch_parts, _metadata_rows, _valid_wrong_query_pairs
from .losses import _target_spans, _pad_empty_gt


def add_local_plans(context, outputs, batch, epoch, pair):
    bits = context["new_bits"]
    B, L, _ = outputs.span_logits.shape
    context["local_plans"] = [None] * B
    context["local_failures"] = ["disabled"] * B
    context["new_support_pairs"] = []
    context["direction_query_valid"] = [False] * B
    context["direction_missing_text"] = [False] * B
    if not bits:
        return
    inp, _, meta = _batch_parts(batch)
    targets = _batch_parts(batch)[1]
    rows = _metadata_rows(meta, B)
    spans, gm = _target_spans(outputs, _pad_empty_gt(batch))
    spans = spans.cpu()
    gm = gm.cpu()
    geo = context["geometry"]
    valid = geo.valid.cpu()
    iou = geo.all_iou.cpu()
    step = geo.one_step.cpu()
    pad = inp["video_padding_mask"].cpu()
    sal = targets.get("saliency_all_labels")
    sal = sal.cpu() if isinstance(sal, torch.Tensor) else None
    ids = torch.arange(L * L)
    starts = ids // L
    ends = ids % L + 1
    clips = torch.arange(L)
    permutation, _ = _valid_wrong_query_pairs(meta, B)
    video = inp.get("src_vid", inp.get("video_features")).detach().cpu()[..., :512]
    for b, row in enumerate(rows):
        if row is None:
            continue
        gs = gm[b].nonzero().flatten().tolist()
        rated = torch.zeros(L, dtype=torch.bool)
        for t in row.get("relevant_clip_ids", []):
            if int(t) != t or not 0 <= int(t) < L:
                raise ValueError("invalid rated clip ID")
            rated[int(t)] = True
        rated &= ~pad[b]
        if sal is not None and (
            sal.shape != (B, L) or not torch.isfinite(sal[b, rated]).all()
        ):
            raise ValueError("invalid actual saliency labels")
        if bits & 4:
            text = row.get("query")
            wrongrow = rows[int(permutation[b])]
            wrong = wrongrow.get("query") if wrongrow else None
            missing = not isinstance(text, str) or not isinstance(wrong, str)
            identical = (
                False
                if missing
                else " ".join(text.split()).casefold()
                == " ".join(wrong.split()).casefold()
            )
            context["direction_missing_text"][b] = missing
            context["direction_query_valid"][b] = (
                pair is not None
                and bool(pair[b])
                and not identical
                and context.get("wrong_field") is not None
            )
        if bits & 1:
            reason = "no_wrong_pair"
            context["local_failures"][b] = reason
            if pair is not None and bool(pair[b]) and gs and row.get("qid") is not None:
                order = sorted(gs, key=lambda g: tuple(spans[b, g].tolist()))
                key = str(row.get("vid", "")) + "|" + str(row["qid"])
                g = order[
                    (
                        int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
                        + int(epoch)
                        - 1
                    )
                    % len(order)
                ]
                a = float(spans[b, g].min() / step[b])
                z = float(spans[b, g].max() / step[b])
                inside = (clips < z) & (clips + 1 > a) & ~pad[b]
                k = max(1, int(inside.sum()) // 4)
                known = (inside & rated).nonzero().flatten().tolist()
                ranked = (
                    sorted(known, key=lambda t: (float(sal[b, t]), t))
                    if sal is not None
                    else []
                )
                low = ranked[:k]
                high = ranked[-k:]
                safe = ~pad[b]
                for h in gs:
                    a = float(spans[b, h].min() / step[b]) - 1
                    z = float(spans[b, h].max() / step[b]) + 1
                    safe &= ~((clips < z) & (clips + 1 > a))
                norms = video[b].norm(dim=-1)
                donor = (
                    (safe & torch.isfinite(norms) & norms.gt(1e-12))
                    .nonzero()
                    .flatten()
                    .tolist()
                )
                if len(ranked) < 2 * k:
                    reason = "insufficient_explicit_ratings"
                elif float(sal[b, low].max()) >= float(sal[b, high].min()):
                    reason = "no_strict_high_low_separation"
                elif len(donor) < k:
                    reason = "insufficient_protected_nonzero_donors"
                elif not (
                    torch.isfinite(norms[low + high]) & norms[low + high].gt(1e-12)
                ).all():
                    reason = "invalid_receiver_norm"
                else:
                    reason = "local"
                    context["local_plans"][b] = {
                        "gt": g,
                        "high": high,
                        "low": low,
                        "donors": donor[:k],
                        "k": k,
                        "high_min": float(sal[b, high].min()),
                        "low_max": float(sal[b, low].max()),
                    }
                context["local_failures"][b] = reason
        if bits & 2 and sal is not None:
            for g in gs:
                v = valid[b].flatten()
                if not v.any():
                    continue
                anchor = int(iou[b, :, :, g].flatten().masked_fill(~v, -1).argmax())
                s = int(starts[anchor])
                e = int(ends[anchor])
                width = e - s
                a = float(spans[b, g].min() / step[b])
                z = float(spans[b, g].max() / step[b])
                known = (
                    ((clips < z) & (clips + 1 > a) & rated & (clips >= s) & (clips < e))
                    .nonzero()
                    .flatten()
                    .tolist()
                )
                if not known:
                    continue
                evidence = max(known, key=lambda t: (float(sal[b, t]), -t))
                if float(sal[b, evidence]) <= 0:
                    continue
                other = [h for h in gs if h != g]
                relaxed_fallback = bool(
                    context.get("support_shift_relaxed_fallback", False)
                )
                for name, sign in [("shift_left", -1), ("shift_right", 1)]:
                    shift = starts - s
                    base = (
                        v
                        & (ends - starts == width)
                        & (shift * sign > 1)
                        & iou[b, :, :, g]
                        .flatten()
                        .le(float(iou[b].reshape(L * L, -1)[anchor, g]) - 0.1)
                    )
                    pool = base & (starts <= evidence) & (ends > evidence)
                    # round45 H14: width=1 anchors can never satisfy the evidence-containment
                    # requirement together with the >=2-clip shift magnitude (mathematically
                    # impossible -- see module docstring). When the strict pool is empty,
                    # retry without that requirement so these queries get their first-ever
                    # "shifted-away-is-worse" support signal instead of none at all.
                    if not pool.any() and relaxed_fallback:
                        pool = base
                    if other:
                        pool &= (
                            iou[b].reshape(L * L, -1)[:, other].max(1).values.lt(0.7)
                        )
                    for h in other:
                        a = float(spans[b, h].min() / step[b]) - 1
                        z = float(spans[b, h].max() / step[b]) + 1
                        left = (
                            (starts < s)
                            & (starts < z)
                            & (torch.minimum(ends, torch.full_like(ends, s)) > a)
                        )
                        right = (
                            (ends > e)
                            & (torch.maximum(starts, torch.full_like(starts, e)) < z)
                            & (ends > a)
                        )
                        pool &= ~(left | right)
                    if pool.any():
                        candidates = ids[pool]
                        negative = int(candidates[shift[candidates].abs().argmin()])
                        context["new_support_pairs"].append(
                            (b, g, name, anchor, negative)
                        )


def merge_local_masks(context, event, background, selected, video):
    branches = []
    for b, plan in enumerate(context["local_plans"]):
        if plan is not None:
            event[b].zero_()
            background[b].zero_()
            event[b, plan["high"]] = True
            background[b, plan["low"]] = True
            selected[b] = plan["gt"]
            branches.append("local")
        else:
            branches.append("whole" if selected[b] >= 0 else "fullwrong")
    context["event_mask"] = event.to(video.device)
    context["background_mask"] = background.to(video.device)
    context["selected_gt"] = selected.to(video.device)
    context["local_branches"] = branches


def replace_local_rows(original, masked, context, is_event, channels):
    out = masked.clone()
    for b, plan in enumerate(context["local_plans"]):
        if plan is None:
            continue
        target = plan["high" if is_event else "low"]
        donor = original[b, plan["donors"], :channels]
        norm = original[b, target, :channels].norm(dim=-1, keepdim=True)
        out[b, target, :channels] = donor / donor.norm(dim=-1, keepdim=True) * norm
    # Actual same-forward invariants, not a hard-coded declaration.
    context.setdefault("replacement_checks", []).append(
        {
            "TEF_exact": bool(
                torch.equal(out[..., channels:], original[..., channels:])
            ),
            "finite": bool(torch.isfinite(out).all()),
            "local_norm_max_error": float(
                torch.stack(
                    [
                        (
                            out[
                                b, (p["high"] if is_event else p["low"]), :channels
                            ].norm(dim=-1)
                            - original[
                                b, (p["high"] if is_event else p["low"]), :channels
                            ].norm(dim=-1)
                        )
                        .abs()
                        .max()
                        for b, p in enumerate(context["local_plans"])
                        if p is not None
                    ]
                ).max()
            )
            if any(p is not None for p in context["local_plans"])
            else 0.0,
        }
    )
    return out


def replace_support_pairs(context):
    if "original_support_pairs" not in context:
        context["original_support_pairs"] = context["support_pairs"]
    new = context["new_support_pairs"]
    groups = {(b, g) for b, g, *_ in new}
    context["support_pairs"] = [
        r for r in context["original_support_pairs"] if r[:2] not in groups
    ] + list(new)


def query_direction_loss(context, field, b, gt, endpoint, ai, ci, old):
    used = bool(context["new_bits"] & 4) and context["direction_query_valid"][b]
    f = getattr(field, "transition_" + endpoint)[b].flatten()
    if used:
        w = getattr(context["wrong_field"], "transition_" + endpoint)[b].flatten()
        dc = 0.5 * (f[ai] - f[ci])
        dw = 0.5 * (w[ai] - w[ci])
        loss = F.relu(0.2 - dc + dw)
    else:
        dc = 0.5 * (f[ai] - f[ci])
        dw = None
        loss = old
    context.setdefault("direction_measurements", []).append(
        {
            "b": b,
            "gt": gt,
            "endpoint": endpoint,
            "new": used,
            "correct": dc,
            "wrong": dw,
            "loss": loss,
        }
    )
    return loss


def local_probe(model, outputs, batch, terms, context):
    from collections import Counter

    report = {
        "new_bits": context["new_bits"],
        "E_branch_counts": dict(
            Counter(
                context.get(
                    "local_branches", ["old_protocol"] * outputs.span_logits.shape[0]
                )
            )
        ),
        "E_local_failure_counts": dict(Counter(context["local_failures"])),
        "replacement_actual_checks": context.get("replacement_checks", []),
        "S_new_pair_count": len(context["new_support_pairs"]),
        "S_new_GT_count": len({r[:2] for r in context["new_support_pairs"]}),
        "T_missing_raw_text_queries": sum(context["direction_missing_text"]),
    }

    def stat(values):
        if not values:
            return {"count": 0, "mean": None, "margin_met_rate": None}
        x = torch.cat([v.reshape(-1) for v in values]).detach().float()
        return {
            "count": x.numel(),
            "mean": float(x.mean()),
            "margin_met_rate": float((x >= 0.2).float().mean()),
            "quantiles": torch.quantile(
                x, torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], device=x.device)
            )
            .cpu()
            .tolist(),
        }

    components = {}
    field = outputs.trifield_output
    report["E_strict_covered_counts"] = {
        k: [0] * len(context["local_plans"]) for k in ("local", "whole")
    }
    for kind in ("local", "whole"):
        gaps = []
        if "event_field" in context:
            allgap = (
                context["background_field"].evidence - context["event_field"].evidence
            )
            for b, branch in enumerate(
                context.get("local_branches", ["whole"] * len(allgap))
            ):
                if branch != kind:
                    continue
                selected = int(context["selected_gt"][b])
                mask = (
                    terms.geometry.valid[b]
                    & terms.geometry.all_iou[b, :, :, selected].ge(0.7)
                    if selected >= 0
                    else torch.zeros_like(terms.geometry.valid[b])
                )
                if kind == "local":
                    plan = context["local_plans"][b]
                    L = mask.shape[-1]
                    idx = torch.arange(L, device=mask.device)
                    positions = plan["high"] + plan["low"]
                    mask &= (idx[:, None] <= min(positions)) & (
                        idx[None, :] >= max(positions)
                    )
                report["E_strict_covered_counts"][kind][b] = int(mask.sum())
                if mask.any():
                    gaps.append(allgap[b][mask])
        report["E_" + kind + "_margin"] = stat(gaps)
        if gaps:
            components["E_" + kind] = torch.relu(0.2 - torch.cat(gaps)).mean()
    for kind, new in [("new", True), ("fallback", False)]:
        sg = [
            field.support[b].flatten()[ai] - field.support[b].flatten()[ci]
            for b, g, t, ai, ci in context["support_pairs"]
            if t.startswith("shift_") == new
        ]
        report["S_" + kind + "_margin"] = stat(sg)
        if sg:
            components["S_" + kind] = torch.relu(0.2 - torch.stack(sg)).mean()
        rows = [r for r in context.get("direction_measurements", []) if r["new"] == new]
        tg = [r["correct"] - r["wrong"] if new else 2 * r["correct"] for r in rows]
        report["T_" + kind + "_direction_margin"] = stat(tg)
        if rows:
            components["T_" + kind] = torch.stack([r["loss"] for r in rows]).mean()
    position = [
        getattr(field, "transition_" + end)[b].flatten()[ai]
        - getattr(field, "transition_" + end)[b].flatten()[ci]
        for b, g, end, family, ai, ci in context["transition_pairs"]
        if family == "position"
    ]
    report["T_position_margin"] = stat(position)
    if position:
        components["T_position"] = torch.relu(0.2 - torch.stack(position)).mean()
    report["T_new_correct_direction"] = stat(
        [r["correct"] for r in context.get("direction_measurements", []) if r["new"]]
    )
    report["T_new_wrong_direction"] = stat(
        [r["wrong"] for r in context.get("direction_measurements", []) if r["new"]]
    )
    report["S_fallback_GT_count"] = len(
        {r[:2] for r in context["support_pairs"] if not r[2].startswith("shift_")}
    )
    report["E_high_low_strict_count"] = sum(
        p["low_max"] < p["high_min"] for p in context["local_plans"] if p is not None
    )
    report["E_fullwrong_candidate_count"] = int(
        context["report"].get("E_total_positive_count", 0)
    ) - sum(sum(x) for x in report["E_strict_covered_counts"].values())
    report["component_parameter_gradients"] = {}
    params = [p for p in model.parameters() if p.requires_grad]
    for n, loss in components.items():
        grads = torch.autograd.grad(loss, params, allow_unused=True, retain_graph=True)
        active = [g.detach() for g in grads if g is not None]
        report["component_parameter_gradients"][n] = {
            "finite": all(bool(torch.isfinite(g).all()) for g in active),
            "connected": len(active),
            "raw_l2": float(
                torch.sqrt(
                    sum(
                        (g.float().square().sum() for g in active),
                        loss.detach().new_zeros(()),
                    )
                )
            ),
        }
    report["gradient_scope"] = (
        "standalone family means; actual GT/type/query weighted objective is in five-loss gradient probe"
    )
    return report


def restrict_local_candidates(active, context):
    result = active.clone()
    L = active.shape[-1]
    idx = torch.arange(L, device=active.device)
    for b, plan in enumerate(context["local_plans"]):
        if plan is None:
            continue
        receivers = plan["high"] + plan["low"]
        result[b] &= (idx[:, None] <= min(receivers)) & (idx[None, :] >= max(receivers))
    context["report"]["local_receiver_excluded_candidates"] = int(
        (active & ~result).sum()
    )
    return result
