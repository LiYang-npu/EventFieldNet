"""Round31 geometry-only counterfactual labels and graph-connected objectives."""

import hashlib
import random
import torch
from torch.nn import functional as F
from .losses import _target_spans, _pad_empty_gt
from field_core.losses import candidate_geometry
from field_core.adapter import _batch_parts, _metadata_rows


def _zero(t):
    return t.float().sum() * 0


def _balanced(rows, reference):
    # rows are query -> GT -> type -> list of scalar losses.
    queries = []
    for groups in rows.values():
        gmeans = []
        for types in groups.values():
            tmeans = [torch.stack(v).mean() for v in types.values() if v]
            if tmeans:
                gmeans.append(torch.stack(tmeans).mean())
        if gmeans:
            queries.append(torch.stack(gmeans).mean())
    return torch.stack(queries).mean() if queries else _zero(reference)


def _append(rows, b, g, t, value):
    rows.setdefault(b, {}).setdefault(g, {}).setdefault(t, []).append(value)


def geometry_pairs(outputs, batch):
    """Discrete IDs only; detached CPU label construction avoids GPU scalar loops."""
    geometry = candidate_geometry(outputs, _pad_empty_gt(batch))
    spans, gm = _target_spans(outputs, _pad_empty_gt(batch))
    valid = geometry.valid.detach().cpu()
    ious = geometry.all_iou.detach().cpu()
    spans = spans.detach().cpu()
    gm = gm.detach().cpu()
    step = geometry.one_step.detach().cpu()
    bsz, l, _ = valid.shape
    ids = torch.arange(l * l)
    starts = ids // l
    ends = ids % l + 1
    width = ends - starts
    support = []
    transition = []
    anchors = []
    expanded_rejected = 0
    transition_conflicts = 0
    saliency_gt_count = 0
    inputs, targets, _ = _batch_parts(batch)
    sal = targets.get("saliency_all_labels")
    sal = (
        sal.detach().cpu()
        if isinstance(sal, torch.Tensor) and sal.ndim == 2 and sal.shape == (bsz, l)
        else None
    )
    for b in range(bsz):
        ground = gm[b].nonzero().flatten().tolist()
        v = valid[b].flatten()
        for g in ground:
            if not v.any():
                continue
            ai = int(ious[b, :, :, g].flatten().masked_fill(~v, -1).argmax())
            s, e = ai // l, ai % l + 1
            w = e - s
            anchors.append((b, g, ai))
            other = [h for h in ground if h != g]
            clean = v.clone()
            if other:
                clean &= ious[b].reshape(l * l, -1)[:, other].max(1).values.lt(0.7)
            # One minimum-size-change candidate per perturbation type; flat ID tie.
            expand = clean & starts.le(s) & ends.ge(e) & width.ge(w * 1.2) & width.gt(w)
            for h in other:
                gs = float(spans[b, h].min() / step[b]) - 1
                ge = float(spans[b, h].max() / step[b]) + 1
                left = (
                    (starts < s)
                    & (starts.float() < ge)
                    & (torch.minimum(ends, torch.full_like(ends, s)).float() > gs)
                )
                right = (
                    (ends > e)
                    & (torch.maximum(starts, torch.full_like(starts, e)).float() < ge)
                    & (ends.float() > gs)
                )
                expanded_rejected += int((expand & (left | right)).sum())
                expand &= ~(left | right)
            if expand.any():
                candidates = ids[expand]
                ci = int(candidates[(width[candidates] - w).argmin()])
                support.append((b, g, "expand", ai, ci))
            if sal is not None:
                local = sal[b, s:e]
                # Require measured finite nonconstant positive evidence, not an invented midpoint.
                if (
                    local.numel()
                    and torch.isfinite(local).all()
                    and local.max() > 0
                    and local.max() > local.min()
                ):
                    saliency_gt_count += 1
                    anchor = s + int(local.argmax())
                    trim = (
                        clean
                        & starts.ge(s)
                        & ends.le(e)
                        & width.le(w * 0.8)
                        & width.lt(w)
                        & starts.le(anchor)
                        & ends.gt(anchor)
                    )
                    if trim.any():
                        candidates = ids[trim]
                        ci = int(candidates[(w - width[candidates]).argmin()])
                        support.append((b, g, "truncate", ai, ci))
            for typ, position, opposite, col in [
                ("start", s, e - 1, 0),
                ("end", e - 1, s, 1),
            ]:
                if not valid[b].flatten()[position * l + position]:
                    continue
                true = float(spans[b, g, col] / step[b]) - (1 if typ == "end" else 0)
                for d in (-2, 2):
                    p = position + d
                    if not 0 <= p < l or abs(p - true) <= 1:
                        continue
                    if any(
                        abs(
                            p
                            - (
                                float(spans[b, h, col] / step[b])
                                - (1 if typ == "end" else 0)
                            )
                        )
                        <= 1
                        for h in other
                    ):
                        transition_conflicts += 1
                        continue
                    ci = p * l + p
                    if valid[b].flatten()[ci]:
                        transition.append(
                            (b, g, typ, "position", position * l + position, ci)
                        )
                if abs(position - opposite) > 2:
                    conflict = any(
                        abs(
                            opposite
                            - (
                                float(spans[b, h, col] / step[b])
                                - (1 if typ == "end" else 0)
                            )
                        )
                        <= 1
                        for h in other
                    )
                    ci = opposite * l + opposite
                    transition_conflicts += int(conflict)
                    if not conflict and valid[b].flatten()[ci]:
                        transition.append(
                            (b, g, typ, "direction", position * l + position, ci)
                        )
    return (
        geometry,
        anchors,
        support,
        transition,
        {
            "saliency_available": sal is not None,
            "saliency_usable_GT_count": saliency_gt_count,
            "expanded_otherGT_protection_rejected_candidates": expanded_rejected,
            "transition_same_type_conflicts": transition_conflicts,
            "anchor_count": len(anchors),
            "support_expand_count": sum(x[2] == "expand" for x in support),
            "support_truncate_count": sum(x[2] == "truncate" for x in support),
            "transition_position_count": sum(x[3] == "position" for x in transition),
            "transition_direction_count": sum(x[3] == "direction" for x in transition),
        },
    )


def _isolated_mask_forward(model, inputs, mask, key, rng, context=None, is_event=True):
    cp = dict(inputs)
    value = inputs[key]
    # Interface-specific masking dimensions are verified before this helper is enabled.
    channels = getattr(model, "counterfactual_visual_channels", None)
    if channels is None:
        if key != "video_features":
            raise RuntimeError(
                "verified visual channel count required for src_vid/TEF input"
            )
        channels = value.shape[-1]
    if not 0 < int(channels) <= value.shape[-1]:
        raise ValueError("invalid visual channel count")
    cp[key] = torch.cat(
        (value[..., :channels].masked_fill(mask[..., None], 0), value[..., channels:]),
        -1,
    )
    if context is not None and context["new_bits"] & 1:
        from .local_counterfactual import replace_local_rows

        cp[key] = replace_local_rows(value, cp[key], context, is_event, int(channels))
    torch.random.set_rng_state(rng["torch"])
    if rng["cuda"] is not None:
        torch.cuda.set_rng_state_all(rng["cuda"])
    random.setstate(rng["python"])
    if rng["numpy"] is not None:
        import numpy as np

        np.random.set_state(rng["numpy"])
    from torch.func import functional_call

    state = dict(model.parent_model.named_parameters())
    state.update({n: v.detach().clone() for n, v in model.parent_model.named_buffers()})
    result = functional_call(model.parent_model, state, (cp,))
    model._attach_raw_inputs(result.extension_output.state, cp)
    return model.selector.raw_from_state(
        result.extension_output.state,
        result.span_valid_mask.bool(),
        result.span_logits,
        video_padding_mask=inputs.get("video_padding_mask"),
    )


def _build_counterfactual_context(model, outputs, batch, epoch, pair):
    new_bits = 6
    bits = 7
    context = {
        "bits": bits,
        "new_bits": new_bits,
        "report": {
            "enabled_E": bool(bits & 1),
            "enabled_S": bool(bits & 2),
            "enabled_T": bool(bits & 4),
            "extra_backbone_forwards": 0,
            "new_E": bool(new_bits & 1),
            "new_S": bool(new_bits & 2),
            "new_T": bool(new_bits & 4),
        },
    }
    context["round31_bits"] = 7
    context["round31_mechanism_bits"] = int(getattr(model.options, "mechanism_bits", 0))
    if not bits:
        return context
    geometry, anchors, sp, tp, report = geometry_pairs(outputs, batch)
    context.update(geometry=geometry, support_pairs=sp, transition_pairs=tp)
    context["report"].update(report)
    context["wrong_field"] = getattr(model, "_last_wrong_query_field", None)
    context["support_shift_relaxed_fallback"] = bool(
        getattr(model, "support_shift_relaxed_fallback", False)
    )
    from .local_counterfactual import add_local_plans

    add_local_plans(context, outputs, batch, epoch, pair)
    if not bits & 1:
        return context
    inputs, targets, metadata = _batch_parts(batch)
    # Real input configuration must explicitly be verified, never infer TEF dimensions.
    key = getattr(model, "counterfactual_video_key", None)
    if key is None or key not in inputs:
        raise RuntimeError(
            "Round31 E requires verified counterfactual_video_key; do not guess visual/position channel layout"
        )
    video = inputs[key]
    b, l, _ = video.shape
    spans, gm = _target_spans(outputs, _pad_empty_gt(batch))
    spans = spans.detach().cpu()
    gm = gm.detach().cpu()
    step = geometry.one_step.detach().cpu()
    pad = inputs["video_padding_mask"].detach().cpu()
    rows = _metadata_rows(metadata, b)
    ev = torch.zeros(b, l, dtype=torch.bool)
    bg = ev.clone()
    selected = torch.full((b,), -1, dtype=torch.long)
    for bi in range(b):
        if pair is None or not bool(pair[bi]):
            continue
        gids = sorted(
            gm[bi].nonzero().flatten().tolist(),
            key=lambda gi: tuple(spans[bi, gi].tolist()),
        )
        if not gids:
            continue
        row = rows[bi]
        if row is None or row.get("qid") is None:
            continue
        stable = str(row.get("vid", "")) + "|" + str(row["qid"])
        gi = gids[
            (
                int.from_bytes(hashlib.sha256(stable.encode()).digest()[:8], "big")
                + int(epoch)
                - 1
            )
            % len(gids)
        ]
        s = float(spans[bi, gi].min() / step[bi])
        e = float(spans[bi, gi].max() / step[bi])
        idx = torch.arange(l)
        event = (idx.float() < e) & ((idx + 1).float() > s) & ~pad[bi]
        safe = ~pad[bi]
        for h in gids:
            gs = float(spans[bi, h].min() / step[bi]) - 1
            ge = float(spans[bi, h].max() / step[bi]) + 1
            safe &= ~((idx.float() < ge) & ((idx + 1).float() > gs))
        count = int(event.sum())
        available = safe.nonzero().flatten()
        if count and len(available) >= count:
            ev[bi] = event
            bg[bi, available[:count]] = True
            selected[bi] = gi
    context.update(
        selected_gt=selected.to(video.device),
        event_mask=ev.to(video.device),
        background_mask=bg.to(video.device),
    )
    if new_bits & 1:
        from .local_counterfactual import merge_local_masks

        merge_local_masks(context, ev, bg, selected, video)
    if not context["event_mask"].any():
        return context
    rng = {
        "torch": torch.random.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "python": random.getstate(),
        "numpy": None,
    }
    try:
        import numpy as np

        rng["numpy"] = np.random.get_state()
    except ImportError:
        pass
    try:
        context["event_field"] = _isolated_mask_forward(
            model, inputs, context["event_mask"], key, rng, context, True
        )
        context["background_field"] = _isolated_mask_forward(
            model, inputs, context["background_mask"], key, rng, context, False
        )
    finally:
        torch.random.set_rng_state(rng["torch"])
        if rng["cuda"] is not None:
            torch.cuda.set_rng_state_all(rng["cuda"])
        random.setstate(rng["python"])
        if rng["numpy"] is not None:
            np.random.set_state(rng["numpy"])
    context["report"].update(
        extra_backbone_forwards=2,
        mask_replacement=(
            "per_protocol_local_norm_preserving_or_whole_zero" if new_bits & 1 else 0.0
        ),
        mask_position_padding_unchanged=True,
        paired_masks_share_rng=True,
        extra_rng_restored=True,
        masked_event_counts=ev.sum(1).tolist(),
        masked_background_counts=bg.sum(1).tolist(),
        selected_gt=selected.tolist(),
    )
    return context


def apply_counterfactual_losses(
    outputs, batch, g, wrong, pair, evidence, support, transition, context
):
    bits = context["bits"]
    field = outputs.trifield_output
    report = context["report"]
    context["components"] = {}
    context["direction_measurements"] = []
    if bits & 1 and wrong is not None:
        mask = g.valid & g.max_iou.ge(0.7)
        if pair is not None:
            mask &= pair[:, None, None]
        hq = F.relu(0.2 - (field.evidence - wrong))
        active = torch.zeros_like(mask)
        if "event_field" in context:
            selected = context["selected_gt"]
            iou = g.all_iou.gather(
                -1, selected.clamp_min(0)[:, None, None, None].expand(*mask.shape, 1)
            ).squeeze(-1)
            active = mask & selected.ge(0)[:, None, None] & iou.ge(0.7)
            if context.get("new_bits", 0) & 1:
                from .local_counterfactual import restrict_local_candidates

                active = restrict_local_candidates(active, context)
            hm = F.relu(
                0.2
                - (
                    context["background_field"].evidence
                    - context["event_field"].evidence
                )
            )
            combined = torch.where(active, 0.5 * hq + 0.5 * hm, hq)
            context["components"]["E_mask_hinge"] = (
                hm[active].mean() if active.any() else _zero(field.raw_evidence)
            )
            report["E_mask_hinge_mean"] = float(
                context["components"]["E_mask_hinge"].detach()
            )
            evidence = (
                combined[mask].mean() if mask.any() else _zero(field.raw_evidence)
            )
        if context.get("round31_bits", 0) & 1:
            from .interaction import local_e_mask

            local = local_e_mask(outputs, batch, g)
            if pair is not None:
                local &= pair[:, None, None]
            union = mask | local
            combined = (
                torch.where(active, 0.5 * hq + 0.5 * hm, hq)
                if "event_field" in context
                else hq
            )
            evidence = (
                combined[union].mean() if union.any() else _zero(field.raw_evidence)
            )
            context["round31_local_mask"] = local
            report.update(
                round31_E_old_count=int(mask.sum()),
                round31_E_added_count=int((local & ~mask).sum()),
                round31_E_union_count=int(union.sum()),
            )
            mask = union
        context["components"]["E_wrong_query_hinge"] = (
            hq[mask].mean() if mask.any() else _zero(field.raw_evidence)
        )
        report["E_positive_counts"] = mask.flatten(1).sum(1).detach().cpu().tolist()
        report["E_new_counts"] = active.flatten(1).sum(1).detach().cpu().tolist()
        report.update(
            E_new_candidate_fraction=float(active.sum() / mask.sum().clamp_min(1)),
            E_bg_matched_query_count=int(active.flatten(1).any(1).sum()),
            E_fallback_query_count=int(
                (mask.flatten(1).any(1) & ~active.flatten(1).any(1)).sum()
            ),
            E_new_candidate_count=int(active.sum()),
            E_old_candidate_count=int((mask & ~active).sum()),
            E_total_positive_count=int(mask.sum()),
        )
    if context.get("new_bits", 0) & 2:
        from .local_counterfactual import replace_support_pairs

        replace_support_pairs(context)
    if bits & 2:
        rows = {}
        for b, gt, typ, ai, ci in context["support_pairs"]:
            f = field.support[b].flatten()
            _append(rows, b, gt, typ, F.relu(0.2 - (f[ai] - f[ci])))
        support = _balanced(rows, field.raw_support)
        report["S_active_queries"] = len(rows)
        report["S_active_GT_count"] = sum(len(gs) for gs in rows.values())
        report["S_loss"] = float(support.detach())
        context["components"]["S_counterfactual"] = support
    if bits & 4:
        # Each endpoint type balances position/direction families; then GT/query.
        rows = {}
        for b, gt, endpoint, family, ai, ci in context["transition_pairs"]:
            f = getattr(field, "transition_" + endpoint)[b].flatten()
            loss = F.relu(0.2 - (f[ai] - f[ci]))
            if family == "direction":
                from .local_counterfactual import query_direction_loss

                loss = query_direction_loss(
                    context, field, b, gt, endpoint, ai, ci, loss
                )
            rows.setdefault(b, {}).setdefault(gt, {}).setdefault(
                endpoint, {}
            ).setdefault(family, []).append(loss)
        balanced = {}
        for b, gs in rows.items():
            for gt, ends in gs.items():
                for endpoint, families in ends.items():
                    values = [torch.stack(v).mean() for v in families.values() if v]
                    if values:
                        _append(balanced, b, gt, endpoint, torch.stack(values).mean())
        transition = _balanced(
            balanced, field.raw_transition_start + field.raw_transition_end
        )
        report["T_active_queries"] = len(balanced)
        report["T_active_GT_count"] = sum(len(gs) for gs in balanced.values())
        report["T_loss"] = float(transition.detach())
        context["components"]["T_counterfactual"] = transition
    return (
        evidence,
        support,
        transition,
        {
            "counterfactual/" + k: v
            for k, v in report.items()
            if isinstance(v, (int, float, bool))
        },
    )


def build_counterfactual_context(model, outputs, batch, epoch, pair):
    # Scoped by the full probe runner only; never persist a training graph.
    cache = getattr(model, "_round31_probe_context_cache", None)
    key = (id(outputs), id(batch), int(epoch))
    if cache is not None and key in cache:
        return cache[key]
    result = _build_counterfactual_context(model, outputs, batch, epoch, pair)
    if cache is not None:
        cache[key] = result
    return result
