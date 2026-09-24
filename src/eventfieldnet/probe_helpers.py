"""R68 final-objective, actual parameter-route and target-matched S probes."""

import torch
from .quality_probes import actual_probe as base_actual_probe, scalar_tree
from .precision import snapshot_precision
from .gradient_routing import local_parameters
from .model_factory import (
    HINGE_OBJECTIVE,
    NEUTRAL_OBJECTIVE,
    LEGACY_PROJECTION_ROUTE,
    route_parameters,
)
from . import coverage_loss as s_helpers


def norm(value):
    return float(value.detach().float().norm()) if value is not None else 0.0


def norm_list(values):
    return (
        sum(float(v.detach().float().square().sum()) for v in values if v is not None)
        ** 0.5
    )


def derivative(value, tensor):
    if not value.requires_grad or not tensor.requires_grad:
        return None
    with torch.autocast(device_type=value.device.type, enabled=False):
        return torch.autograd.grad(value, tensor, retain_graph=True, allow_unused=True)[
            0
        ]


def gradients(value, named):
    if not value.requires_grad:
        return [torch.zeros_like(p) for _, p in named]
    with torch.autocast(device_type=value.device.type, enabled=False):
        grads = torch.autograd.grad(
            value, [p for _, p in named], retain_graph=True, allow_unused=True
        )
    return [torch.zeros_like(p) if g is None else g for (_, p), g in zip(named, grads)]


def cosine(a, b):
    den = norm_list(a) * norm_list(b)
    return (
        sum(
            float((x.detach().float() * y.detach().float()).sum()) for x, y in zip(a, b)
        )
        / den
        if den
        else None
    )


def routing_probe(
    model, field, terms, result, named, scale, scope, outside_error, strict=True
):
    parts = field.r68_loss_parts
    raw = gradients(parts["reference_rank"], named)
    applied = gradients(terms.rank, named)
    aux = gradients(model.loss_weights["support"] * terms.support, named)
    plain_total = gradients(parts["reference_total"], named)
    actual_total = gradients(result.loss, named)
    expected_total = [
        g + model.loss_weights["rank"] * (scale - 1.0) * r
        for g, r in zip(plain_total, raw)
    ]
    expected_rank_norm = norm_list([scale * g for g in raw])
    rank_error = norm_list([a - scale * b for a, b in zip(applied, raw)])
    total_error = norm_list([a - b for a, b in zip(actual_total, expected_total)])
    if strict:
        assert rank_error <= max(1e-8, expected_rank_norm * 2e-4), (
            scope,
            rank_error,
            expected_rank_norm,
        )
        assert total_error <= max(1e-8, norm_list(expected_total) * 2e-4), (
            scope,
            total_error,
            norm_list(expected_total),
        )
    weighted_raw = [model.loss_weights["rank"] * v for v in raw]
    weighted_applied = [model.loss_weights["rank"] * v for v in applied]
    aux_norm = norm_list(aux)
    return dict(
        scope=scope,
        declared_scale=scale,
        raw_rank_norm=norm_list(weighted_raw),
        applied_rank_norm=norm_list(weighted_applied),
        support_aux_norm=aux_norm,
        raw_s_aux_norm=aux_norm,
        rank_to_aux_ratio=norm_list(weighted_applied) / aux_norm if aux_norm else None,
        raw_rank_aux_cosine=cosine(weighted_raw, aux),
        applied_rank_aux_cosine=cosine(weighted_applied, aux),
        actual_scale_gradient_error=rank_error,
        expected_unweighted_rank_norm=expected_rank_norm,
        actual_total_local_error=total_error,
        expected_total_local_norm=norm_list(expected_total),
        outside_route_gradient_error=outside_error,
        outside_scope="all trainable parameters outside the complete declared local plus optional s_projection route; edge_head included only for declared followup edge route",
        parameters={
            name: dict(
                raw_rank_norm=norm(r), applied_rank_norm=norm(a), auxiliary_norm=norm(s)
            )
            for (name, _), r, a, s in zip(named, raw, applied, aux)
        },
        parameter_norm_scope="per-parameter rank entries unweighted; group rank and all auxiliary entries weighted",
    )


def declared_optimizer_policy(model):
    parent = {id(p) for p in model.parent_model.parameters() if p.requires_grad}
    groups = [
        dict(
            name=g.name,
            lr=g.lr,
            weight_decay=g.weight_decay,
            parameters=len(g.params),
            parent_parameters=sum(id(p) in parent for p in g.params),
            reference_lr_multipliers=[1.0],
        )
        for g in model.parameter_groups()
    ]
    return dict(
        scope="initial model-declared groups unchanged; this is not actual optimizer telemetry",
        late_parent_enabled=model.r68_late_parent_enabled,
        first_affected_epoch=11,
        multiplier_epoch10=model.r68_parent_lr_multiplier(10),
        multiplier_epoch11=model.r68_parent_lr_multiplier(11),
        actual_current_multiplier_requires_runtime_receipt=True,
        groups=groups,
    )


@torch.no_grad()
def readout_probe(model, field):
    head = model.selector.r66_centered_s_readout
    details = field.r59_support_details
    altered = dict(details, density_ratio=1.0 - details["density_ratio"])
    altered["quality"] = details["coverage"] * altered["density_ratio"]
    before, features = head(details)
    after, _ = head(altered)
    return dict(
        type=type(head).__name__,
        parameter_values=head.weight.detach().cpu().tolist(),
        feature_width=features.shape[-1],
        local_density_intervention_max_abs_change=float((after - before).abs().max()),
        scope="same J07 three-feature readout for all R68 arms; no density-independence claim",
    )


def coverage_gradient_probe(model, field, terms, strict=True):
    payload = getattr(field, "r68_coverage_payload", None)
    if payload is None:
        return None
    gap, target = payload["gap"], payload["target"]
    tr, ex = payload["truncation_mask"], payload["expansion_mask"]
    grad = derivative(model.loss_weights["support"] * terms.support, gap)
    if grad is None:
        assert not bool(payload["mask"].any()), (
            "Active actual coverage loss disconnected from its gaps"
        )
        grad = torch.zeros_like(gap)
    over = tr & (gap[..., :2].detach() > target[..., :2] + 1e-7)
    under = tr & (gap[..., :2].detach() < target[..., :2] - 1e-7)
    over_max = float(grad[..., :2][over].abs().max()) if over.any() else None
    bad_tr = int(((grad[..., :2] > 1e-9) & under).sum())
    bad_ex = int(((grad[..., 2:] * gap[..., 2:].detach() < -1e-9) & ex).sum())
    upper_enabled = getattr(model, "long_s_spec", {}).get("upper", False)
    upper = tr & (gap[..., :2].detach() > target[..., :2] + 0.050001)
    middle = (
        tr
        & (gap[..., :2].detach() > target[..., :2] + 1e-7)
        & (gap[..., :2].detach() < target[..., :2] + 0.05 - 1e-7)
    )
    if upper_enabled:
        assert not bool((grad[..., :2][upper] <= 0).any()), (
            "Upper-band derivative must reduce excessive S gap"
        )
        assert not bool((grad[..., :2][middle] != 0).any()), (
            "Inside band must be neutral"
        )
    analytic_error = 0.0
    if "gt_weights" in payload:
        mask = payload["mask"]
        fm = mask.reshape(*mask.shape[:2], 2, 2)
        fc = fm.sum(-1)
        active_gt = payload["active_gt"]
        active_q = active_gt.any(-1)
        qw = payload["query_weights"] * active_q
        gw = payload["gt_weights"] * active_gt
        coefficient = qw / qw.sum().clamp_min(1.0)
        coefficient = (
            coefficient[:, None] * gw / gw.sum(-1, keepdim=True).clamp_min(1.0)
        )
        coefficient = coefficient[:, :, None] / (fc > 0).sum(
            -1, keepdim=True
        ).clamp_min(1.0)
        coefficient = (
            coefficient[:, :, :, None] / fc[:, :, :, None].clamp_min(1)
        ).expand_as(fm).reshape_as(gap) * mask
        dg = torch.cat(
            (
                -(gap[..., :2].detach() < target[..., :2]).float(),
                (gap[..., 2:].detach() / 0.05).clamp(-1.0, 1.0),
            ),
            -1,
        )
        if upper_enabled:
            dg[..., :2] += (
                0.25 * (gap[..., :2].detach() > target[..., :2] + 0.05).float()
            )
        expected = model.loss_weights["support"] * coefficient * dg
        analytic_error = norm(grad - expected)
        if strict:
            assert analytic_error <= max(1e-8, norm(expected) * 2e-5), (
                analytic_error,
                norm(expected),
            )
    if strict:
        assert bool(torch.isfinite(grad).all()) and bad_tr == 0 and bad_ex == 0
        if (
            model.r68_mode in HINGE_OBJECTIVE
            and not upper_enabled
            and over_max is not None
        ):
            assert over_max == 0.0, "Hinge still penalizes an over-margin truncation"
    return dict(
        scope="actual weighted support loss derivative with respect to the very gaps used in its forward",
        hinge_truncation=model.r68_mode in HINGE_OBJECTIVE and not upper_enabled,
        upper_band_enabled=upper_enabled,
        upper_band_pairs=int(upper.sum()),
        actual_analytic_gradient_error=analytic_error,
        truncation_pairs=int(tr.sum()),
        expansion_pairs=int(ex.sum()),
        over_margin_truncation_pairs=int(over.sum()),
        under_margin_truncation_pairs=int(under.sum()),
        actual_over_margin_gradient_maxabs=over_max,
        truncation_gradient_norm=norm(grad[..., :2]),
        expansion_gradient_norm=norm(grad[..., 2:]),
        truncation_wrong_direction_count=bad_tr,
        expansion_wrong_direction_count=bad_ex,
        actual_pair_gap_gradient_finite=bool(torch.isfinite(grad).all()),
    )


def short_kl_probe(model, field, strict=True):
    p = getattr(field, "r68_short_kl_payload", None)
    if p is None:
        return None
    active, weights = p["active"], p["query_weight"]
    effective = weights * active
    normalized = effective / effective.sum().clamp_min(1.0)
    query, length, value = p["query_kl"], p["length_factor"], p["weighted_kl"]
    dq = derivative(value, query)
    expected = normalized * length
    query_error = (
        norm(dq - expected)
        if dq is not None
        else (norm(expected) if active.any() else 0.0)
    )
    dlength = derivative(value, length)
    length_error = norm(dlength - normalized * query) if dlength is not None else None
    reference = field.r68_loss_parts["reference_mixed_kl"]
    reference_error = float((p["reference_kl"] - reference).detach().abs())
    changed = derivative(model.loss_weights["rank"] * (value - reference), field.score)
    count_short = int(p["short_mask"].sum())
    count_valid = int(p["valid_gt_mask"].sum())
    rows = []
    for i in range(query.shape[0]):
        rows.append(
            dict(
                active=bool(active[i]),
                valid_gt=int(p["valid_gt_mask"][i].sum()),
                short_gt=int(p["short_mask"][i].sum()),
                known_duration_gt=int(
                    (p["valid_gt_mask"][i] & p["known_duration_mask"][i]).sum()
                ),
                short_fraction=float(p["short_fraction"][i]),
                weight=float(weights[i]),
                normalized_weight=float(normalized[i]),
                query_kl=float(query[i].detach()),
                length_factor=float(length[i].detach()),
                actual_dKL_dquery=float(dq[i].detach()) if dq is not None else None,
                analytic_dKL_dlength=float((normalized[i] * query[i]).detach()),
            )
        )
    groups = {}
    for name, member in [
        ("contains_short_gt", p["short_mask"].any(-1)),
        ("no_short_gt", ~p["short_mask"].any(-1)),
    ]:
        contribution = (query * length * normalized * member).sum()
        grad = derivative(model.loss_weights["rank"] * contribution, field.score)
        groups[name] = dict(
            active_queries=int((member & active).sum()),
            actual_score_gradient_norm=norm(grad),
        )
    if strict:
        assert reference_error < 1e-6, (
            "L06 reference query KL differs from actual replaced KL"
        )
        assert query_error <= max(1e-8, norm(expected) * 1e-5), (
            query_error,
            norm(expected),
        )
        if length_error is not None:
            assert length_error <= max(1e-8, norm(normalized * query) * 1e-5)
        assert (
            bool(torch.isfinite(weights).all())
            and bool((weights >= 1.0).all())
            and bool((weights <= 2.0).all())
        )
    return dict(
        scope="actual mixed-KL only; fixed existing length multiplier, GT-only weights; no inference GT",
        valid_gt=count_valid,
        short_gt=count_short,
        known_duration_gt=int((p["valid_gt_mask"] & p["known_duration_mask"]).sum()),
        unknown_duration_gt=int((p["valid_gt_mask"] & ~p["known_duration_mask"]).sum()),
        active_queries=int(active.sum()),
        active_queries_with_short_gt=int((active & p["short_mask"].any(-1)).sum()),
        active_weight_sum=float(effective.sum()),
        normalized_weight_sum=float(normalized.sum()),
        actual_short_kl_change_norm=norm(changed),
        reference_kl_error=reference_error,
        actual_query_gradient_error=query_error,
        actual_length_gradient_error=length_error,
        expected_query_gradient_norm=norm(expected),
        expected_length_gradient_norm=norm(normalized * query),
        length_factor_requires_grad=bool(length.requires_grad),
        length_gradient_scope="when GT-derived length does not require grad, analytic sensitivity is reported separately, never called an actual training derivative",
        per_query=rows,
        normalized_weights=normalized.detach().cpu().tolist(),
        score_gradient_by_query_group=groups,
        old=float(reference.detach()),
        new=float(value.detach()),
    )


def long_kl_probe(model, field, strict=True):
    p = getattr(field, "r68_long_kl_payload", None)
    if p is None:
        return None
    active, weights = p["active"], p["query_weight"]
    effective = weights * active
    normalized = effective / effective.sum().clamp_min(1.0)
    query, length, value = p["query_kl"], p["length_factor"], p["weighted_kl"]
    dq = derivative(value, query)
    expected = normalized * length
    query_error = (
        norm(dq - expected)
        if dq is not None
        else (norm(expected) if active.any() else 0.0)
    )
    dlength = derivative(value, length)
    length_error = norm(dlength - normalized * query) if dlength is not None else None
    reference = field.r68_loss_parts["reference_mixed_kl"]
    reference_error = float((p["reference_kl"] - reference).detach().abs())
    changed = derivative(model.loss_weights["rank"] * (value - reference), field.score)
    count_long = int(p["long_mask"].sum())
    count_valid = int(p["valid_gt_mask"].sum())
    rows = []
    for i in range(query.shape[0]):
        rows.append(
            dict(
                active=bool(active[i]),
                valid_gt=int(p["valid_gt_mask"][i].sum()),
                long_gt=int(p["long_mask"][i].sum()),
                known_duration_gt=int(
                    (p["valid_gt_mask"][i] & p["known_duration_mask"][i]).sum()
                ),
                long_fraction=float(p["long_fraction"][i]),
                weight=float(weights[i]),
                normalized_weight=float(normalized[i]),
                query_kl=float(query[i].detach()),
                length_factor=float(length[i].detach()),
                actual_dKL_dquery=float(dq[i].detach()) if dq is not None else None,
                analytic_dKL_dlength=float((normalized[i] * query[i]).detach()),
            )
        )
    groups = {}
    for name, member in [
        ("contains_long_gt", p["long_mask"].any(-1)),
        ("no_long_gt", ~p["long_mask"].any(-1)),
    ]:
        contribution = (query * length * normalized * member).sum()
        grad = derivative(model.loss_weights["rank"] * contribution, field.score)
        groups[name] = dict(
            active_queries=int((member & active).sum()),
            actual_score_gradient_norm=norm(grad),
        )
    if strict:
        assert reference_error < 1e-6, (
            "L06 reference query KL differs from actual replaced KL"
        )
        assert query_error <= max(1e-8, norm(expected) * 1e-5), (
            query_error,
            norm(expected),
        )
        if length_error is not None:
            assert length_error <= max(1e-8, norm(normalized * query) * 1e-5)
        assert (
            bool(torch.isfinite(weights).all())
            and bool((weights >= 1.0).all())
            and bool((weights <= 2.0).all())
        )
    return dict(
        scope="actual mixed-KL only; fixed existing length multiplier, GT-only weights; no inference GT",
        valid_gt=count_valid,
        long_gt=count_long,
        known_duration_gt=int((p["valid_gt_mask"] & p["known_duration_mask"]).sum()),
        unknown_duration_gt=int((p["valid_gt_mask"] & ~p["known_duration_mask"]).sum()),
        active_queries=int(active.sum()),
        active_queries_with_long_gt=int((active & p["long_mask"].any(-1)).sum()),
        active_weight_sum=float(effective.sum()),
        normalized_weight_sum=float(normalized.sum()),
        actual_long_kl_change_norm=norm(changed),
        reference_kl_error=reference_error,
        actual_query_gradient_error=query_error,
        actual_length_gradient_error=length_error,
        expected_query_gradient_norm=norm(expected),
        expected_length_gradient_norm=norm(normalized * query),
        length_factor_requires_grad=bool(length.requires_grad),
        length_gradient_scope="when GT-derived length does not require grad, analytic sensitivity is reported separately, never called an actual training derivative",
        per_query=rows,
        normalized_weights=normalized.detach().cpu().tolist(),
        score_gradient_by_query_group=groups,
        old=float(reference.detach()),
        new=float(value.detach()),
    )


def actual_probe(model, field, terms, result, strict=True):
    record = base_actual_probe(model, field, terms, result, strict=strict)
    record["inherited_s_encoder_probe_scope"] = (
        "inherited raw_rank label differentiates already-routed terms.rank; exact unrouted final rank and actual correction are reported under r68_s_projection_routing"
    )
    record["routing_zero_value_only_not_zero_gradient"] = True
    record.update(
        r66_mode=model.r66_mode,
        r68_mode=model.r68_mode,
        precision_policy=snapshot_precision(),
        r66_actual_objective=scalar_tree(model.r66_last_training),
        r68_actual_objective=scalar_tree(model.r68_last_training),
    )
    assert (
        not record["precision_policy"]["matmul_allow_tf32"]
        and not record["precision_policy"]["cudnn_allow_tf32"]
    )
    assert record["precision_policy"]["float32_matmul_precision"] == "highest"
    inherited = field.r66_loss_parts
    expected_base_rank = inherited["reference_rank"] + (
        inherited["actual_mixed_kl"] - inherited["old_mixed_kl"]
    )
    parts = field.r68_loss_parts
    expected_rank = expected_base_rank
    expected_support = parts.get(
        "actual_coverage_objective", parts["reference_support"]
    )
    expected_total = parts["base_reference_total"]
    if model.r68_mode in NEUTRAL_OBJECTIVE:
        expected_total = expected_total + model.loss_weights["support"] * (
            expected_support - parts["reference_support"]
        )
    if model.r68_mode == "short_balanced_kl" or getattr(model, "followup_spec", {}).get(
        "long_kl", False
    ):
        change = parts["actual_mixed_kl"] - parts["reference_mixed_kl"]
        expected_rank = expected_rank + change
        expected_total = expected_total + model.loss_weights["rank"] * change
    correction = parts["rank_gradient_correction"]
    expected_rank = expected_rank + correction
    expected_total = expected_total + model.loss_weights["rank"] * correction
    identities = dict(
        base_rank_value_error=float(
            (parts["base_reference_rank"] - expected_base_rank).detach().abs()
        ),
        rank_value_error=float((terms.rank - expected_rank).detach().abs()),
        support_value_error=float((terms.support - expected_support).detach().abs()),
        actual_total_error=float((result.loss - expected_total).detach().abs()),
    )
    dg = derivative(terms.rank, field.score)
    eg = derivative(expected_rank, field.score)
    if strict:
        assert (dg is None) == (eg is None)
    identities["rank_score_derivative_error"] = (
        norm(dg - eg) if dg is not None and eg is not None else 0.0
    )
    if strict:
        assert max(identities.values()) < 1e-5, identities
    record["r68_objective_identities"] = identities
    record["r66_objective_identities"] = dict(identities)
    inherited_kl_delta = model.loss_weights["rank"] * (
        inherited["actual_mixed_kl"] - inherited["old_mixed_kl"]
    )
    record["actual_kl_substitution"] = dict(
        old=float(inherited["old_mixed_kl"].detach()),
        new=float(inherited["actual_mixed_kl"].detach()),
        target_exponent=2.0,
        candidate_pool_unchanged=True,
        actual_score_gradient_change_norm=norm(
            derivative(inherited_kl_delta, field.score)
        ),
        scope="inherited exponent4-to2 actual mixed-KL substitution; L06 extra query weights are separately measured in r68_short_kl",
    )
    named_route = route_parameters(model)
    routed_ids = {id(p) for _, p in named_route}
    outside = [
        (n, p)
        for n, p in model.named_parameters()
        if p.requires_grad and id(p) not in routed_ids
    ]
    outside_error = norm_list(gradients(correction, outside))
    if strict:
        assert outside_error == 0.0, (
            "R68 correction leaks outside its allowed parameters"
        )
    local = local_parameters(model)
    projection = [
        ("selector.s_projection." + n, p)
        for n, p in model.selector.s_projection.named_parameters()
        if p.requires_grad
    ]
    record["r68_local_routing"] = routing_probe(
        model,
        field,
        terms,
        result,
        local,
        0.1,
        "local token gate and readout",
        outside_error,
        strict,
    )
    ps = 0.1 if model.r68_mode in LEGACY_PROJECTION_ROUTE else 1.0
    record["r68_s_projection_routing"] = routing_probe(
        model,
        field,
        terms,
        result,
        projection,
        ps,
        "selector.s_projection only; excludes selector.edge_head",
        outside_error,
        strict,
    )
    record["r68_routing"] = dict(
        scope="complete explicitly allowed correction set",
        names=[n for n, _ in named_route],
        outside_route_gradient_error=outside_error,
        correction_numeric_value=float(correction.detach()),
        local_scale=0.1,
        s_projection_scale=ps,
        edge_head_scale=0.1
        if getattr(model, "followup_spec", {}).get("edge_route", False)
        else 1.0,
    )
    record["r68_optimizer_policy"] = declared_optimizer_policy(model)
    record["r68_readout"] = readout_probe(model, field)
    record["r68_coverage_gradient"] = coverage_gradient_probe(
        model, field, terms, strict
    )
    record["r68_short_kl"] = short_kl_probe(model, field, strict)
    record["r68_long_kl"] = long_kl_probe(model, field, strict)
    edge = [
        ("selector.edge_head." + n, p)
        for n, p in model.selector.edge_head.named_parameters()
        if p.requires_grad
    ]
    es = 0.1 if getattr(model, "followup_spec", {}).get("edge_route", False) else 1.0
    record["followup_edge_route"] = routing_probe(
        model,
        field,
        terms,
        result,
        edge,
        es,
        "edge head rank-only route; auxiliary unchanged",
        outside_error,
        strict,
    )
    parameters = [
        (n, p)
        for n, p in model.named_parameters()
        if p.requires_grad
        and any(
            x in n
            for x in (
                "selector.s_projection.",
                "selector.edge_head.",
                "r66_centered_s_readout.",
                "r58_calibration_bias",
                "r59_local_support.weight",
            )
        )
    ]
    record["r68_structure_derivatives"] = {}
    for key, value in [
        ("rank", model.loss_weights["rank"] * terms.rank),
        ("support", model.loss_weights["support"] * terms.support),
        ("total", result.loss),
    ]:
        gs = gradients(value, parameters)
        record["r68_structure_derivatives"][key] = {
            n: dict(norm=norm(g), finite=bool(torch.isfinite(g).all()))
            for (n, _), g in zip(parameters, gs)
        }
    record["r66_structure_derivatives"] = record["r68_structure_derivatives"]
    record["r66_local_s"] = scalar_tree(field.r66_s_stats)
    return record


@torch.no_grad()
def coverage_consistency_panel(
    field, geometry, spans, gt_mask, batch, *, hinge_truncation=False
):
    from .length_objective import durations

    duration = durations(batch, spans)[:, None]
    seconds = (spans[..., 1] - spans[..., 0]).clamp_min(0.0) * duration
    known = gt_mask & torch.isfinite(duration) & (duration > 0.0)
    buckets = {
        "all": gt_mask,
        "short_0_10s": known & (seconds <= 10.0),
        "middle_10_30s": known & (seconds > 10.0) & (seconds <= 30.0),
        "long_over30s": known & (seconds > 30.0),
    }
    result = dict(
        scope="K06 exact counterfactual pairs; whole-S neutrality, all other GT exclusions retained; fixed panel, not official MR",
        hinge_truncation=hinge_truncation,
        neutral_tolerance=0.02,
        buckets={},
    )
    for name, report_mask in buckets.items():
        _, stats = s_helpers.coverage_gap_objective(
            field,
            geometry,
            spans,
            gt_mask,
            report_gt_mask=report_mask,
            hinge_truncation=hinge_truncation,
        )
        result["buckets"][name] = dict(
            report_gt_count=int(report_mask.sum()),
            pair_counts=dict(
                truncation=int(stats["truncation/pairs"]),
                expansion=int(stats["expansion/pairs"]),
            ),
            truncation_positive_gap_fraction=float(
                stats["truncation/positive_gap_fraction"]
            ),
            truncation_s_gap=float(stats["truncation/s_gap"]),
            truncation_target_gap=float(stats["truncation/target_gap"]),
            truncation_coverage_gap=float(stats["truncation/coverage_gap"]),
            neutral_abs_gap=float(stats["expansion/absolute_error"]),
            expansion_s_gap=float(stats["expansion/s_gap"]),
            expansion_coverage_gap=float(stats["expansion/coverage_gap"]),
            expansion_coverage_gap_abs=float(stats["expansion/coverage_gap_abs"]),
            expansion_nonneutral_fraction=float(stats["expansion/nonneutral_fraction"]),
            actual_objective_stats=scalar_tree(stats),
        )
    return result
