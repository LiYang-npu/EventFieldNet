"""Bounded diagnostics for the frozen Round31 factors, not semantic labels."""

import torch


def projection_probe(model, field, alive):
    e = model.selector.e_interaction
    s = getattr(model.selector, "s_projection", None)
    independent = bool(model.options.independent_s_projection)
    if independent != (s is not None):
        raise RuntimeError("Independent S projection does not match configuration")
    result = {"independent": independent, "additional_parameters": 0}
    if s is not None:
        named = dict(s.named_parameters())
        expected = {"video.weight", "key.weight", "value.weight"}
        temporal = bool(getattr(model, "r48_spec", {}).get("temporal", False))
        if temporal:
            expected.add("r48_temporal.weight")
        result["temporal_context_enabled"] = temporal
        if set(named) != expected:
            raise RuntimeError("S encoder must contain only three projection matrices")
        other = dict(e.named_parameters())
        result["additional_parameters"] = sum(p.numel() for p in named.values())
        result["separate_storage"] = all(
            p is not other[n] and p.data_ptr() != other[n].data_ptr()
            for n, p in named.items()
        )
        groups = model.parameter_groups()
        result["same_optimizer_groups_as_E"] = all(
            [i for i, g in enumerate(groups) if any(p is v for v in g.params)]
            == [i for i, g in enumerate(groups) if any(other[n] is v for v in g.params)]
            for n, p in named.items()
        )
        result["parameter_distance_from_E"] = {
            n: float((p.detach() - other[n].detach()).norm()) for n, p in named.items()
        }
        if (
            result["additional_parameters"] != (98304 + (192 if temporal else 0))
            or not result["separate_storage"]
            or not result["same_optimizer_groups_as_E"]
        ):
            raise RuntimeError(
                "Independent S ownership or optimizer group contract failed"
            )
    ze, zs = field.round31_z_e.detach(), field.round31_z_s.detach()
    result["representation_distance_l2"] = float((ze - zs)[alive].norm())
    result["representation_initial_equality_is_not_a_training_constraint"] = True
    return result


def query_strata(probe):
    if "detail" not in probe:
        return {"enabled": False}
    per = probe["detail"]["per_query"]
    sims = probe["jaccard_by_query"]
    result = {"enabled": True, "coverage": probe["coverage"], "strata": {}}
    for name, lo, hi in [
        ("lt025", 0.0, 0.25),
        ("025_05", 0.25, 0.5),
        ("ge05", 0.5, 1.01),
    ]:
        selected = (
            torch.tensor(
                [s is not None and lo <= s < hi for s in sims],
                device=per["relative"].device,
            )
            & per["relative"]
        )
        result["strata"][name] = {"active_relative_queries": int(selected.sum())}
        for key in (
            "clean_hinge",
            "difference_hinge",
            "clean_gap",
            "wrong_gap",
            "difference_gap",
        ):
            result["strata"][name][key] = (
                float(per[key][selected].mean()) if selected.any() else None
            )
    return result


def span_aggregation_probe(model, field, alive):
    """Check actual deployed VJP against an independent softmax/mean reference.

    Three deterministic valid spans in at most three queries bound autograd cost. Full-grid
    deployment identity remains in semantic_probe. Never runs another parent.
    """
    e = field.round31_e_token
    rows = []
    queries = field.valid.flatten(1).any(1).nonzero().flatten().tolist()
    queries = (
        sorted(set(queries[j] for j in (0, len(queries) // 2, len(queries) - 1)))
        if queries
        else []
    )
    for b in queries:
        candidates = field.valid[b].nonzero()
        if not len(candidates):
            continue
        lengths = candidates[:, 1] - candidates[:, 0] + 1
        order = lengths.argsort(stable=True)
        picks = sorted(
            set([int(order[0]), int(order[len(order) // 2]), int(order[-1])])
        )
        for ix in picks:
            start, end = (int(x) for x in candidates[ix])
            value = field.evidence[b, start, end]
            reference = torch.zeros_like(e)
            local = e[b, start : end + 1].detach().float()
            weights = (
                torch.softmax(local / 0.2, 0)
                if model.options.e_span_lme
                else torch.full_like(local, 1.0 / len(local))
            )
            reference[b, start : end + 1] = weights
            grad = torch.autograd.grad(value, e, retain_graph=True, allow_unused=True)[
                0
            ]
            if grad is None:
                raise RuntimeError("Deployed E disconnected from token E")
            error = float((grad - reference).abs().max())
            detached = float(value.detach())
            rows.append(
                dict(
                    query=b,
                    start=start,
                    end=end,
                    length=len(local),
                    vjp_max_abs_error=error,
                    gradient_sum=float(grad.sum()),
                    padding_gradient_max=float(grad[~alive].abs().max())
                    if (~alive).any()
                    else 0.0,
                    within_token_range=bool(
                        local.min() - 2e-6 <= detached <= local.max() + 2e-6
                    ),
                    effective_tokens=float(1.0 / weights.square().sum()),
                    maximum_token_weight=float(weights.max()),
                    gain_over_mean=detached - float(local.mean()),
                )
            )
    return {
        "mode": "log_mean_exp_tau_0.2" if model.options.e_span_lme else "mean",
        "scope": "at most nine VJPs: first/middle/last active query, each shortest/median/longest candidate",
        "rows": rows,
        "vjp_passed": all(
            r["vjp_max_abs_error"] < 3e-5
            and abs(r["gradient_sum"] - 1.0) < 3e-5
            and r["padding_gradient_max"] == 0.0
            and r["within_token_range"]
            for r in rows
        ),
    }


def aggregation_anchor_probe(field, geometry):
    """Compare E around each best-IoU grid anchor; labels only select probes."""
    from field_core.selector import span_mean

    means = span_mean(field.round31_e_token.detach()[..., None]).squeeze(-1)
    rows = []
    length = field.valid.shape[-1]
    for b in range(field.valid.shape[0]):
        for g in range(geometry.all_iou.shape[-1]):
            iou = geometry.all_iou[b, ..., g]
            flat = iou.masked_fill(~field.valid[b], -1).flatten()
            if not bool(flat.max() > 0):
                continue
            index = int(flat.argmax())
            start, end = divmod(index, length)
            spans = {
                "anchor": (start, end),
                "truncate_left": (start + 1, end),
                "truncate_right": (start, end - 1),
                "overwide": (max(0, start - 1), min(length - 1, end + 1)),
            }
            record = {"query": b, "gt_index": g, "spans": {}}
            for name, (s, e) in spans.items():
                if 0 <= s <= e < length and bool(field.valid[b, s, e]):
                    record["spans"][name] = dict(
                        start=s,
                        end=e,
                        iou=float(iou[s, e]),
                        deployed_E=float(field.evidence[b, s, e].detach()),
                        mean_E=float(means[b, s, e]),
                    )
            rows.append(record)
    return {
        "scope": "fixed-panel GT best-IoU grid anchors and one-clip perturbations; not official MR",
        "rows": rows,
    }


SUPPORT_READOUTS = ("edge_mean", "pooled", "unordered_halves", "ordered_halves")


def support_reference(prepared_tokens, head, mode):
    """Independent direct-slice reference; input is already token-RMS prepared.

    No production dispatcher call, no second RMS, no padding admitted here.
    """
    if mode not in SUPPORT_READOUTS:
        raise ValueError("unknown support readout")
    u = prepared_tokens.float()
    if u.ndim != 2 or u.shape[-1] != 64:
        raise ValueError("reference requires [tokens,64]")
    if len(u) < 2:
        return u.sum() * 0.0

    def g(a, b):
        return head(torch.cat((a, b, b - a, a * b), -1)).squeeze(-1).float()

    def joint(a, b):
        za, zb = torch.zeros_like(a), torch.zeros_like(b)
        return g(a, b) - g(a, zb) - g(za, b) + g(za, zb)

    if mode == "edge_mean":
        return joint(u[:-1], u[1:]).tanh().mean()
    m = u.mean(0)
    if mode == "pooled":
        return joint(m, m).tanh()
    cut = len(u) // 2
    a, b = u[:cut].mean(0), u[cut:].mean(0)
    raw = joint(a, b)
    if mode == "unordered_halves":
        raw = (raw + joint(b, a)) * 0.5
    return raw.tanh()


def support_readout_probe(model, field, alive):
    """Bounded real-deployment checks; synthetic algebra is not semantic accuracy."""
    mode = str(model.options.support_readout)
    if mode not in SUPPORT_READOUTS:
        raise RuntimeError("invalid actual support_readout")
    h = field.round31_z_s
    with torch.autocast(device_type=h.device.type, enabled=False):
        u = h.float() / (h.float().square().mean(-1, keepdim=True) + 1e-6).sqrt()
        head = model.selector.edge_head
        parent = field.valid.bool()
        length = parent.shape[-1]
        prefix = torch.nn.functional.pad(alive.long().cumsum(-1), (1, 0))
        ids = torch.arange(length, device=h.device)
        widths = ids[None, :] - ids[:, None] + 1
        interval = (widths > 0)[None] & (
            (prefix[:, ids + 1][:, None, :] - prefix[:, ids][:, :, None])
            == widths[None]
        )
        support_valid = parent & interval
        holes = parent & ~interval
        rows = []
        qs = parent.flatten(1).any(1).nonzero().flatten().tolist()
        qs = sorted(set(qs[j] for j in (0, len(qs) // 2, len(qs) - 1))) if qs else []
        for q in qs:
            spans = parent[q].nonzero()
            order = (spans[:, 1] - spans[:, 0]).argsort(stable=True)
            picks = sorted(
                set(int(order[j]) for j in (0, len(order) // 2, len(order) - 1))
            )
            for ix in picks:
                start, end = map(int, spans[ix])
                ok = bool(support_valid[q, start, end])
                ref = (
                    support_reference(u[q, start : end + 1], head, mode)
                    if ok
                    else u[q].sum() * 0.0
                )
                actual = field.support[q, start, end]
                ge = None
                if torch.is_grad_enabled() and actual.requires_grad:
                    params = [h] + list(head.parameters())
                    ga = torch.autograd.grad(
                        actual, params, retain_graph=True, allow_unused=True
                    )
                    gr = torch.autograd.grad(
                        ref, params, retain_graph=True, allow_unused=True
                    )
                    ge = max(
                        float(
                            (
                                (torch.zeros_like(p) if a is None else a)
                                - (torch.zeros_like(p) if b is None else b)
                            )
                            .detach()
                            .abs()
                            .max()
                        )
                        for p, a, b in zip(params, ga, gr)
                    )
                rows.append(
                    dict(
                        query=q,
                        start=start,
                        end=end,
                        length=end - start + 1,
                        support_interval_valid=ok,
                        value_max_abs_error=float((actual - ref).detach().abs()),
                        input_and_all_head_vjp_max_abs_error=ge,
                    )
                )
        holemax = (
            float(field.support[holes].detach().abs().max()) if holes.any() else 0.0
        )
        return dict(
            mode=mode,
            formula_source="independent direct-slice prepared-token reference",
            scope="at most nine actual candidates; not full-grid gradient verification or semantic labels",
            parent_candidate_count=int(parent.sum()),
            support_interval_valid_count=int(support_valid.sum()),
            internal_hole_candidate_count=int(holes.sum()),
            internal_hole_support_max_abs=holemax,
            parent_mask_source="field.valid; never narrowed to support validity",
            head_parameter_count=sum(p.numel() for p in head.parameters()),
            single_token_zero=bool(
                field.support.diagonal(dim1=1, dim2=2)[parent.diagonal(dim1=1, dim2=2)]
                .eq(0)
                .all()
            ),
            candidate_pool_structure=candidate_pool_structure(u, head, mode, rows),
            rows=rows,
            reference_passed=all(
                r["value_max_abs_error"] < 3e-5
                and (
                    r["input_and_all_head_vjp_max_abs_error"] is None
                    or r["input_and_all_head_vjp_max_abs_error"] < 1e-4
                )
                for r in rows
            )
            and holemax == 0.0,
            gradient_checked=bool(rows)
            and all(
                r["input_and_all_head_vjp_max_abs_error"] is not None for r in rows
            ),
            semantic_accuracy_status="not_identified",
        )


def support_pair_dispatch_probe(model, field):
    """Check the actual length-two donor API, not donor eligibility or semantics."""
    h = field.round31_z_s
    valid = field.round31_edge_valid.bool()
    coords = valid.nonzero()
    picks = sorted(set((0, len(coords) // 2, len(coords) - 1))) if len(coords) else []
    rows = []
    with torch.autocast(device_type=h.device.type, enabled=False):
        for ix in picks:
            q, t = map(int, coords[ix])
            tokens = h[q, t : t + 2]
            u = (
                tokens.float()
                / (tokens.float().square().mean(-1, keepdim=True) + 1e-6).sqrt()
            )
            reference = support_reference(
                u, model.selector.edge_head, model.options.support_readout
            )
            actual = model.selector.support_score_from_pairs(tokens[0], tokens[1])
            ge = None
            if torch.is_grad_enabled() and actual.requires_grad:
                params = [h] + list(model.selector.edge_head.parameters())
                ga = torch.autograd.grad(
                    actual, params, retain_graph=True, allow_unused=True
                )
                gr = torch.autograd.grad(
                    reference, params, retain_graph=True, allow_unused=True
                )
                ge = max(
                    float(
                        (
                            (torch.zeros_like(p) if a is None else a)
                            - (torch.zeros_like(p) if b is None else b)
                        )
                        .detach()
                        .abs()
                        .max()
                    )
                    for p, a, b in zip(params, ga, gr)
                )
            rows.append(
                dict(
                    query=q,
                    start=t,
                    value_max_abs_error=float((actual - reference).detach().abs()),
                    input_and_all_head_vjp_max_abs_error=ge,
                )
            )
    return dict(
        mode=model.options.support_readout,
        scope="up to three actual valid adjacent inputs through donor length-two API; not fallback coverage or semantic truth",
        rows=rows,
        value_tolerance=3e-5,
        vjp_tolerance=1e-4,
        reference_passed=all(
            r["value_max_abs_error"] < 3e-5
            and (
                r["input_and_all_head_vjp_max_abs_error"] is None
                or r["input_and_all_head_vjp_max_abs_error"] < 1e-4
            )
            for r in rows
        ),
        gradient_checked=bool(rows)
        and all(r["input_and_all_head_vjp_max_abs_error"] is not None for r in rows),
    )


@torch.no_grad()
def candidate_pool_structure(prepared, head, mode, rows):
    """Given prepared halves only: no full-sequence reversal/semantic assertion."""
    samples = []
    ms = []
    ls = []
    rs = []

    def joint(a, b):
        z = torch.zeros_like(a)

        def g(x, y):
            return head(torch.cat((x, y, y - x, x * y), -1)).squeeze(-1).float()

        return g(a, b) - g(a, z) - g(z, b) + g(z, z)

    for row in rows:
        if not row["support_interval_valid"] or row["length"] < 2:
            continue
        q, start, end = row["query"], row["start"], row["end"]
        u = prepared[q, start : end + 1].detach().float()
        cut = len(u) // 2
        m = u.mean(0)
        l = u[:cut].mean(0)
        r = u[cut:].mean(0)
        lr, rl = joint(l, r), joint(r, l)
        ordered, swapped = lr.tanh(), rl.tanh()
        unordered = ((lr + rl) * 0.5).tanh()
        unordered_swap = ((rl + lr) * 0.5).tanh()
        ms.append(m)
        ls.append(l)
        rs.append(r)
        samples.append(
            dict(
                query=q,
                start=start,
                end=end,
                length=len(u),
                left_length=cut,
                right_length=len(u) - cut,
                left_right_l2=float((l - r).norm()),
                left_right_rms=float((l - r).square().mean().sqrt()),
                ordered_value=float(ordered),
                ordered_given_halves_swap_value=float(swapped),
                ordered_swap_signed_delta=float(swapped - ordered),
                unordered_value=float(unordered),
                unordered_given_halves_swap_value=float(unordered_swap),
                unordered_swap_abs_error=float((unordered_swap - unordered).abs()),
                pooled_value=float(joint(m, m).tanh()),
            )
        )

    def geometry(values):
        if not values:
            return dict(
                sample_count=0,
                available=False,
                channel_variance_mean=None,
                effective_energy_rank=None,
            )
        x = torch.stack(values).float()
        centered = x - x.mean(0, keepdim=True)
        energy = torch.linalg.svdvals(centered).square()
        total = energy.sum()
        p = energy / total.clamp_min(1e-30)
        rank = (
            float(torch.exp(-(p * p.clamp_min(1e-30).log()).sum()))
            if bool(total > 0)
            else 0.0
        )
        return dict(
            sample_count=len(values),
            available=True,
            finite=bool(torch.isfinite(x).all()),
            signed_mean=float(x.mean()),
            abs_mean=float(x.abs().mean()),
            rms=float(x.square().mean().sqrt()),
            channel_variance_mean=float(centered.square().mean()),
            effective_energy_rank=rank,
            rank_definition="entropy of squared singular values after centering across sampled candidates; zero when no centered energy",
        )

    return dict(
        sample_count=len(samples),
        max_sample_count=9,
        scope="same sampled real valid candidates n>=2; token-RMS once before pooling; detach/no_grad, no extra model forward",
        deployment_mode=mode,
        pool_statistics_role="bounded pooled-vector diagnostics; see per-input roles",
        pool_input_roles={
            "mean_input": "deployment input" if mode == "pooled" else "reference only",
            "left_input": "deployment input"
            if mode in ("unordered_halves", "ordered_halves")
            else "reference only",
            "right_input": "deployment input"
            if mode in ("unordered_halves", "ordered_halves")
            else "reference only",
        },
        swap_scope="given prepared l/r exchange; not whole-sequence reversal (especially odd lengths)",
        swap_deployment_role="unordered invariance"
        if mode == "unordered_halves"
        else "ordered response, no nonzero requirement"
        if mode == "ordered_halves"
        else "reference only",
        mean_input=geometry(ms),
        left_input=geometry(ls),
        right_input=geometry(rs),
        samples=samples,
        semantic_accuracy_status="not_identified",
    )
