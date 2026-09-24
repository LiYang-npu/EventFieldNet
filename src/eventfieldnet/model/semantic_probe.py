"""Round31 semantic diagnostics; measurements only, no semantic ground-truth claims."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from field_core.selector import span_mean


def _stats(x: torch.Tensor | None) -> dict[str, Any]:
    if not isinstance(x, torch.Tensor):
        return {
            "available": False,
            "count": 0,
            "finite": False,
            "abs_mean": None,
            "std": None,
            "saturation_095": None,
        }
    x = x.detach().float().flatten()
    return {
        "available": True,
        "count": x.numel(),
        "finite": bool(torch.isfinite(x).all()),
        "mean": float(x.mean()) if x.numel() else None,
        "abs_mean": float(x.abs().mean()) if x.numel() else None,
        "std": float(x.std(unbiased=False)) if x.numel() else None,
        "min": float(x.min()) if x.numel() else None,
        "max": float(x.max()) if x.numel() else None,
        "p10": float(torch.quantile(x, 0.10)) if x.numel() else None,
        "p50": float(torch.quantile(x, 0.50)) if x.numel() else None,
        "p90": float(torch.quantile(x, 0.90)) if x.numel() else None,
        "negative_fraction": float(x.lt(0).float().mean()) if x.numel() else None,
        "saturation_095": float(x.abs().ge(0.95).float().mean()) if x.numel() else None,
    }


def _masked_stats(value: torch.Tensor, mask: torch.Tensor) -> dict[str, Any]:
    if tuple(value.shape) != tuple(mask.shape):
        raise RuntimeError(
            f"round31 probe mask shape {tuple(mask.shape)} != value shape {tuple(value.shape)}"
        )
    return _stats(value[mask.bool()])


def _require_tensor(
    owner: Any, name: str, shape: tuple[int, ...] | None = None
) -> torch.Tensor:
    value = getattr(owner, name, None)
    if not isinstance(value, torch.Tensor):
        raise RuntimeError(f"round31 probe requires field.{name} tensor")
    if shape is not None and tuple(value.shape) != tuple(shape):
        raise RuntimeError(
            f"round31 probe field.{name} shape {tuple(value.shape)} != {tuple(shape)}"
        )
    return value


def _controls(model: Any) -> dict[str, Any]:
    """Validate the fixed R31 Y0..Y7 route matrix fail-closed.

    R31 keeps the R30 field construction fixed (edge_mean, wide/donor S,
    independent S projection); its three bits only choose rank-stop-S,
    auxiliary input detach, and candidate-pair P.  Do not reuse the R30 X
    four-readout mapping here: doing so can make a probe report a valid
    configuration while computing a different loss graph.
    """
    options = getattr(model, "options", None)
    if options is None:
        raise RuntimeError("round31 probe requires model.options")
    required = (
        "variant",
        "support_readout",
        "a_local_evidence",
        "b_edge_support",
        "c_latent_composition",
        "support_deployment",
        "support_input",
        "support_pair_mode",
        "support_wide_pair",
        "support_length_center",
        "rank_candidate_pair",
        "rank_kl_half",
        "rank_stop_s",
        "aux_detach_input",
        "independent_s_projection",
        "query_relative_e",
        "e_span_lme",
    )
    missing = [name for name in required if not hasattr(options, name)]
    if missing:
        raise RuntimeError(f"round31 probe options missing exact fields: {missing}")
    variant = str(options.variant).lower()
    r48_routes = {
        "r48_control": 4,
        "r48_rank_stop_s": 5,
        "r48_aux_detach": 6,
        "r48_soft_target": 4,
        "r48_no_pair": 0,
        "r48_e_lme": 4,
    }
    if variant in r48_routes:
        index = r48_routes[variant]
    elif variant in (
        "y8",
        "y9",
        "y10",
        "y11",
        "y12",
        "y13",
        "y14",
        "y15",
        "y16",
        "y17",
        "y18",
        "y19",
        "y20",
        "y21",
        "y22",
        "y23",
        "y24",
        "y25",
        "y26",
        "y27",
        "y31",
        "y32",
        "y33",
    ):
        # round32 extension: y8/y9 and the y10-y16 MR57 structural ablation
        # group are all exactly y4's route cell (rank_stop_s=False,
        # aux_detach_input=False, rank_candidate_pair=True) with either
        # different loss weights (y8/y9, validated separately in
        # adapter.py) or exactly one mechanism flag flipped (y10-y16,
        # validated as fixed per-variant in Round31Options.from_values).
        # Alias to index 4 so every bit check below reduces to y4's
        # already-audited values.
        index = 4
    elif len(variant) == 2 and variant[0] == "y" and variant[1] in "01234567":
        index = int(variant[1])
    else:
        raise RuntimeError(f"invalid R31 variant {variant!r}; expected y0..y7, y8-y16")
    readout = str(options.support_readout).lower()
    if readout != "edge_mean":
        raise RuntimeError(f"R31 fixes support_readout=edge_mean, got {readout!r}")
    expected_rank_stop = bool(index & 1)
    expected_aux_detach = bool(index & 2)
    expected_pair = bool(index & 4)
    # round32 extension: y12 is the one MR57 structural variant that
    # deliberately flips e_span_lme (validated as fixed per-variant in
    # Round31Options.from_values); every other variant (aliased or not)
    # keeps the original R31 fixed value of False.
    expected_e_span_lme = variant in ("y12", "r48_e_lme")
    checks = {
        "a_local_evidence": (
            type(options.a_local_evidence) is bool and options.a_local_evidence is True
        ),
        "b_edge_support": (
            type(options.b_edge_support) is bool and options.b_edge_support is True
        ),
        "c_latent_composition": (
            type(options.c_latent_composition) is bool
            and options.c_latent_composition is False
        ),
        "support_wide_pair": (
            type(options.support_wide_pair) is bool
            and options.support_wide_pair is True
        ),
        "support_length_center": (
            type(options.support_length_center) is bool
            and options.support_length_center is False
        ),
        "rank_candidate_pair": (
            type(options.rank_candidate_pair) is bool
            and options.rank_candidate_pair is expected_pair
        ),
        "rank_kl_half": (
            type(options.rank_kl_half) is bool and options.rank_kl_half is False
        ),
        "rank_stop_s": (
            type(options.rank_stop_s) is bool
            and options.rank_stop_s is expected_rank_stop
        ),
        "aux_detach_input": (
            type(options.aux_detach_input) is bool
            and options.aux_detach_input is expected_aux_detach
        ),
        "independent_s_projection": (
            type(options.independent_s_projection) is bool
            and options.independent_s_projection is True
        ),
        "query_relative_e": (
            type(options.query_relative_e) is bool and options.query_relative_e is False
        ),
        "e_span_lme": (
            type(options.e_span_lme) is bool
            and options.e_span_lme is expected_e_span_lme
        ),
    }
    bad = [name for name, ok in checks.items() if not ok]
    if bad:
        raise RuntimeError(f"R31 fixed-field/route mismatch for {bad}")
    deployment = str(options.support_deployment).lower()
    support_input = str(options.support_input).lower()
    support_pair_mode = str(options.support_pair_mode).lower()
    if deployment != "in_score":
        raise RuntimeError("R31 fixes support_deployment=in_score")
    if support_input != "rms":
        raise RuntimeError(f"R31 fixes support_input=rms, got {support_input!r}")
    if support_pair_mode != "joint_residual":
        raise RuntimeError("R31 fixes support_pair_mode=joint_residual")
    return {
        "variant": str(options.variant),
        "new_E": True,
        "B_on": True,
        "C_on": False,
        "C_expected_off": True,
        "support_deployment": deployment,
        "support_input": support_input,
        "support_pair_mode": support_pair_mode,
        "support_wide_pair": True,
        "support_length_center": False,
        "rank_candidate_pair": expected_pair,
        "rank_kl_half": False,
        "rank_stop_s": expected_rank_stop,
        "aux_detach_input": expected_aux_detach,
        "factor_bits": index,
        "route_bits": index,
        "support_readout": readout,
        "readout_index": 0,
        "rank_background": "K+.5P" if expected_pair else "K",
        "sources": {
            "support_readout": "options.support_readout",
            "new_E": "options.a_local_evidence",
            "support_deployment": "options.support_deployment",
            "support_input": "options.support_input",
            "support_pair_mode": "options.support_pair_mode",
            "support_wide_pair": "options.support_wide_pair",
            "support_length_center": "options.support_length_center",
            "rank_candidate_pair": "options.rank_candidate_pair",
            "rank_kl_half": "options.rank_kl_half",
            "rank_stop_s": "options.rank_stop_s",
            "aux_detach_input": "options.aux_detach_input",
        },
    }


def _padding_probe(
    value: torch.Tensor,
    valid: torch.Tensor,
    alive: torch.Tensor,
    *,
    name: str,
) -> dict[str, Any]:
    if tuple(value.shape) == tuple(valid.shape):
        padding = ~valid.bool()
        scope = "candidate_span"
    elif value.ndim == 2 and tuple(value.shape) == tuple(
        alive.shape[:-1] + (alive.shape[-1] - 1,)
    ):
        padding = ~(alive[:, :-1] & alive[:, 1:])
        scope = "adjacent_edge"
    elif tuple(value.shape) == tuple(alive.shape):
        padding = ~alive.bool()
        scope = "token"
    else:
        raise RuntimeError(
            f"round31 probe field.{name} has unsupported shape {tuple(value.shape)}; "
            f"expected candidate-span {tuple(valid.shape)}, edge {tuple(alive[:, :-1].shape)}, "
            f"or token {tuple(alive.shape)}"
        )
    padding_values = value.detach()[padding]
    return {
        "scope": scope,
        "padding_count": int(padding.sum()),
        "padding_finite": bool(torch.isfinite(padding_values).all()),
        "padding_zero": bool(padding_values.eq(0).all()),
    }


def _edge_probe(field: Any, alive: torch.Tensor) -> dict[str, Any]:
    edge = _require_tensor(field, "round31_edge_score")
    edge_valid = _require_tensor(field, "round31_edge_valid", tuple(edge.shape)).bool()
    expected_valid = alive[:, :-1] & alive[:, 1:]
    if not torch.equal(edge_valid, expected_valid):
        raise RuntimeError(
            "round31 field.round31_edge_valid disagrees with video padding"
        )
    per_query = []
    centered_values = []
    for index in range(edge.shape[0]):
        values = edge[index][edge_valid[index]].detach().float()
        if values.numel() == 0:
            continue
        centered = values - values.mean()
        per_query.append(centered.std(unbiased=False))
        centered_values.append(centered)
    padding_values = edge.detach()[~edge_valid]
    return {
        "available": True,
        "edge_count": int(edge_valid.sum()),
        "query_count": len(per_query),
        "per_query_centered_std": _stats(torch.stack(per_query) if per_query else None),
        "centered_edge_values": _stats(
            torch.cat(centered_values) if centered_values else None
        ),
        "padding_finite": bool(torch.isfinite(padding_values).all()),
        "padding_zero": bool(padding_values.eq(0).all()),
        "scope": "deployed edge score, centered independently within each query",
    }


def _support_inputs(
    field: Any, options: Any, valid: torch.Tensor, alive: torch.Tensor
) -> dict[str, Any]:
    mode = getattr(field, "round31_support_input", None)
    if not isinstance(mode, str) or mode not in ("raw", "rms"):
        raise RuntimeError(
            "round31 probe requires field.round31_support_input in {raw,rms}"
        )
    if mode != str(options.support_input):
        raise RuntimeError(
            f"round31 field.round31_support_input={mode!r} != options.support_input={options.support_input!r}"
        )
    before = _require_tensor(
        field, "round31_support_input_rms_before", tuple(alive.shape)
    )
    after = _require_tensor(
        field, "round31_support_input_rms_after", tuple(alive.shape)
    )
    raw_vector = _require_tensor(field, "round31_z_s")
    if raw_vector.ndim != 3 or tuple(raw_vector.shape[:2]) != tuple(alive.shape):
        raise RuntimeError("round31 field.round31_z_e must have [batch,time,dim] shape")
    eps = 1.0e-6
    raw_float = raw_vector.float()
    expected_before = torch.sqrt(raw_float.square().mean(-1) + eps).masked_fill(
        ~alive, 0
    )
    if mode == "raw":
        expected_after = expected_before
    else:
        normalized = raw_float / expected_before.clamp_min(eps).unsqueeze(-1)
        expected_after = torch.sqrt(normalized.square().mean(-1) + eps).masked_fill(
            ~alive, 0
        )

    def identity(observed: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
        delta = (observed.float() - expected.float())[alive].detach().abs()
        return {
            "max_abs_error": float(delta.max()) if delta.numel() else 0.0,
            "within_fp32_tolerance": bool(not delta.numel() or delta.max() < 2e-6),
        }

    before_probe = {
        **_masked_stats(before, alive),
        **_padding_probe(before, valid, alive, name="round31_support_input_rms_before"),
        "source_field": "field.round31_support_input_rms_before",
        "identity": identity(before, expected_before),
    }
    after_probe = {
        **_masked_stats(after, alive),
        **_padding_probe(after, valid, alive, name="round31_support_input_rms_after"),
        "source_field": "field.round31_support_input_rms_after",
        "identity": identity(after, expected_after),
    }
    return {
        "mode": mode,
        "mode_matches_options": True,
        "raw": {
            "available": True,
            "vector_stats": _stats(raw_vector[alive]),
            "rms_before": before_probe,
            "source_field": "field.round31_z_s",
        },
        "rms": {
            "available": True,
            "rms_after": after_probe,
            "source_field": "field.round31_support_input_rms_after",
        },
        "selected_input": "field.round31_z_s"
        if mode == "raw"
        else "single token-RMS prepared S input; no post-pool RMS",
    }


def _score_comparison(
    field: Any, valid: torch.Tensor, deployment: str
) -> dict[str, Any]:
    score = _require_tensor(field, "score", tuple(valid.shape))
    zero = _require_tensor(field, "round31_score_without_support", tuple(valid.shape))
    deployed = _require_tensor(field, "round31_deployed_support", tuple(valid.shape))
    valid = valid.bool()
    valid_score = score[valid]
    valid_zero = zero[valid]
    finite = bool(
        torch.isfinite(valid_score).all() and torch.isfinite(valid_zero).all()
    )
    if not finite:
        return {
            "available": True,
            "deployment": deployment,
            "valid_mask_equal": True,
            "finite": False,
            "common_valid_count": int(valid.sum()),
            "active_queries": int(valid.flatten(1).any(1).sum()),
            "rankings_available": False,
            "scope": "same forward score and stored zero-support score; non-finite valid value",
        }

    expected = (
        field.support if deployment == "in_score" else torch.zeros_like(field.support)
    )
    deployment_error = (score - zero - deployed)[valid].detach().float().abs()
    deployed_error = (deployed - expected)[valid].detach().float().abs()
    flat_score = score.float().flatten(1).masked_fill(~valid.flatten(1), -torch.inf)
    flat_zero = zero.float().flatten(1).masked_fill(~valid.flatten(1), -torch.inf)
    delta = score.float() - zero.float()
    top1_changed = 0
    top10_order_changed = 0
    top10_set_changed = 0
    active_queries = 0
    top1_deltas = []
    top10_deltas = []
    for row in range(score.shape[0]):
        count = int(valid[row].sum())
        if count == 0:
            continue
        active_queries += 1
        order_score = torch.argsort(flat_score[row], descending=True, stable=True)
        order_zero = torch.argsort(flat_zero[row], descending=True, stable=True)
        k = min(10, count)
        top_score = order_score[:k]
        top_zero = order_zero[:k]
        if int(top_score[0]) != int(order_zero[0]):
            top1_changed += 1
        if not torch.equal(top_score, top_zero):
            top10_order_changed += 1
        if set(top_score.tolist()) != set(top_zero.tolist()):
            top10_set_changed += 1
        row_delta = delta[row].flatten()
        top1_deltas.append(row_delta[top_score[0]].reshape(1))
        top10_deltas.append(row_delta[top_score])

    denominator = max(active_queries, 1)
    return {
        "available": True,
        "deployment": deployment,
        "valid_mask_equal": True,
        "finite": True,
        "common_valid_count": int(valid.sum()),
        "active_queries": active_queries,
        "top1_changed_queries": top1_changed,
        "top1_change_rate": top1_changed / denominator,
        "top10_order_changed_queries": top10_order_changed,
        "top10_order_change_rate": top10_order_changed / denominator,
        "top10_set_changed_queries": top10_set_changed,
        "top10_set_change_rate": top10_set_changed / denominator,
        "top1_score_delta_at_actual_top1": _stats(
            torch.cat(top1_deltas) if top1_deltas else None
        ),
        "top10_score_delta_at_actual_top10": _stats(
            torch.cat(top10_deltas) if top10_deltas else None
        ),
        "score_delta_all_common_valid": _stats(delta[valid]),
        "actual_deployment_identity": {
            "expected_delta": "support" if deployment == "in_score" else "zero",
            "max_abs_error": float(deployment_error.max())
            if deployment_error.numel()
            else 0.0,
            "within_fp32_tolerance": bool(
                not deployment_error.numel() or deployment_error.max() < 2e-6
            ),
            "deployed_support_max_abs_error": float(deployed_error.max())
            if deployed_error.numel()
            else 0.0,
            "deployed_support_within_fp32_tolerance": bool(
                not deployed_error.numel() or deployed_error.max() < 2e-6
            ),
            "source_field": "field.round31_deployed_support",
        },
        "scope": "same forward score and stored zero-support score; stable flat-ID tie order; valid candidates only",
    }


def _length_strata(
    field: Any, valid: torch.Tensor, score_delta: torch.Tensor
) -> dict[str, Any]:
    support = _require_tensor(field, "support", tuple(valid.shape))
    length = valid.shape[-1]
    starts = torch.arange(length, device=valid.device)[:, None]
    ends = torch.arange(length, device=valid.device)[None, :]
    span_length = (ends - starts + 1).expand_as(valid)
    buckets = {
        "singleton": span_length.eq(1),
        "length_2": span_length.eq(2),
        "length_3_4": span_length.ge(3) & span_length.le(4),
        "length_5_8": span_length.ge(5) & span_length.le(8),
        "length_9_plus": span_length.ge(9),
    }
    raw = getattr(field, "round31_uncentered_support", None)
    if isinstance(raw, torch.Tensor) and tuple(raw.shape) != tuple(valid.shape):
        raise RuntimeError("round31 uncentered support shape mismatch")
    result = {}
    for name, bucket in buckets.items():
        mask = valid.bool() & bucket
        row = {
            "count": int(mask.sum()),
            "support_deployed": _masked_stats(support, mask),
            "F_actual_support_contribution": _masked_stats(score_delta, mask),
        }
        if isinstance(raw, torch.Tensor):
            row["support_uncentered"] = _masked_stats(raw, mask)
            row["centered_delta"] = _masked_stats(support - raw, mask)
        result[name] = row
    singleton = valid.bool() & buckets["singleton"]

    centering = getattr(field, "round31_support_centering", None)
    centering_result: dict[str, Any] = {
        "enabled": isinstance(centering, Mapping),
        "source_field": "field.round31_support_centering",
        "centered_output_is_same_tensor": support is raw,
        "centered_output_value_identity": (
            bool(torch.equal(support.detach(), raw.detach()))
            if isinstance(raw, torch.Tensor)
            else None
        ),
    }
    if isinstance(centering, Mapping):
        means = centering.get("means")
        counts = centering.get("counts")
        if not isinstance(means, torch.Tensor) or not isinstance(counts, torch.Tensor):
            raise RuntimeError(
                "round31 support centering requires means and counts tensors"
            )
        if (
            means.ndim != 2
            or counts.shape != means.shape
            or means.shape[0] != valid.shape[0]
        ):
            raise RuntimeError("round31 support centering tensors have invalid shape")
        expected_counts = valid.bool().flatten(1).sum(1)
        observed_counts = counts.float().sum(1)
        active_groups = counts.gt(0)
        group = (ends - starts).clamp_min(0).flatten().expand(valid.shape[0], -1)
        valid_flat = valid.bool().flatten(1)
        support_flat = support.float().flatten(1)
        count_at_candidate = counts.float().gather(1, group)
        centered_values = support_flat.masked_fill(~valid_flat, 0.0)
        centered_group_sum = (
            means.float().new_zeros(means.shape).scatter_add(1, group, centered_values)
        )
        centered_group_mean = centered_group_sum / counts.float().clamp_min(1.0)
        centered_group_abs = centered_group_mean[active_groups].detach().abs()
        single_group_mask = valid_flat & count_at_candidate.eq(1.0)
        single_group_values = support_flat[single_group_mask].detach().abs()
        centering_result.update(
            {
                "means": _stats(means[active_groups]),
                "counts": _stats(counts[active_groups]),
                "active_length_groups": int(active_groups.sum()),
                "counts_match_valid_candidates": bool(
                    torch.equal(
                        observed_counts.detach().to(expected_counts.dtype),
                        expected_counts.detach(),
                    )
                ),
                "finite": bool(
                    torch.isfinite(means).all() and torch.isfinite(counts).all()
                ),
            }
        )
        if isinstance(raw, torch.Tensor):
            expected = (
                (raw.float().flatten(1) - means.float().gather(1, group))
                .masked_fill(~valid.flatten(1), 0.0)
                .reshape_as(valid)
            )
            error = (expected - support.float())[valid].detach().abs()
            centering_result["identity_max_abs_error"] = (
                float(error.max()) if error.numel() else 0.0
            )
            centering_result["identity_within_fp32_tolerance"] = bool(
                not error.numel() or error.max() < 2.0e-6
            )
        centering_result["singleton_zero_fraction"] = (
            float(support[singleton].eq(0).float().mean()) if singleton.any() else None
        )
        centering_result.update(
            {
                "group_mean_abs_max": float(centered_group_abs.max())
                if centered_group_abs.numel()
                else 0.0,
                "group_mean_zero": bool(
                    not centered_group_abs.numel() or centered_group_abs.max() < 2.0e-6
                ),
                "single_candidate_group_count": int(single_group_values.numel()),
                "single_candidate_group_zero_fraction": (
                    float(single_group_values.le(2.0e-6).float().mean())
                    if single_group_values.numel()
                    else None
                ),
                "single_candidate_group_max_abs": (
                    float(single_group_values.max())
                    if single_group_values.numel()
                    else 0.0
                ),
                "means_requires_grad": bool(means.requires_grad),
                "means_has_grad_fn": bool(means.grad_fn is not None),
            }
        )
    else:
        centering_result["identity_within_fp32_tolerance"] = centering_result[
            "centered_output_value_identity"
        ]
        centering_result["reason"] = (
            "length centering disabled; zero/default statistics are not active"
        )

    return {
        "available": True,
        "buckets": result,
        "singleton_zero_fraction": float(support[singleton].eq(0).float().mean())
        if singleton.any()
        else None,
        "length_centering": centering_result,
        "scope": "valid candidate spans grouped by inclusive clip length; signed statistics are reported",
    }


def _batch_inputs(batch: Any) -> dict[str, Any]:
    if isinstance(batch, dict):
        return batch.get("inputs", {})
    return getattr(batch, "inputs", {})


def _e_readout_scale_probe(
    model: Any, field: Any, alive: torch.Tensor
) -> dict[str, Any]:
    """Panel-only measurements; independently reconstruct the scale formula."""
    z = _require_tensor(field, "round31_z_e").detach().float()
    u = (
        _require_tensor(field, "round31_e_readout_input", tuple(z.shape))
        .detach()
        .float()
    )
    if z.ndim != 3 or tuple(z.shape[:2]) != tuple(alive.shape):
        raise RuntimeError("E scale probe shape mismatch")
    if not bool(alive.any(1).all()):
        raise RuntimeError("E scale probe rejects empty queries")
    mode = model.options.e_readout_scale
    masked = z.masked_fill(~alive[..., None], 0)
    if mode == "raw":
        expected, gains = z, torch.ones_like(alive, dtype=torch.float32)
    elif mode == "fixed_init":
        gains = torch.full_like(alive, 974.43609777058, dtype=torch.float32)
        expected = masked * gains[..., None]
    elif mode == "query_rms":
        rms = (masked.square().sum((1, 2)) / (alive.sum(1) * z.shape[-1]) + 1e-6).sqrt()
        gains = rms.reciprocal()[:, None].expand_as(alive)
        expected = masked / rms[:, None, None]
    elif mode == "token_rms":
        rms = (masked.square().mean(-1) + 1e-6).sqrt()
        gains = rms.reciprocal()
        expected = masked / rms[..., None]
    else:
        raise RuntimeError(f"Unknown E scale {mode}")
    gain = _require_tensor(field, "round31_e_readout_gain").detach().float()
    if gain.ndim == 3 and gain.shape[-1] == 1:
        gain = gain.squeeze(-1)
    gain = gain.expand_as(alive)
    observed = gain[alive]
    quantiles = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], device=z.device)

    def geometry(value):
        tokens = value[alive]
        singular = torch.linalg.svdvals(tokens)
        energy = singular.square()
        p = energy / energy.sum().clamp_min(1e-30)
        effective_rank = (
            torch.exp(-(p * p.clamp_min(1e-30).log()).sum())
            if bool(energy.sum() > 0)
            else energy.new_zeros(())
        )
        return dict(
            global_rms=float(tokens.square().mean().sqrt()),
            token_rms=_stats(tokens.square().mean(-1).sqrt()),
            channel_variance_mean=float(tokens.var(0, unbiased=False).mean()),
            effective_energy_rank=float(effective_rank),
            mean_within_query_std=float(
                torch.stack(
                    [
                        row[mask].std(0, unbiased=False).mean()
                        for row, mask in zip(value, alive)
                    ]
                ).mean()
            ),
        )

    padding_zero = bool(u[~alive].eq(0).all())
    formula = bool(torch.allclose(u, expected, atol=1e-6, rtol=1e-6))
    gain_identity = bool(
        torch.allclose(gain[alive], gains[alive], atol=1e-5, rtol=1e-6)
    )
    finite = bool(torch.isfinite(u).all() and torch.isfinite(observed).all())
    if not (padding_zero and formula and gain_identity and finite):
        raise RuntimeError(
            "Actual E readout scale failed formula, gain, finite, or padding contract"
        )
    if not bool((observed <= 1000.0001).all()):
        raise RuntimeError("E readout scale gain exceeds frozen bound")
    return dict(
        mode=mode,
        epsilon=1e-6,
        fixed_init_gain=974.43609777058,
        input=geometry(z),
        output=geometry(u),
        gain_quantiles=dict(
            zip(
                ["min", "q25", "median", "q75", "max"],
                torch.quantile(observed, quantiles).cpu().tolist(),
            )
        ),
        valid_tokens=int(alive.sum()),
        padding_zero=padding_zero,
        formula_matches=formula,
        gain_matches=gain_identity,
        finite=finite,
        raw_exact=torch.equal(z, u) if mode == "raw" else None,
        readout_weight_l2=float(
            model.selector.e_interaction.readout.weight.detach().float().norm()
        ),
        scope="fixed training panel; exact steps in runtime step telemetry; no semantic labels",
    )


def semantic_probe(model: Any, outputs: Any, batch: Any) -> dict[str, Any]:
    """Return exact Round31 field diagnostics; never infer semantic truth."""
    field = getattr(outputs, "trifield_output", None)
    if field is None:
        raise RuntimeError("round31 probe requires outputs.trifield_output")
    controls = _controls(model)
    field_deployment = getattr(field, "round31_support_deployment", None)
    if field_deployment != controls["support_deployment"]:
        raise RuntimeError(
            f"round31 field.round31_support_deployment={field_deployment!r} != "
            f"options.support_deployment={controls['support_deployment']!r}"
        )
    if not controls["B_on"]:
        raise RuntimeError("Round31 probe requires b_edge_support=true")
    if controls["C_on"]:
        raise RuntimeError("Round31 probe requires c_latent_composition=false")
    valid = _require_tensor(field, "valid").bool()
    for name in ("score", "evidence", "support", "transition_start", "transition_end"):
        _require_tensor(field, name, tuple(valid.shape))
    inputs = _batch_inputs(batch)
    pad = inputs.get("video_padding_mask")
    if isinstance(pad, torch.Tensor):
        alive = ~pad.bool()
    else:
        alive = valid.any(-1) | valid.any(-2)
    if (
        alive.ndim != 2
        or alive.shape[0] != valid.shape[0]
        or alive.shape[1] != valid.shape[1]
    ):
        raise RuntimeError(
            "round31 probe video padding shape does not match field.valid"
        )
    edges = alive[:, :-1] & alive[:, 1:]

    def error(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> dict[str, Any]:
        if tuple(a.shape) != tuple(b.shape) or tuple(a.shape) != tuple(mask.shape):
            raise RuntimeError("round31 probe identity shape mismatch")
        v = (a - b)[mask].detach().float().abs()
        return {
            "max_abs_error": float(v.max()) if v.numel() else 0.0,
            "within_fp32_tolerance": bool(not v.numel() or v.max() < 2e-6),
        }

    def ablate(score: torch.Tensor) -> dict[str, Any]:
        def top(x: torch.Tensor) -> torch.Tensor:
            return x.masked_fill(~valid, -torch.inf).flatten(1).argmax(1)

        active = valid.flatten(1).any(1)
        return {
            "delta": _stats((field.score - score)[valid]),
            "top1_changed_queries": int(
                ((top(field.score) != top(score)) & active).sum()
            ),
        }

    edge_info = _edge_probe(field, alive)
    support_inputs = _support_inputs(field, model.options, valid, alive)
    edge = _require_tensor(field, "round31_edge_score", tuple(edges.shape))
    uncentered = getattr(field, "round31_uncentered_support", None)
    if not isinstance(uncentered, torch.Tensor):
        raise RuntimeError("actual R30 deployed support readout tensor missing")
    if tuple(uncentered.shape) != tuple(valid.shape):
        raise RuntimeError("round31 uncentered support shape does not match candidates")
    expected_support = uncentered
    centering_detail = getattr(field, "round31_support_centering", None)
    if controls["support_length_center"]:
        if not isinstance(centering_detail, Mapping):
            raise RuntimeError(
                "length-centered variant did not expose centering detail"
            )
        means = centering_detail.get("means")
        if (
            not isinstance(means, torch.Tensor)
            or means.ndim != 2
            or means.shape[0] != valid.shape[0]
        ):
            raise RuntimeError("length-centered variant has invalid means tensor")
        starts = torch.arange(valid.shape[-1], device=valid.device)[:, None]
        ends = torch.arange(valid.shape[-1], device=valid.device)[None, :]
        group = (ends - starts).clamp_min(0).flatten().expand(valid.shape[0], -1)
        expected_support = (
            uncentered.float().flatten(1) - means.float().gather(1, group)
        ).reshape_as(valid)
        expected_support = expected_support.masked_fill(~valid, 0.0)
    support_identity = error(expected_support, field.support, valid)
    from .factor_probes import support_readout_probe

    readout_check = support_readout_probe(model, field, alive)
    support_identity["scope"] = (
        "deployment tensor closure; independent mathematics in readout_reference"
    )

    score_compare = _score_comparison(field, valid, controls["support_deployment"])
    zero = _require_tensor(field, "round31_score_without_support", tuple(valid.shape))
    delta = field.score.float() - zero.float()
    r = {
        "scope": "fixed training panel, not official validation or semantic ground truth",
        "controls": controls,
        "mechanisms": {
            "query_relative_e": model.options.query_relative_e,
            "e_span_lme": model.options.e_span_lme,
            "independent_s_projection": model.options.independent_s_projection,
            "support_wide_pair": controls["support_wide_pair"],
            "support_length_center": controls["support_length_center"],
            "rank_candidate_pair": controls["rank_candidate_pair"],
            "rank_kl_half": controls["rank_kl_half"],
            "A_local_evidence": bool(controls["new_E"]),
            "B_edge_support": bool(controls["B_on"]),
            "C_latent_composition": bool(controls["C_on"]),
        },
        "E": {
            "deployed": _stats(field.evidence[valid]),
            "readout_scale": _e_readout_scale_probe(model, field, alive),
            "role": "local query-conditioned evidence"
            if controls["new_E"]
            else "legacy candidate E plus learned residual",
        },
        "S": {
            "deployed": _stats(field.round31_deployed_support[valid]),
            "auxiliary_support": _stats(field.support[valid]),
            "role": "candidate S readout: " + controls["support_readout"],
            "support_readout": controls["support_readout"],
            "readout_reference": readout_check,
            "edge_reference_scope": "joint primitive reference; not candidate S for non-edge_mean",
            "support_deployment": controls["support_deployment"],
            "support_input": support_inputs,
            "support_pair_mode": controls["support_pair_mode"],
            "support_wide_pair": controls["support_wide_pair"],
            "support_length_center": controls["support_length_center"],
            "rank_candidate_pair": controls["rank_candidate_pair"],
            "rank_kl_half": controls["rank_kl_half"],
            "edge_centered_std": edge_info,
            "edges": _stats(edge[edges]),
            "deployment_identity": support_identity,
            "singleton_zero": bool(
                field.support.diagonal(dim1=1, dim2=2)[valid.diagonal(dim1=1, dim2=2)]
                .eq(0)
                .all()
            ),
            "valid_edge_count": int(edges.sum()),
            "score_without_support": score_compare,
            "F_actual_support_contribution": _masked_stats(delta, valid),
            "S_auxiliary_original": _masked_stats(field.support, valid),
            "S_uncentered": _masked_stats(uncentered, valid),
            "length_strata": _length_strata(field, valid, delta),
        },
        "T": {
            "start": _stats(field.transition_start[valid]),
            "end": _stats(field.transition_end[valid]),
            "internal_stage_labels_available": False,
            "role": "exterior start/end; natural internal transitions not identified",
        },
        "semantic_identification": {
            "same_entities_wrong_action": "not_identified: no action/relation annotation",
            "low_E_same_process_bridge": "not_identified: low evidence does not label continuity",
            "unrelated_insert": "weak_counterfactual: outside GT is not proof of unrelated event",
            "internal_vs_exterior_change": "not_identified: latent modes are not stage labels",
        },
    }

    if controls["new_E"]:
        e_token = _require_tensor(field, "round31_e_token")
        if model.options.e_span_lme:
            from .candidate_primitives import log_mean_exp_span

            expected, _ = log_mean_exp_span(e_token, alive)
        else:
            expected = span_mean(e_token[..., None]).squeeze(-1)
        r["E"]["local_values"] = _stats(e_token[alive])
        r["E"]["deployment_identity"] = error(expected, field.evidence, valid)
        r["E"]["old_candidate_branch_used"] = False

    wrong = getattr(model, "_last_wrong_query_field", None)
    if wrong is not None:
        r["query_response"] = {
            key: _stats((getattr(field, key) - getattr(wrong, key))[valid])
            for key in ("evidence", "support", "transition_start", "transition_end")
        }
        r["query_response"]["scope"] = (
            "mismatched query response, not action-specific correctness"
        )
    return r
