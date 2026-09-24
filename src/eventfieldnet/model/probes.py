"""Round31 probes for objective, fields, structure, short events, and panel."""

from __future__ import annotations
import random
from typing import Any, Mapping
import torch
from torch import Tensor
import field_core.probes as _v1
from field_core.probes import (
    probe_candidate_membership,
    probe_conditioner_ablation,
    probe_endpoint_responsibility,
    probe_multigt_duplicate,
    probe_ordinal_tie_subgradient,
    probe_parameter_groups,
)
from field_core.losses import (
    FiveLossTerms,
    candidate_geometry,
    official_threshold_grades,
)
from .losses import compute_loss_terms, transition_targets


def _parts(batch: Any) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    if isinstance(batch, Mapping):
        return batch.get("inputs", {}), batch.get("targets", {})
    return getattr(batch, "inputs", {}), getattr(batch, "targets", {})


def _rows(batch: Any, n: int) -> list[Mapping[str, Any] | None]:
    metadata = (
        batch.get("metadata", {})
        if isinstance(batch, Mapping)
        else getattr(batch, "metadata", {})
    )
    rows = (
        metadata.get("metas", metadata.get("records"))
        if isinstance(metadata, Mapping)
        else None
    )
    if isinstance(rows, Mapping):
        rows = [rows] * n
    if not isinstance(rows, (list, tuple)):
        return [None] * n
    out = [x if isinstance(x, Mapping) else None for x in rows[:n]]
    return out + [None] * (n - len(out))


def _finite(x: Tensor) -> bool:
    return bool(torch.isfinite(x.detach().float()).all())


def _round31_feature_requires_grad(field: Any) -> dict[str, Any]:
    """Report learned feature graph status, separately from raw inputs."""
    result: dict[str, Any] = {}
    for name in ("z_e", "z_s", "h_aux"):
        value = getattr(field, "round31_" + name, None) if field is not None else None
        result[name] = {
            "available": isinstance(value, Tensor),
            "requires_grad": bool(value.requires_grad)
            if isinstance(value, Tensor)
            else None,
            "shape": list(value.shape) if isinstance(value, Tensor) else None,
        }
    return result


def _round31_raw_input_probe(field: Any) -> dict[str, Any]:
    """Verify requires_grad on the actual source tensors consumed by R31.

    The selector stores the source tuple on the field as a reference. Missing
    metadata is reported as unverified rather than converted to False, so a
    smoke exporter cannot pass by omitting the source tensors.
    """
    raw = getattr(field, "round31_raw_inputs", None) if field is not None else None
    if not isinstance(raw, (tuple, list)) or len(raw) < 2:
        return {
            "schema": "round31_raw_inputs_v1",
            "available": False,
            "verified": False,
            "all_requires_grad_false": None,
            "sources": {},
            "reason": "field.round31_raw_inputs missing",
        }
    labels = ("src_vid[..., :512]", "src_txt")
    source_rows: dict[str, Any] = {}
    for label, value in zip(labels, raw[:2]):
        if not isinstance(value, Tensor):
            source_rows[label] = {
                "available": False,
                "requires_grad": None,
                "shape": None,
                "finite": None,
            }
            continue
        source_rows[label] = {
            "available": True,
            "requires_grad": bool(value.requires_grad),
            "shape": list(value.shape),
            "finite": bool(torch.isfinite(value.detach().float()).all()),
        }
    available = all(bool(row.get("available")) for row in source_rows.values())
    all_false = (
        all(not bool(row["requires_grad"]) for row in source_rows.values())
        if available
        else None
    )
    return {
        "schema": "round31_raw_inputs_v1",
        "available": available,
        "verified": available and all_false is True,
        "all_requires_grad_false": all_false,
        "sources": source_rows,
        "source_tuple": "state.round31_raw_inputs=(src_vid[..., :512],src_txt,padding,query_padding)",
    }


def _probe_stats(value: Tensor | None) -> dict[str, Any]:
    if not isinstance(value, Tensor):
        return {
            "count": 0,
            "finite": False,
            "mean": None,
            "abs_mean": None,
            "std": None,
            "min": None,
            "max": None,
            "p10": None,
            "p50": None,
            "p90": None,
            "negative_fraction": None,
        }
    x = value.detach().float().flatten()
    finite = bool(torch.isfinite(x).all())
    if not x.numel():
        return {
            "count": 0,
            "finite": finite,
            "mean": None,
            "abs_mean": None,
            "std": None,
            "min": None,
            "max": None,
            "p10": None,
            "p50": None,
            "p90": None,
            "negative_fraction": None,
        }
    return {
        "count": int(x.numel()),
        "finite": finite,
        "mean": float(x.mean()),
        "abs_mean": float(x.abs().mean()),
        "std": float(x.std(unbiased=False)),
        "min": float(x.min()),
        "max": float(x.max()),
        "p10": float(torch.quantile(x, 0.10)),
        "p50": float(torch.quantile(x, 0.50)),
        "p90": float(torch.quantile(x, 0.90)),
        "negative_fraction": float(x.lt(0).float().mean()),
    }


def _norm(gs: tuple[Tensor | None, ...], ref: Tensor) -> Tensor:
    xs = [g.float().square().sum() for g in gs if g is not None]
    return torch.stack(xs).sum().sqrt() if xs else ref.detach().float().sum() * 0.0


def _gradient_report(
    model: Any, terms: FiveLossTerms, score: Tensor | None
) -> Mapping[str, Any]:
    params = [p for p in model.parameters() if p.requires_grad]
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    base = ("rank", "evidence", "support", "transition", "endpoint")
    all_terms = base + ("total",)
    configured = getattr(model, "loss_weights", {})
    weights = {n: float(configured.get(n, 1.0)) for n in base}
    weights["total"] = 1.0
    grads = {}
    for n in all_terms:
        v = getattr(terms, n)
        grads[n] = (
            torch.autograd.grad(v, params, retain_graph=True, allow_unused=True)
            if params and v.requires_grad
            else tuple(None for _ in params)
        )

    def dot(a, b):
        vals = [
            x.float().mul(y.float()).sum()
            for x, y in zip(a, b)
            if x is not None and y is not None
        ]
        return torch.stack(vals).sum() if vals else terms.total.detach().float() * 0.0

    buckets = {
        "edge_head": [
            i for i, n in enumerate(names) if n.startswith("selector.edge_head.")
        ],
        "latent_composition": [
            i for i, n in enumerate(names) if n.startswith("selector.composition.")
        ],
        "learned_e_interaction": [
            i for i, n in enumerate(names) if n.startswith("selector.e_interaction.")
        ],
        "learned_t_interaction": [
            i for i, n in enumerate(names) if n.startswith("selector.t_interaction.")
        ],
        "learned_e_projection": [
            i
            for i, n in enumerate(names)
            if n.startswith("selector.e_interaction.") and ".readout." not in n
        ],
        "independent_s_projection": [
            i for i, n in enumerate(names) if n.startswith("selector.s_projection.")
        ],
        "learned_e_readout": [
            i
            for i, n in enumerate(names)
            if n.startswith("selector.e_interaction.readout.")
        ],
        "learned_t_projection": [
            i
            for i, n in enumerate(names)
            if n.startswith("selector.t_interaction.") and "readout." not in n
        ],
        "scalar_calibration": [
            i for i, n in enumerate(names) if n == "selector.linear_weights"
        ],
        "pairwise_interactions": [
            i for i, n in enumerate(names) if n == "selector.pair_weights"
        ],
        "input_conditioner": [
            i
            for i, n in enumerate(names)
            if n.startswith("parent_model.input_conditioner.")
        ],
        "shared_head": [
            i for i, n in enumerate(names) if n.startswith("parent_model.shared_head.")
        ],
        "parent_start_head": [
            i
            for i, n in enumerate(names)
            if n.startswith("parent_model.stem.start")
            or n.startswith("parent_model.shared_head.start")
        ],
        "parent_end_head": [
            i
            for i, n in enumerate(names)
            if n.startswith("parent_model.stem.end")
            or n.startswith("parent_model.shared_head.end")
        ],
        "encoder": [
            i
            for i, n in enumerate(names)
            if n.startswith("parent_model.stem.temporal_encoder.")
        ],
        "selector_evidence": [
            i
            for i, n in enumerate(names)
            if n.startswith(
                (
                    "selector.evidence.",
                    "selector.evidence_pool_branch.",
                    "selector.evidence_context_branch.",
                )
            )
        ],
        "selector_evidence_base": [
            i for i, n in enumerate(names) if n.startswith("selector.evidence.")
        ],
        "selector_evidence_pool": [
            i
            for i, n in enumerate(names)
            if n.startswith("selector.evidence_pool_branch.")
        ],
        "selector_evidence_context": [
            i
            for i, n in enumerate(names)
            if n.startswith("selector.evidence_context_branch.")
        ],
        "selector_support": [
            i
            for i, n in enumerate(names)
            if n.startswith(
                (
                    "selector.support",
                    "selector.context_branch.",
                    "selector.dispersion_branch.",
                )
            )
        ],
        "selector_support_base": [
            i for i, n in enumerate(names) if n.startswith("selector.support")
        ],
        "selector_support_context": [
            i for i, n in enumerate(names) if n.startswith("selector.context_branch.")
        ],
        "selector_support_dispersion": [
            i
            for i, n in enumerate(names)
            if n.startswith("selector.dispersion_branch.")
        ],
        "selector_transition_start": [
            i for i, n in enumerate(names) if n.startswith("selector.transition_start.")
        ],
        "selector_transition_end": [
            i for i, n in enumerate(names) if n.startswith("selector.transition_end.")
        ],
    }
    norms = {n: _norm(grads[n], terms.total) for n in all_terms}
    result = {}
    for n in all_terms:
        modules = {
            k: _norm(tuple(grads[n][i] for i in ix), terms.total).detach()
            for k, ix in buckets.items()
        }
        result[n] = {
            "weight": weights[n],
            "raw": {
                "global_norm": norms[n].detach(),
                "finite": all(g is None or _finite(g) for g in grads[n]),
                "nonzero_parameter_tensors": sum(
                    int(g is not None and bool(g.detach().abs().max().gt(0)))
                    for g in grads[n]
                ),
                "module_norms": modules,
            },
            "effective": {
                "global_norm": (norms[n] * weights[n]).detach(),
                "module_norms": {k: v * weights[n] for k, v in modules.items()},
            },
        }
    cos = {}
    for i, left in enumerate(all_terms):
        for right in all_terms[i + 1 :]:
            den = norms[left] * norms[right]
            defined = bool(den.detach().gt(1.0e-12))
            cos[left + "__" + right] = {
                "defined": defined,
                "value": (dot(grads[left], grads[right]) / den).detach()
                if defined
                else None,
            }
    result["pairwise_cosine"] = cos
    recomposed = []
    for i, p in enumerate(params):
        pieces = [grads[n][i] * weights[n] for n in base if grads[n][i] is not None]
        value = pieces[0] if pieces else torch.zeros_like(p)
        for piece in pieces[1:]:
            value = value + piece
        recomposed.append(value)
    total_grad = tuple(
        torch.zeros_like(p) if g is None else g for p, g in zip(params, grads["total"])
    )
    diff = tuple(a - b for a, b in zip(total_grad, recomposed))
    err, total_norm = _norm(diff, terms.total), _norm(total_grad, terms.total)
    rel = err / total_norm.clamp_min(1.0e-12)
    result["weighted_recomposition"] = {
        "weights": dict(weights),
        "raw_norm": _norm(tuple(recomposed), terms.total).detach(),
        "total_norm": total_norm.detach(),
        "absolute_error": err.detach(),
        "relative_error": rel.detach(),
        "diagnostic_precision": "fp32_forward_required",
        "within_relative_tolerance": bool(rel.detach().le(1.0e-4)),
        "near_zero_absolute_guard": bool(
            total_norm.detach().le(1.0e-6) and err.detach().le(1.0e-6)
        ),
    }
    if isinstance(score, Tensor):
        fs, fv, fi = (
            score.float().flatten(1),
            terms.geometry.valid.flatten(1),
            terms.geometry.max_iou.flatten(1),
        )
        grades = official_threshold_grades(
            terms.geometry.max_iou, terms.geometry.valid
        ).flatten(1)
        eligible = [
            i
            for i in range(fs.shape[0])
            if bool(fv[i].any()) and int(grades[i][fv[i]].unique().numel()) > 1
        ]
        if eligible:
            ix = torch.tensor(eligible, device=fs.device, dtype=torch.long)
            good = fi[ix].masked_fill(~fv[ix], -1).argmax(1)
            bad = fi[ix].masked_fill(~fv[ix], 2).argmin(1)
            gap = (fs[ix, good] - fs[ix, bad]).mean()
            gg = (
                torch.autograd.grad(gap, params, retain_graph=True, allow_unused=True)
                if params and gap.requires_grad
                else tuple(None for _ in params)
            )
            raw = {n: dot(grads[n], gg).detach() for n in all_terms}
            result.update(
                {
                    "good_bad_gap": gap.detach(),
                    "good_bad_gap_eligible_row_count": len(eligible),
                    "good_bad_gap_raw_dot": raw,
                    "good_bad_gap_downhill_dot": {n: -v for n, v in raw.items()},
                    "good_bad_gap_grad_norm": _norm(gg, terms.total).detach(),
                }
            )
        else:
            result["good_bad_gap_skipped"] = "no row with two distinct official grades"
    return result


def probe_target_consistency(
    outputs: Any, batch: Any, terms: FiveLossTerms, mode="independent_max"
) -> Mapping[str, Any]:
    expected = transition_targets(outputs, batch, terms.geometry, mode)
    valid = terms.geometry.valid.bool()
    se, ee = (
        (terms.transition_start_target - expected.start).abs()[valid],
        (terms.transition_end_target - expected.end).abs()[valid],
    )
    both = torch.cat((se, ee))
    zero = outputs.span_logits.new_tensor(0.0)
    return {
        "target_source": mode + "_actual",
        "start_target_exact": bool(
            torch.equal(terms.transition_start_target, expected.start)
        ),
        "end_target_exact": bool(
            torch.equal(terms.transition_end_target, expected.end)
        ),
        "start_max_abs_error": se.max().detach() if se.numel() else zero,
        "end_max_abs_error": ee.max().detach() if ee.numel() else zero,
        "max_abs_error": both.max().detach() if both.numel() else zero,
        "independent_best_gt_conflict_rate": expected.conflict[valid]
        .float()
        .mean()
        .detach()
        if bool(valid.any())
        else zero,
        "valid_candidate_count": valid.sum().detach(),
    }


def probe_field_saturation(outputs: Any) -> Mapping[str, Any]:
    field = getattr(outputs, "trifield_output", None)
    if field is None:
        return {"available": False}
    valid, result = field.valid.bool(), {"available": True}
    values = {
        "carrier": field.carrier,
        "evidence": field.evidence,
        "support": field.support,
        "transition_start": field.transition_start,
        "transition_end": field.transition_end,
    }
    for name, value in values.items():
        x = value.float()[valid]
        if x.numel():
            result.update(
                {
                    name + "/finite": _finite(x),
                    name + "/saturation_rate_0.95": x.abs()
                    .ge(0.95)
                    .float()
                    .mean()
                    .detach(),
                    name + "/saturation_rate_0.99": x.abs()
                    .ge(0.99)
                    .float()
                    .mean()
                    .detach(),
                    name + "/p01": torch.quantile(x, 0.01).detach(),
                    name + "/p99": torch.quantile(x, 0.99).detach(),
                }
            )
        else:
            result.update(
                {
                    name + "/finite": True,
                    name + "/saturation_rate_0.95": 0.0,
                    name + "/saturation_rate_0.99": 0.0,
                    name + "/p01": 0.0,
                    name + "/p99": 0.0,
                }
            )
    return result


def probe_round31_support_pair(model, outputs):
    """Probe the actual edge operator without changing model state.

    The reported references use the same selector and RMS endpoint helper as
    deployment. In plain mode the extra three g calls are diagnostic only; the
    training forward keeps the exact Round23 operation graph.
    """
    field = getattr(outputs, "trifield_output", None)
    selector = getattr(model, "selector", None)
    if field is None or selector is None:
        return {"available": False, "reason": "missing trifield output or selector"}
    edge = getattr(field, "round31_edge_score", None)
    raw_forward = getattr(field, "round31_edge_raw", None)
    edge_valid = getattr(field, "round31_edge_valid", None)
    hidden = getattr(field, "round31_h", None)
    mode = str(
        getattr(
            field,
            "round31_support_pair_mode",
            getattr(selector, "support_pair_mode", "plain"),
        )
    )
    result = {
        "available": isinstance(edge, Tensor) and isinstance(raw_forward, Tensor),
        "mode": mode,
        "formula": (
            "tanh(g(a,b)-g(a,0)-g(0,b)+g(0,0))"
            if mode == "joint_residual"
            else "tanh(g(a,b))"
        ),
        "diagnostic_only_extra_references": mode == "plain",
        "scope": "same forward edge grid; no optimizer update; GT-free",
    }
    if not result["available"] or not isinstance(edge_valid, Tensor):
        result["reason"] = "edge field/decomposition unavailable"
        return result
    edge_valid = edge_valid.bool()
    if edge.shape != edge_valid.shape or raw_forward.shape != edge_valid.shape:
        raise RuntimeError("Round31 support-pair probe edge shape mismatch")
    if not isinstance(hidden, Tensor) or hidden.ndim != 3 or hidden.shape[-1] != 64:
        result["reason"] = "missing [B,L,64] edge hidden state"
        return result
    if hidden.shape[1] < 2:
        result["valid_edge_count"] = 0
        result["single_endpoint_zero_identity"] = {
            "applicable": mode == "joint_residual",
            "left_max_abs": 0.0,
            "right_max_abs": 0.0,
            "within_tolerance": True,
        }
        return result

    left, right = hidden[:, :-1], hidden[:, 1:]
    references = selector.edge_raw_components_from_pairs(
        left, right, include_references=True
    )
    raw = references["raw"].float()
    bounded = torch.tanh(raw)
    valid = edge_valid

    def stats(value):
        if value is None:
            return None
        values = value.detach().float()[valid]
        if not values.numel():
            return {"count": 0, "finite": True}
        levels = values.new_tensor([0.0, 0.1, 0.5, 0.9, 1.0])
        per_query = [
            row.detach().float()[row_valid].std(unbiased=False)
            for row, row_valid in zip(value, valid)
            if bool(row_valid.any())
        ]
        centered = torch.stack(per_query) if per_query else values.new_zeros((0,))
        return {
            "count": int(values.numel()),
            "finite": bool(torch.isfinite(values).all()),
            "mean": float(values.mean()),
            "mean_abs": float(values.abs().mean()),
            "std": float(values.std(unbiased=False)),
            "quantiles": [float(x) for x in torch.quantile(values, levels)],
            "per_query_centered_std": {
                "query_count": int(centered.numel()),
                "mean": float(centered.mean()) if centered.numel() else 0.0,
                "quantiles": [float(x) for x in torch.quantile(centered, levels)]
                if centered.numel()
                else [],
            },
        }

    def max_abs(value):
        if value is None:
            return 0.0
        values = value.detach().float()[valid]
        return float(values.abs().max()) if values.numel() else 0.0

    joint_raw = (
        references["g_ab"]
        - references["g_a0"]
        - references["g_0b"]
        + references["g_00"]
    )
    expected_raw = references["g_ab"] if mode == "plain" else joint_raw
    formula_error = (raw - expected_raw).float()
    forward_error = raw - raw_forward.float()
    bounded_error = bounded - edge.float()
    components = {
        name: stats(references[name]) for name in ("g_ab", "g_a0", "g_0b", "g_00")
    }
    components["configured_raw"] = stats(raw)
    components["joint_residual_raw"] = stats(joint_raw)
    result.update(
        {
            "valid_edge_count": int(valid.sum()),
            "raw_forward_identity_max_abs": max_abs(forward_error),
            "bounded_tanh_identity_max_abs": max_abs(bounded_error),
            "formula_identity_max_abs": max_abs(formula_error),
            "formula_identity_within_fp32_tolerance": bool(
                max_abs(formula_error) < 2.0e-6
            ),
            "forward_identity_within_fp32_tolerance": bool(
                max_abs(forward_error) < 2.0e-6
            ),
            "components": components,
            "configured_raw_name": "g_ab" if mode == "plain" else "joint_residual_raw",
            "bounded": stats(bounded),
            "configured_raw_abs_mean": float(raw.detach().float()[valid].abs().mean())
            if bool(valid.any())
            else 0.0,
            "configured_raw_nonzero_fraction": float(
                raw.detach().float()[valid].abs().gt(1.0e-8).float().mean()
            )
            if bool(valid.any())
            else 0.0,
            "joint_residual_raw_abs_mean": float(
                joint_raw.detach().float()[valid].abs().mean()
            )
            if bool(valid.any())
            else 0.0,
            "joint_residual_raw_nonzero_fraction": float(
                joint_raw.detach().float()[valid].abs().gt(1.0e-8).float().mean()
            )
            if bool(valid.any())
            else 0.0,
        }
    )

    # Algebra of raw joint primitive, deliberately NOT the length-two dispatcher:
    # pooled([a,0]) is generally nonzero and must not inherit this zero gate.
    def primitive(a, b):
        return (
            selector.edge_raw_components_from_pairs(a, b, include_references=True)[
                "raw"
            ]
            .float()
            .tanh()
        )

    zero_left = primitive(left, torch.zeros_like(right))
    zero_right = primitive(torch.zeros_like(left), right)
    zero_both = primitive(torch.zeros_like(left), torch.zeros_like(right))
    result["operator_scope"] = "raw joint primitive reference, not candidate dispatcher"
    result["support_readout"] = str(model.options.support_readout)

    result["single_endpoint_zero_identity"] = {
        "applicable": mode == "joint_residual",
        "left_max_abs": max_abs(zero_left),
        "right_max_abs": max_abs(zero_right),
        "both_max_abs": max_abs(zero_both),
        "within_tolerance": bool(
            mode != "joint_residual"
            or (
                max_abs(zero_left) < 2.0e-6
                and max_abs(zero_right) < 2.0e-6
                and max_abs(zero_both) < 2.0e-6
            )
        ),
        "reference": "post-RMS zero endpoint; bounded edge output",
    }

    gradients = {
        "available": False,
        "scope": "autograd.grad on single-endpoint zero outputs; existing .grad untouched",
    }
    if mode == "joint_residual" and torch.is_grad_enabled() and hidden.requires_grad:
        count = int(valid.sum())
        params = [p for p in selector.edge_head.parameters() if p.requires_grad]

        def vjp_norms(reduction):
            left_values, right_values = zero_left[valid], zero_right[valid]
            left_scalar = (
                getattr(left_values, reduction)() if count else zero_left.sum() * 0.0
            )
            right_scalar = (
                getattr(right_values, reduction)() if count else zero_right.sum() * 0.0
            )
            left_grads = torch.autograd.grad(
                left_scalar, [left] + params, retain_graph=True, allow_unused=True
            )
            right_grads = torch.autograd.grad(
                right_scalar, [right] + params, retain_graph=True, allow_unused=True
            )

            def norm(values):
                return float(
                    torch.sqrt(
                        sum(
                            (
                                x.detach().float().square().sum()
                                for x in values
                                if x is not None
                            ),
                            left_scalar.detach().float().new_zeros(()),
                        )
                    )
                )

            return {
                "left_endpoint_l2": norm(left_grads[:1]),
                "right_endpoint_l2": norm(right_grads[:1]),
                "left_parameter_l2": norm(left_grads[1:]),
                "right_parameter_l2": norm(right_grads[1:]),
            }

        # Keep the original sum diagnostic visible; its absolute residual scales
        # with panel size. The release check uses an explicitly mean-reduced VJP.
        summed = vjp_norms("sum")
        averaged = vjp_norms("mean")

        def flags(values, tolerance):
            return {
                "left_zero_gradient": values["left_endpoint_l2"] < tolerance,
                "right_zero_gradient": values["right_endpoint_l2"] < tolerance,
                "left_parameter_zero_gradient": values["left_parameter_l2"] < tolerance,
                "right_parameter_zero_gradient": values["right_parameter_l2"]
                < tolerance,
            }

        gradients = {
            "available": count > 0,
            "schema": "round31_zero_endpoint_mean_vjp_v2",
            "reduction": "valid_edge_mean",
            "valid_edge_count": count,
            "scope": "autograd.grad; no parameter update; existing .grad untouched",
            **averaged,
            "zero_gradient_tolerance": 1.0e-6,
            **flags(averaged, 1.0e-6),
            "legacy_sum_diagnostic": {
                "reduction": "valid_edge_sum",
                **summed,
                "zero_gradient_tolerance": 2.0e-5,
                **flags(summed, 2.0e-5),
                "used_for_release_gate": False,
            },
        }
    result["single_endpoint_zero_gradients"] = gradients
    from .factor_probes import support_pair_dispatch_probe

    result["length_two_dispatch_reference"] = support_pair_dispatch_probe(model, field)
    return result


def _duration(row):
    try:
        v = float(row.get("duration")) if row is not None else 0.0
        return v if v > 0 else None
    except (TypeError, ValueError):
        return None


def _gt_width_seconds(row, gi, span):
    windows = row.get("relevant_windows") if row is not None else None
    if isinstance(windows, (list, tuple)) and gi < len(windows):
        item = windows[gi]
        if isinstance(item, Tensor):
            item = item.detach().cpu().tolist()
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            try:
                return abs(float(item[1]) - float(item[0])), "metadata_seconds"
            except (TypeError, ValueError):
                pass
    d = _duration(row)
    if d is not None:
        return abs(float(span[1]) - float(span[0])) * d, "normalized_times_duration"
    return None, "unavailable"


def _short_scopes(outputs, batch):
    _, targets = _parts(batch)
    spans, mask = targets.get("gt_spans"), targets.get("gt_span_mask")
    b = int(outputs.span_logits.shape[0])
    if spans is None:
        spans = outputs.span_logits.new_zeros((b, 1, 2))
    if mask is None:
        mask = torch.zeros(spans.shape[:2], dtype=torch.bool, device=spans.device)
    spans, mask = (
        spans.to(outputs.span_logits.device).float(),
        mask.to(outputs.span_logits.device).bool(),
    )
    m10, m2, sources = torch.zeros_like(mask), torch.zeros_like(mask), set()
    for bi, row in enumerate(_rows(batch, b)):
        for gi in range(spans.shape[1]):
            if not bool(mask[bi, gi]):
                continue
            width, source = _gt_width_seconds(row, gi, spans[bi, gi])
            sources.add(source)
            if width is not None:
                m10[bi, gi], m2[bi, gi] = width <= 10.0, width <= 2.0
    source = (
        "unavailable"
        if not sources
        else next(iter(sources))
        if len(sources) == 1
        else "mixed:" + ",".join(sorted(sources))
    )
    return {"all": mask, "short_le_10s": m10, "short_le_2s": m2}, source


def _duplicates(top_iou, top_valid, scope, threshold):
    rates, hits = [], []
    for bi in range(top_iou.shape[0]):
        best, assigned = top_iou[bi].masked_fill(~scope[bi][None, :], -1.0).max(-1)
        ids = assigned[top_valid[bi] & best.ge(threshold)]
        count = int(ids.numel())
        rates.append(
            top_iou.new_tensor(
                float(max(0, count - int(ids.unique().numel()))) / max(1, count)
            )
        )
        hits.append(top_iou.new_tensor(float(count)))
    if not rates:
        z = top_iou.sum() * 0.0
        return z, z
    return torch.stack(rates).mean(), torch.stack(hits).sum()


def probe_short_gt_coverage(
    outputs: Any, batch: Any, threshold: float = 0.70
) -> Mapping[str, Any]:
    geometry, field = (
        candidate_geometry(outputs, batch),
        getattr(outputs, "trifield_output", None),
    )
    score, valid = (
        (outputs.span_logits.float() if field is None else field.score.float()),
        geometry.valid.bool(),
    )
    scopes, source = _short_scopes(outputs, batch)
    all_flat, valid_flat, score_flat = (
        geometry.all_iou.float().flatten(1, 2),
        valid.flatten(1),
        score.flatten(1),
    )
    safe, cache = score_flat.masked_fill(~valid_flat, float("-inf")), {}
    for k0 in (10, 30):
        k = min(k0, score_flat.shape[1])
        index = safe.topk(k, dim=1).indices
        cache[k0] = (
            all_flat.gather(1, index[..., None].expand(-1, -1, all_flat.shape[-1])),
            valid_flat.gather(1, index),
        )
    result = {
        "available": True,
        "iou_threshold": float(threshold),
        "short_scope_source": source,
        "candidate_count_mean": valid_flat.sum(1).float().mean().detach(),
        "full_grid_candidate_count": int(score.shape[1] * score.shape[2]),
    }
    for name, scope in scopes.items():
        result["coverage/" + name + "/gt_count"] = scope.sum().float().detach()
        full, candidate = [], []
        for bi in range(scope.shape[0]):
            for gi in torch.nonzero(scope[bi], as_tuple=False).flatten().tolist():
                full.append(
                    all_flat[bi, :, gi].max().clamp_min(0).ge(threshold).float()
                )
                candidate.append(
                    all_flat[bi, :, gi]
                    .masked_fill(~valid_flat[bi], -1)
                    .max()
                    .clamp_min(0)
                    .ge(threshold)
                    .float()
                )
        result["coverage/" + name + "/full_recall@0.70"] = (
            torch.stack(full).mean().detach() if full else score.new_tensor(0)
        )
        result["coverage/" + name + "/candidate_recall@0.70"] = (
            torch.stack(candidate).mean().detach() if candidate else score.new_tensor(0)
        )
        for k0 in (10, 30):
            top_iou, top_valid = cache[k0]
            vals = []
            for bi in range(scope.shape[0]):
                ids = torch.nonzero(scope[bi], as_tuple=False).flatten()
                if ids.numel():
                    vals.extend(
                        list(
                            top_iou[bi][:, ids]
                            .masked_fill(~top_valid[bi][:, None], -1.0)
                            .max(0)
                            .values.ge(threshold)
                            .float()
                        )
                    )
            result["coverage/" + name + "/top" + str(k0) + "_recall@0.70"] = (
                torch.stack(vals).mean().detach() if vals else score.new_tensor(0)
            )
            rate, hit = _duplicates(top_iou, top_valid, scope, threshold)
            result["coverage/" + name + "/top" + str(k0) + "_duplicate_rate@0.70"] = (
                rate.detach()
            )
            result["coverage/" + name + "/top" + str(k0) + "_positive_hit_count"] = (
                hit.detach()
            )
        if name.startswith("short_"):
            result[name + "/full_recall@0.70"] = result[
                "coverage/" + name + "/full_recall@0.70"
            ]
            result[name + "/candidate_recall@0.70"] = result[
                "coverage/" + name + "/candidate_recall@0.70"
            ]
    return result


def _round31_scalar(value: Any, default: float = 0.0) -> float:
    if isinstance(value, Tensor):
        value = value.detach().float()
        return float(value.item()) if value.numel() == 1 else default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _round31_optional_scalar(value: Any) -> float | None:
    """Convert an optional metric to a JSON-safe scalar without inventing activity."""
    if value is None:
        return None
    if isinstance(value, Tensor):
        value = value.detach().float()
        if value.numel() != 1:
            return None
        return float(value.item())
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _round31_pair_metric_block(
    metrics: Mapping[str, Any],
    prefix: str,
    detail: Mapping[str, Any] | None = None,
    pair_margins: Tensor | None = None,
) -> dict[str, Any]:
    """Expose actual pair metrics and validate their detached reduction."""
    source = f"{prefix}/"
    out: dict[str, Any] = {
        "gap_mean": _round31_optional_scalar(metrics.get(source + "gap_mean")),
        "margin_mean": _round31_optional_scalar(metrics.get(source + "margin_mean")),
        "margin_met_rate": _round31_optional_scalar(
            metrics.get(source + "margin_met_rate")
        ),
        "metric_source": source.rstrip("/"),
    }
    if not isinstance(detail, Mapping):
        out["recomputed"] = False
        return out
    available = detail.get("available")
    gaps = detail.get("gaps")
    if not isinstance(available, Tensor) or not isinstance(gaps, Tensor):
        out["recomputed"] = False
        return out
    available = available.detach().bool()
    observed_gaps = gaps.detach().float()[available]
    if isinstance(pair_margins, Tensor) and tuple(pair_margins.shape) == tuple(
        available.shape
    ):
        observed_margins = pair_margins.detach().float()[available]
    else:
        observed_margins = observed_gaps.new_full(observed_gaps.shape, 0.2)
    observed_gap = float(observed_gaps.mean()) if observed_gaps.numel() else 0.0
    observed_margin = (
        float(observed_margins.mean()) if observed_margins.numel() else 0.0
    )
    observed_met = (
        float((observed_gaps >= observed_margins).float().mean())
        if observed_gaps.numel()
        else 0.0
    )
    out.update(
        {
            "recomputed": True,
            "available_pair_count": int(observed_gaps.numel()),
            "recomputed_gap_mean": observed_gap,
            "recomputed_margin_mean": observed_margin,
            "recomputed_margin_met_rate": observed_met,
            "gap_mean_abs_error": abs(
                (out["gap_mean"] if out["gap_mean"] is not None else observed_gap)
                - observed_gap
            ),
            "margin_mean_abs_error": abs(
                (
                    out["margin_mean"]
                    if out["margin_mean"] is not None
                    else observed_margin
                )
                - observed_margin
            ),
            "margin_met_rate_abs_error": abs(
                (
                    out["margin_met_rate"]
                    if out["margin_met_rate"] is not None
                    else observed_met
                )
                - observed_met
            ),
        }
    )
    out["within_fp32_tolerance"] = bool(
        out["gap_mean_abs_error"] <= 2.0e-6
        and out["margin_mean_abs_error"] <= 2.0e-6
        and out["margin_met_rate_abs_error"] <= 2.0e-6
    )
    return out


def _round31_parameter_groups(model: Any) -> dict[str, list[tuple[str, Any]]]:
    """Return stable R31 gradient buckets.

    The route report names the S head/projection separately from E, T,
    carrier, and legacy support parameters.  The two compatibility aliases
    (``edge_head``/``score_head``) are retained for older panel consumers;
    they point at the same parameter sets and do not change the loss graph.
    """
    groups: dict[str, list[tuple[str, Any]]] = {
        "all_trainable": [],
        "s_head": [],
        "s_projection": [],
        "e_interaction": [],
        "t_interaction": [],
        "carrier": [],
        "parent_carrier": [],
        "legacy_support": [],
        "legacy": [],
        "score_head": [],
        "edge_head": [],
        "other": [],
    }
    named_parameters = getattr(model, "named_parameters", None)
    if not callable(named_parameters):
        return groups
    for name, parameter in named_parameters():
        if not parameter.requires_grad:
            continue
        groups["all_trainable"].append((name, parameter))
        if name.startswith(("selector.edge_head.", "edge_head.")):
            groups["s_head"].append((name, parameter))
            groups["edge_head"].append((name, parameter))
        if name.startswith(("selector.s_projection.", "s_projection.")):
            groups["s_projection"].append((name, parameter))
        if name.startswith(("selector.e_interaction.", "e_interaction.")):
            groups["e_interaction"].append((name, parameter))
        if name.startswith(("selector.t_interaction.", "t_interaction.")):
            groups["t_interaction"].append((name, parameter))
        if not name.startswith("parent_model.") and (
            "carrier" in name.lower()
            or name.endswith("carrier_bias")
            or name.startswith("selector.carrier.")
        ):
            groups["carrier"].append((name, parameter))
        # Keep the actual parent path separate from legacy selector support
        # residuals.  Parent parameters are not evidence that S leaked into
        # rank/LS; they are the ordinary carrier/shared model path.
        if name.startswith("parent_model."):
            groups["parent_carrier"].append((name, parameter))
            groups["score_head"].append((name, parameter))
        if name.startswith(
            (
                "selector.support.",
                "selector.context_branch.",
                "selector.dispersion_branch.",
            )
        ):
            groups["legacy_support"].append((name, parameter))
        if name.startswith(("shared_head.", "score_head.")):
            groups["legacy_support"].append((name, parameter))
            groups["score_head"].append((name, parameter))
        if name.startswith(
            (
                "selector.support.",
                "selector.context_branch.",
                "selector.dispersion_branch.",
                "shared_head.",
                "score_head.",
            )
        ):
            groups["legacy"].append((name, parameter))
    assigned = {id(parameter) for name, parameter in groups["s_head"]}
    assigned.update(id(parameter) for name, parameter in groups["s_projection"])
    assigned.update(id(parameter) for name, parameter in groups["e_interaction"])
    assigned.update(id(parameter) for name, parameter in groups["t_interaction"])
    assigned.update(id(parameter) for name, parameter in groups["carrier"])
    assigned.update(id(parameter) for name, parameter in groups["parent_carrier"])
    assigned.update(id(parameter) for name, parameter in groups["legacy_support"])
    assigned.update(id(parameter) for name, parameter in groups["legacy"])
    groups["other"] = [
        (name, parameter)
        for name, parameter in groups["all_trainable"]
        if id(parameter) not in assigned
    ]
    return groups


def _round31_grad_tuple(loss: Any, parameters: list[Any]) -> tuple[Tensor | None, ...]:
    if not parameters or not isinstance(loss, Tensor) or not loss.requires_grad:
        return tuple(None for _ in parameters)
    return torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)


def _round31_grad_norm(grads: tuple[Tensor | None, ...]) -> float:
    values = [g.detach().float().square().sum() for g in grads if isinstance(g, Tensor)]
    return float(torch.stack(values).sum().sqrt()) if values else 0.0


def _round31_gradient_component(
    loss: Any,
    outer_weight: float,
    parameters: list[tuple[str, Any]],
) -> dict[str, Any]:
    names = [name for name, _ in parameters]
    grads = _round31_grad_tuple(loss, [parameter for _, parameter in parameters])
    connected = [g for g in grads if isinstance(g, Tensor)]
    finite = all(bool(torch.isfinite(g.detach()).all()) for g in connected)
    raw_norm = _round31_grad_norm(grads)
    zero_count = sum(
        int(isinstance(g, Tensor) and not bool(g.detach().abs().gt(0).any()))
        for g in grads
    )
    return {
        "parameter_names": names,
        "parameter_count": len(grads),
        "connected_parameter_count": len(connected),
        "none_parameter_count": len(grads) - len(connected),
        "zero_parameter_count": zero_count,
        "finite": finite,
        "raw_l2": raw_norm,
        "outer_weight": float(outer_weight),
        "weighted_l2": raw_norm * float(outer_weight),
        "requires_grad": bool(isinstance(loss, Tensor) and loss.requires_grad),
    }


def _round31_component_gradient_recomposition(
    target: Any,
    components: Mapping[str, Any],
    outer_weight: float,
    model: Any,
) -> dict[str, Any]:
    """Compare live target VJP with the sum of its named loss components."""
    groups = _round31_parameter_groups(model)
    result: dict[str, Any] = {
        "available": bool(components),
        "outer_weight": float(outer_weight),
        "components": {},
        "groups": {},
    }
    for component_name, component in components.items():
        result["components"][component_name] = {}
        for group_name, parameters in groups.items():
            result["components"][component_name][group_name] = (
                _round31_gradient_component(component, outer_weight, parameters)
            )
    for group_name, parameters in groups.items():
        if not parameters:
            result["groups"][group_name] = {
                "available": False,
                "reason": "no active parameters",
            }
            continue
        params = [parameter for _, parameter in parameters]
        target_grads = _round31_grad_tuple(target, params)
        component_grads = {
            name: _round31_grad_tuple(value, params)
            for name, value in components.items()
        }
        recomposed: list[Tensor] = []
        target_weighted: list[Tensor] = []
        for index, parameter in enumerate(params):
            target_grad = target_grads[index]
            target_weighted.append(
                torch.zeros_like(parameter)
                if target_grad is None
                else target_grad * float(outer_weight)
            )
            value = torch.zeros_like(parameter)
            for gradients in component_grads.values():
                gradient = gradients[index]
                if gradient is not None:
                    value = value + gradient * float(outer_weight)
            recomposed.append(value)
        error = _round31_grad_norm(
            tuple(a - b for a, b in zip(target_weighted, recomposed))
        )
        target_norm = _round31_grad_norm(tuple(target_weighted))
        relative = error / max(target_norm, 1.0e-12)
        result["groups"][group_name] = {
            "available": True,
            "target_raw_l2": _round31_grad_norm(target_grads),
            "target_weighted_l2": target_norm,
            "recomposed_weighted_l2": _round31_grad_norm(tuple(recomposed)),
            "absolute_error": error,
            "relative_error": relative,
            "within_tolerance": bool(
                relative <= 1.0e-4 or (target_norm <= 1.0e-6 and error <= 1.0e-6)
            ),
            "diagnostic_precision": "fp32 autograd VJP",
        }
    return result


def _round31_terms_scalar_recomposition(
    terms: FiveLossTerms, weights: Mapping[str, Any]
) -> dict[str, Any]:
    names = ("rank", "evidence", "support", "transition", "endpoint")
    values = {
        name: _round31_optional_scalar(getattr(terms, name, None)) for name in names
    }
    weighted = {
        name: (values[name] * float(weights.get(name, 1.0)))
        if values[name] is not None
        else None
        for name in names
    }
    if any(value is None for value in weighted.values()):
        return {"available": False, "values": values, "weighted_values": weighted}
    expected = sum(weighted.values())
    actual = _round31_optional_scalar(getattr(terms, "total", None))
    error = abs(expected - actual) if actual is not None else None
    return {
        "available": actual is not None,
        "values": values,
        "weighted_values": weighted,
        "weighted_sum": expected,
        "terms_total": actual,
        "absolute_error": error,
        "within_fp32_tolerance": bool(error is not None and error <= 2.0e-6),
    }


def _round31_component_scalar_recomposition(
    target: Any,
    components: Mapping[str, Any],
    outer_weight: float = 1.0,
) -> dict[str, Any]:
    values = {
        name: _round31_optional_scalar(value) for name, value in components.items()
    }
    active = {name: value for name, value in values.items() if value is not None}
    target_value = _round31_optional_scalar(target)
    result: dict[str, Any] = {
        "available": bool(active) and target_value is not None,
        "components": values,
        "target": target_value,
        "outer_weight": float(outer_weight),
    }
    if not result["available"]:
        return result
    recomposed = sum(active.values())
    error = abs(recomposed - target_value)
    result.update(
        {
            "recomposed": recomposed,
            "weighted_components": {
                name: value * float(outer_weight) for name, value in active.items()
            },
            "weighted_recomposed": recomposed * float(outer_weight),
            "weighted_target": target_value * float(outer_weight),
            "absolute_error": error,
            "weighted_absolute_error": error * abs(float(outer_weight)),
            "within_fp32_tolerance": bool(error <= 2.0e-6),
        }
    )
    return result


def _round31_scalar_gradient_snapshot(
    value: Any,
    model: Any,
    *,
    source: str,
    outer_weight: float = 1.0,
) -> dict[str, Any]:
    """Record raw/effective scalar VJPs without changing model state.

    K/P references and effective components must stay visibly separate.  In
    particular, beta=0 permits a live raw P reference gradient while its
    effective P gradient must be zero.
    """
    groups = _round31_parameter_groups(model)
    named_fn = getattr(model, "named_parameters", None)
    named_parameters = (
        [(name, parameter) for name, parameter in named_fn() if parameter.requires_grad]
        if callable(named_fn)
        else []
    )
    parameters = [parameter for _, parameter in named_parameters]
    if not isinstance(value, Tensor):
        return {
            "available": False,
            "source": source,
            "reason": "missing scalar tensor",
            "outer_weight": float(outer_weight),
        }
    if not parameters or not value.requires_grad:
        grads: tuple[Tensor | None, ...] = tuple(None for _ in parameters)
    else:
        grads = torch.autograd.grad(
            value, parameters, retain_graph=True, allow_unused=True
        )
    finite = all(
        gradient is None or bool(torch.isfinite(gradient.detach()).all())
        for gradient in grads
    )
    raw_norm = _round31_grad_norm(grads)
    by_name = {name: gradient for (name, _), gradient in zip(named_parameters, grads)}
    module_norms: dict[str, Any] = {}
    for group_name, group_parameters in groups.items():
        selected = tuple(by_name.get(name) for name, _ in group_parameters)
        group_norm = _round31_grad_norm(selected)
        connected_count = sum(gradient is not None for gradient in selected)
        zero_count = sum(
            int(
                isinstance(gradient, Tensor)
                and not bool(gradient.detach().abs().gt(0).any())
            )
            for gradient in selected
        )
        module_norms[group_name] = {
            "raw_l2": group_norm,
            "weighted_l2": group_norm * abs(float(outer_weight)),
            "parameter_count": len(group_parameters),
            "connected_parameter_count": connected_count,
            "none_parameter_count": len(group_parameters) - connected_count,
            "zero_parameter_count": zero_count,
            "finite": all(
                gradient is None or bool(torch.isfinite(gradient.detach()).all())
                for gradient in selected
            ),
        }
    return {
        "available": True,
        "source": source,
        "scalar": _round31_optional_scalar(value),
        "outer_weight": float(outer_weight),
        "raw_l2": raw_norm,
        "weighted_l2": raw_norm * abs(float(outer_weight)),
        "finite": finite,
        "parameter_count": len(parameters),
        "connected_parameter_count": sum(gradient is not None for gradient in grads),
        "module_norms": module_norms,
    }


def _round31_rank_budget_probe(
    model: Any,
    terms: FiveLossTerms,
    rank_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and expose the actual R29 K/P budget returned by losses.py."""
    required = (
        "original_kl",
        "kl_query",
        "kl_active",
        "pair_query",
        "pair_available",
        "pair_active",
        "fallback_active",
        "excluded",
        "mixed_query",
        "kl_component",
        "pair_component",
        "pair_reference",
        "pair_reference_active_only",
        "alpha",
        "beta",
        "kl_coefficients",
        "pair_coefficients",
        "rank_pair_enabled",
        "rank_kl_half",
        "pair_reference_only",
    )
    missing = [name for name in required if name not in rank_payload]
    if missing:
        raise RuntimeError("round31 rank budget missing fields: " + ", ".join(missing))

    def tensor(name: str) -> Tensor:
        value = rank_payload[name]
        if not isinstance(value, Tensor):
            raise RuntimeError(f"round31 rank budget field {name} must be a tensor")
        return value

    kl_query = tensor("kl_query").detach().float()
    kl_active = tensor("kl_active").detach().bool()
    pair_query = tensor("pair_query").detach().float()
    pair_available = tensor("pair_available").detach().bool()
    pair_active = tensor("pair_active").detach().bool()
    fallback_active = tensor("fallback_active").detach().bool()
    excluded = tensor("excluded").detach().bool()
    mixed_query = tensor("mixed_query").detach().float()
    kl_coefficients = tensor("kl_coefficients").detach().float()
    pair_coefficients = tensor("pair_coefficients").detach().float()
    query_values = (
        kl_query,
        kl_active,
        pair_query,
        pair_available,
        pair_active,
        fallback_active,
        excluded,
        mixed_query,
        kl_coefficients,
        pair_coefficients,
    )
    query_shapes = {tuple(value.shape) for value in query_values}
    if len(query_shapes) != 1:
        raise RuntimeError(f"round31 rank budget query shape mismatch: {query_shapes}")

    expected_pair_active = kl_active & pair_available
    expected_fallback = kl_active & ~expected_pair_active
    expected_excluded = ~kl_active
    if not torch.equal(pair_active, expected_pair_active):
        raise RuntimeError("round31 pair_active is not KL-active AND pair-available")
    if not torch.equal(fallback_active, expected_fallback):
        raise RuntimeError(
            "round31 fallback_active does not partition KL-active queries"
        )
    if not torch.equal(excluded, expected_excluded):
        raise RuntimeError(
            "round31 excluded does not equal complement of KL-active queries"
        )

    alpha = float(_round31_scalar(rank_payload["alpha"], default=float("nan")))
    beta = float(_round31_scalar(rank_payload["beta"], default=float("nan")))
    rank_kl_half = bool(rank_payload["rank_kl_half"])
    rank_pair_enabled = bool(rank_payload["rank_pair_enabled"])
    expected_alpha = 0.5 if rank_kl_half else 1.0
    expected_beta = 0.5 if rank_pair_enabled else 0.0
    if alpha != expected_alpha or beta != expected_beta:
        raise RuntimeError(
            f"round31 rank budget factor mismatch: alpha={alpha}, beta={beta}, "
            f"expected {expected_alpha}/{expected_beta}"
        )

    expected_k = torch.where(
        pair_active,
        kl_query.new_full(kl_query.shape, expected_alpha),
        torch.ones_like(kl_query),
    ).masked_fill(~kl_active, 0.0)
    expected_p = torch.where(
        pair_active,
        pair_query.new_full(pair_query.shape, expected_beta),
        torch.zeros_like(pair_query),
    )
    if not torch.equal(kl_coefficients, expected_k):
        raise RuntimeError(
            "round31 KL coefficients do not match alpha/fallback contract"
        )
    if not torch.equal(pair_coefficients, expected_p):
        raise RuntimeError(
            "round31 P coefficients do not match beta/eligibility contract"
        )

    expected_mixed = kl_coefficients * kl_query + pair_coefficients * pair_query
    mixed_delta = (mixed_query - expected_mixed).abs().masked_select(kl_active)
    mixed_error = (
        mixed_delta.max() if mixed_delta.numel() else mixed_query.new_zeros(())
    )
    if mixed_query.numel() and float(mixed_error) > 2.0e-6:
        raise RuntimeError(
            f"round31 mixed K/P query recomposition error={float(mixed_error):g}"
        )

    denominator = int(kl_active.sum())
    eligible_pair_queries = int(pair_active.sum())
    fallback_queries = int(fallback_active.sum())
    excluded_queries = int(excluded.sum())
    rank_weight = float(getattr(model, "loss_weights", {}).get("rank", 1.0))
    components = {
        "K_effective": rank_payload["kl_component"],
        "P_effective": rank_payload["pair_component"],
    }
    scalar_recomposition = _round31_component_scalar_recomposition(
        getattr(terms, "rank", None), components, rank_weight
    )
    gradient_recomposition = _round31_component_gradient_recomposition(
        getattr(terms, "rank", None), components, rank_weight, model
    )
    raw_k = _round31_scalar_gradient_snapshot(
        rank_payload["original_kl"],
        model,
        source="rank.original_kl_reference",
        outer_weight=rank_weight,
    )
    raw_p = _round31_scalar_gradient_snapshot(
        rank_payload["pair_reference"],
        model,
        source="rank.pair_reference_original_KL_denominator",
        outer_weight=rank_weight,
    )
    effective_k = _round31_scalar_gradient_snapshot(
        rank_payload["kl_component"],
        model,
        source="rank.kl_component_effective",
        outer_weight=rank_weight,
    )
    effective_p = _round31_scalar_gradient_snapshot(
        rank_payload["pair_component"],
        model,
        source="rank.pair_component_effective",
        outer_weight=rank_weight,
    )
    effective_pair_zero = bool(beta == 0.0 and effective_p["raw_l2"] <= 1.0e-12)
    return {
        "available": True,
        "decomposition_enabled_all_cells": True,
        "rank_pair_enabled": rank_pair_enabled,
        "rank_kl_half": rank_kl_half,
        "alpha": alpha,
        "beta": beta,
        "original_kl_active_count": denominator,
        "pair_available_query_count": int(pair_available.sum()),
        "eligible_pair_query_count": eligible_pair_queries,
        "fallback_query_count": fallback_queries,
        "excluded_or_no_gt_query_count": excluded_queries,
        "denominator": {
            "name": "original_KL_active_queries",
            "count": denominator,
        },
        "pair_reference_only": bool(rank_payload["pair_reference_only"]),
        "query_eligibility": {
            "kl_active": kl_active.cpu().tolist(),
            "pair_available": pair_available.cpu().tolist(),
            "pair_active": pair_active.cpu().tolist(),
            "fallback_active": fallback_active.cpu().tolist(),
            "excluded": excluded.cpu().tolist(),
        },
        "raw_reference": {
            "K": {
                "source": "rank.kl_query",
                "per_query": kl_query.cpu().tolist(),
                "scalar": _round31_optional_scalar(rank_payload["original_kl"]),
                "gradient": raw_k,
            },
            "P": {
                "source": "rank.pair_query",
                "per_query": pair_query.cpu().tolist(),
                "scalar_original_KL_denominator": _round31_optional_scalar(
                    rank_payload["pair_reference"]
                ),
                "scalar_pair_active_only": _round31_optional_scalar(
                    rank_payload["pair_reference_active_only"]
                ),
                "gradient": raw_p,
            },
        },
        "effective_components": {
            "K": {
                "coefficient_per_query": kl_coefficients.cpu().tolist(),
                "per_query": (kl_coefficients * kl_query).cpu().tolist(),
                "scalar": _round31_optional_scalar(rank_payload["kl_component"]),
                "gradient": effective_k,
            },
            "P": {
                "coefficient_per_query": pair_coefficients.cpu().tolist(),
                "per_query": (pair_coefficients * pair_query).cpu().tolist(),
                "scalar": _round31_optional_scalar(rank_payload["pair_component"]),
                "gradient": effective_p,
            },
            "pair_gradient_zero_when_beta_zero": effective_pair_zero,
        },
        "mixed_query": mixed_query.cpu().tolist(),
        "scalar_recomposition": scalar_recomposition,
        "gradient_recomposition": gradient_recomposition,
        "coefficient_closure": {
            "kl_coefficients_exact": True,
            "pair_coefficients_exact": True,
            "mixed_query_max_abs_error": float(mixed_error),
            "within_fp32_tolerance": bool(float(mixed_error) <= 2.0e-6),
        },
        # Stable aliases consumed by the R31 route exporter.  Keep the
        # detailed raw_reference/effective_components blocks above for older
        # readers, while making the K/P graph distinction easy to audit.
        "route_gradients": {
            "raw": {"K": raw_k, "P": raw_p},
            "effective": {"K": effective_k, "P": effective_p},
            "mixed": _round31_scalar_gradient_snapshot(
                rank_payload["mixed_query"].masked_select(kl_active).mean()
                if bool(kl_active.any())
                else rank_payload["mixed_query"].sum() * 0.0,
                model,
                source="rank.mixed_query_effective",
                outer_weight=rank_weight,
            ),
        },
    }


def probe_round31_candidate_pairs(
    model: Any,
    outputs: Any,
    terms: FiveLossTerms,
) -> Mapping[str, Any]:
    """Report the all-cell cache and actual support/rank budget.

    Candidate qualification is always built.  C only controls whether P is
    an effective training component; B only controls alpha on pair-eligible
    queries.  The raw K/P references are retained in every cell.
    """
    field = getattr(outputs, "trifield_output", None)
    payload = getattr(field, "round31_candidate_pair_probe", None)
    opt = getattr(model, "options", None)
    support_enabled = bool(getattr(opt, "support_wide_pair", False))
    rank_pair_enabled = bool(getattr(opt, "rank_candidate_pair", False))
    rank_kl_half = bool(getattr(opt, "rank_kl_half", False))
    result: dict[str, Any] = {
        "available": False,
        "enabled": False,
        "support_enabled": support_enabled,
        "rank_pair_enabled": rank_pair_enabled,
        "rank_kl_half": rank_kl_half,
        "support": {
            "enabled": support_enabled,
            "available": False,
            "reason": (
                "disabled by round31 factor"
                if not support_enabled
                else "candidate pair cache unavailable"
            ),
        },
        "rank": {
            "enabled": True,
            "pair_term_enabled": rank_pair_enabled,
            "available": False,
            "reason": "candidate pair cache unavailable",
        },
        "terms_scalar_recomposition": _round31_terms_scalar_recomposition(
            terms, getattr(model, "loss_weights", {})
        ),
    }
    if not isinstance(payload, Mapping) or not isinstance(
        payload.get("pairs"), Mapping
    ):
        result["reason"] = "round31 candidate pair cache missing"
        result["gradient_recomposition"] = {
            "support": {"available": False, "reason": result["support"]["reason"]},
            "rank": {"available": False, "reason": result["rank"]["reason"]},
        }
        return result

    pairs = payload["pairs"]
    available_raw = pairs.get("available")
    positive_valid = pairs.get("positive_valid")
    gt_mask = pairs.get("gt_mask")
    if not all(
        isinstance(value, Tensor) for value in (available_raw, positive_valid, gt_mask)
    ):
        raise RuntimeError("round31 candidate pair probe has incomplete cache")
    available = available_raw.detach().bool()
    positive_valid = positive_valid.detach().bool()
    gt_mask = gt_mask.detach().bool()
    margin = pairs.get("margin")
    pool_counts = pairs.get("pool_counts")
    cache_support = bool(payload.get("support_enabled", False))
    cache_rank = bool(payload.get("rank_enabled", False))
    cache_rank_pair = bool(payload.get("rank_pair_enabled", rank_pair_enabled))
    cache_rank_half = bool(payload.get("rank_kl_half", rank_kl_half))
    result.update(
        {
            "available": True,
            "enabled": True,
            "cache_support_enabled": cache_support,
            "cache_rank_enabled": cache_rank,
            "cache_rank_pair_enabled": cache_rank_pair,
            "cache_rank_kl_half": cache_rank_half,
            "configuration_match": bool(
                cache_support == support_enabled
                and cache_rank is True
                and cache_rank_pair == rank_pair_enabled
                and cache_rank_half == rank_kl_half
            ),
            "batch_size": int(available.shape[0]),
            "gt_slots": int(available.shape[1]),
            "candidate_positive_eligible_gt_count": int(positive_valid.sum()),
            "candidate_gt_count": int(gt_mask.sum()),
            "short_available_count": int(available[..., 0].sum()),
            "wide_available_count": int(available[..., 1].sum()),
            "available_direction_count": int(available.sum()),
            "available_query_count": int(available.any(-1).any(-1).sum()),
            "short_query_count": int(available[..., 0].any(-1).sum()),
            "wide_query_count": int(available[..., 1].any(-1).sum()),
            "selection_is_detached": not available_raw.requires_grad,
            "selection_source": (
                "detached field.score/IoU candidate cache; GT is used only by "
                "loss selection"
            ),
        }
    )
    if isinstance(margin, Tensor):
        selected_margin = margin.detach().float()[available]
        result["selected_margin_mean"] = (
            float(selected_margin.mean()) if selected_margin.numel() else 0.0
        )
        result["selected_margin_min"] = (
            float(selected_margin.min()) if selected_margin.numel() else 0.0
        )
    if isinstance(pool_counts, Tensor):
        pools = pool_counts.detach().float()[available]
        result["pool_count_mean"] = float(pools.mean()) if pools.numel() else 0.0
        result["pool_count_max"] = int(pools.max()) if pools.numel() else 0

    metrics = terms.metrics
    support_payload = payload.get("support")
    rank_payload = payload.get("rank")
    support_payload = support_payload if isinstance(support_payload, Mapping) else {}
    rank_payload = rank_payload if isinstance(rank_payload, Mapping) else {}
    support_detail = support_payload.get("detail")
    rank_detail = rank_payload.get("detail")
    support_metrics = {
        "active_query_count": _round31_scalar(
            metrics.get("candidate_support/active_query_count")
        ),
        "active_gt_count": _round31_scalar(
            metrics.get("candidate_support/active_gt_count")
        ),
        "wide_pair_count": _round31_scalar(
            metrics.get("candidate_support/wide_pair_count")
        ),
        "fallback_query_count": _round31_scalar(
            metrics.get("candidate_support/fallback_query_count")
        ),
        "zero_query_count": _round31_scalar(
            metrics.get("candidate_support/zero_query_count")
        ),
        "wide_component": _round31_scalar(
            metrics.get("candidate_support/wide_component")
        ),
        "donor_component": _round31_scalar(
            metrics.get("candidate_support/donor_component")
        ),
    }
    rank_metrics = {
        "pair_active_query_count": _round31_scalar(
            metrics.get("candidate_rank/active_query_count")
        ),
        "pair_active_gt_count": _round31_scalar(
            metrics.get("candidate_rank/active_gt_count")
        ),
        "pair_count": _round31_scalar(metrics.get("candidate_rank/pair_count")),
        "fallback_query_count": _round31_scalar(
            metrics.get("candidate_rank/fallback_query_count")
        ),
        "kl_component": _round31_scalar(metrics.get("candidate_rank/kl_component")),
        "pair_component": _round31_scalar(metrics.get("candidate_rank/pair_component")),
    }
    support_result = {
        "enabled": support_enabled,
        "available": bool(support_enabled and isinstance(support_detail, Mapping)),
        **support_metrics,
        "term_source": ("candidate wide pairs with explicit donor fallback"),
    }
    rank_result = {
        "enabled": True,
        "pair_term_enabled": rank_pair_enabled,
        "available": bool(isinstance(rank_detail, Mapping)),
        **rank_metrics,
        "term_source": (
            "raw K/P split; effective alpha*K+beta*P on pair-active "
            "queries, original K fallback"
        ),
    }

    if support_enabled:
        support_result["pair_metrics"] = _round31_pair_metric_block(
            metrics,
            "candidate_support",
            support_detail,
            margin * getattr(model, "r48_support_margin_scale", 1.0),
        )
        fallback_active = support_payload.get("fallback_active")
        mixed_active = support_payload.get("active")
        support_result["donor"] = {
            "enabled": True,
            "reference_only_not_active_fallback": True,
            "active_fallback_query_count": (
                int(fallback_active.detach().bool().sum())
                if isinstance(fallback_active, Tensor)
                else None
            ),
            "mixed_active_query_count": (
                int(mixed_active.detach().bool().sum())
                if isinstance(mixed_active, Tensor)
                else None
            ),
            "gap_mean": _round31_optional_scalar(
                metrics.get("donor_reference/pair_weighted_margin_mean")
            ),
            "margin_mean": (
                0.2 if support_metrics["wide_pair_count"] is not None else None
            ),
            "margin_met_rate": _round31_optional_scalar(
                metrics.get("donor_reference/pair_weighted_margin_met_rate")
            ),
            "pair_count": _round31_optional_scalar(
                metrics.get("donor_reference/pair_orientation_count")
            ),
            "metric_source": (
                "donor_reference/* from current readout length-two dispatcher; all donor-eligible queries, not fallback-only"
            ),
            "active_metric_source": "support.fallback_active/mixed_active",
            "support_readout": str(model.options.support_readout),
            "fallback_only_gap_status": "not_exported; reference gap must not stand in for active fallback",
        }
        support_components = {
            "wide": support_payload.get("wide_component"),
            "donor": support_payload.get("donor_component"),
        }
        for name in tuple(support_components):
            if support_components[name] is None:
                support_components[name] = metrics.get(
                    "candidate_support/" + name + "_component"
                )
        support_result["scalar_recomposition"] = (
            _round31_component_scalar_recomposition(
                getattr(terms, "support", None),
                support_components,
                float(getattr(model, "loss_weights", {}).get("support", 1.0)),
            )
        )
        support_weight = float(getattr(model, "loss_weights", {}).get("support", 1.0))
        support_result["gradient_recomposition"] = (
            _round31_component_gradient_recomposition(
                getattr(terms, "support", None),
                support_components,
                support_weight,
                model,
            )
        )
        support_result["route_gradients"] = {
            "raw": {
                "wide": _round31_scalar_gradient_snapshot(
                    support_components["wide"],
                    model,
                    source="support.wide_component_raw",
                    outer_weight=1.0,
                ),
                "donor": _round31_scalar_gradient_snapshot(
                    support_components["donor"],
                    model,
                    source="support.donor_component_raw",
                    outer_weight=1.0,
                ),
                "mixed": _round31_scalar_gradient_snapshot(
                    getattr(terms, "support", None),
                    model,
                    source="support.mixed_loss_raw",
                    outer_weight=1.0,
                ),
            },
            "effective": {
                "wide": _round31_scalar_gradient_snapshot(
                    support_components["wide"],
                    model,
                    source="support.wide_component_effective",
                    outer_weight=support_weight,
                ),
                "donor": _round31_scalar_gradient_snapshot(
                    support_components["donor"],
                    model,
                    source="support.donor_component_effective",
                    outer_weight=support_weight,
                ),
                "mixed": _round31_scalar_gradient_snapshot(
                    getattr(terms, "support", None),
                    model,
                    source="support.mixed_loss_effective",
                    outer_weight=support_weight,
                ),
            },
            "outer_weight": support_weight,
        }
    else:
        support_result["reason"] = (
            "disabled by round31 factor; zero metrics are not active coverage"
        )

    if isinstance(rank_payload, Mapping) and rank_payload:
        rank_result["pair_metrics"] = _round31_pair_metric_block(
            metrics, "candidate_rank", rank_detail, margin
        )
        rank_components = {
            "K_effective": rank_payload.get("kl_component"),
            "P_effective": rank_payload.get("pair_component"),
        }
        rank_result["scalar_recomposition"] = _round31_component_scalar_recomposition(
            getattr(terms, "rank", None),
            rank_components,
            float(getattr(model, "loss_weights", {}).get("rank", 1.0)),
        )
        rank_result["gradient_recomposition"] = (
            _round31_component_gradient_recomposition(
                getattr(terms, "rank", None),
                rank_components,
                float(getattr(model, "loss_weights", {}).get("rank", 1.0)),
                model,
            )
        )
        rank_result.update(_round31_rank_budget_probe(model, terms, rank_payload))
    else:
        rank_result["reason"] = "rank budget payload unavailable"

    result["support"] = support_result
    result["rank"] = rank_result
    result["gradient_recomposition"] = {
        "support": support_result.get(
            "gradient_recomposition",
            {"available": False, "reason": support_result.get("reason")},
        ),
        "rank": rank_result.get(
            "gradient_recomposition",
            {"available": False, "reason": rank_result.get("reason")},
        ),
    }
    result["scalar_recomposition"] = {
        "support": support_result.get(
            "scalar_recomposition",
            {"available": False, "reason": support_result.get("reason")},
        ),
        "rank": rank_result.get(
            "scalar_recomposition",
            {"available": False, "reason": rank_result.get("reason")},
        ),
        "terms": result["terms_scalar_recomposition"],
    }
    rank_routes = rank_result.get(
        "route_gradients",
        {
            "raw": {},
            "effective": {},
            "mixed": {"available": False, "reason": "rank payload unavailable"},
        },
    )
    support_routes = support_result.get(
        "route_gradients",
        {
            "raw": {},
            "effective": {},
            "mixed": {"available": False, "reason": support_result.get("reason")},
        },
    )
    result["route_gradients"] = {
        "schema": "round31_loss_route_gradients_v1",
        "groups": [
            "s_head",
            "s_projection",
            "e_interaction",
            "t_interaction",
            "carrier",
            "parent_carrier",
            "legacy_support",
            "legacy",
            "other",
        ],
        "rank": rank_routes,
        "support": support_routes,
        "route_flags": {
            "rank_stop_s": bool(getattr(opt, "rank_stop_s", False)),
            "aux_detach_input": bool(getattr(opt, "aux_detach_input", False)),
            "rank_candidate_pair": bool(getattr(opt, "rank_candidate_pair", False)),
        },
        "feature_requires_grad": _round31_feature_requires_grad(
            getattr(outputs, "trifield_output", None)
        ),
        "raw_inputs": _round31_raw_input_probe(
            getattr(outputs, "trifield_output", None)
        ),
        "scope": "same actual forward/loss graph; autograd VJPs only; no optimizer update",
    }
    # A second name helps exporters that use the noun-first schema while the
    # canonical field remains route_gradients for this round.
    result["gradient_routes"] = result["route_gradients"]
    return result


def _round31_vjp_summary(
    gradient: Tensor | None,
    source: str,
) -> dict[str, Any]:
    if not isinstance(gradient, Tensor):
        return {
            "source": source,
            "available": False,
            "connected": False,
            "finite": None,
            "l2": None,
        }
    value = gradient.detach().float()
    return {
        "source": source,
        "available": True,
        "connected": True,
        "finite": bool(torch.isfinite(value).all()),
        "l2": float(value.square().sum().sqrt()),
        "nonzero_fraction": float(value.abs().gt(0).float().mean()),
        "max_abs": float(value.abs().max()) if value.numel() else 0.0,
    }


def probe_round31_support_centering(model: Any, outputs: Any) -> Mapping[str, Any]:
    """Validate same-length S centering, activity, and its differentiable VJP."""
    field = getattr(outputs, "trifield_output", None)
    if field is None:
        return {"available": False, "reason": "missing trifield output"}
    valid = getattr(field, "valid", None)
    support = getattr(field, "support", None)
    raw = getattr(field, "round31_uncentered_support", None)
    detail = getattr(field, "round31_support_centering", None)
    enabled = bool(
        getattr(getattr(model, "options", None), "support_length_center", False)
    )
    if (
        not isinstance(valid, Tensor)
        or not isinstance(support, Tensor)
        or not isinstance(raw, Tensor)
    ):
        return {
            "available": False,
            "enabled": enabled,
            "reason": "missing uncentered or deployed support",
        }
    if tuple(valid.shape) != tuple(raw.shape) or tuple(valid.shape) != tuple(
        support.shape
    ):
        raise RuntimeError("round31 support centering shape mismatch")
    valid_bool = valid.bool()
    tensor_identity = support is raw
    value_identity = bool(torch.equal(support.detach(), raw.detach()))
    if not enabled:
        return {
            "available": True,
            "enabled": False,
            "source_field": "field.round31_uncentered_support",
            "uncentered_count": int(raw[valid_bool].numel()),
            "centered_output_is_same_tensor": tensor_identity,
            "centered_output_value_identity": value_identity,
            "identity_within_fp32_tolerance": value_identity,
            "reason": "length centering disabled; zero/default statistics are not active",
        }
    if not isinstance(detail, Mapping):
        raise RuntimeError("length-centered R28 field did not expose centering detail")
    means, counts = detail.get("means"), detail.get("counts")
    if not isinstance(means, Tensor) or not isinstance(counts, Tensor):
        raise RuntimeError("length-centered R28 field has no means/counts tensors")
    if (
        means.ndim != 2
        or counts.shape != means.shape
        or means.shape[0] != valid.shape[0]
    ):
        raise RuntimeError("length-centered R28 centering tensors have invalid shape")
    b, length, width = valid.shape
    if width != length or means.shape[1] != length:
        raise RuntimeError("length-centered R28 centering dimension mismatch")
    starts = torch.arange(length, device=valid.device)[:, None]
    ends = torch.arange(length, device=valid.device)[None, :]
    group = (ends - starts).clamp_min(0).flatten().expand(b, -1)
    valid_flat = valid_bool.flatten(1)
    raw_flat = raw.float().flatten(1)
    support_flat = support.float().flatten(1)
    means_float = means.float()
    counts_float = counts.float()
    expected = (
        (raw_flat - means_float.gather(1, group))
        .masked_fill(~valid_flat, 0.0)
        .reshape_as(valid)
    )
    error = (expected - support.float())[valid_bool].detach().abs()
    active_groups = counts_float.gt(0)
    centered_values = support_flat.masked_fill(~valid_flat, 0.0)
    group_sum = means_float.new_zeros(means_float.shape).scatter_add(
        1, group, centered_values
    )
    group_mean = group_sum / counts_float.clamp_min(1.0)
    group_mean_abs = group_mean[active_groups].detach().abs()
    candidate_group_count = counts_float.gather(1, group)
    single_mask = valid_flat & candidate_group_count.eq(1.0)
    single_values = support_flat[single_mask].detach().abs()
    singleton_mask = valid_bool & torch.eye(
        length, device=valid.device, dtype=torch.bool
    )[None].expand_as(valid_bool)
    # Prefer a non-singleton, length>=2 group so the mean subtraction and
    # candidate-minus-group VJP are both exercised.  A short/singleton
    # fallback is still reported explicitly when the panel has no such group.
    preferred_groups = active_groups & counts_float.gt(1.0)
    preferred_groups[:, :1] = False
    selection_groups = (
        preferred_groups if bool(preferred_groups.any()) else active_groups
    )
    selection_mode = (
        "length_ge_2_multi_candidate"
        if bool(preferred_groups.any())
        else "available_group_fallback"
    )
    mean_vjp: dict[str, Any]
    centered_vjp: dict[str, Any]
    vjp_selection: dict[str, Any] = {
        "mode": selection_mode,
        "preferred_group_available": bool(preferred_groups.any()),
    }
    if (
        means.requires_grad
        and raw.requires_grad
        and bool(selection_groups.any())
        and torch.is_grad_enabled()
    ):
        first_group = torch.nonzero(selection_groups, as_tuple=False)[0]
        query_index, group_index = int(first_group[0]), int(first_group[1])
        group_candidates = valid_flat[query_index] & group[query_index].eq(group_index)
        candidate_ids = torch.nonzero(group_candidates, as_tuple=False).flatten()
        if not candidate_ids.numel():
            raise RuntimeError("selected centering group has no valid candidate")
        candidate_id = int(candidate_ids[0])
        mean_sample = means[query_index, group_index]
        mean_gradient = torch.autograd.grad(
            mean_sample, raw, retain_graph=True, allow_unused=True
        )[0]
        expected_mean_gradient = torch.zeros_like(raw)
        expected_mean_gradient_flat = expected_mean_gradient.float().flatten(1)
        expected_mean_gradient_flat[query_index, candidate_ids] = 1.0 / float(
            candidate_ids.numel()
        )
        expected_mean_gradient = expected_mean_gradient_flat.reshape_as(raw)
        mean_vjp = _round31_vjp_summary(
            mean_gradient, "one active length-group mean -> raw support"
        )
        if isinstance(mean_gradient, Tensor):
            mean_vjp["reference_max_abs_error"] = float(
                (mean_gradient.float() - expected_mean_gradient.float()).abs().max()
            )
            mean_vjp["reference_within_tolerance"] = bool(
                mean_vjp["reference_max_abs_error"] < 2.0e-6
            )
        candidate_sample = support_flat[query_index, candidate_id]
        candidate_gradient = torch.autograd.grad(
            candidate_sample, raw, retain_graph=True, allow_unused=True
        )[0]
        centered_vjp = _round31_vjp_summary(
            candidate_gradient, "one centered candidate -> raw support"
        )
        expected_centered_gradient = torch.zeros_like(raw)
        expected_centered_flat = expected_centered_gradient.float().flatten(1)
        expected_centered_flat[query_index, candidate_ids] = -1.0 / float(
            candidate_ids.numel()
        )
        expected_centered_flat[query_index, candidate_id] += 1.0
        expected_centered_gradient = expected_centered_flat.reshape_as(raw)
        if isinstance(candidate_gradient, Tensor):
            centered_vjp["reference_max_abs_error"] = float(
                (candidate_gradient.float() - expected_centered_gradient.float())
                .abs()
                .max()
            )
            centered_vjp["reference_within_tolerance"] = bool(
                centered_vjp["reference_max_abs_error"] < 2.0e-6
            )
        vjp_selection.update(
            {
                "query": query_index,
                "group": group_index,
                "inclusive_length": group_index + 1,
                "group_candidate_count": int(candidate_ids.numel()),
                "candidate_flat_index": candidate_id,
            }
        )
    else:
        mean_vjp = {
            "source": "one active length-group mean -> raw support",
            "available": False,
            "connected": False,
            "finite": None,
            "l2": None,
            "reason": "means/raw are detached, no active group, or grad mode is disabled",
        }
        centered_vjp = {
            "source": "one centered candidate -> raw support",
            "available": False,
            "connected": False,
            "finite": None,
            "l2": None,
            "reason": "raw is detached, no selected candidate, or grad mode is disabled",
        }
    return {
        "available": True,
        "enabled": True,
        "source_field": "field.round31_support_centering",
        "formula": "S(c)=raw_span(c)-mean(raw_span | same inclusive clip length, valid query candidates)",
        "means": _probe_stats(means[active_groups]),
        "counts": _probe_stats(counts[active_groups]),
        "active_length_group_count": int(active_groups.sum()),
        "identity_max_abs_error": float(error.max()) if error.numel() else 0.0,
        "identity_within_fp32_tolerance": bool(
            not error.numel() or error.max() < 2.0e-6
        ),
        "centered_output_is_same_tensor": tensor_identity,
        "centered_output_value_identity": value_identity,
        "group_mean_abs_max": float(group_mean_abs.max())
        if group_mean_abs.numel()
        else 0.0,
        "group_mean_zero": bool(
            not group_mean_abs.numel() or group_mean_abs.max() < 2.0e-6
        ),
        "single_candidate_group_count": int(single_values.numel()),
        "single_candidate_group_zero_fraction": (
            float(single_values.le(2.0e-6).float().mean())
            if single_values.numel()
            else None
        ),
        "single_candidate_group_max_abs": (
            float(single_values.max()) if single_values.numel() else 0.0
        ),
        "singleton_zero_fraction": (
            float(support[singleton_mask].detach().eq(0).float().mean())
            if bool(singleton_mask.any())
            else None
        ),
        "padding_zero": bool(support[~valid_bool].detach().eq(0).all()),
        "means_requires_grad": bool(means.requires_grad),
        "means_has_grad_fn": bool(means.grad_fn is not None),
        "means_vjp": mean_vjp,
        "sampled_centered_vjp": centered_vjp,
    }


def probe_round31_score_bounds(model: Any, outputs: Any) -> Mapping[str, Any]:
    field = getattr(outputs, "trifield_output", None)
    if field is None:
        return {"available": False}
    valid = field.valid.bool()
    support_bound = (
        2.0 if bool(getattr(model.options, "support_length_center", False)) else 1.0
    )
    final_bound = 5.0 if support_bound > 1.0 else 4.0
    values = {
        "carrier": (field.carrier, 1.0),
        "evidence": (field.evidence, 1.0),
        "support": (field.support, support_bound),
        "transition_start": (field.transition_start, 1.0),
        "transition_end": (field.transition_end, 1.0),
        "score": (field.score, final_bound),
    }
    result = {
        "available": True,
        "expected_bounds": {k: [-bound, bound] for k, (_, bound) in values.items()},
        "fields": {},
    }
    for name, (value, bound) in values.items():
        selected = value.float()[valid]
        finite = bool(torch.isfinite(selected).all())
        observed = float(selected.detach().abs().max()) if selected.numel() else 0.0
        result["fields"][name] = {
            "finite": finite,
            "observed_abs_max": observed,
            "bound": bound,
            "within_bound": bool(finite and observed <= bound + 2.0e-5),
        }
    result["support_bound"] = support_bound
    result["final_score_bound"] = final_bound
    result["source"] = "actual deployed FP32 field tensors on valid candidates"
    return result


def probe_round31_all_losses(model, outputs, batch, epoch=1):
    modes = [(m, bool(m.training)) for m in model.modules()]
    cpu = torch.random.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    py_rng = random.getstate()
    try:
        import numpy as np

        np_rng = np.random.get_state()
    except (ImportError, ModuleNotFoundError):
        np, np_rng = None, None
    caches = {
        n: getattr(model, n)
        for n in (
            "_last_inputs",
            "_last_field_output",
            "_last_evidence_pairs",
            "_last_wrong_query_field",
        )
        if hasattr(model, n)
    }
    try:
        wrong, pair = (
            model._wrong_query_evidence(batch, outputs)
            if hasattr(model, "_wrong_query_evidence")
            else (None, None)
        )
        opt = getattr(model, "options", None)
        from .counterfactual import build_counterfactual_context

        context = build_counterfactual_context(model, outputs, batch, epoch, pair)
        context["query_relative_e"] = opt.query_relative_e
        terms = compute_loss_terms(
            outputs,
            batch,
            wrong_evidence=wrong,
            evidence_pair_mask=pair,
            counterfactual=context,
            evidence_reduction=opt.evidence_reduction,
            evidence_supervision=opt.evidence_supervision,
            loss_weights=getattr(model, "loss_weights", None),
            rank_mode=str(getattr(opt, "rank_mode", "gt_balanced_kl")),
            rank_form=opt.rank_form,
            rank_competition=opt.rank_competition,
            rank_logit_scale=float(getattr(opt, "rank_logit_scale", 1.0)),
            rank_target_threshold=float(getattr(opt, "rank_target_threshold", 0.5)),
            rank_target_exponent=float(getattr(opt, "rank_target_exponent", 2.0)),
            rank_margin=float(getattr(opt, "rank_margin", 0.1)),
            quality_stratified=bool(getattr(opt, "quality_stratified", False)),
            transition_target_mode=getattr(
                opt, "transition_target_mode", "independent_max"
            ),
            a_local_evidence=opt.a_local_evidence,
            b_edge_support=opt.b_edge_support,
            support_wide_pair=bool(getattr(opt, "support_wide_pair", False)),
            rank_candidate_pair=bool(getattr(opt, "rank_candidate_pair", False)),
            rank_kl_half=bool(getattr(opt, "rank_kl_half", False)),
            rank_stop_s=bool(getattr(opt, "rank_stop_s", False)),
            aux_detach_input=bool(getattr(opt, "aux_detach_input", False)),
            selector=model.selector,
            epoch=epoch,
            negative_selection_mode=model.negative_selection_mode,
            rank_length_reweight_gain=model.rank_length_reweight_gain,
            duration_aux_weight=model.duration_aux_weight,
            length_conditional_gain=model.length_conditional_gain,
            carrier_desaturation_weight=model.carrier_desaturation_weight,
            rank_topk_restrict=model.rank_topk_restrict,
            short_relief_gain=model.short_relief_gain,
            counterfactual_support_mix=model.counterfactual_support_mix,
            rank_protect_gt=getattr(model, "r48_protect_gt", False),
            support_margin_scale=getattr(model, "r48_support_margin_scale", 1.0),
        )
        weights = getattr(model, "loss_weights", {})
        result = {
            "loss_terms": {
                n: getattr(terms, n).detach()
                for n in (
                    "rank",
                    "evidence",
                    "support",
                    "transition",
                    "endpoint",
                    "total",
                )
            },
            "loss_probes": {
                n: {
                    "loss_name": n,
                    "finite": _finite(getattr(terms, n)),
                    "scalar": getattr(terms, n).ndim == 0,
                    "raw": getattr(terms, n).detach(),
                    "weight": float(weights.get(n, 1.0)),
                    "effective": getattr(terms, n).detach()
                    * float(weights.get(n, 1.0)),
                }
                for n in ("rank", "evidence", "support", "transition", "endpoint")
            },
            "loss_metrics": {
                n: (v.detach() if isinstance(v, Tensor) else v)
                for n, v in terms.metrics.items()
            },
            "gradient_probe": _gradient_report(
                model,
                terms,
                getattr(getattr(outputs, "trifield_output", None), "score", None),
            ),
            "legacy_dense_target_reference": probe_target_consistency(
                outputs,
                batch,
                terms,
                getattr(opt, "transition_target_mode", "independent_max"),
            ),
            "epoch": int(epoch),
            "diagnostic_precision": "fp32_forward_required",
        }
        # These three summaries are derived from the same loss call and the
        # same field tensors; no second model forward or optimizer mutation.
        result["candidate_pairs"] = probe_round31_candidate_pairs(model, outputs, terms)
        result["support_centering"] = probe_round31_support_centering(model, outputs)
        result["score_bounds"] = probe_round31_score_bounds(model, outputs)
        result["rank_decomposition"] = result["candidate_pairs"].get(
            "rank", {"enabled": False}
        )
        result["legacy_counterfactual_reference"] = dict(context.get("report", {}))
        result["query_relative_e"] = context.get(
            "query_relative_e_probe", {"enabled": False}
        )
        if opt.query_relative_e and "detail" in result["query_relative_e"]:
            per = result["query_relative_e"]["detail"]["per_query"]
            each = torch.where(
                per["relative"],
                0.5 * (per["clean_hinge"] + per["difference_hinge"]),
                per["clean_hinge"],
            )
            expected = (
                each[per["active"]].mean() if per["active"].any() else each.sum() * 0
            )
            result["query_relative_e"]["per_query_recomposition_error"] = float(
                (expected - terms.evidence.detach()).abs()
            )
            wrong_token = getattr(context.get("wrong_field"), "round31_e_token", None)
            if isinstance(wrong_token, Tensor) and wrong_token.requires_grad:
                g = torch.autograd.grad(
                    terms.evidence, wrong_token, retain_graph=True, allow_unused=True
                )[0]
                result["query_relative_e"]["wrong_token_vjp"] = {
                    "connected": g is not None,
                    "finite": bool(g is None or torch.isfinite(g).all()),
                    "l2": float(g.detach().norm()) if g is not None else 0.0,
                }
        # Probe only: no extra component backward in the training loss path.
        pnames, params = zip(
            *[(n, p) for n, p in model.named_parameters() if p.requires_grad]
        )
        result["legacy_counterfactual_reference"]["component_parameter_gradients"] = {}
        for name, loss in context.get("components", {}).items():
            grads = (
                torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
                if loss.requires_grad
                else [None] * len(params)
            )
            active = [g.detach() for g in grads if g is not None]
            result["legacy_counterfactual_reference"]["component_parameter_gradients"][
                name
            ] = {
                "connected_parameter_count": len(active),
                "finite": all(bool(torch.isfinite(g).all()) for g in active),
                "raw_l2": float(
                    torch.sqrt(
                        sum(
                            (g.float().square().sum() for g in active),
                            loss.detach().new_zeros(()),
                        )
                    )
                ),
            }

        result["actual_loss_sources"] = {
            "evidence": "explicit rated-token ordinal pairs; active-query mean; no legacy fallback"
            if opt.a_local_evidence
            else "legacy counterfactual E candidate mean",
            "support": (
                "wide deployed candidate-S hinge with per-query length-two dispatcher donor fallback; pair/GT/query means"
                if bool(getattr(opt, "support_wide_pair", False))
                else "edge insertion hinge; sides then edges then GT then active-query mean"
            )
            if opt.b_edge_support
            else "legacy support counterfactual",
            "transition": "legacy exterior transition counterfactual",
            "endpoint": "parent endpoint objective",
            "rank": (
                "raw K/P always computed; effective "
                f"alpha={0.5 if bool(getattr(opt, 'rank_kl_half', False)) else 1.0:g}*K + "
                f"beta={0.5 if bool(getattr(opt, 'rank_candidate_pair', False)) else 0.0:g}*P "
                "on pair-eligible KL-active queries; original K fallback and "
                "original KL-active denominator; beta=0 P is reference-only"
            ),
        }
        result["evidence_reduction"] = {
            "scope": result["actual_loss_sources"]["evidence"]
        }
        from .semantic_probe import semantic_probe

        result["semantic_fields"] = semantic_probe(model, outputs, batch)
        from .factor_probes import (
            query_strata,
            span_aggregation_probe,
            aggregation_anchor_probe,
            projection_probe,
        )

        result["query_relative_strata"] = query_strata(result["query_relative_e"])
        field = outputs.trifield_output
        inputs = _parts(batch)[0]
        alive = ~inputs["video_padding_mask"].bool()
        result["span_aggregation"] = span_aggregation_probe(model, field, alive)
        result["aggregation_anchors"] = aggregation_anchor_probe(field, terms.geometry)
        result["field_projection"] = projection_probe(model, field, alive)
        if opt.query_relative_e:
            result["actual_loss_sources"]["evidence"] = (
                "anchored query-relative rated ordinal; full clean ordinal fallback per query"
            )
            result["evidence_reduction"]["scope"] = result["actual_loss_sources"][
                "evidence"
            ]
        result["configured_rank_objective"] = probe_rank_objective(
            model, outputs, batch, terms
        )
        result["rank_probe"] = {
            "mode": str(getattr(opt, "rank_mode", "ordinal_margin")),
            "score_span": terms.metrics.get("rank/score_span", terms.rank.detach() * 0),
            "violation_rate": terms.metrics.get(
                "rank/violating_pair_rate", terms.rank.detach() * 0
            ),
            "margin": terms.metrics.get("rank/margin", terms.rank.detach() * 0),
            "fallback_gt_rate": terms.metrics.get(
                "rank/fallback_gt_rate", terms.rank.detach() * 0
            ),
        }
        result["support_probe"] = {
            k: (v.detach() if isinstance(v, Tensor) else v)
            for k, v in terms.metrics.items()
            if k.startswith("edge_support/" if opt.b_edge_support else "support/")
        }
        result["transition_probe"] = {
            k: (v.detach() if isinstance(v, Tensor) else v)
            for k, v in terms.metrics.items()
            if k.startswith("transition/")
        }
        return result
    finally:
        for m, state in modes:
            m.training = state
        torch.random.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)
        random.setstate(py_rng)
        if np is not None and np_rng is not None:
            np.random.set_state(np_rng)
        for n, value in caches.items():
            setattr(model, n, value)


def _run_round31_probes_impl(model, outputs, batch, epoch=1):
    from .losses import _pad_empty_gt

    batch = _pad_empty_gt(batch)
    inputs, _ = _parts(batch)
    parent, head = getattr(model, "parent_model", None), None
    head = getattr(parent, "shared_head", None)
    minimum = int(getattr(head, "min_span_clips", 2))
    result = {
        "legacy_support_structure": {
            "active": False,
            "replaced_by": "semantic_fields.S",
        }
        if model.options.b_edge_support
        else probe_support_structure(model, outputs, batch, epoch),
        "legacy_evidence_structure": {
            "active": False,
            "replaced_by": "semantic_fields.E",
        }
        if model.options.a_local_evidence
        else probe_evidence_structure(model, outputs, batch, epoch),
        "scalar_residual": probe_scalar_residual(model, outputs),
        "residual_selection": probe_residual_selection(model, outputs, batch),
        "transition_structure": probe_transition_structure(model, outputs, batch),
        "candidate_membership": probe_candidate_membership(
            outputs,
            inputs.get("video_padding_mask") if isinstance(inputs, Mapping) else None,
            minimum,
        ),
        "score_identity": probe_round31_score_identity(model, outputs),
        "three_fields": probe_round31_three_fields(model, outputs, batch),
        "conditioner_ablation": probe_conditioner_ablation(model, inputs, outputs),
        "field_ablation": probe_round31_field_ablation(model, outputs, batch),
        "multigt_duplicate_synthetic": probe_multigt_duplicate(),
        "ordinal_tie_synthetic": probe_ordinal_tie_subgradient(),
        "endpoint_responsibility_synthetic": probe_endpoint_responsibility(),
        "parameter_groups": probe_parameter_groups(model),
        "carrier_strata": probe_carrier_strata(model, outputs, batch),
        "field_saturation": probe_field_saturation(outputs),
        "support_pair": probe_round31_support_pair(model, outputs),
        "short_gt_coverage": probe_short_gt_coverage(outputs, batch),
        "short_gt_matching": probe_short_gt_matching(outputs, batch),
    }
    field = getattr(outputs, "trifield_output", None)
    score = outputs.span_logits.float() if field is None else field.score.float()
    from .losses import candidate_geometry

    valid = candidate_geometry(outputs, batch).valid.bool().flatten(1)
    flat = score.flatten(1)
    baseline = flat.masked_fill(~valid, float("-inf"))
    scaled = (float(model.options.rank_logit_scale) * flat).masked_fill(
        ~valid, float("-inf")
    )
    result["rank_scale_identity"] = {
        "same_forward": True,
        "valid_candidate_order_exact": bool(
            torch.equal(
                torch.argsort(baseline, dim=1, descending=True, stable=True),
                torch.argsort(scaled, dim=1, descending=True, stable=True),
            )
        ),
        "prediction_score_unchanged": True,
        "scale": float(getattr(model.options, "rank_logit_scale", 1.0)),
    }
    loss_probe = probe_round31_all_losses(model, outputs, batch, epoch)
    result["loss_gradients"] = loss_probe
    for _name in (
        "candidate_pairs",
        "support_centering",
        "score_bounds",
        "rank_decomposition",
    ):
        result[_name] = loss_probe.get(_name, {"available": False})
    opt = getattr(model, "options", None)
    result["round31_variant"] = getattr(opt, "variant", "missing")
    result["round31_rank_mode"] = getattr(opt, "rank_mode", "ordinal_margin")
    result["round31_route"] = {
        "rank_stop_s": bool(getattr(opt, "rank_stop_s", False)),
        "aux_detach_input": bool(getattr(opt, "aux_detach_input", False)),
        "rank_candidate_pair": bool(getattr(opt, "rank_candidate_pair", False)),
        "rank_score_source": str(
            getattr(outputs.trifield_output, "round31_rank_score_source", "missing")
        ),
        "deployed_score_requires_grad": bool(
            getattr(outputs.trifield_output, "score", torch.empty(())).requires_grad
        ),
        "rank_score_requires_grad": bool(
            getattr(
                outputs.trifield_output, "round31_rank_score", torch.empty(())
            ).requires_grad
        ),
        "aux_support_requires_grad": bool(
            getattr(outputs.trifield_output, "round31_aux_support_requires_grad", False)
        ),
        "feature_requires_grad": _round31_feature_requires_grad(
            outputs.trifield_output
        ),
        "raw_inputs": _round31_raw_input_probe(outputs.trifield_output),
    }
    result["round31_factors"] = {
        "support_readout": str(opt.support_readout),
        "readout_index": (
            "edge_mean",
            "pooled",
            "unordered_halves",
            "ordered_halves",
        ).index(str(opt.support_readout)),
        "rank_background": "K+.5P" if opt.rank_candidate_pair else "K",
        "support_wide_pair": bool(getattr(opt, "support_wide_pair", False)),
        "support_length_center": bool(getattr(opt, "support_length_center", False)),
        "rank_candidate_pair": bool(getattr(opt, "rank_candidate_pair", False)),
        "rank_kl_half": bool(getattr(opt, "rank_kl_half", False)),
        "alpha": 0.5 if bool(getattr(opt, "rank_kl_half", False)) else 1.0,
        "beta": 0.5 if bool(getattr(opt, "rank_candidate_pair", False)) else 0.0,
        "factor_bits": int(getattr(opt, "factor_bits", 0)),
    }
    result["diagnostic_precision"] = "fp32_forward_required"
    return result


class Round31ProbeSuite(_v1.TrifieldProbeSuite):
    """Fixed 4x64 panel with isolated loader and FP32 diagnostics."""

    def run(self, request):
        bundle = request.bundle
        loader = getattr(bundle, "train_loader", None)
        prepare = getattr(bundle, "prepare_batch", None)
        if loader is None or not callable(prepare):
            raise RuntimeError(
                "Round31ProbeSuite requires train_loader and prepare_batch"
            )
        seed = getattr(request.config, "seed", None)
        if seed is None:
            raise RuntimeError("fixed panel requires config.seed")
        panel_info = self._panel_section(request.fixed_panel, seed)
        probe_loader, expected = self._fixed_loader(loader, request.fixed_panel, seed)
        cpu = torch.random.get_rng_state()
        cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        py_rng = random.getstate()
        try:
            import numpy as np

            np_rng = np.random.get_state()
        except (ImportError, ModuleNotFoundError):
            np, np_rng = None, None
        modes = [(m, bool(m.training)) for m in request.model.modules()]
        random.seed(self.PROBE_RNG_SEED)
        torch.manual_seed(self.PROBE_RNG_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.PROBE_RNG_SEED)
        if np is not None:
            np.random.seed(self.PROBE_RNG_SEED)
        request.model.eval()
        observed = []
        sizes = []
        batches = []
        try:
            for bi, raw in enumerate(probe_loader):
                if bi >= self.PANEL_BATCHES:
                    raise RuntimeError("fixed panel yielded more than four batches")
                prepared = prepare(raw, request.device)
                inputs, _, metadata = self._prepared_parts(prepared)
                qids = self._batch_qids(metadata)
                if len(qids) != self.PANEL_BATCH_SIZE:
                    raise RuntimeError("fixed panel batch is not 64")
                observed.extend(qids)
                sizes.append(len(qids))
                with torch.enable_grad():
                    out = request.model(inputs)
                    probe = run_round31_probes(
                        request.model, out, prepared, int(request.epoch)
                    )
                batches.append(
                    {
                        "batch_index": bi,
                        "batch_size": len(qids),
                        "qids": qids,
                        "probe": probe,
                    }
                )
                del raw, prepared, out, probe
        finally:
            for m, state in modes:
                m.training = state
            torch.random.set_rng_state(cpu)
            if cuda is not None:
                torch.cuda.set_rng_state_all(cuda)
            random.setstate(py_rng)
            if np is not None and np_rng is not None:
                np.random.set_state(np_rng)
        if (
            len(batches) != self.PANEL_BATCHES
            or sizes != [self.PANEL_BATCH_SIZE] * self.PANEL_BATCHES
        ):
            raise RuntimeError("fixed panel is not exactly 4x64")
        order = observed == expected
        observed_hash = self._qid_order_sha256(
            panel_info["raw_qids"] if order else observed
        )
        hash_match = observed_hash == panel_info["qid_order_sha256"]
        if not order or not hash_match:
            raise RuntimeError("fixed panel QID order/hash mismatch")
        return {
            "schema": "trifield_round31_probe_suite_v1",
            "epoch": int(request.epoch),
            "phase": str(request.phase),
            "diagnostic_precision": "fp32",
            "probe_rng_seed": self.PROBE_RNG_SEED,
            "batch_count": len(batches),
            "batch_size": self.PANEL_BATCH_SIZE,
            "batch_sizes": sizes,
            "batch_qids": observed,
            "fixed_panel": self._panel_summary(request.fixed_panel, seed),
            "fixed_panel_full_order_match": order,
            "fixed_panel_qid_order_sha256": observed_hash,
            "fixed_panel_qid_hash_match": hash_match,
            "batches": batches,
            "trainability": request.model.trainability_audit()
            if callable(getattr(request.model, "trainability_audit", None))
            else {},
            "experiment_contract": request.model.experiment_contract()
            if callable(getattr(request.model, "experiment_contract", None))
            else {},
        }


def build_probe_suite(config: Any | None = None, **kwargs: Any) -> Round31ProbeSuite:
    return Round31ProbeSuite(config=config, **kwargs)


__all__ = [
    "Round31ProbeSuite",
    "build_probe_suite",
    "probe_round31_all_losses",
    "probe_round31_candidate_pairs",
    "probe_round31_support_centering",
    "probe_round31_score_bounds",
    "probe_round31_support_pair",
    "probe_field_saturation",
    "probe_short_gt_coverage",
    "probe_target_consistency",
    "run_round31_probes",
]


def probe_short_gt_matching(outputs, batch):
    """Fixed-panel GT matching after label-independent raw/NMS selection."""
    import numpy as np
    from ..evaluator import select_full_grid_candidates, _matching_coverage

    geometry = candidate_geometry(outputs, batch)
    field = getattr(outputs, "trifield_output", None)
    scores = outputs.span_logits.float() if field is None else field.score.float()
    flat = scores.detach().flatten(1)
    valid = geometry.valid.bool().flatten(1)
    windows = (
        torch.stack((geometry.candidate_start, geometry.candidate_end), -1)
        .detach()
        .flatten(1, 2)
        .cpu()
        .numpy()
    )
    ious = geometry.all_iou.detach().flatten(1, 2).cpu().numpy()
    scopes, source = _short_scopes(outputs, batch)
    counts = {
        name: {"gt_count": 0, "raw_matches": 0, "nms_matches": 0} for name in scopes
    }
    for bi in range(flat.shape[0]):
        k = min(30, int(valid[bi].sum()))
        native = (
            flat[bi]
            .masked_fill(~valid[bi], float("-inf"))
            .topk(k)
            .indices.cpu()
            .tolist()
        )
        raw, _, nms = select_full_grid_candidates(
            windows[bi],
            flat[bi].cpu().numpy(),
            valid[bi].cpu().numpy(),
            native_order=native,
        )
        for name, scope in scopes.items():
            gt = torch.nonzero(scope[bi], as_tuple=False).flatten().cpu().numpy()
            counts[name]["gt_count"] += len(gt)
            for label, ids in (("raw", raw), ("nms", nms)):
                matrix = ious[bi][np.ix_(ids, gt)]
                counts[name][label + "_matches"] += _matching_coverage(matrix, 0.7)
    for row in counts.values():
        for label in ("raw", "nms"):
            row[label + "_recall@0.70"] = row[label + "_matches"] / max(
                1, row["gt_count"]
            )
    return {
        "scope_source": source,
        "panel_only": True,
        "nms_threshold": 0.5,
        "max_candidates": 30,
        "selection_uses_gt": False,
        "scopes": counts,
    }


def probe_carrier_strata(model, outputs, batch):
    geometry = candidate_geometry(outputs, batch)
    field = outputs.trifield_output
    valid = geometry.valid.bool()
    quality = geometry.max_iou.float()
    raw = field.carrier_raw.float()
    mode = model.options.carrier_mode
    derivative = (
        1 - torch.tanh(raw).square() if mode == "tanh" else (1 + raw.square()).pow(-1.5)
    )
    masks = {
        "iou_lt03": valid & (quality < 0.3),
        "iou_03_to07": valid & (quality >= 0.3) & (quality < 0.7),
        "iou_ge07": valid & (quality >= 0.7),
    }
    scores = (
        field.score.float().flatten(1).masked_fill(~valid.flatten(1), float("-inf"))
    )
    indices = scores.topk(min(10, scores.shape[1]), dim=1).indices
    top = (
        torch.zeros_like(valid.flatten(1)).scatter(1, indices, True).reshape_as(valid)
        & valid
    )
    masks["top10"] = top
    result = {
        "carrier_mode": mode,
        "top10_uses_gt": False,
        "quality_scope": "max IoU over GT; diagnostics only",
        "strata": {},
    }
    for name, mask in masks.items():
        row = {"count": int(mask.sum())}
        if bool(mask.any()):
            for label, value in [
                ("raw", raw),
                ("activation_derivative", derivative),
                ("final_score", field.score.float()),
            ]:
                x = value[mask].detach()
                row[label] = {
                    f"p{int(q * 100)}": torch.quantile(x, q) for q in (0.1, 0.5, 0.9)
                }
            row["saturation_rate_095"] = (
                field.carrier[mask].abs().ge(0.95).float().mean().detach()
            )
        result["strata"][name] = row
    return result


def probe_transition_structure(model, outputs, batch):
    from .losses import (
        candidate_geometry,
        matched_endpoint_targets,
        _target_spans,
        configured_rank_loss,
        _pad_empty_gt,
    )
    from .endpoint_pair_diagnostic import diagnose_query, pair_semantics

    batch = _pad_empty_gt(batch)
    g = candidate_geometry(outputs, batch)
    field = outputs.trifield_output
    inputs, _ = _parts(batch)
    target = matched_endpoint_targets(outputs, batch, g)

    def conditional(value, axis):
        count = g.valid.sum(axis)
        safe = value.masked_fill(~g.valid, 0)
        mean = safe.sum(axis) / count.clamp_min(1)
        var = (value - mean.unsqueeze(axis)).square().masked_fill(~g.valid, 0).sum(
            axis
        ) / count.clamp_min(1)
        chosen = var[count > 1]
        return {
            "mean": chosen.mean().detach() if chosen.numel() else None,
            "count": int(chosen.numel()),
        }

    state = outputs.extension_output.state
    rs, re = model.selector.transition_readouts(
        state, field.valid, inputs.get("video_padding_mask"), context=False
    )
    no_context = model.selector.compose_score(
        field,
        {
            "transition_start": torch.tanh(rs.float()),
            "transition_end": torch.tanh(re.float()),
        },
    )
    valid = field.valid

    def changed(score):
        a = field.score.masked_fill(~valid, -torch.inf).flatten(1).argmax(1)
        b = score.masked_fill(~valid, -torch.inf).flatten(1).argmax(1)
        return int(((a != b) & valid.flatten(1).any(1)).sum())

    spans, mask = _target_spans(outputs, batch)
    # Differentiate the ACTUAL rank formula and weight, not a hand-coded direction.
    rank, _ = configured_rank_loss(
        field.score.float() * model.options.rank_logit_scale,
        g,
        mask,
        model.options.rank_target_threshold,
        model.options.rank_target_exponent,
        rank_form=model.options.rank_form,
        rank_competition=model.options.rank_competition,
    )
    grad = torch.autograd.grad(
        rank * model.loss_weights["rank"],
        field.score,
        retain_graph=True,
        allow_unused=False,
    )[0]
    rows = []
    for b in range(field.score.shape[0]):
        if not bool(mask[b].any()):
            rows.append(
                {
                    "qid": str(b),
                    "gt_count": 0,
                    "skipped": "no valid GT; no geometric label or rank-pair supervision",
                }
            )
            continue
        q = {
            "qid": str(b),
            "candidate_start": g.candidate_start[b].flatten().detach().cpu().tolist(),
            "candidate_end": g.candidate_end[b].flatten().detach().cpu().tolist(),
            "valid": field.valid[b].flatten().cpu().tolist(),
            "score": field.score[b].flatten().detach().cpu().tolist(),
            "transition_start": field.transition_start[b]
            .flatten()
            .detach()
            .cpu()
            .tolist(),
            "transition_end": field.transition_end[b].flatten().detach().cpu().tolist(),
            "gt_spans": spans[b].cpu().tolist(),
            "gt_mask": mask[b].cpu().tolist(),
            "one_step": float(g.one_step[b]),
        }
        q["without_transition_score"] = (
            model.selector.compose_score(
                field,
                {
                    "transition_start": torch.zeros_like(field.transition_start),
                    "transition_end": torch.zeros_like(field.transition_end),
                },
            )[b]
            .flatten()
            .detach()
            .cpu()
            .tolist()
        )
        row = diagnose_query(q)
        row["transition_ablation_scope"] = (
            "Ts/Te zeroed with scalar calibration and pairwise interactions recomputed; flip includes their mediated score effect"
        )
        gts = [x for x, keep in zip(q["gt_spans"], q["gt_mask"]) if keep]
        cross = [
            i
            for i, keep in enumerate(q["valid"])
            if keep
            and pair_semantics(
                (q["candidate_start"][i], q["candidate_end"][i]), gts, q["one_step"]
            ).get("strong_cross", False)
        ]
        flatgrad = grad[b].flatten()
        directions = []
        for gtrow in row["ranking"]["0.7"]:
            if gtrow["reachable"]:
                good = gtrow["best_good_index"]
                directions.extend(
                    float((flatgrad[i] - flatgrad[good]).detach()) for i in cross
                )
        row["actual_rank_bad_minus_good_gradient"] = {
            "pair_count": len(directions),
            "positive_count": sum(x > 0 for x in directions),
            "mean": sum(directions) / len(directions) if directions else None,
            "meaning": "positive means a score-space descent step lowers bad-minus-good; pairs can share candidates/GT",
        }
        rows.append(row)
    return {
        "context_mode": model.options.transition_context_mode,
        "start_prediction_conditional_variance": conditional(field.transition_start, 2),
        "end_prediction_conditional_variance": conditional(field.transition_end, 1),
        "start_matched_target_conditional_variance": conditional(target.start, 2),
        "end_matched_target_conditional_variance": conditional(target.end, 1),
        "remove_context_top1_changes": changed(no_context),
        "remove_context_valid_max_delta": (no_context - field.score)[valid]
        .abs()
        .max()
        .detach()
        if valid.any()
        else None,
        "remove_start_top1_changes": changed(
            model.selector.compose_score(
                field, {"transition_start": torch.zeros_like(field.transition_start)}
            )
        ),
        "remove_end_top1_changes": changed(
            model.selector.compose_score(
                field, {"transition_end": torch.zeros_like(field.transition_end)}
            )
        ),
        "queries": rows,
        "scope": "fixed training panel, not official validation MR",
    }


def probe_round31_score_identity(model, outputs):
    result = {}
    field = outputs.trifield_output
    recomposed = model.selector.compose_score(field)
    err = (recomposed - field.score)[field.valid].abs()
    support_on = model.options.support_deployment == "in_score"
    result["equation"] = (
        "4/(1+2a+b)*(C+a*(E+d*S)+b*(Ts+Te)/2), d=1 for in_score, 0 for zero"
    )
    result["support_deployment"] = model.options.support_deployment
    result["support_input"] = model.options.support_input
    result["region_gain"] = model.options.region_gain
    result["transition_gain"] = model.options.transition_gain
    a, b = model.options.region_gain, model.options.transition_gain
    normalizer = 4.0 / (1.0 + 2 * a + b)
    weights = {
        "carrier": normalizer,
        "evidence": normalizer * a,
        "support": normalizer * a * float(support_on),
        "transition_start": normalizer * b / 2,
        "transition_end": normalizer * b / 2,
    }
    result["composition_coefficients"] = weights
    result["coefficient_sum"] = sum(weights.values())
    result["weighted_contributions"] = {}
    for name, weight in weights.items():
        value = (getattr(field, name).detach() * weight)[field.valid]
        result["weighted_contributions"][name] = {
            "abs_mean": float(value.abs().mean()) if value.numel() else None,
            "std": float(value.std(unbiased=False)) if value.numel() else None,
        }
    support_bound = (
        2.0 if bool(getattr(model.options, "support_length_center", False)) else 1.0
    )
    score_bound = 5.0 if support_bound > 1.0 else 4.0
    observed_score_abs = (
        field.score[field.valid].float().abs().max()
        if field.valid.any()
        else field.score.new_zeros(())
    )
    result["score_identity"] = bool(not err.numel() or err.max() < 2.0e-6)
    # The legacy four-unit check remains diagnostic; centered S has a five-unit bound.
    result["score_bound_4"] = bool(
        not field.valid.any() or observed_score_abs <= 4.000002
    )
    result["score_bound"] = bool(
        not field.valid.any() or observed_score_abs <= score_bound + 2.0e-5
    )
    result["expected_score_bound"] = score_bound
    result["expected_support_bound"] = support_bound
    result["equation_max_abs_error"] = err.max().detach() if err.numel() else 0.0
    result["equation_within_fp32_tolerance"] = bool(
        not err.numel() or err.max() < 2.0e-6
    )
    return result


def probe_round31_field_ablation(model, outputs, batch):
    from .losses import _pad_empty_gt

    g = candidate_geometry(outputs, _pad_empty_gt(batch))
    field = outputs.trifield_output
    valid = g.valid
    active = valid.flatten(1).any(1)

    def top(score):
        return score.flatten(1).masked_fill(~valid.flatten(1), -torch.inf).argmax(1)

    original = top(field.score)
    row = torch.arange(len(original), device=original.device)
    baseline = (
        g.max_iou.flatten(1)[row, original][active].mean() if active.any() else None
    )
    result = {
        "interaction_recomputed": True,
        "scope": "fixed training panel coarse geometry; not MR",
        "intact_top1_iou_mean": baseline.detach() if baseline is not None else None,
    }
    names = {
        "carrier": ("carrier",),
        "evidence": ("evidence",),
        "support": ("support",),
        "transition": ("transition_start", "transition_end"),
        "transition_start": ("transition_start",),
        "transition_end": ("transition_end",),
    }
    for name, parts in names.items():
        score = (
            field.round31_score_without_support
            if name == "support"
            else model.selector.compose_score(
                field, {p: torch.zeros_like(getattr(field, p)) for p in parts}
            )
        )
        chosen = top(score)
        row = torch.arange(len(chosen), device=chosen.device)
        changed_iou = (
            g.max_iou.flatten(1)[row, chosen][active].mean() if active.any() else None
        )
        result[name] = {
            "delta_top1_iou_mean": (changed_iou - baseline).detach()
            if baseline is not None
            else None,
            "top1_changed_query_count": int(((chosen != original) & active).sum()),
            "top1_iou_mean": g.max_iou.flatten(1)[row, chosen][active].mean().detach()
            if active.any()
            else None,
            "score_abs_delta_mean": (score - field.score)[valid].abs().mean().detach()
            if valid.any()
            else None,
        }
    return result


def probe_round31_three_fields(model, outputs, batch):
    field = getattr(outputs, "trifield_output", None)
    state = getattr(getattr(outputs, "extension_output", None), "state", None)
    result = {
        "three_fields_present": field is not None and state is not None,
        "shuffle_interaction_recomputed": True,
    }
    if field is None or state is None:
        return result
    valid = field.valid
    role = getattr(state, "role_updates", None)
    result["role_updates_shape"] = (
        tuple(role.shape) if isinstance(role, Tensor) else None
    )
    result["role_updates_finite"] = bool(
        isinstance(role, Tensor) and torch.isfinite(role).all()
    )
    start_input = getattr(state, "transition_start_input", None)
    end_input = getattr(state, "transition_end_input", None)
    result["transition_inputs_distinct"] = bool(
        isinstance(start_input, Tensor)
        and isinstance(end_input, Tensor)
        and not torch.equal(start_input.detach(), end_input.detach())
    )
    for name in ("evidence", "support", "transition_start", "transition_end"):
        value = getattr(field, name)
        chosen = value[valid]
        _bound = (
            2.0
            if (
                name == "support"
                and bool(getattr(model.options, "support_length_center", False))
            )
            else 1.0
        )
        result[name] = {
            "finite": bool(torch.isfinite(chosen).all()),
            "bound": _bound,
            "range_ok": bool(
                not chosen.numel() or chosen.abs().max() <= _bound + 2.0e-5
            ),
            "nonzero_fraction": chosen.abs().gt(1.0e-8).float().mean().detach()
            if chosen.numel()
            else None,
        }
        # One deterministic spatial permutation per field; same actual recomposer.
        shuffled = value.clone()
        for b in range(len(value)):
            flat = shuffled[b].flatten()
            ids = torch.nonzero(valid[b].flatten(), as_tuple=False).flatten()
            if ids.numel():
                flat[ids] = value[b].flatten()[ids.flip(0)]
        score = model.selector.compose_score(field, {name: shuffled})
        a = field.score.flatten(1).masked_fill(~valid.flatten(1), -torch.inf).argmax(1)
        z = score.flatten(1).masked_fill(~valid.flatten(1), -torch.inf).argmax(1)
        result[name]["shuffle_top1_changes"] = int(
            ((a != z) & valid.flatten(1).any(1)).sum()
        )
    result["ablation"] = probe_round31_field_ablation(model, outputs, batch)
    return result


def probe_scalar_residual(model, outputs):
    field = outputs.trifield_output
    linear, pair = model.selector.residuals(field)
    valid = field.valid
    result = {
        "linear_enabled": model.selector.linear_calibration,
        "pair_enabled": model.selector.pairwise_interactions,
        "inputs": "E,S,Ts,Te only; carrier excluded",
        "parameter_initialization": "zeros; no RNG draw",
    }

    def summarize(value):
        value = value[valid].float().detach()
        return {
            "count": int(value.numel()),
            "finite": bool(torch.isfinite(value).all()),
            "mean": value.mean() if value.numel() else None,
            "std": value.std(unbiased=False) if value.numel() else None,
            "quantiles": torch.quantile(
                value, value.new_tensor([0.0, 0.1, 0.5, 0.9, 1.0])
            )
            .detach()
            .cpu()
            .tolist()
            if value.numel()
            else None,
            "quantile_levels": [0.0, 0.1, 0.5, 0.9, 1.0],
        }

    values = [
        getattr(field, name).float()
        for name in ("evidence", "support", "transition_start", "transition_end")
    ]
    x = torch.stack(values, -1)
    products = torch.stack(
        [
            values[i] * values[j]
            for i, j in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
        ],
        -1,
    )
    result["coefficient_order"] = {
        "linear": ["E", "S", "Ts", "Te"],
        "pair": ["E*S", "E*Ts", "E*Te", "S*Ts", "S*Te", "Ts*Te"],
    }
    result["linear_coefficients"] = (
        model.selector.linear_weights.detach().cpu().tolist()
        if model.selector.linear_calibration
        else None
    )
    result["pair_coefficients"] = (
        model.selector.pair_weights.detach().cpu().tolist()
        if model.selector.pairwise_interactions
        else None
    )
    result["u_linear"] = (
        summarize((x * model.selector.linear_weights).sum(-1) / 4)
        if model.selector.linear_calibration
        else None
    )
    result["u_pair"] = (
        summarize((products * model.selector.pair_weights).sum(-1) / 6)
        if model.selector.pairwise_interactions
        else None
    )
    result["residual_distributions"] = {
        name: summarize(value)
        for name, value in [
            ("linear", linear),
            ("pair", pair),
            ("total", linear + pair),
        ]
    }
    for name, value in (("linear", linear), ("pair", pair), ("total", linear + pair)):
        selected = value[valid]
        result[name] = {
            "max_abs": selected.abs().max().detach() if selected.numel() else None,
            "mean_abs": selected.abs().mean().detach() if selected.numel() else None,
            "saturation_095": selected.abs()
            .ge(0.475 if name != "total" else 0.95)
            .float()
            .mean()
            .detach()
            if selected.numel()
            else None,
        }
    return result


def probe_residual_selection(model, outputs, batch):
    import numpy as np
    from .losses import _pad_empty_gt
    from ..evaluator import select_full_grid_candidates, _matching_coverage

    g = candidate_geometry(outputs, _pad_empty_gt(batch))
    field = outputs.trifield_output
    scores = {
        "deployed": field.score,
        "without_linear": model.selector.compose_score(field, disable_linear=True),
        "without_pair": model.selector.compose_score(field, disable_pair=True),
        "without_both": model.selector.compose_score(
            field, disable_linear=True, disable_pair=True
        ),
    }
    result = {
        "geometry_scope": "same-forward coarse training panel grid/discretized GT; recomputed raw/NMS; not official validation MR",
        "selection_score_protocol": "FP32 softmax(masked logits), native probability topk, probability scores passed to selector; identical eval protocol",
        "modes": {},
    }
    reference = {}
    for mode, score in scores.items():
        buckets = {
            f"{p}{k}": {
                "gt_count": 0,
                "independent_gt_covered_07": 0,
                "one_to_one_matches_07": 0,
                "queries_selection_changed": 0,
            }
            for p in ("raw", "nms")
            for k in (10, 30)
        }
        for b in range(len(score)):
            valid = field.valid[b].flatten()
            flat = score[b].flatten()
            probabilities = (
                torch.softmax(flat.masked_fill(~valid, -torch.inf).float(), dim=0)
                if valid.any()
                else torch.zeros_like(flat, dtype=torch.float32)
            )
            native = (
                probabilities.topk(min(30, int(valid.sum())))
                .indices.detach()
                .cpu()
                .numpy()
            )
            windows = (
                torch.stack((g.candidate_start[b], g.candidate_end[b]), -1)
                .flatten(0, 1)
                .cpu()
                .numpy()
            )
            raw, _, nms = select_full_grid_candidates(
                windows,
                probabilities.detach().cpu().numpy(),
                valid.cpu().numpy(),
                native_order=native,
            )
            _, targets = _parts(batch)
            mask = targets.get("gt_span_mask")
            gtids = (
                torch.nonzero(mask[b].bool(), as_tuple=False).flatten().cpu().numpy()
                if mask is not None
                else np.array([], dtype=int)
            )
            matrix = g.all_iou[b].flatten(0, 1).cpu().numpy()
            for p, ids in (("raw", raw), ("nms", nms)):
                for k in (10, 30):
                    chosen = ids[:k]
                    key = f"{p}{k}"
                    values = matrix[np.ix_(chosen, gtids)]
                    bucket = buckets[key]
                    bucket["gt_count"] += len(gtids)
                    bucket["independent_gt_covered_07"] += (
                        int((values.max(0) >= 0.7).sum()) if len(chosen) else 0
                    )
                    bucket["one_to_one_matches_07"] += _matching_coverage(values, 0.7)
                    if mode == "deployed":
                        reference[(b, key)] = chosen
                    else:
                        bucket["queries_selection_changed"] += int(
                            not np.array_equal(reference[(b, key)], chosen)
                        )
        result["modes"][mode] = buckets
    return result


def probe_rank_objective(model, outputs, batch, terms):
    """Actual configured rank gradient and same-forward positive-domain diagnostics."""
    import numpy as np
    from .losses import rank_partition, _gt_mask_for_batch
    from eventfieldnet.evaluator import select_full_grid_candidates

    f = outputs.trifield_output
    g = terms.geometry
    mask = _gt_mask_for_batch(batch, g)
    qg, valid, P, D, active, _, ld, lp = rank_partition(
        2 * f.score, g, mask, model.options.rank_competition
    )
    z = (2 * f.score).flatten(1)
    global_p = torch.softmax(z.masked_fill(~valid, -torch.inf), 1)
    global_p = torch.nan_to_num(global_p)
    bagmass = (global_p[..., None] * P).sum(1)
    # Concentration inside each positive bag, distinct from mass allocated to the bag.
    maxinside = global_p[..., None].expand_as(qg).masked_fill(~P, 0).amax(
        1
    ) / bagmass.clamp_min(1e-30)
    entg = -(qg * qg.clamp_min(1e-30).log()).sum(1)
    mixture = qg.sum(-1) / mask.sum(1).clamp_min(1).float()[:, None]
    entmix = -(mixture * mixture.clamp_min(1e-30).log()).sum(1)

    def summary(x):
        v = x.detach()[active].float()
        return {
            "count": int(v.numel()),
            "mean": float(v.mean()) if v.numel() else None,
            "quantiles": torch.quantile(v, v.new_tensor([0, 0.1, 0.5, 0.9, 1]))
            .cpu()
            .tolist()
            if v.numel()
            else [],
        }

    rank_score = getattr(f, "round31_rank_score", f.score)
    if not isinstance(rank_score, Tensor) or tuple(rank_score.shape) != tuple(
        f.score.shape
    ):
        raise RuntimeError("round31 rank diagnostic requires field.round31_rank_score")
    grad = torch.autograd.grad(
        model.loss_weights["rank"] * terms.rank,
        rank_score,
        retain_graph=True,
        allow_unused=True,
    )[0]
    if grad is None:
        grad = torch.zeros_like(f.score)
    result = {
        "rank_form": model.options.rank_form,
        "rank_competition": model.options.rank_competition,
        "gradient_source": "actual configured weighted rank autograd through field.round31_rank_score",
        "rank_score_source": getattr(f, "round31_rank_score_source", "missing"),
        "rank_stop_s": bool(getattr(model.options, "rank_stop_s", False)),
        "aux_detach_input": bool(getattr(model.options, "aux_detach_input", False)),
        "geometry_scope": "training panel candidate_geometry; not official MR",
        "global_positive_mass": summary(bagmass),
        "conditional_positive_mass": summary((lp - ld).exp()),
        "positive_top1_concentration": summary(maxinside),
        "excluded_candidates": summary((valid[..., None] & ~D).sum(1).float()),
        "per_GT_entropy": summary(entg),
        "mixture_entropy_query_mean": float(entmix[active.any(1)].mean())
        if active.any()
        else None,
        "entropy_warning": "dense conditional uses mean per-GT entropy; control uses mixture entropy; raw losses not directly comparable",
        "actual_rank_score_grad_norm": float(grad.detach().norm()),
        "coverage": {},
        "wrong_direction": {},
    }
    buckets = {}
    fs = f.score.detach().flatten(1)
    gg = grad.detach().flatten(1).cpu()
    fs_cpu = fs.cpu()
    iou_cpu = g.all_iou.detach().cpu()
    maxiou_cpu = g.max_iou.detach().cpu()
    mask_cpu = mask.detach().cpu()
    for b in range(len(fs)):
        vv = valid[b]
        probs = (
            torch.softmax(fs[b].masked_fill(~vv, -torch.inf).float(), 0)
            if vv.any()
            else torch.zeros_like(fs[b])
        )
        native = probs.topk(min(30, int(vv.sum()))).indices.cpu().numpy()
        windows = (
            torch.stack((g.candidate_start[b], g.candidate_end[b]), -1)
            .flatten(0, 1)
            .cpu()
            .numpy()
        )
        raw, order, nms = select_full_grid_candidates(
            windows, probs.cpu().numpy(), vv.cpu().numpy(), native_order=native
        )
        iou = iou_cpu[b].flatten(0, 1)
        ids = torch.nonzero(mask_cpu[b], as_tuple=False).flatten().tolist()
        maxiou = maxiou_cpu[b].flatten()
        for policy, selected in (("raw", raw), ("nms", nms)):
            from eventfieldnet.evaluator import _matching_coverage

            for k in (10, 30):
                matrix = iou[selected[:k]][:, ids].cpu().numpy()
                key = policy + str(k)
                bb = result["coverage"].setdefault(
                    key, {"gt_count": 0, "covered_075": 0, "one_to_one_075": 0}
                )
                bb["gt_count"] += len(ids)
                bb["covered_075"] += (
                    int((matrix.max(0) >= 0.75).sum()) if len(matrix) else 0
                )
                bb["one_to_one_075"] += _matching_coverage(matrix, 0.75)
            for gt in ids:
                good = [int(i) for i in order if iou[i, gt] >= 0.75]
                if not good:
                    continue
                best = max(good, key=lambda i: float(fs_cpu[b, i]))
                covered = any(iou[i, gt] >= 0.75 for i in selected[:10])
                if covered:
                    continue
                for bad in selected[:10]:
                    bad = int(bad)
                    if iou[bad, gt] >= 0.5 or fs_cpu[b, bad] <= fs_cpu[b, best]:
                        continue
                    category = (
                        "globally_bad"
                        if maxiou[bad] < 0.5
                        else (
                            "good_for_other_GT"
                            if maxiou[bad] >= 0.75
                            else "partial_for_other_GT"
                        )
                    )
                    key = policy + "10/" + category
                    bucket = buckets.setdefault(key, [])
                    bucket.append((b, gt, float(gg[b, bad] - gg[b, best])))
    for key, rows in buckets.items():
        result["wrong_direction"][key] = {
            "pairs": len(rows),
            "unique_query_count": len({r[0] for r in rows}),
            "unique_GT_count": len({r[:2] for r in rows}),
            "positive": sum(r[2] > 1e-8 for r in rows),
            "negative": sum(r[2] < -1e-8 for r in rows),
            "near_zero": sum(abs(r[2]) <= 1e-8 for r in rows),
            "d_score_quantiles": np.quantile(
                [r[2] for r in rows], [0, 0.1, 0.5, 0.9, 1]
            ).tolist(),
        }
    return result


def probe_support_structure(model, outputs, batch, epoch=1):
    field = outputs.trifield_output
    valid = field.valid
    active = valid.flatten(1).any(1)
    top = lambda score: (
        score.flatten(1).masked_fill(~valid.flatten(1), -torch.inf).argmax(1)
    )
    original = top(field.score)

    def stats(value):
        value = value[valid].detach().float()
        return {
            "finite": bool(torch.isfinite(value).all()),
            "mean_abs": value.abs().mean() if value.numel() else None,
            "max_abs": value.abs().max() if value.numel() else None,
        }

    result = {
        "same_forward": True,
        "scope": "fixed training panel; not official MR",
        "neighbor_valid_fraction": field.support_neighbor_count[valid]
        .gt(0)
        .float()
        .mean()
        .detach(),
        "variance": stats(field.support_variance_mean),
        "variance_nonnegative": bool(field.support_variance_mean[valid].ge(0).all()),
    }
    params = [
        (name, param)
        for name, param in model.selector.named_parameters()
        if name.startswith(("context_branch.", "dispersion_branch."))
    ]
    gradients = {}
    if params and torch.is_grad_enabled() and field.score.requires_grad:
        loss = model.compute_loss(outputs, batch, None, epoch).loss
        grads = torch.autograd.grad(
            loss, [p for _, p in params], allow_unused=True, retain_graph=True
        )
        gradients = {
            name: {
                "connected": grad is not None,
                "finite": bool(torch.isfinite(grad).all())
                if grad is not None
                else None,
                "l2": grad.float().norm().detach() if grad is not None else None,
            }
            for (name, _), grad in zip(params, grads)
        }
    result["actual_total_loss_gradients"] = gradients
    for name in ("context", "dispersion"):
        delta = getattr(field, "support_" + name + "_delta")
        score = model.selector.score_without_support_branch(field, name)
        result[name] = {
            "enabled": bool(getattr(model.options, "support_" + name)),
            "raw_delta": stats(delta),
            "top1_changed_query_count": int(((top(score) != original) & active).sum()),
            "score_change": stats(score - field.score),
        }
    return result


def probe_evidence_structure(model, outputs, batch, epoch=1):
    field = outputs.trifield_output
    valid = field.valid
    active = valid.flatten(1).any(1)
    top = lambda score: (
        score.flatten(1).masked_fill(~valid.flatten(1), -torch.inf).argmax(1)
    )

    def stats(x):
        x = x[valid].detach().float()
        return {
            "finite": bool(torch.isfinite(x).all()),
            "mean_abs": x.abs().mean() if x.numel() else None,
            "max_abs": x.abs().max() if x.numel() else None,
        }

    params = [
        (n, p)
        for n, p in model.selector.named_parameters()
        if n.startswith(("evidence_pool_branch.", "evidence_context_branch."))
    ]
    gradients = {}
    if params and torch.is_grad_enabled() and field.score.requires_grad:
        loss = model.compute_loss(outputs, batch, None, epoch).loss
        gs = torch.autograd.grad(
            loss, [p for _, p in params], allow_unused=True, retain_graph=True
        )
        gradients = {
            n: {
                "connected": g is not None,
                "finite": bool(torch.isfinite(g).all()) if g is not None else None,
                "l2": g.float().norm().detach() if g is not None else None,
            }
            for (n, _), g in zip(params, gs)
        }
    result = {
        "same_forward": True,
        "recentered_ablation": True,
        "scope": "fixed training panel; not official MR",
        "actual_total_loss_gradients": gradients,
        "neighbor_valid_fraction": field.evidence_neighbor_count[valid]
        .gt(0)
        .float()
        .mean()
        .detach()
        if bool(valid.any())
        else None,
    }
    for name in ("pool", "context"):
        score = model.selector.score_without_evidence_branch(field, name)
        result[name] = {
            "enabled": bool(getattr(model.options, "evidence_" + name)),
            "raw_delta": stats(getattr(field, "evidence_" + name + "_delta")),
            "score_change": stats(score - field.score),
            "top1_changed_query_count": int(
                ((top(score) != top(field.score)) & active).sum()
            ),
        }
    return result


def probe_evidence_reduction(model, outputs, batch, terms, wrong, pair):
    """No new forward: compare reductions on the actual wrong-query E graph."""
    from .losses import evidence_objective_values

    field = outputs.trifield_output
    positive = terms.geometry.valid & terms.geometry.max_iou.ge(0.70)
    paired = (
        torch.zeros(positive.shape[0], dtype=torch.bool, device=positive.device)
        if wrong is None
        else (
            torch.ones(positive.shape[0], dtype=torch.bool, device=positive.device)
            if pair is None
            else pair.bool()
        )
    )
    mask = positive & paired[:, None, None]
    candidate, query, counts, means, hinge, bg = evidence_objective_values(
        outputs,
        batch,
        terms.geometry,
        field.evidence if wrong is None else wrong,
        mask,
        model.options.evidence_supervision,
    )
    active = counts > 0
    cw = counts.float() / counts.sum().clamp_min(1)
    qw = active.float() / active.sum().clamp_min(1)

    def values(t):
        return t.detach().cpu().tolist()

    def gradnorm(loss):
        named = [("correct_E", field.evidence)] + (
            [] if wrong is None else [("wrong_E", wrong)]
        )
        named = [(n, t) for n, t in named if t.requires_grad]
        grads = (
            torch.autograd.grad(
                loss, [t for _, t in named], retain_graph=True, allow_unused=True
            )
            if named and loss.requires_grad
            else [None] * len(named)
        )
        return {
            n: {
                "connected": g is not None,
                "raw_l2": 0.0 if g is None else float(g.detach().norm()),
                "weighted_l2": 0.0
                if g is None
                else float(g.detach().norm()) * float(model.loss_weights["evidence"]),
            }
            for (n, t), g in zip(named, grads)
        }

    mode = model.options.evidence_reduction
    expected = query if mode == "query_mean" else candidate
    matched = bg["matched"]
    length = field.evidence.shape[-1]
    flat_ids = (
        torch.arange(length * length, device=mask.device)
        .reshape(1, length, length)
        .expand_as(mask)
    )
    same_length = (flat_ids % length - flat_ids // length) == (
        bg["ids"] % length - bg["ids"] // length
    )
    from .losses import _target_spans, evidence_background_indices

    spans, gtmask = _target_spans(outputs, batch)
    _, bg_valid = evidence_background_indices(terms.geometry, spans, gtmask)
    protected = (
        bg_valid.flatten(1)
        .gather(1, bg["ids"].flatten(1).clamp_min(0))
        .reshape_as(mask)
    )
    wrong_hinge_loss = (
        bg["query_hinge"][mask].mean() if mask.any() else field.raw_evidence.sum() * 0
    )
    bg_hinge_loss = (
        bg["background_hinge"][matched].mean()
        if matched.any()
        else field.raw_evidence.sum() * 0
    )
    offset = 0.375
    shifted_query = torch.relu(
        0.20
        - (
            (field.evidence + offset)
            - ((field.evidence if wrong is None else wrong) + offset)
        )
    )
    original_bg = (
        field.evidence.flatten(1)
        .gather(1, bg["ids"].flatten(1).clamp_min(0))
        .reshape_as(mask)
    )
    shifted_bg = torch.relu(0.20 - ((field.evidence + offset) - (original_bg + offset)))
    offset_error = (
        (shifted_query - bg["query_hinge"]).abs()[mask].max()
        if mask.any()
        else field.evidence.new_zeros(())
    )
    if matched.any():
        offset_error = torch.maximum(
            offset_error, (shifted_bg - bg["background_hinge"]).abs()[matched].max()
        )
    return {
        "scope": "same actual forward; output E tensor gradients, not parameter gradients; parameter gradients in five-loss probe",
        "selected_mode": mode,
        "supervision": model.options.evidence_supervision,
        "background_same_length": bool(same_length[matched].all()),
        "background_all_gt_protection": bool(protected[matched].all()),
        "background_match_count": int(matched.sum()),
        "background_coverage": float(matched.sum() / mask.sum().clamp_min(1)),
        "common_E_offset_max_error": float(offset_error.detach()),
        "common_E_offset_scope": "bounded E values correct/wrong/background shifted together; no extra forward",
        "wrong_query_margin_met_fraction": float(
            (bg["query_hinge"][mask] == 0).float().mean()
        )
        if mask.any()
        else None,
        "background_margin_met_fraction": float(
            (bg["background_hinge"][matched] == 0).float().mean()
        )
        if matched.any()
        else None,
        "component_gradient_scope": "standalone unweighted hinge means on respective eligible sets; actual mixed E gradient in reduction gradients",
        "wrong_query_hinge_gradients": gradnorm(wrong_hinge_loss),
        "background_hinge_gradients": gradnorm(bg_hinge_loss),
        "background_rule": "ascending legal same-width flat IDs; positive flat ID modulo count; expanded all-GT protection 1clip; zero overlap including touching allowed",
        "background_matched_counts": values(bg["matched"].flatten(1).sum(1)),
        "background_fallback_counts": values((mask & ~bg["matched"]).flatten(1).sum(1)),
        "wrong_query_hinge_mean": float(bg["query_hinge"][mask].detach().mean())
        if mask.any()
        else None,
        "background_hinge_matched_mean": float(
            bg["background_hinge"][bg["matched"]].detach().mean()
        )
        if bg["matched"].any()
        else None,
        "positive_counts_before_pairing": values(positive.flatten(1).sum(1)),
        "eligible_counts": values(counts),
        "paired_queries": values(paired),
        "valid_pair_rate": float(paired.float().mean()),
        "supervised_query_count": int(active.sum()),
        "candidate_mean_weights": values(cw),
        "query_mean_weights": values(qw),
        "candidate_effective_queries": float(1 / cw.square().sum())
        if active.any()
        else 0.0,
        "query_effective_queries": float(1 / qw.square().sum())
        if active.any()
        else 0.0,
        "query_hinge": [
            v if ok else None for v, ok in zip(values(means), values(active))
        ],
        "query_violation_fraction": [
            v if ok else None
            for v, ok in zip(
                values(((hinge > 0) & mask).flatten(1).sum(1) / counts.clamp_min(1)),
                values(active),
            )
        ],
        "candidate_mean_loss": float(candidate.detach()),
        "query_mean_loss": float(query.detach()),
        "selected_loss_exact": bool(
            torch.equal(expected.detach(), terms.evidence.detach())
        ),
        "candidate_mean_gradients": gradnorm(candidate),
        "query_mean_gradients": gradnorm(query),
    }


def run_round31_probes(model, outputs, batch, epoch=1):
    sentinel = object()
    previous = getattr(model, "_round31_probe_context_cache", sentinel)
    model._round31_probe_context_cache = {}
    old_wrong = getattr(model, "_last_wrong_query_field", None)
    try:
        return _run_round31_probes_impl(model, outputs, batch, epoch)
    finally:
        model._last_wrong_query_field = old_wrong
        if previous is sentinel:
            delattr(model, "_round31_probe_context_cache")
        else:
            model._round31_probe_context_cache = previous
