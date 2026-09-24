"""Executable, low-cost probes for the base's losses and structures."""

from __future__ import annotations

from contextlib import nullcontext
import copy
import hashlib
import json
import random

from dataclasses import fields as dataclass_fields
from dataclasses import is_dataclass, replace
from types import SimpleNamespace
from typing import Any, Mapping

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Subset

from .losses import (
    FiveLossTerms,
    candidate_geometry,
    compute_five_loss_terms,
    official_threshold_grades,
    official_ordinal_violation_loss,
    quality_targets,
)


def _finite(value: Tensor) -> bool:
    return bool(torch.isfinite(value.detach().float()).all())


def _selected(value: Tensor, valid: Tensor) -> Tensor:
    return value.detach().float()[valid.bool()]


def _batch_parts(batch: Any) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    if isinstance(batch, Mapping):
        return batch.get("inputs", {}), batch.get("targets", {})
    return getattr(batch, "inputs", {}), getattr(batch, "targets", {})


def probe_candidate_membership(
    outputs: Any,
    video_padding_mask: Tensor | None = None,
    min_span_clips: int = 2,
) -> Mapping[str, Any]:
    """Check that the selector leaves the parent's coordinate mask untouched."""
    valid = outputs.span_valid_mask.bool()
    b, length, _ = valid.shape
    if video_padding_mask is None:
        video_padding_mask = torch.zeros(
            (b, length), dtype=torch.bool, device=valid.device
        )
    pad = video_padding_mask.bool().to(valid.device)
    index = torch.arange(length, device=valid.device)
    expected = (
        (index[None, :] >= index[:, None] + int(min_span_clips) - 1)
        & (~pad[:, :, None])
        & (~pad[:, None, :])
    )
    mismatch = valid ^ expected
    return {
        "candidate_membership_exact": bool(not mismatch.any()),
        "candidate_membership_mismatch_count": int(mismatch.sum().item()),
        "candidate_count_per_row_mean": valid.float().sum((-1, -2)).mean().detach(),
        "min_span_clips": int(min_span_clips),
    }


def probe_score_identity(outputs: Any) -> Mapping[str, Any]:
    field = getattr(outputs, "trifield_output", None)
    if field is None:
        return {"score_identity": False, "reason": "missing trifield_output"}
    valid = field.valid.bool()
    output_score = outputs.span_logits.float()[valid]
    field_score = field.score.float()[valid]
    carrier = field.carrier.float()[valid]
    final = field.score.float()[valid]
    return {
        "score_identity": bool(torch.equal(output_score, field_score)),
        "score_max_abs_error": (output_score - field_score).abs().max().detach()
        if output_score.numel()
        else 0.0,
        "carrier_bounded": bool(
            carrier.numel() == 0 or ((carrier >= -1.0).all() and (carrier <= 1.0).all())
        ),
        "carrier_min": carrier.min().detach() if carrier.numel() else 0.0,
        "carrier_max": carrier.max().detach() if carrier.numel() else 0.0,
        "final_min": final.min().detach() if final.numel() else 0.0,
        "final_max": final.max().detach() if final.numel() else 0.0,
        "all_field_finite": all(
            _finite(value)
            for value in (
                field.raw_evidence,
                field.raw_support,
                field.raw_transition_start,
                field.raw_transition_end,
                field.score,
            )
        ),
    }


def _score_with_ablation(field: Any, name: str) -> Tensor:
    score = field.carrier.float()
    if name != "evidence":
        score = score + field.evidence.float()
    if name != "support":
        score = score + field.support.float()
    if name != "transition":
        score = score + 0.5 * (
            field.transition_start.float() + field.transition_end.float()
        )
    if name == "carrier":
        score = (
            field.evidence.float()
            + field.support.float()
            + 0.5 * (field.transition_start.float() + field.transition_end.float())
        )
    return score.masked_fill(~field.valid.bool(), 0.0)


def _top1_index(score: Tensor, valid: Tensor) -> Tensor:
    return (
        score.float().flatten(1).masked_fill(~valid.bool().flatten(1), -1.0e4).argmax(1)
    )


def _top1_metrics(score: Tensor, iou: Tensor, valid: Tensor) -> Mapping[str, Tensor]:
    flat_score, flat_iou, _flat_valid = (
        score.float().flatten(1),
        iou.float().flatten(1),
        valid.bool().flatten(1),
    )
    top = _top1_index(score, valid)
    row = torch.arange(flat_score.shape[0], device=score.device)
    top_iou = flat_iou[row, top]
    result: dict[str, Tensor] = {
        "top1_iou": top_iou.mean(),
        "top1_grade": (top_iou[:, None] >= score.new_tensor((0.50, 0.70, 0.95)))
        .float()
        .mean(0),
    }
    for threshold in (0.50, 0.70, 0.95):
        result[f"hit@{threshold:.2f}"] = (top_iou >= threshold).float().mean()
    return result


def probe_field_ablation(outputs: Any, batch: Any) -> Mapping[str, Any]:
    """Measure actual final-score decision changes after zeroing each field."""
    field = getattr(outputs, "trifield_output", None)
    if field is None:
        return {"field_ablation_available": False}
    geometry = candidate_geometry(outputs, batch)
    valid = field.valid.bool()
    baseline = _top1_metrics(field.score, geometry.max_iou, valid)
    result: dict[str, Any] = {"field_ablation_available": True}
    for name in ("carrier", "evidence", "support", "transition"):
        score = _score_with_ablation(field, name)
        metrics = _top1_metrics(score, geometry.max_iou, valid)
        result[f"{name}/proxy_top1_iou"] = metrics["top1_iou"].detach()
        result[f"{name}/proxy_top1_iou_delta"] = (
            metrics["top1_iou"] - baseline["top1_iou"]
        ).detach()
        for threshold in (0.50, 0.70, 0.95):
            key = f"hit@{threshold:.2f}"
            result[f"{name}/proxy_top1_threshold_recall@{threshold:.2f}"] = metrics[
                key
            ].detach()
            result[f"{name}/proxy_top1_threshold_recall_delta@{threshold:.2f}"] = (
                metrics[key] - baseline[key]
            ).detach()
        result[f"{name}/decision_change_rate"] = (
            metrics["top1_iou"].new_tensor(0.0)
            if not valid.any()
            else (_top1_index(score, valid) != _top1_index(field.score, valid))
            .float()
            .mean()
        )
    result["baseline/proxy_top1_iou"] = baseline["top1_iou"].detach()
    for threshold in (0.50, 0.70, 0.95):
        result[f"baseline/proxy_top1_threshold_recall@{threshold:.2f}"] = baseline[
            f"hit@{threshold:.2f}"
        ].detach()
    return result


def _zero_conditioner_state(state: Any) -> Any:
    """Return a detached zero-field state for the independent conditioner probe."""
    input_semantic = getattr(state, "input_semantic", None)
    values: dict[str, Any] = {}
    if is_dataclass(state):
        for item in dataclass_fields(state):
            name = item.name
            value = getattr(state, name)
            if isinstance(value, Tensor):
                if name == "conditioned_semantic" and isinstance(
                    input_semantic, Tensor
                ):
                    values[name] = input_semantic
                elif name == "input_semantic":
                    values[name] = value
                else:
                    values[name] = torch.zeros_like(value)
        try:
            return replace(state, **values)
        except (TypeError, ValueError):
            return state
    if hasattr(state, "__dict__"):
        clone = SimpleNamespace(**vars(state))
        for name, value in vars(clone).items():
            if isinstance(value, Tensor):
                if name == "conditioned_semantic" and isinstance(
                    input_semantic, Tensor
                ):
                    setattr(clone, name, input_semantic)
                elif name != "input_semantic":
                    setattr(clone, name, torch.zeros_like(value))
        return clone
    return state


def _find_input_conditioner(model: Any) -> tuple[Any | None, str | None]:
    root = getattr(model, "parent_model", model)
    direct = getattr(root, "input_conditioner", None)
    if direct is not None and hasattr(direct, "register_forward_hook"):
        return direct, "input_conditioner"
    named_modules = getattr(root, "named_modules", None)
    if callable(named_modules):
        for name, module in named_modules():
            if name.rsplit(".", 1)[-1] == "input_conditioner" and hasattr(
                module, "register_forward_hook"
            ):
                return module, name
    return None, None


def probe_conditioner_ablation(
    model: Any,
    inputs: Mapping[str, Any] | None,
    baseline_outputs: Any,
) -> Mapping[str, Any]:
    """Run an independent zero-conditioner forward and compare decisions.

    The hook replaces only the conditioner output for this diagnostic forward:
    the video tensor is returned unchanged, all conditioner field tensors are
    zeroed, and the candidate grid comes from the parent's own output.  The
    model mode, RNG streams, wrapper caches, and input mapping are restored.
    """
    conditioner, conditioner_name = _find_input_conditioner(model)
    if conditioner is None:
        return {
            "conditioner_probe_available": False,
            "conditioner_probe_reason": "parent input_conditioner module not found",
        }
    if not isinstance(inputs, Mapping):
        return {
            "conditioner_probe_available": False,
            "conditioner_probe_reason": "prepared input mapping unavailable",
        }
    baseline_field = getattr(baseline_outputs, "trifield_output", None)
    baseline_valid = getattr(baseline_outputs, "span_valid_mask", None)
    if baseline_field is None or not isinstance(baseline_valid, Tensor):
        return {
            "conditioner_probe_available": False,
            "conditioner_probe_reason": "baseline trifield output or mask unavailable",
        }
    baseline_score = baseline_field.score.float()
    training_modes = [(module, bool(module.training)) for module in model.modules()]
    cpu_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    cache_names = ("_last_inputs", "_last_field_output", "_last_evidence_pairs")
    saved_cache = {
        name: getattr(model, name) for name in cache_names if hasattr(model, name)
    }

    def hook(_module: Any, module_inputs: tuple[Any, ...], module_output: Any) -> Any:
        if not isinstance(module_output, (tuple, list)) or len(module_output) < 2:
            return module_output
        raw_video = (
            module_inputs[0]
            if module_inputs and isinstance(module_inputs[0], Tensor)
            else module_output[0]
        )
        zero_state = _zero_conditioner_state(module_output[1])
        values = list(module_output)
        values[0], values[1] = raw_video, zero_state
        return tuple(values) if isinstance(module_output, tuple) else values

    handle = conditioner.register_forward_hook(hook)
    try:
        model.eval()
        with torch.no_grad():
            ablated_outputs = model(inputs)
        ablated_field = getattr(ablated_outputs, "trifield_output", None)
        ablated_valid = getattr(ablated_outputs, "span_valid_mask", None)
        result: dict[str, Any] = {
            "conditioner_probe_available": ablated_field is not None
            and isinstance(ablated_valid, Tensor),
            "conditioner_module": conditioner_name,
            "conditioner_ablation_independent_forward": True,
            "conditioner_ablation_field_output_shuffle_is_separate": True,
        }
        if ablated_field is None or not isinstance(ablated_valid, Tensor):
            result["conditioner_probe_reason"] = (
                "ablated forward did not expose trifield output or mask"
            )
            return result
        same_shape = baseline_valid.shape == ablated_valid.shape
        result["conditioner_ablation_candidate_membership_unchanged"] = bool(
            same_shape and torch.equal(baseline_valid.bool(), ablated_valid.bool())
        )
        result["conditioner_ablation_coordinate_contract"] = (
            "parent span grid and mask are read from each forward"
        )
        if not same_shape:
            result["conditioner_probe_reason"] = "ablated output mask shape changed"
            return result
        valid = baseline_valid.bool()
        score = ablated_field.score.float()
        result["conditioner_ablation_score_abs_delta"] = (
            (score[valid] - baseline_score[valid]).abs().mean().detach()
            if valid.any()
            else score.new_tensor(0.0)
        )
        result["conditioner_ablation_score_finite"] = _finite(score)
        result["conditioner_ablation_top1_change_rate"] = (
            (_top1_index(score, valid) != _top1_index(baseline_score, valid))
            .float()
            .mean()
            if valid.any()
            else score.new_tensor(0.0)
        )
        baseline_state = getattr(
            getattr(baseline_outputs, "extension_output", None), "state", None
        )
        ablated_state = getattr(
            getattr(ablated_outputs, "extension_output", None), "state", None
        )
        baseline_role = getattr(baseline_state, "role_updates", None)
        ablated_role = getattr(ablated_state, "role_updates", None)
        if isinstance(baseline_role, Tensor) and isinstance(ablated_role, Tensor):
            result["conditioner_ablation_baseline_role_norm"] = (
                baseline_role.float().norm(dim=-1).mean().detach()
            )
            result["conditioner_ablation_zeroed_role_norm"] = (
                ablated_role.float().norm(dim=-1).mean().detach()
            )
            result["conditioner_ablation_role_shape_unchanged"] = (
                baseline_role.shape == ablated_role.shape
            )
        return result
    finally:
        handle.remove()
        for module, training in training_modes:
            module.training = training
        torch.random.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        for name, value in saved_cache.items():
            setattr(model, name, value)


def _shuffle_valid(value: Tensor, valid: Tensor) -> Tensor:
    result = value.detach().clone()
    for row in range(value.shape[0]):
        flat = result[row].reshape(-1)
        mask = valid[row].reshape(-1)
        if int(mask.sum()) > 1:
            flat[mask] = torch.roll(flat[mask], shifts=1, dims=0)
    return result


def probe_three_fields(outputs: Any, batch: Any | None = None) -> Mapping[str, Any]:
    """Check E/S/T state, ranges, and decision dependence under ablations/shuffles."""
    field = getattr(outputs, "trifield_output", None)
    state = getattr(getattr(outputs, "extension_output", None), "state", None)
    result: dict[str, Any] = {
        "three_fields_present": field is not None and state is not None
    }
    if field is None or state is None:
        return result
    valid = field.valid.bool()
    for name in ("evidence", "support", "transition_start", "transition_end"):
        value = getattr(field, name)
        selected = _selected(value, valid)
        result[f"{name}_finite"] = _finite(value)
        result[f"{name}_range_ok"] = bool(
            selected.numel() == 0
            or ((selected >= -1.0).all() and (selected <= 1.0).all())
        )
        result[f"{name}_nonzero_fraction"] = (
            selected.abs().gt(1.0e-8).float().mean().detach()
            if selected.numel()
            else 0.0
        )
    role = getattr(state, "role_updates", None)
    result["role_updates_shape"] = (
        tuple(role.shape) if isinstance(role, Tensor) else None
    )
    result["role_updates_finite"] = bool(isinstance(role, Tensor) and _finite(role))
    start_input = getattr(state, "transition_start_input", None)
    end_input = getattr(state, "transition_end_input", None)
    result["transition_inputs_distinct"] = bool(
        isinstance(start_input, Tensor)
        and isinstance(end_input, Tensor)
        and not torch.equal(start_input.detach(), end_input.detach())
    )
    field_values = {
        "evidence": field.evidence.float(),
        "support": field.support.float(),
        "transition": 0.5
        * (field.transition_start.float() + field.transition_end.float()),
    }
    base = field.carrier.float() + sum(field_values.values())
    for name, value in field_values.items():
        shuffled = _shuffle_valid(value, valid)
        shuffled_score = (base - value + shuffled).masked_fill(~valid, 0.0)
        result[f"{name}/shuffle_score_abs_delta"] = (
            (shuffled_score[valid] - base[valid]).abs().mean().detach()
            if valid.any()
            else 0.0
        )
        result[f"{name}/shuffle_top1_change_rate"] = (
            (
                shuffled_score.flatten(1)
                .masked_fill(~valid.flatten(1), -1.0e4)
                .argmax(1)
                != base.flatten(1).masked_fill(~valid.flatten(1), -1.0e4).argmax(1)
            )
            .float()
            .mean()
            if valid.any()
            else 0.0
        )
        result[f"{name}/zero_score_abs_delta"] = (
            value[valid].abs().mean().detach() if valid.any() else 0.0
        )
    if batch is not None:
        result.update(probe_field_ablation(outputs, batch))
    return result


def _term_gradient_report(
    model: Any, terms: FiveLossTerms, score: Tensor | None = None
) -> Mapping[str, Any]:
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    module_buckets = {
        "input_conditioner": [
            i
            for i, name in enumerate(names)
            if name.startswith("parent_model.input_conditioner.")
        ],
        "shared_head": [
            i
            for i, name in enumerate(names)
            if name.startswith("parent_model.shared_head.")
        ],
        "parent_start_head": [
            i
            for i, name in enumerate(names)
            if name.startswith("parent_model.stem.start")
            or name.startswith("parent_model.shared_head.start")
        ],
        "parent_end_head": [
            i
            for i, name in enumerate(names)
            if name.startswith("parent_model.stem.end")
            or name.startswith("parent_model.shared_head.end")
        ],
        "encoder": [
            i
            for i, name in enumerate(names)
            if name.startswith("parent_model.stem.temporal_encoder.")
        ],
        "selector_evidence": [
            i for i, name in enumerate(names) if name.startswith("selector.evidence.")
        ],
        "selector_support": [
            i for i, name in enumerate(names) if name.startswith("selector.support")
        ],
        "selector_transition_start": [
            i
            for i, name in enumerate(names)
            if name.startswith("selector.transition_start.")
        ],
        "selector_transition_end": [
            i
            for i, name in enumerate(names)
            if name.startswith("selector.transition_end.")
        ],
    }
    term_names = ("rank", "evidence", "support", "transition", "endpoint", "total")
    weights = {
        "rank": 1.0,
        "evidence": 0.1,
        "support": 0.1,
        "transition": 0.1,
        "endpoint": 0.1,
        "total": 1.0,
    }
    gradients: dict[str, tuple[Tensor | None, ...]] = {}
    for name in term_names:
        value = getattr(terms, name)
        if value.requires_grad:
            gradients[name] = torch.autograd.grad(
                value, params, retain_graph=True, allow_unused=True
            )
        else:
            gradients[name] = tuple(None for _ in params)

    def dot(
        left: tuple[Tensor | None, ...], right: tuple[Tensor | None, ...]
    ) -> Tensor:
        values = [
            a.float().mul(b.float()).sum()
            for a, b in zip(left, right)
            if a is not None and b is not None
        ]
        return (
            torch.stack(values).sum() if values else terms.total.detach().float() * 0.0
        )

    def norm(value: tuple[Tensor | None, ...]) -> Tensor:
        values = [a.float().square().sum() for a in value if a is not None]
        return (
            torch.stack(values).sum().sqrt()
            if values
            else terms.total.detach().float() * 0.0
        )

    def bucket_norm(gradient: tuple[Tensor | None, ...], indices: list[int]) -> Tensor:
        return norm(tuple(gradient[i] for i in indices))

    result: dict[str, Any] = {}
    norms = {name: norm(value) for name, value in gradients.items()}
    for name in term_names:
        grad = gradients[name]
        raw_modules = {
            bucket: bucket_norm(grad, indices).detach()
            for bucket, indices in module_buckets.items()
        }
        weight = weights[name]
        result[name] = {
            "weight": weight,
            "raw": {
                "global_norm": norms[name].detach(),
                "finite": all(g is None or _finite(g) for g in grad),
                "nonzero_parameter_tensors": sum(
                    int(g is not None and bool(g.detach().abs().max().gt(0.0)))
                    for g in grad
                ),
                "module_norms": raw_modules,
            },
            "effective": {
                "global_norm": (norms[name] * weight).detach(),
                "module_norms": {
                    bucket: (value * weight).detach()
                    for bucket, value in raw_modules.items()
                },
            },
        }

    cosine: dict[str, Any] = {}
    for i, left_name in enumerate(term_names):
        for right_name in term_names[i + 1 :]:
            denominator = norms[left_name] * norms[right_name]
            defined = bool(denominator.detach().gt(1.0e-12))
            cosine[f"{left_name}__{right_name}"] = {
                "defined": defined,
                "value": (
                    dot(gradients[left_name], gradients[right_name]) / denominator
                ).detach()
                if defined
                else None,
            }
    result["pairwise_cosine"] = cosine

    weighted_names = ("rank", "evidence", "support", "transition", "endpoint")
    recomposed: list[Tensor] = []
    for index in range(len(params)):
        pieces = [gradients["rank"][index]]
        for name in weighted_names[1:]:
            pieces.append(
                None if gradients[name][index] is None else gradients[name][index] * 0.1
            )
        value = None
        for piece in pieces:
            if piece is not None:
                value = piece if value is None else value + piece
        recomposed.append(torch.zeros_like(params[index]) if value is None else value)
    total_grad = tuple(
        torch.zeros_like(parameter) if gradient is None else gradient
        for parameter, gradient in zip(params, gradients["total"])
    )
    diff = tuple(left - right for left, right in zip(total_grad, recomposed))
    result["weighted_recomposition"] = {
        "raw_norm": norm(tuple(recomposed)).detach(),
        "total_norm": norm(total_grad).detach(),
        "absolute_error": norm(diff).detach(),
        "relative_error": (norm(diff) / norm(total_grad).clamp_min(1.0e-12)).detach(),
    }

    geometry = terms.geometry
    flat_valid = geometry.valid.flatten(1)
    flat_iou = geometry.max_iou.flatten(1)
    score_value = (
        score
        if isinstance(score, Tensor)
        else getattr(getattr(model, "_last_field_output", None), "score", None)
    )
    flat_score = (
        score_value.float().flatten(1)
        if isinstance(score_value, Tensor) and score_value.ndim > 1
        else score_value.float()
        if isinstance(score_value, Tensor)
        else None
    )
    eligible_rows: list[int] = []
    if flat_score is not None:
        grades = official_threshold_grades(geometry.max_iou, geometry.valid).flatten(1)
        for row in range(flat_score.shape[0]):
            row_valid = flat_valid[row]
            if (
                bool(row_valid.any())
                and int(grades[row][row_valid].unique().numel()) > 1
            ):
                eligible_rows.append(row)
        if eligible_rows:
            row_index = torch.tensor(
                eligible_rows, device=flat_score.device, dtype=torch.long
            )
            good_idx = (
                flat_iou[row_index].masked_fill(~flat_valid[row_index], -1.0).argmax(1)
            )
            bad_idx = (
                flat_iou[row_index].masked_fill(~flat_valid[row_index], 2.0).argmin(1)
            )
            gap = (
                flat_score[row_index, good_idx] - flat_score[row_index, bad_idx]
            ).mean()
            gap_grad = (
                torch.autograd.grad(gap, params, retain_graph=True, allow_unused=True)
                if gap.requires_grad
                else tuple(None for _ in params)
            )
            raw_dot = {
                name: dot(gradients[name], gap_grad).detach() for name in term_names
            }
            result["good_bad_gap"] = gap.detach()
            result["good_bad_gap_eligible_row_count"] = len(eligible_rows)
            result["good_bad_gap_raw_dot"] = raw_dot
            result["good_bad_gap_downhill_dot"] = {
                name: (-value).detach() for name, value in raw_dot.items()
            }
            result["good_bad_gap_grad_norm"] = norm(gap_grad).detach()
        else:
            result["good_bad_gap_skipped"] = "no row with two distinct official grades"
    return result


def probe_all_losses(
    model: Any, outputs: Any, batch: Any, epoch: int = 1
) -> Mapping[str, Any]:
    """Run every loss and read gradients without changing ``.grad`` or mode."""
    training_modes = [(module, bool(module.training)) for module in model.modules()]
    cpu_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        wrong, pair = (
            model._wrong_query_evidence(batch, outputs)
            if hasattr(model, "_wrong_query_evidence")
            else (None, None)
        )
        terms = compute_five_loss_terms(
            outputs, batch, wrong_evidence=wrong, evidence_pair_mask=pair
        )
        result: dict[str, Any] = {
            "loss_terms": {
                name: getattr(terms, name).detach()
                for name in (
                    "rank",
                    "evidence",
                    "support",
                    "transition",
                    "endpoint",
                    "total",
                )
            }
        }
        result["loss_probes"] = {
            "rank": probe_rank_loss(terms),
            "evidence": probe_evidence_loss(terms),
            "support": probe_support_loss(terms),
            "transition": probe_transition_loss(terms),
            "endpoint": probe_endpoint_loss(terms),
        }
        result["gradient_probe"] = _term_gradient_report(
            model,
            terms,
            getattr(getattr(outputs, "trifield_output", None), "score", None),
        )
        result["epoch"] = int(epoch)
        return result
    finally:
        for module, training in training_modes:
            module.training = training
        torch.random.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)


def probe_parameter_groups(model: Any) -> Mapping[str, Any]:
    groups = list(model.parameter_groups())
    expected = {
        id(parameter): name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    seen: dict[int, str] = {}
    duplicate: list[str] = []
    for group in groups:
        for parameter in group.params:
            if id(parameter) in seen:
                duplicate.append(f"{seen[id(parameter)]}->{group.name}")
            seen[id(parameter)] = group.name
    missing = [name for ident, name in expected.items() if ident not in seen]
    return {
        "group_count": len(groups),
        "parameter_group_exact_cover": not missing and not duplicate,
        "missing": missing,
        "duplicates": duplicate,
        "group_names": [group.name for group in groups],
    }


def probe_real_top10_gt_coverage(
    outputs: Any, batch: Any, top_k: int = 10
) -> Mapping[str, Any]:
    """Measure actual top-k coverage of every real GT and duplicate assignments."""
    geometry = candidate_geometry(outputs, batch)
    field = getattr(outputs, "trifield_output", None)
    score = outputs.span_logits.float() if field is None else field.score.float()
    valid = outputs.span_valid_mask.bool()
    flat_score, flat_valid = score.flatten(1), valid.flatten(1)
    k = min(int(top_k), flat_score.shape[1])
    top_index = flat_score.masked_fill(~flat_valid, -1.0e4).topk(k, dim=1).indices
    all_iou = geometry.all_iou.flatten(1, 2)
    top_iou = all_iou.gather(1, top_index[..., None].expand(-1, -1, all_iou.shape[-1]))
    _, targets = _batch_parts(batch)
    gt_mask = targets.get("gt_span_mask")
    if gt_mask is None:
        gt_mask = torch.zeros(
            (score.shape[0], all_iou.shape[-1]), dtype=torch.bool, device=score.device
        )
    gt_mask = gt_mask.bool().to(score.device)
    gt_coverage = top_iou.max(1).values
    selected_gt = geometry.best_gt_index.flatten(1).gather(1, top_index)
    duplicate_excess = []
    for row in range(score.shape[0]):
        # Count duplicate assignments only among selected candidates with a
        # real positive IoU; invalid GT slots were excluded by best assignment.
        positive = top_iou[row].max(-1).values.gt(0.0)
        ids = selected_gt[row][positive]
        duplicate_excess.append(max(0, int(ids.numel()) - int(ids.unique().numel())))
    return {
        "top_k": k,
        "gt_count": gt_mask.sum(1).float().mean().detach(),
        "coverage@0.50": (gt_coverage[gt_mask] >= 0.50).float().mean().detach()
        if gt_mask.any()
        else 0.0,
        "coverage@0.70": (gt_coverage[gt_mask] >= 0.70).float().mean().detach()
        if gt_mask.any()
        else 0.0,
        "mean_best_iou_in_topk": gt_coverage[gt_mask].mean().detach()
        if gt_mask.any()
        else 0.0,
        "duplicate_assignment_excess_mean": score.new_tensor(
            duplicate_excess, dtype=torch.float32
        )
        .mean()
        .detach()
        if duplicate_excess
        else 0.0,
        "candidate_supervision": "all valid candidates for rank; top-k only for this diagnostic",
    }


def probe_multigt_duplicate() -> Mapping[str, Any]:
    """Synthetic sanity check for stable first-index assignment."""
    valid = torch.ones((1, 4, 4), dtype=torch.bool)
    outputs = SimpleNamespace(span_logits=torch.zeros((1, 4, 4)), span_valid_mask=valid)
    batch = SimpleNamespace(
        inputs={"video_padding_mask": torch.zeros((1, 4), dtype=torch.bool)},
        targets={
            "gt_spans": torch.tensor([[[0.25, 0.75], [0.25, 0.75]]]),
            "gt_span_mask": torch.tensor([[True, True]]),
        },
    )
    geometry = candidate_geometry(outputs, batch)
    return {
        "duplicate_iou_equal": torch.equal(
            geometry.all_iou[..., 0], geometry.all_iou[..., 1]
        ),
        "stable_first_tie": bool(
            (geometry.best_gt_index[geometry.max_iou.gt(0.0)] == 0).all()
        ),
        "unique_matching": False,
        "documented_limitation": "max-IoU duplicate GTs share supervision; no unique matching",
    }


def probe_ordinal_tie_subgradient() -> Mapping[str, Any]:
    score = torch.zeros((1, 2), requires_grad=True)
    iou = torch.tensor([[0.9, 0.5]])
    valid = torch.ones_like(score, dtype=torch.bool)
    loss, _ = official_ordinal_violation_loss(score, iou, valid)
    loss.backward()
    tie_gradient = score.grad.detach().clone()
    score2 = torch.tensor([[1.0, 0.0]], requires_grad=True)
    loss2, _ = official_ordinal_violation_loss(score2, iou, valid)
    loss2.backward()
    return {
        "tie_loss_zero": bool(loss.detach().eq(0.0)),
        "tie_separating_subgradient_nonzero": bool(tie_gradient.abs().sum().gt(0.0)),
        "non_tie_loss_zero_when_ordered": bool(loss2.detach().eq(0.0)),
        "tie_gradient": tie_gradient,
    }


def probe_endpoint_responsibility() -> Mapping[str, Any]:
    """Synthetic single-end perturbation: changing one endpoint affects its T label."""
    valid = torch.ones((1, 3, 3), dtype=torch.bool)
    outputs = SimpleNamespace(span_logits=torch.zeros((1, 3, 3)), span_valid_mask=valid)
    batch = SimpleNamespace(
        inputs={"video_padding_mask": torch.zeros((1, 3), dtype=torch.bool)},
        targets={
            "gt_spans": torch.tensor([[[0.0, 1.0]]]),
            "gt_span_mask": torch.tensor([[True]]),
        },
    )
    geometry = candidate_geometry(outputs, batch)
    support, ts, te = quality_targets(geometry)
    start_denominator = geometry.selected_width.clamp_min(
        geometry.one_step[:, None, None]
    )
    changed_start_target = (
        2.0
        * (
            1.0
            - (geometry.candidate_start + 0.25 - geometry.selected_start).abs()
            / start_denominator
        ).clamp(0.0, 1.0)
        - 1.0
    )
    unchanged_end_target = (
        2.0
        * (
            1.0
            - (geometry.candidate_end - geometry.selected_end).abs() / start_denominator
        ).clamp(0.0, 1.0)
        - 1.0
    )
    return {
        "support_present": _finite(support),
        "start_label_responds": bool((ts - changed_start_target).abs().sum().gt(0.0)),
        "end_label_unchanged_by_start_perturbation": bool(
            (te - unchanged_end_target).abs().sum().eq(0.0)
        ),
    }


def _single_loss_probe(terms: FiveLossTerms, name: str) -> Mapping[str, Any]:
    value = getattr(terms, name)
    return {
        "loss_name": name,
        "finite": _finite(value),
        "scalar": value.ndim == 0,
        "value": value.detach(),
    }


def probe_rank_loss(terms: FiveLossTerms) -> Mapping[str, Any]:
    return _single_loss_probe(terms, "rank")


def probe_evidence_loss(terms: FiveLossTerms) -> Mapping[str, Any]:
    return _single_loss_probe(terms, "evidence")


def probe_support_loss(terms: FiveLossTerms) -> Mapping[str, Any]:
    return _single_loss_probe(terms, "support")


def probe_transition_loss(terms: FiveLossTerms) -> Mapping[str, Any]:
    return _single_loss_probe(terms, "transition")


def probe_endpoint_loss(terms: FiveLossTerms) -> Mapping[str, Any]:
    return _single_loss_probe(terms, "endpoint")


def run_base_probes(
    model: Any, outputs: Any, batch: Any, epoch: int = 1
) -> Mapping[str, Any]:
    """Collect the cheap structural, intervention, coverage, and gradient probes."""
    inputs, _ = _batch_parts(batch)
    pad = inputs.get("video_padding_mask") if isinstance(inputs, Mapping) else None
    parent = getattr(model, "parent_model", None)
    head = getattr(parent, "shared_head", None)
    min_span = int(getattr(head, "min_span_clips", 2))
    result: dict[str, Any] = {
        "candidate_membership": probe_candidate_membership(outputs, pad, min_span),
        "score_identity": probe_score_identity(outputs),
        "three_fields": probe_three_fields(outputs, batch),
        "conditioner_ablation": probe_conditioner_ablation(model, inputs, outputs),
        "field_ablation": probe_field_ablation(outputs, batch),
        "real_top10_gt_coverage": probe_real_top10_gt_coverage(outputs, batch),
        "multigt_duplicate_synthetic": probe_multigt_duplicate(),
        "ordinal_tie_synthetic": probe_ordinal_tie_subgradient(),
        "endpoint_responsibility_synthetic": probe_endpoint_responsibility(),
        "parameter_groups": probe_parameter_groups(model),
    }
    result["loss_gradients"] = probe_all_losses(model, outputs, batch, epoch)
    return result


class TrifieldProbeSuite:
    """Run the fixed training panel without consuming the training sampler.

    The formal panel is exactly four batches of 64 examples. A shallow copy
    of the map-style dataset is used with a deep-copied data list because the
    QVHighlights item reader normalizes metadata in place. The original loader
    and its sampler/generator are never iterated.
    """

    PANEL_BATCHES = 4
    PANEL_BATCH_SIZE = 64
    PROBE_RNG_SEED = 2024

    def __init__(
        self, config: Any | None = None, max_batches: int | None = None, **_: Any
    ) -> None:
        self.config = config
        # Keep the old keyword loadable for callers, but do not allow it to
        # silently truncate the formally required 4x64 panel.
        if max_batches is not None and int(max_batches) != self.PANEL_BATCHES:
            raise ValueError(
                f"fixed trifield probe requires {self.PANEL_BATCHES} batches; "
                f"got max_batches={max_batches}"
            )

    @staticmethod
    def _prepared_parts(prepared: Any) -> tuple[Any, Any, Any]:
        if isinstance(prepared, Mapping):
            return (
                prepared.get("inputs", {}),
                prepared.get("targets", {}),
                prepared.get("metadata", {}),
            )
        return (
            getattr(prepared, "inputs", {}),
            getattr(prepared, "targets", {}),
            getattr(prepared, "metadata", {}),
        )

    @staticmethod
    def _panel_summary(panel: Mapping[str, Any] | None, seed: Any) -> Mapping[str, Any]:
        if not isinstance(panel, Mapping):
            return {"available": False, "reason": "fixed panel not supplied"}
        section = panel.get(str(seed), panel.get(seed))
        if not isinstance(section, Mapping):
            return {"available": False, "reason": f"seed section {seed!r} not found"}
        qids = section.get("qids")
        return {
            "available": isinstance(qids, list),
            "seed": str(seed),
            "count": len(qids) if isinstance(qids, list) else 0,
            "qid_order_sha256": section.get("qid_order_sha256"),
            "source": section.get("source"),
            "source_field": section.get("source_field"),
        }

    @staticmethod
    def _qid_order_sha256(qids: list[Any] | tuple[Any, ...]) -> str:
        # Fixed-panel protocol: compact JSON preserves integer QIDs and
        # avoids ambiguity from delimiters or whitespace.
        payload = json.dumps(list(qids), separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _panel_section(panel: Mapping[str, Any] | None, seed: Any) -> Mapping[str, Any]:
        if not isinstance(panel, Mapping):
            raise RuntimeError("ProbeSuite requires a fixed panel mapping")
        section = panel.get(str(seed), panel.get(seed))
        if not isinstance(section, Mapping):
            raise RuntimeError(f"fixed panel has no seed section for {seed!r}")
        raw_qids = section.get("qids")
        if not isinstance(raw_qids, list):
            raise RuntimeError("fixed panel seed section must contain a qids list")
        expected_count = (
            TrifieldProbeSuite.PANEL_BATCHES * TrifieldProbeSuite.PANEL_BATCH_SIZE
        )
        if len(raw_qids) != expected_count:
            raise RuntimeError(
                f"fixed panel must contain exactly {expected_count} qids"
            )
        if len({str(value) for value in raw_qids}) != len(raw_qids):
            raise RuntimeError("fixed panel qids must be unique")
        computed_hash = TrifieldProbeSuite._qid_order_sha256(raw_qids)
        expected_hash = section.get("qid_order_sha256")
        if expected_hash not in (None, "") and str(expected_hash) != computed_hash:
            raise RuntimeError(
                "fixed panel qid_order_sha256 does not match its ordered qids: "
                f"expected={expected_hash} computed={computed_hash}"
            )
        return {
            "section": section,
            "raw_qids": raw_qids,
            "qids": [str(value) for value in raw_qids],
            "qid_order_sha256": computed_hash,
        }

    @staticmethod
    def _dataset_qid_index(dataset: Any) -> dict[str, int]:
        """Build a QID-to-index map from QVHighlights.data without __getitem__."""
        rows = getattr(dataset, "data", None)
        if not isinstance(rows, (list, tuple)):
            raise RuntimeError(
                "fixed panel requires a map-style dataset with metadata list dataset.data"
            )
        index: dict[str, int] = {}
        duplicates: list[str] = []
        for row_index, row in enumerate(rows):
            if isinstance(row, Mapping):
                qid = row.get("qid", row.get("query_id"))
            else:
                qid = getattr(row, "qid", getattr(row, "query_id", None))
            if qid is None:
                raise RuntimeError(f"dataset.data[{row_index}] has no qid metadata")
            key = str(qid)
            if key in index:
                duplicates.append(key)
            else:
                index[key] = int(row_index)
        if duplicates:
            raise RuntimeError(
                "dataset.data contains duplicate qids; fixed panel index is ambiguous: "
                f"{duplicates[:5]}"
            )
        return index

    @staticmethod
    def _fixed_loader(
        loader: Any, panel: Mapping[str, Any] | None, seed: Any
    ) -> tuple[Any, list[str]]:
        dataset = getattr(loader, "dataset", None)
        if dataset is None:
            raise RuntimeError("fixed panel requires bundle.train_loader.dataset")
        panel_info = TrifieldProbeSuite._panel_section(panel, seed)
        qid_index = TrifieldProbeSuite._dataset_qid_index(dataset)
        missing = [qid for qid in panel_info["qids"] if qid not in qid_index]
        if missing:
            raise RuntimeError(
                f"fixed panel contains {len(missing)} qids absent from train dataset; "
                f"first={missing[:5]}"
            )
        indices = [qid_index[qid] for qid in panel_info["qids"]]

        # QVHighlights.__getitem__ updates a nested metadata dict in place.
        # Isolate that mutation while retaining feature paths and the
        # reviewed collate function from the original loader.
        probe_dataset = copy.copy(dataset)
        try:
            probe_dataset.data = copy.deepcopy(getattr(dataset, "data"))
        except (AttributeError, TypeError) as exc:
            raise RuntimeError(
                "fixed panel dataset.data cannot be copied safely"
            ) from exc
        probe_subset = Subset(probe_dataset, indices)
        generator = torch.Generator().manual_seed(TrifieldProbeSuite.PROBE_RNG_SEED)
        probe_loader = DataLoader(
            probe_subset,
            batch_size=TrifieldProbeSuite.PANEL_BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            collate_fn=getattr(loader, "collate_fn", None),
            pin_memory=bool(getattr(loader, "pin_memory", False)),
            drop_last=False,
            generator=generator,
        )
        if len(probe_loader) != TrifieldProbeSuite.PANEL_BATCHES:
            raise RuntimeError(
                f"fixed panel loader has {len(probe_loader)} batches, expected "
                f"{TrifieldProbeSuite.PANEL_BATCHES}"
            )
        return probe_loader, panel_info["qids"]

    @staticmethod
    def _batch_qids(metadata: Any) -> list[str]:
        if not isinstance(metadata, Mapping):
            return []
        rows = metadata.get("metas", metadata.get("records"))
        if not isinstance(rows, (list, tuple)):
            return []
        result = []
        for row in rows:
            if isinstance(row, Mapping) and row.get("qid") is not None:
                result.append(str(row.get("qid")))
        return result

    def run(self, request: Any) -> Mapping[str, Any]:
        bundle = request.bundle
        loader = getattr(bundle, "train_loader", None)
        prepare_batch = getattr(bundle, "prepare_batch", None)
        if loader is None or not callable(prepare_batch):
            raise RuntimeError(
                "ProbeSuite requires bundle.train_loader and bundle.prepare_batch"
            )
        seed = getattr(request.config, "seed", None)
        if seed is None:
            raise RuntimeError("fixed panel probe requires config.seed")
        panel_info = self._panel_section(request.fixed_panel, seed)
        probe_loader, expected_qids = self._fixed_loader(
            loader, request.fixed_panel, seed
        )

        # run_probe_suite normally supplies this guard, but keeping the suite
        # self-contained also makes direct unit/integration calls safe.
        torch_rng = torch.random.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        python_rng = random.getstate()
        try:
            import numpy as np

            numpy_rng = np.random.get_state()
        except (ImportError, ModuleNotFoundError):
            np = None
            numpy_rng = None

        training_modes = [
            (module, bool(module.training)) for module in request.model.modules()
        ]
        # Keep every receipt comparable across epochs. The outer child guard
        # also restores these states, while this local guard makes direct
        # TrifieldProbeSuite.run calls safe.
        random.seed(self.PROBE_RNG_SEED)
        torch.manual_seed(self.PROBE_RNG_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.PROBE_RNG_SEED)
        if np is not None:
            np.random.seed(self.PROBE_RNG_SEED)
        request.model.eval()

        precision = str(getattr(request.config, "precision", "bf16"))
        context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if precision == "bf16" and request.device.type == "cuda"
            else nullcontext()
        )
        observed_qids: list[str] = []
        batch_sizes: list[int] = []
        batch_results: list[Mapping[str, Any]] = []
        try:
            for batch_index, raw_batch in enumerate(probe_loader):
                if batch_index >= self.PANEL_BATCHES:
                    raise RuntimeError("fixed panel yielded more than four batches")
                prepared = prepare_batch(raw_batch, request.device)
                inputs, _, metadata = self._prepared_parts(prepared)
                if not isinstance(inputs, Mapping):
                    raise TypeError("prepared probe batch inputs must be a mapping")
                qids = self._batch_qids(metadata)
                if len(qids) != self.PANEL_BATCH_SIZE:
                    raise RuntimeError(
                        f"fixed panel batch {batch_index} has {len(qids)} qids, expected "
                        f"{self.PANEL_BATCH_SIZE}"
                    )
                batch_sizes.append(len(qids))
                observed_qids.extend(qids)
                with torch.enable_grad(), context:
                    outputs = request.model(inputs)
                    probes = run_base_probes(
                        request.model, outputs, prepared, int(request.epoch)
                    )
                batch_results.append(
                    {
                        "batch_index": batch_index,
                        "batch_size": len(qids),
                        "qids": qids,
                        "probe": probes,
                    }
                )
                del outputs, probes, prepared, raw_batch
        finally:
            for module, training in training_modes:
                module.training = training
            torch.random.set_rng_state(torch_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
            random.setstate(python_rng)
            if np is not None and numpy_rng is not None:
                np.random.set_state(numpy_rng)

        if len(batch_results) != self.PANEL_BATCHES:
            raise RuntimeError(
                f"fixed panel yielded {len(batch_results)} batches, expected {self.PANEL_BATCHES}"
            )
        if batch_sizes != [self.PANEL_BATCH_SIZE] * self.PANEL_BATCHES:
            raise RuntimeError(f"fixed panel batch sizes are not 4x64: {batch_sizes}")
        full_order_match = observed_qids == expected_qids
        # Observed metadata may expose integer QIDs as strings. Once the exact
        # ordered comparison succeeds, hash the panel's original JSON scalar
        # types so it is directly comparable to the protocol hash.
        observed_hash = self._qid_order_sha256(
            panel_info["raw_qids"] if full_order_match else observed_qids
        )
        hash_match = observed_hash == panel_info["qid_order_sha256"]
        if not full_order_match or not hash_match:
            first_mismatch = next(
                (
                    i
                    for i, (actual, expected) in enumerate(
                        zip(observed_qids, expected_qids)
                    )
                    if actual != expected
                ),
                min(len(observed_qids), len(expected_qids)),
            )
            raise RuntimeError(
                "fixed panel QID order/hash mismatch: "
                f"order_match={full_order_match} hash_match={hash_match} "
                f"first_mismatch_index={first_mismatch}"
            )
        panel = self._panel_summary(request.fixed_panel, seed)
        return {
            "schema": "trifield_base_v1_probe_suite_v2",
            "epoch": int(request.epoch),
            "phase": str(request.phase),
            "probe_rng_seed": self.PROBE_RNG_SEED,
            "batch_count": len(batch_results),
            "batch_size": self.PANEL_BATCH_SIZE,
            "batch_sizes": batch_sizes,
            "batch_qids": observed_qids,
            "fixed_panel": panel,
            "fixed_panel_full_order_match": full_order_match,
            "fixed_panel_qid_order_sha256": observed_hash,
            "fixed_panel_qid_hash_match": hash_match,
            "batches": batch_results,
            "trainability": request.model.trainability_audit()
            if callable(getattr(request.model, "trainability_audit", None))
            else {},
            "experiment_contract": request.model.experiment_contract()
            if callable(getattr(request.model, "experiment_contract", None))
            else {},
        }


def build_probe_suite(config: Any | None = None, **kwargs: Any) -> TrifieldProbeSuite:
    """Config-first factory used by child.py's formal probe receipts."""
    return TrifieldProbeSuite(config=config, **kwargs)


__all__ = [
    "probe_candidate_membership",
    "probe_score_identity",
    "probe_field_ablation",
    "probe_three_fields",
    "probe_conditioner_ablation",
    "probe_all_losses",
    "probe_parameter_groups",
    "probe_real_top10_gt_coverage",
    "probe_multigt_duplicate",
    "probe_ordinal_tie_subgradient",
    "probe_endpoint_responsibility",
    "probe_rank_loss",
    "probe_evidence_loss",
    "probe_support_loss",
    "probe_transition_loss",
    "probe_endpoint_loss",
    "run_base_probes",
    "TrifieldProbeSuite",
    "build_probe_suite",
]
