"""Round1 probes for every objective term, field, and structural intervention."""

from __future__ import annotations

import random
from typing import Any, Mapping

import torch
from torch import Tensor

import field_core.probes as _v1_probes
from field_core.probes import (
    probe_candidate_membership,
    probe_conditioner_ablation,
    probe_endpoint_responsibility,
    probe_field_ablation,
    probe_multigt_duplicate,
    probe_ordinal_tie_subgradient,
    probe_parameter_groups,
    probe_real_top10_gt_coverage,
    probe_score_identity,
    probe_three_fields,
)
from field_core.losses import FiveLossTerms, official_threshold_grades

from .losses import compute_round1_loss_terms


def _finite(value: Tensor) -> bool:
    return bool(torch.isfinite(value.detach().float()).all())


def _metric(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach()
    return value


def _single_loss_probe(
    terms: FiveLossTerms,
    name: str,
    weights: Mapping[str, float],
) -> Mapping[str, Any]:
    value = getattr(terms, name)
    weight = float(weights.get(name, 1.0))
    return {
        "loss_name": name,
        "finite": _finite(value),
        "scalar": value.ndim == 0,
        "raw": value.detach(),
        "weight": weight,
        "effective": (value.detach() * weight),
        "value": value.detach(),
    }


def _safe_norm(values: list[Tensor], reference: Tensor) -> Tensor:
    if not values:
        return reference.detach().float().sum() * 0.0
    return torch.stack([value.float().square().sum() for value in values]).sum().sqrt()


def _gradient_report(
    model: Any,
    terms: FiveLossTerms,
    score: Tensor | None,
) -> Mapping[str, Any]:
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    term_names = ("rank", "evidence", "support", "transition", "endpoint", "total")
    configured = getattr(model, "loss_weights", {})
    weights = {
        name: float(configured.get(name, 1.0))
        for name in ("rank", "evidence", "support", "transition", "endpoint")
    }
    weights["total"] = 1.0
    gradients: dict[str, tuple[Tensor | None, ...]] = {}
    for name in term_names:
        value = getattr(terms, name)
        if value.requires_grad and params:
            gradients[name] = torch.autograd.grad(
                value,
                params,
                retain_graph=True,
                allow_unused=True,
            )
        else:
            gradients[name] = tuple(None for _ in params)

    reference = terms.total

    def norm(gradient: tuple[Tensor | None, ...]) -> Tensor:
        return _safe_norm(
            [value for value in gradient if value is not None],
            reference,
        )

    def dot(
        left: tuple[Tensor | None, ...],
        right: tuple[Tensor | None, ...],
    ) -> Tensor:
        pieces = [
            a.float().mul(b.float()).sum()
            for a, b in zip(left, right)
            if a is not None and b is not None
        ]
        return torch.stack(pieces).sum() if pieces else reference.detach().float() * 0.0

    # Names are deliberately split so the probe answers which actual route
    # receives each term. Shared parent tensors can appear in several buckets.
    buckets = {
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
    norms = {name: norm(gradient) for name, gradient in gradients.items()}
    result: dict[str, Any] = {}
    for name in term_names:
        gradient = gradients[name]
        raw_modules = {
            bucket: norm(tuple(gradient[index] for index in indices)).detach()
            for bucket, indices in buckets.items()
        }
        weight = weights[name]
        result[name] = {
            "weight": weight,
            "raw": {
                "global_norm": norms[name].detach(),
                "finite": all(value is None or _finite(value) for value in gradient),
                "nonzero_parameter_tensors": sum(
                    int(value is not None and bool(value.detach().abs().max().gt(0.0)))
                    for value in gradient
                ),
                "module_norms": raw_modules,
            },
            "effective": {
                "global_norm": (norms[name] * weight).detach(),
                "module_norms": {
                    bucket: value * weight for bucket, value in raw_modules.items()
                },
            },
        }

    cosine: dict[str, Any] = {}
    for index, left_name in enumerate(term_names):
        for right_name in term_names[index + 1 :]:
            denominator = norms[left_name] * norms[right_name]
            defined = bool(denominator.detach().gt(1.0e-12))
            cosine[f"{left_name}__{right_name}"] = {
                "defined": defined,
                "value": (
                    (
                        dot(gradients[left_name], gradients[right_name]) / denominator
                    ).detach()
                    if defined
                    else None
                ),
            }
    result["pairwise_cosine"] = cosine

    recomposed: list[Tensor] = []
    base_names = ("rank", "evidence", "support", "transition", "endpoint")
    for index, parameter in enumerate(params):
        pieces = [
            gradients[name][index] * weights[name]
            for name in base_names
            if gradients[name][index] is not None
        ]
        recomposed.append(
            sum(pieces[1:], pieces[0]) if pieces else torch.zeros_like(parameter)
        )
    total_gradient = tuple(
        torch.zeros_like(parameter) if gradient is None else gradient
        for parameter, gradient in zip(params, gradients["total"])
    )
    difference = tuple(left - right for left, right in zip(total_gradient, recomposed))
    raw_error = norm(difference)
    total_norm = norm(total_gradient)
    relative_error = raw_error / total_norm.clamp_min(1.0e-12)
    result["weighted_recomposition"] = {
        "weights": dict(weights),
        "raw_norm": norm(tuple(recomposed)).detach(),
        "total_norm": total_norm.detach(),
        "absolute_error": raw_error.detach(),
        "relative_error": relative_error.detach(),
        "diagnostic_precision": "fp32_forward_required",
        "within_relative_tolerance": bool(relative_error.detach().le(1.0e-4)),
        "near_zero_absolute_guard": bool(
            total_norm.detach().le(1.0e-6) and raw_error.detach().le(1.0e-6)
        ),
    }

    if isinstance(score, Tensor):
        flat_score = score.float().flatten(1)
        geometry = terms.geometry
        flat_valid = geometry.valid.flatten(1)
        flat_iou = geometry.max_iou.flatten(1)
        grades = official_threshold_grades(
            geometry.max_iou,
            geometry.valid,
        ).flatten(1)
        eligible_rows = [
            row
            for row in range(flat_score.shape[0])
            if bool(flat_valid[row].any())
            and int(grades[row][flat_valid[row]].unique().numel()) > 1
        ]
        if eligible_rows:
            row_index = torch.tensor(
                eligible_rows,
                device=flat_score.device,
                dtype=torch.long,
            )
            good = (
                flat_iou[row_index].masked_fill(~flat_valid[row_index], -1.0).argmax(1)
            )
            bad = flat_iou[row_index].masked_fill(~flat_valid[row_index], 2.0).argmin(1)
            gap = (flat_score[row_index, good] - flat_score[row_index, bad]).mean()
            gap_gradient = (
                torch.autograd.grad(
                    gap,
                    params,
                    retain_graph=True,
                    allow_unused=True,
                )
                if gap.requires_grad and params
                else tuple(None for _ in params)
            )
            gap_dot = {
                name: dot(gradients[name], gap_gradient).detach() for name in term_names
            }
            result["good_bad_gap"] = gap.detach()
            result["good_bad_gap_eligible_row_count"] = len(eligible_rows)
            result["good_bad_gap_downhill_dot"] = {
                name: (-value).detach() for name, value in gap_dot.items()
            }
        else:
            result["good_bad_gap_skipped"] = "no row with two distinct official grades"
    return result


def probe_round1_all_losses(
    model: Any,
    outputs: Any,
    batch: Any,
    epoch: int = 1,
) -> Mapping[str, Any]:
    """Probe all five terms with read-only autograd and configured weights."""

    training_modes = [(module, bool(module.training)) for module in model.modules()]
    cpu_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    python_rng = random.getstate()
    try:
        import numpy as np

        numpy_rng = np.random.get_state()
    except (ImportError, ModuleNotFoundError):
        np = None
        numpy_rng = None
    cache_names = (
        "_last_inputs",
        "_last_field_output",
        "_last_evidence_pairs",
    )
    saved_cache = {
        name: getattr(model, name) for name in cache_names if hasattr(model, name)
    }
    try:
        wrong, pair = (
            model._wrong_query_evidence(batch, outputs)
            if hasattr(model, "_wrong_query_evidence")
            else (None, None)
        )
        options = getattr(model, "options", None)
        terms = compute_round1_loss_terms(
            outputs,
            batch,
            wrong_evidence=wrong,
            evidence_pair_mask=pair,
            loss_weights=getattr(model, "loss_weights", None),
            rank_margin=float(getattr(options, "rank_margin", 0.0)),
            transition_target_mode=str(
                getattr(options, "transition_target_mode", "matched")
            ),
            quality_per_query=bool(getattr(options, "quality_per_query", False)),
            quality_stratified=bool(getattr(options, "quality_stratified", False)),
        )
        weights = getattr(
            model,
            "loss_weights",
            {
                "rank": 1.0,
                "evidence": 0.1,
                "support": 0.1,
                "transition": 0.1,
                "endpoint": 0.1,
            },
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
            },
            "loss_probes": {
                name: _single_loss_probe(terms, name, weights)
                for name in (
                    "rank",
                    "evidence",
                    "support",
                    "transition",
                    "endpoint",
                )
            },
            "loss_metrics": {
                name: _metric(value) for name, value in terms.metrics.items()
            },
            "gradient_probe": _gradient_report(
                model,
                terms,
                getattr(getattr(outputs, "trifield_output", None), "score", None),
            ),
            "epoch": int(epoch),
        }
        result["rank_probe"] = {
            "score_span": _metric(terms.metrics["rank/score_span"]),
            "violation_rate": _metric(terms.metrics["rank/violating_pair_rate"]),
            "margin": _metric(terms.metrics["rank/margin"]),
        }
        result["support_probe"] = {
            key: _metric(value)
            for key, value in terms.metrics.items()
            if key.startswith("support/")
        }
        result["transition_probe"] = {
            key: _metric(value)
            for key, value in terms.metrics.items()
            if key.startswith("transition/")
        }
        return result
    finally:
        for module, training in training_modes:
            module.training = training
        torch.random.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        random.setstate(python_rng)
        if np is not None and numpy_rng is not None:
            np.random.set_state(numpy_rng)
        for name, value in saved_cache.items():
            setattr(model, name, value)


def run_round1_probes(
    model: Any,
    outputs: Any,
    batch: Any,
    epoch: int = 1,
) -> Mapping[str, Any]:
    """Run V1 structural probes plus round1 objective-specific probes."""

    inputs = (
        batch.get("inputs", {})
        if isinstance(batch, Mapping)
        else getattr(batch, "inputs", {})
    )
    parent = getattr(model, "parent_model", None)
    head = getattr(parent, "shared_head", None)
    min_span = int(getattr(head, "min_span_clips", 2))
    result: dict[str, Any] = {
        "candidate_membership": probe_candidate_membership(
            outputs,
            inputs.get("video_padding_mask") if isinstance(inputs, Mapping) else None,
            min_span,
        ),
        "score_identity": probe_score_identity(outputs),
        "three_fields": probe_three_fields(outputs, batch),
        "conditioner_ablation": probe_conditioner_ablation(
            model,
            inputs,
            outputs,
        ),
        "field_ablation": probe_field_ablation(outputs, batch),
        "real_top10_gt_coverage": probe_real_top10_gt_coverage(
            outputs,
            batch,
        ),
        "multigt_duplicate_synthetic": probe_multigt_duplicate(),
        "ordinal_tie_synthetic": probe_ordinal_tie_subgradient(),
        "endpoint_responsibility_synthetic": probe_endpoint_responsibility(),
        "parameter_groups": probe_parameter_groups(model),
    }
    result["loss_gradients"] = probe_round1_all_losses(
        model,
        outputs,
        batch,
        epoch,
    )
    result["round1_variant"] = getattr(
        getattr(model, "options", None), "variant", "r0_control"
    )
    result["diagnostic_precision"] = "fp32_forward_required"
    return result


class Round1ProbeSuite(_v1_probes.TrifieldProbeSuite):
    """Fixed-panel suite using round1 loss probes and FP32 diagnostic forward."""

    def run(self, request: Any) -> Mapping[str, Any]:
        bundle = request.bundle
        loader = getattr(bundle, "train_loader", None)
        prepare_batch = getattr(bundle, "prepare_batch", None)
        if loader is None or not callable(prepare_batch):
            raise RuntimeError(
                "Round1ProbeSuite requires bundle.train_loader and prepare_batch"
            )
        seed = getattr(request.config, "seed", None)
        if seed is None:
            raise RuntimeError("fixed panel probe requires config.seed")
        panel_info = self._panel_section(request.fixed_panel, seed)
        probe_loader, expected_qids = self._fixed_loader(
            loader,
            request.fixed_panel,
            seed,
        )

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
        probe_seed = self.PROBE_RNG_SEED
        random.seed(probe_seed)
        torch.manual_seed(probe_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(probe_seed)
        if np is not None:
            np.random.seed(probe_seed)
        request.model.eval()

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
                    raise TypeError("prepared probe inputs must be a mapping")
                qids = self._batch_qids(metadata)
                if len(qids) != self.PANEL_BATCH_SIZE:
                    raise RuntimeError(
                        f"fixed panel batch {batch_index} has {len(qids)} qids, "
                        f"expected {self.PANEL_BATCH_SIZE}"
                    )
                batch_sizes.append(len(qids))
                observed_qids.extend(qids)
                # No autocast: this receipt is an FP32 diagnostic forward,
                # independent of the BF16 train step that preceded it.
                with torch.enable_grad():
                    outputs = request.model(inputs)
                    probes = run_round1_probes(
                        request.model,
                        outputs,
                        prepared,
                        int(request.epoch),
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
                f"fixed panel yielded {len(batch_results)} batches, expected "
                f"{self.PANEL_BATCHES}"
            )
        if batch_sizes != [self.PANEL_BATCH_SIZE] * self.PANEL_BATCHES:
            raise RuntimeError(f"fixed panel batch sizes are not 4x64: {batch_sizes}")
        full_order_match = observed_qids == expected_qids
        observed_hash = self._qid_order_sha256(
            panel_info["raw_qids"] if full_order_match else observed_qids
        )
        hash_match = observed_hash == panel_info["qid_order_sha256"]
        if not full_order_match or not hash_match:
            first_mismatch = next(
                (
                    index
                    for index, (actual, expected) in enumerate(
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
        return {
            "schema": "trifield_round1_probe_suite_v1",
            "epoch": int(request.epoch),
            "phase": str(request.phase),
            "diagnostic_precision": "fp32",
            "probe_rng_seed": probe_seed,
            "batch_count": len(batch_results),
            "batch_size": self.PANEL_BATCH_SIZE,
            "batch_sizes": batch_sizes,
            "batch_qids": observed_qids,
            "fixed_panel": self._panel_summary(request.fixed_panel, seed),
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


def build_probe_suite(config: Any | None = None, **kwargs: Any) -> Round1ProbeSuite:
    return Round1ProbeSuite(config=config, **kwargs)


__all__ = [
    "Round1ProbeSuite",
    "build_probe_suite",
    "probe_round1_all_losses",
    "run_round1_probes",
]
