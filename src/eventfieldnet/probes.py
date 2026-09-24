"""R68 actual objective and field panels with preserved RNG/module modes.

Both train and validation panels are diagnostics only; no optimizer updates.
Official full1550 selection is a separate strictFP32 evaluation.
"""

import random
import torch
import numpy as np
from .model.probes import Round31ProbeSuite
from .candidate_pools import candidate_pools, structural_pair_loss
from .model.losses import _target_spans
from .geometry_probes import field_geometry_panel
from .probe_helpers import actual_probe, coverage_consistency_panel
from .length_objective import durations
from .model_factory import HINGE_OBJECTIVE


def scalar(x):
    return float(x.detach()) if isinstance(x, torch.Tensor) else x


@torch.no_grad()
def support_counterfactual_panel(field, geometry, spans, gt_mask, batch):
    """Same GT full candidate versus truncation/expansion; labels diagnostic only."""
    batch_size, length, _ = field.support.shape
    gt_count = gt_mask.shape[1]
    valid = geometry.valid.flatten(1)
    iou = geometry.all_iou.float().flatten(1, 2)
    best = iou.masked_fill(~valid[..., None], -1.0).argmax(1)
    positive_quality = iou.gather(1, best[:, None]).squeeze(1)
    start, end = best // length, best % length
    step = ((end - start + 1) // 4).clamp_min(1)
    ns = torch.stack((start + step, start, start - step, start), -1)
    ne = torch.stack((end, end - step, end, end + step), -1)
    bounds = (ns >= 0) & (ne < length) & (ns <= ne)
    ids = ns.clamp(0, length - 1) * length + ne.clamp(0, length - 1)
    flat = ids.flatten(1)
    # A negative that is a good other GT is not treated as a bad interval.
    negative_quality = (
        geometry.max_iou.flatten(1).gather(1, flat).reshape(batch_size, gt_count, 4)
    )
    quality_gap = positive_quality[..., None] - negative_quality
    mask = (
        bounds
        & gt_mask[..., None]
        & valid.gather(1, flat).reshape_as(ids)
        & (positive_quality[..., None] >= 0.7)
        & (quality_gap >= 0.1)
    )

    def gap(value):
        values = value.float().flatten(1)
        return values.gather(1, best)[..., None] - values.gather(1, flat).reshape(
            batch_size, gt_count, 4
        )

    s_gap = gap(field.support)
    full_gap = gap(field.score)
    no_s_gap = full_gap - s_gap
    seconds = (spans[..., 1] - spans[..., 0]).clamp_min(0) * durations(batch, spans)[
        :, None
    ]
    known = durations(batch, spans)[:, None] > 0
    buckets = {
        "all": gt_mask,
        "short_0_10s": gt_mask & known & (seconds <= 10),
        "middle_10_30s": gt_mask & known & (seconds > 10) & (seconds <= 30),
        "long_over30s": gt_mask & known & (seconds > 30),
    }
    result = {
        "scope": "Fixed-panel paired temporal ordering reference; ordering_reference_not_J06_success: pure expansion neutrality is assessed in r66_coverage_consistency; inference never sees GT; not official AP",
        "buckets": {},
    }
    for bucket, member in buckets.items():
        values = {}
        for kind, index in [("truncation", slice(0, 2)), ("expansion", slice(2, 4))]:
            selected = (mask & member[..., None])[..., index]
            count = int(selected.sum())
            row = {"pairs": count}
            if count:
                sg, fg, bg = (
                    s_gap[..., index][selected],
                    full_gap[..., index][selected],
                    no_s_gap[..., index][selected],
                )
                row.update(
                    s_gap=scalar(sg.mean()),
                    s_correct_fraction=scalar((sg > 0).float().mean()),
                    full_gap=scalar(fg.mean()),
                    no_s_gap=scalar(bg.mean()),
                    repair_fraction=scalar(((bg <= 0) & (fg > 0)).float().mean()),
                    damage_fraction=scalar(((bg > 0) & (fg <= 0)).float().mean()),
                )
            values[kind] = row
        result["buckets"][bucket] = values
    return result


class R68ProbeSuite(Round31ProbeSuite):
    def run(self, request):
        model, bundle, seed = request.model, request.bundle, request.config.seed
        train_loader, expected = self._fixed_loader(
            bundle.train_loader, request.fixed_panel, seed
        )
        val_qids = [x["qid"] for x in bundle.val_loader.dataset.data[:256]]
        val_panel = {
            str(seed): {
                "qids": val_qids,
                "qid_order_sha256": self._qid_order_sha256(val_qids),
            }
        }
        val_loader, val_expected = self._fixed_loader(
            bundle.val_loader, val_panel, seed
        )
        cpu = torch.random.get_rng_state()
        cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        py, npstate = random.getstate(), np.random.get_state()
        modes = [(m, m.training) for m in model.modules()]
        capture = getattr(model, "r50_capture_terms", False)
        records = []
        try:
            torch.manual_seed(2024)
            random.seed(2024)
            np.random.seed(2024)
            model.eval()
            model.r50_capture_terms = True
            for split, loader, wanted in [
                ("train", train_loader, expected),
                ("val", val_loader, val_expected),
            ]:
                seen = []
                for bi, raw in enumerate(loader):
                    batch = bundle.prepare_batch(raw, request.device)
                    inputs, _, meta = self._prepared_parts(batch)
                    qids = self._batch_qids(meta)
                    seen.extend(qids)
                    with (
                        torch.enable_grad(),
                        torch.autocast(
                            device_type=torch.device(request.device).type, enabled=False
                        ),
                    ):
                        out = model(inputs)
                        result = model.compute_loss(
                            out, batch, None, max(1, int(request.epoch))
                        )
                        terms, field = model.r50_last_terms, out.trifield_output
                        valid = terms.geometry.valid
                        assert torch.isfinite(result.loss)
                        mask = valid.flatten(1)
                        top = lambda s: (
                            s.flatten(1).masked_fill(~mask, -torch.inf).argmax(1)
                        )
                        quality = terms.geometry.max_iou.flatten(1)
                        initial = top(field.score)
                        base_iou = quality.gather(1, initial[:, None]).mean()
                        record = {
                            "split": split,
                            "batch": bi,
                            "qids": qids,
                            "actual_metrics": {
                                k: scalar(v)
                                for k, v in result.metrics.items()
                                if k.startswith(
                                    (
                                        "candidate_pools/",
                                        "query_objective/",
                                        "candidate_objective/",
                                        "support_objective/",
                                        "joint_training/",
                                        "field_objective/",
                                        "support_geometry/",
                                        "length_objective/",
                                        "quality_calibration/",
                                        "local_support/",
                                        "gradient_routing/",
                                        "model_factory/",
                                    )
                                )
                                or k
                                in [
                                    "rank",
                                    "evidence",
                                    "support",
                                    "transition",
                                    "endpoint",
                                    "total",
                                ]
                            },
                            "fields": {},
                            "loss_gradients": {},
                            "structure_gradients": {},
                        }
                        for name in [
                            "carrier",
                            "evidence",
                            "support",
                            "transition_start",
                            "transition_end",
                        ]:
                            value = getattr(field, name)
                            ids = top(
                                model.selector.compose_score(
                                    field, {name: torch.zeros_like(value)}
                                )
                            )
                            tensor = value.detach()[valid]
                            record["fields"][name] = {
                                "signed_mean": scalar(tensor.mean()),
                                "negative_saturation": scalar(
                                    (tensor < -0.95).float().mean()
                                ),
                                "positive_saturation": scalar(
                                    (tensor > 0.95).float().mean()
                                ),
                                "abs_mean": scalar(tensor.abs().mean()),
                                "saturation": scalar(
                                    (tensor.abs() > 0.95).float().mean()
                                ),
                                "remove_top1_iou_delta": scalar(
                                    quality.gather(1, ids[:, None]).mean() - base_iou
                                ),
                                "top1_changed_rate": scalar(
                                    (ids != initial).float().mean()
                                ),
                            }
                        spans, gm = _target_spans(out, batch)
                        pools = candidate_pools(field, terms.geometry, gm, True)
                        record["deployment_pool_overlap"] = scalar(
                            (pools["raw"] & pools["nms"]).sum(1).float().mean()
                        )
                        record["evidence_wrong_query_gap"] = None
                        wrong = getattr(model, "_last_wrong_query_field", None)
                        positive = valid & (terms.geometry.max_iou >= 0.7)
                        if wrong is not None and positive.any():
                            record["evidence_wrong_query_gap"] = scalar(
                                (field.evidence - wrong.evidence)[positive].mean()
                            )
                        for kind in ["support", "transition"]:
                            value = (
                                field.support
                                if kind == "support"
                                else 0.5
                                * (field.transition_start + field.transition_end)
                            )
                            _, stats = structural_pair_loss(
                                value, terms.geometry, gm, kind
                            )
                            record[kind + "_intervention"] = {
                                k: scalar(v) for k, v in stats.items()
                            }
                        if bi == 0:
                            named = [
                                (n, p)
                                for n, p in model.named_parameters()
                                if p.requires_grad
                            ]
                            for key in [
                                "rank",
                                "evidence",
                                "support",
                                "transition",
                                "endpoint",
                            ]:
                                value = getattr(terms, key) * model.loss_weights[key]
                                gs = (
                                    torch.autograd.grad(
                                        value,
                                        [p for _, p in named],
                                        retain_graph=True,
                                        allow_unused=True,
                                    )
                                    if value.requires_grad
                                    else [None] * len(named)
                                )
                                record["loss_gradients"][key] = {
                                    "norm": sum(
                                        float(g.detach().float().square().sum())
                                        for g in gs
                                        if g is not None
                                    )
                                    ** 0.5,
                                    "none_fraction": sum(g is None for g in gs)
                                    / len(gs),
                                }
                                record["structure_gradients"][key] = {
                                    n: float(g.detach().float().norm())
                                    if g is not None
                                    else None
                                    for (n, p), g in zip(named, gs)
                                    if any(
                                        tag in n
                                        for tag in [
                                            "r50_",
                                            "r51_",
                                            "r53_",
                                            "r58_",
                                            "r59_",
                                            "r65_",
                                            "r66_",
                                            "r67_",
                                            "r68_",
                                        ]
                                    )
                                }
                                del gs
                        geometry = field_geometry_panel(model, out, batch, terms)
                        if (
                            model.r65_mode
                            in ("direct_completeness_s", "direct_quality_s")
                            and "local_s_residual" in geometry
                        ):
                            replacement = geometry.pop("local_s_residual")
                            replacement["scope"] = (
                                "New deployed S versus legacy edge-S readout, fixed S-independent anchors; not residual-only deletion, whole-S deletion or AP"
                            )
                            geometry["s_replacement_vs_legacy"] = replacement
                        record["r59_geometry_panel"] = geometry
                        record["r65_support_counterfactuals"] = (
                            support_counterfactual_panel(
                                field, terms.geometry, spans, gm, batch
                            )
                        )
                        record["r68_coverage_consistency"] = coverage_consistency_panel(
                            field,
                            terms.geometry,
                            spans,
                            gm,
                            batch,
                            hinge_truncation=model.r68_mode in HINGE_OBJECTIVE,
                        )
                        record["r66_coverage_consistency"] = record[
                            "r68_coverage_consistency"
                        ]
                        if bi == 0:
                            record["r68_actual_objective"] = actual_probe(
                                model, field, terms, result, strict=True
                            )
                        record["r59_freeze_policy"] = {
                            "epoch": getattr(model, "r59_epoch", None),
                            "parent_applied_updates_disabled": getattr(
                                model, "r59_parent_frozen", False
                            ),
                            "autograd_grad_is_raw_not_post_hook_applied": True,
                        }
                        records.append(record)
                    model.r50_last_terms = None
                    del out, result, terms, field, batch, raw
                assert seen == wanted, (split, "fixed panel order changed")
        finally:
            model.r50_last_terms = None
            model.r50_capture_terms = capture
            for module, mode in modes:
                module.training = mode
            torch.random.set_rng_state(cpu)
            if cuda is not None:
                torch.cuda.set_rng_state_all(cuda)
            random.setstate(py)
            np.random.set_state(npstate)
        return {
            "schema": "r68_actual_objective_panels_v1",
            "precision": "fp32",
            "epoch": request.epoch,
            "fixed_train_qids": expected,
            "fixed_val_qids": val_expected,
            "scope": "Fixed-panel geometry/gradient diagnostics; official full-validation MR is separate",
            "legacy_probe_replaced_reason": "Use actual final objective/deployed fields and explicit raw/effective gradients",
            "batches": records,
            "experiment_contract": model.experiment_contract(),
        }


def build_probe_suite(config=None, **kwargs):
    return R68ProbeSuite(config=config, **kwargs)
