"""Role-conditioned non-harmful residual correction; unchanged deployed scorer."""

import dataclasses, hashlib
from pathlib import Path
import torch
from torch.nn import functional as F
from . import query_objective, candidate_objective, support_objective
from .model import losses
from field_core.adapter import _batch_parts, _valid_wrong_query_pairs


def residual_correction(field, pairs, kind):
    # S owns missing event content; T owns complete-content endpoint errors.
    mask = pairs["mask"]
    if kind == "support":
        value = field.support
        mask = mask & (pairs["support_delta"] > 0.01)
    else:
        value = 0.5 * (field.transition_start + field.transition_end)
        mask = (
            mask
            & (pairs["support_delta"].abs() < 0.001)
            & ((pairs["start_delta"] + pairs["end_delta"]) > 0.01)
        )
    contribution = candidate_objective.pair_gap(value, pairs)
    rest = candidate_objective.pair_gap(field.score - value, pairs).detach()
    # Do not shrink an already useful large margin (unlike symmetric gap regression).
    needed = (0.1 * pairs["quality_delta"] - rest).clamp(0, 0.25).detach()
    violation = F.relu(needed - contribution)
    loss = candidate_objective.masked_mean(violation, mask)
    stats = {
        "pairs": mask.sum(),
        "gt_coverage": mask.any(1).sum().float() / pairs["gt_mask"].sum().clamp_min(1),
        "harmful_fraction": candidate_objective.masked_mean(
            (contribution < 0).float(), mask
        ),
        "violation_fraction": candidate_objective.masked_mean(
            (violation > 0).float(), mask
        ),
        "required_gap": candidate_objective.masked_mean(needed, mask),
        "actual_gap": candidate_objective.masked_mean(contribution, mask),
        "loss": loss,
    }
    return loss, stats


def semantic_correction(model, field, geo, batch):
    wrong = model._last_wrong_query_field
    inputs, _, metadata = _batch_parts(batch)
    _, pair_cpu = _valid_wrong_query_pairs(metadata, field.score.shape[0])
    pair = pair_cpu.to(field.score.device)
    positive = geo.valid & (geo.max_iou >= 0.7) & pair[:, None, None]
    if wrong is None:
        return field.evidence.sum() * 0, {"valid_queries": field.score.new_tensor(0.0)}
    gap = field.evidence - wrong.evidence
    # Per-candidate hinge prevents a few easy windows hiding harder semantic failures.
    per = (F.relu(0.1 - gap) * positive).flatten(1).sum(1) / positive.flatten(1).sum(
        1
    ).clamp_min(1)
    active = positive.flatten(1).any(1)
    term = query_objective.query_mean(per, active)
    qgap = (gap * positive).flatten(1).sum(1) / positive.flatten(1).sum(1).clamp_min(1)
    return term, {
        "valid_queries": active.sum(),
        "wrong_query_gap": query_objective.query_mean(qgap, active),
        "loss": term,
    }


class Model(support_objective.Model):
    def experiment_contract(self):
        c = super().experiment_contract()
        c.update(
            r54_spec=self.r54_spec,
            r54_inference_uses_gt=False,
            r53_new_head_lr_multiplier=1,
            r54_parent_weights_frozen=bool(self.r54_spec.get("freeze_parent")),
        )
        return c

    def parameter_groups(self):
        groups = super().parameter_groups()
        out = []
        for g in groups:
            ps = [p for p in g.params if p.requires_grad]
            if ps:
                out.append(
                    dataclasses.replace(
                        g,
                        params=ps,
                        lr=g.lr / 20 if g.name == "r53_structure" else g.lr,
                    )
                )
        return out

    def r50_adjust_terms(self, terms, outputs, batch, epoch):
        base = super().r50_adjust_terms(terms, outputs, batch, epoch)
        field = outputs.trifield_output
        spans, gm = losses._target_spans(outputs, batch)
        pairs = candidate_objective.overlap_pairs(field, base.geometry, spans, gm)
        values = {
            k: getattr(base, k)
            for k in ["rank", "evidence", "support", "transition", "endpoint"]
        }
        metrics = dict(base.metrics)
        for kind, flag in [("support", "s"), ("transition", "t")]:
            term, stats = residual_correction(field, pairs, kind)
            if self.r54_spec.get(flag):
                values[kind] = 0.5 * values[kind] + 0.5 * term
            metrics.update(
                {
                    "joint_training/" + kind + "/" + k: v.detach()
                    for k, v in stats.items()
                }
            )
        term, stats = semantic_correction(self, field, base.geometry, batch)
        if self.r54_spec.get("e"):
            values["evidence"] = 0.5 * values["evidence"] + 0.5 * term
        metrics.update(
            {"joint_training/evidence/" + k: v.detach() for k, v in stats.items()}
        )
        total = sum(self.loss_weights[k] * v for k, v in values.items())
        metrics.update({k: v.detach() for k, v in values.items()})
        metrics["total"] = total.detach()
        result = dataclasses.replace(base, **values, total=total, metrics=metrics)
        self.r50_last_terms = (
            result if getattr(self, "r50_capture_terms", False) else None
        )
        return result


def build_model(
    config, *, r54_spec=None, r54_checkpoint=None, r54_sha256=None, **kwargs
):
    spec = dict(r54_spec or {})
    assert not set(spec) - {"e", "s", "t", "freeze_parent"}
    model = support_objective.build_model(config, **kwargs)
    model.__class__ = Model
    model.r54_spec = spec
    if r54_checkpoint:
        p = Path(r54_checkpoint)
        assert hashlib.sha256(p.read_bytes()).hexdigest() == r54_sha256, (
            "R54 checkpoint hash mismatch"
        )
        model.load_state_dict(torch.load(p, map_location="cpu")["model"], strict=True)
    if spec.get("freeze_parent"):
        for p in model.parent_model.parameters():
            p.requires_grad_(False)
    model.r50_probe_dir = str(config.output_dir) + "/r54_gradient_probes"
    return model
