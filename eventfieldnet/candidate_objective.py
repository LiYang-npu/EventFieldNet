"""Overlap-conditioned field roles. GT is used only in training/probes, never inference."""

import dataclasses
import torch
from torch.nn import functional as F
from . import candidate_pools, query_objective
from .model import losses


def masked_mean(x, mask):
    # Equal query, then equal active GT, then equal pair; short events may have no pair.
    pg = (x * mask).sum(1) / mask.sum(1).clamp_min(1)
    ag = mask.any(1)
    pq = (pg * ag).sum(1) / ag.sum(1).clamp_min(1)
    return query_objective.query_mean(pq, ag.any(1))


@torch.no_grad()
def overlap_pairs(field, geo, spans, gm):
    B, L, _ = field.score.shape
    v = geo.valid.flatten(1)
    q = geo.all_iou.float().flatten(1, 2)
    good = q.masked_fill(~v[..., None], -1).argmax(1)
    pq = q.gather(1, good[:, None, :]).squeeze(1)
    cs = geo.candidate_start.flatten(1)
    ce = geo.candidate_end.flatten(1)
    ps = cs.gather(1, good)
    pe = ce.gather(1, good)
    inter = (
        torch.minimum(ce[..., None], pe[:, None])
        - torch.maximum(cs[..., None], ps[:, None])
    ).clamp_min(0)
    ov = inter / ((ce - cs)[..., None] + (pe - ps)[:, None] - inter).clamp_min(1e-8)
    maxq = geo.max_iou.flatten(1)
    pools = candidate_pools.candidate_pools(field, geo, gm, True)
    union = torch.stack(list(pools.values())).any(0)
    # Add deterministic endpoint perturbations, not just score-selected top windows.
    a = good // L
    b = good % L
    w = b - a + 1
    for step in [torch.ones_like(w), (w // 8).clamp_min(1)]:
        for da, db in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1)]:
            aa = a + da * step
            bb = b + db * step
            ok = (aa >= 0) & (bb < L) & (aa <= bb) & gm
            ids = aa.clamp(0, L - 1) * L + bb.clamp(0, L - 1)
            extra = torch.zeros_like(v, dtype=torch.long)
            extra.scatter_add_(1, ids, ok.long())
            union |= extra.gt(0) & v
    mask = (
        union[..., None]
        & v[..., None]
        & gm[:, None, :]
        & (ov >= 0.7)
        & (q >= 0.7)
        & ((pq[:, None] - maxq[..., None]) >= 0.02)
    )
    # Do not demote candidates that accurately match a different annotated event.
    mask &= maxq[..., None] <= q + 0.02
    k = min(8, L * L)
    ids = (
        field.score.detach()
        .flatten(1)[..., None]
        .expand_as(q)
        .masked_fill(~mask, -1e9)
        .topk(k, dim=1)
        .indices
    )
    selected = mask.gather(1, ids)
    gs = torch.minimum(spans[..., 0], spans[..., 1])
    ge = torch.maximum(spans[..., 0], spans[..., 1])
    width = (ge - gs).clamp_min(1e-8)

    def targets(s, e):
        intersection = (
            torch.minimum(e, ge[:, None]) - torch.maximum(s, gs[:, None])
        ).clamp_min(0)
        return (
            intersection / width[:, None],
            (1 - (s - gs[:, None]).abs() / width[:, None]).clamp(0, 1),
            (1 - (e - ge[:, None]).abs() / width[:, None]).clamp(0, 1),
        )

    ns = cs[..., None].expand_as(q).gather(1, ids)
    ne = ce[..., None].expand_as(q).gather(1, ids)
    pt = targets(ps[:, None], pe[:, None])
    nt = targets(ns, ne)
    return {
        "good": good,
        "ids": ids,
        "mask": selected,
        "quality_delta": pq[:, None] - q.gather(1, ids),
        "support_delta": pt[0] - nt[0],
        "start_delta": pt[1] - nt[1],
        "end_delta": pt[2] - nt[2],
        "overlap": ov.gather(1, ids),
        "gt_mask": gm,
    }


def pair_gap(value, pairs):
    z = value.float().flatten(1)
    p = z.gather(1, pairs["good"])[:, None]
    n = z[..., None].expand(-1, -1, pairs["good"].shape[1]).gather(1, pairs["ids"])
    return p - n


def field_role_terms(field, pairs, selector):
    mask = pairs["mask"]
    gaps = {
        k: pair_gap(getattr(field, k), pairs)
        for k in [
            "evidence",
            "support",
            "transition_start",
            "transition_end",
            "carrier",
            "score",
        ]
    }
    # Forward score is exactly the deployed score; only the local gradient route changes.
    st_score = selector.compose_score(
        field, {"evidence": field.evidence.detach(), "carrier": field.carrier.detach()}
    )
    stgap = pair_gap(st_score, pairs)
    terms = {
        "shared_e": masked_mean(gaps["evidence"].square(), mask),
        "coverage_s": masked_mean(
            F.smooth_l1_loss(
                gaps["support"], pairs["support_delta"], reduction="none", beta=0.1
            ),
            mask,
        ),
        "boundary_t": 0.5
        * sum(
            masked_mean(
                F.smooth_l1_loss(gaps[n], pairs[t], reduction="none", beta=0.1), mask
            )
            for n, t in [
                ("transition_start", "start_delta"),
                ("transition_end", "end_delta"),
            ]
        ),
        "local_all": masked_mean(
            F.smooth_l1_loss(
                2 * gaps["score"], pairs["quality_delta"], reduction="none", beta=0.1
            ),
            mask,
        ),
        "local_st": masked_mean(
            F.smooth_l1_loss(
                2 * stgap, pairs["quality_delta"], reduction="none", beta=0.1
            ),
            mask,
        ),
    }
    stats = {
        "pairs": mask.sum().detach(),
        "gt_coverage": mask.any(1).sum().float() / pairs["gt_mask"].sum().clamp_min(1),
        "overlap_mean": masked_mean(pairs["overlap"], mask),
        "quality_delta": masked_mean(pairs["quality_delta"], mask),
        "inversion_rate": masked_mean((gaps["score"] <= 0).float(), mask),
        "shared_e_abs_gap": masked_mean(gaps["evidence"].abs(), mask),
        "forward_route_max_error": (st_score - field.score).abs().max(),
        "truncation_fraction": masked_mean(
            (pairs["support_delta"] > 0.001).float(), mask
        ),
    }
    for name, gap in gaps.items():
        stats[name + "_gap"] = masked_mean(gap, mask)
        weight = 0.5 if name.startswith("transition") else 1.0
        stats["remove_" + name + "_inversion"] = masked_mean(
            ((gaps["score"] - weight * gap) <= 0).float(), mask
        )
    for name, t in terms.items():
        stats[name + "_loss"] = t.detach()
    return terms, stats


class Model(query_objective.Model):
    def experiment_contract(self):
        c = super().experiment_contract()
        c.update(r52_spec=self.r52_spec, r52_inference_uses_gt=False)
        return c

    def r50_adjust_terms(self, terms, outputs, batch, epoch):
        base = super().r50_adjust_terms(terms, outputs, batch, epoch)
        spans, gm = losses._target_spans(outputs, batch)
        pairs = overlap_pairs(outputs.trifield_output, base.geometry, spans, gm)
        local, diag = field_role_terms(outputs.trifield_output, pairs, self.selector)
        values = {
            k: getattr(base, k)
            for k in ["rank", "evidence", "support", "transition", "endpoint"]
        }
        spec = self.r52_spec
        if spec.get("local_rank"):
            values["rank"] = (
                values["rank"] + 0.25 * local["local_" + spec["local_rank"]]
            )
        if spec.get("shared_e"):
            values["evidence"] = 0.75 * values["evidence"] + 0.25 * local["shared_e"]
        if spec.get("coverage_s"):
            values["support"] = 0.5 * values["support"] + 0.5 * local["coverage_s"]
        if spec.get("boundary_t"):
            values["transition"] = (
                0.5 * values["transition"] + 0.5 * local["boundary_t"]
            )
        total = sum(self.loss_weights[k] * v for k, v in values.items())
        metrics = dict(base.metrics)
        metrics.update(
            {
                "candidate_objective/overlap/" + k: v.detach()
                if isinstance(v, torch.Tensor)
                else v
                for k, v in diag.items()
            }
        )
        metrics.update({k: v.detach() for k, v in values.items()})
        metrics["total"] = total.detach()
        result = dataclasses.replace(base, **values, total=total, metrics=metrics)
        self.r50_last_terms = (
            result if getattr(self, "r50_capture_terms", False) else None
        )
        return result


def build_model(config, *, r52_spec=None, **kwargs):
    spec = dict(r52_spec or {})
    assert not set(spec) - {"local_rank", "shared_e", "coverage_s", "boundary_t"}
    assert spec.get("local_rank") in (None, "all", "st")
    model = query_objective.build_model(config, **kwargs)
    model.__class__ = Model
    model.r52_spec = spec
    model.r50_probe_dir = str(config.output_dir) + "/r52_gradient_probes"
    return model
