"""R51: controlled objectives and role-specific structures. No GT at inference."""

import dataclasses
import hashlib
import json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from . import candidate_pools
from .model import losses


def query_mean(x, active):
    return x[active].mean() if active.any() else x.sum() * 0


def quality_loss(score, geo, gm, pools, balanced):
    z = 2 * score.flatten(1).float()
    y = geo.max_iou.flatten(1).float()
    union = torch.stack(list(pools.values())).any(0)
    per = F.binary_cross_entropy_with_logits(z, y, reduction="none")
    if not balanced:
        weights = sum(
            m.float() / m.sum(1).clamp_min(1)[:, None] / len(pools)
            for m in pools.values()
        )
    else:
        qs = geo.all_iou.flatten(1, 2).float().masked_fill(~gm[:, None, :], -1)
        owner = qs.argmax(-1)
        pos = union & (y >= 0.5)
        neg = union & ~pos
        assign = (
            (owner[..., None] == torch.arange(gm.shape[1], device=z.device))
            & pos[..., None]
            & gm[:, None, :]
        )
        mass = assign.float() / assign.sum(1).clamp_min(1)[:, None, :]
        active_gt = assign.any(1) & gm
        positive = (mass * active_gt[:, None, :]).sum(-1) / active_gt.sum(1).clamp_min(
            1
        )[:, None]
        negative = neg.float() / neg.sum(1).clamp_min(1)[:, None]
        # Each active GT gets equal positive mass, positives/background each half.
        weights = 0.5 * positive + 0.5 * negative
        weights = weights / weights.sum(1).clamp_min(1e-9)[:, None]
    q = (per * weights).sum(1)
    return q, {
        "weighted_brier": ((z.sigmoid() - y).square() * weights).sum(1).mean().detach(),
        "positive_mass": (weights * (y >= 0.7)).sum(1).mean().detach(),
    }


def precision_pair(score, geo, gm, union):
    """Per-event hard overlapping near-positive candidates; satisfied hinge is zero."""
    z = 2 * score.flatten(1).float()
    q = geo.all_iou.flatten(1, 2).float()
    valid = geo.valid.flatten(1)
    good = q.masked_fill(~valid[..., None], -1).argmax(1)
    pq = q.gather(1, good[:, None, :]).squeeze(1)
    positive = z.gather(1, good)
    # max-over-all-GT rejects windows that accurately cover another annotated event.
    maxq = geo.max_iou.flatten(1).float()
    bad = (
        union[..., None]
        & (q >= 0.5)
        & ((pq[:, None, :] - maxq[..., None]) >= 0.02)
        & gm[:, None, :]
    )
    k = min(8, z.shape[1])
    ids = z[..., None].expand_as(q).masked_fill(~bad, -1e9).topk(k, dim=1).indices
    mask = bad.gather(1, ids)
    nq = maxq[..., None].expand_as(q).gather(1, ids)
    negative = z[..., None].expand_as(q).gather(1, ids)
    gap = positive[:, None, :] - negative
    margin = (pq[:, None, :] - nq).clamp(0.02, 0.5)
    hinge = F.relu(margin - gap)
    pergt = (hinge * mask).sum(1) / mask.sum(1).clamp_min(1)
    ag = mask.any(1) & gm
    perquery = (pergt * ag).sum(1) / ag.sum(1).clamp_min(1)
    return query_mean(perquery, ag.any(1)), {
        "pairs": mask.sum().detach(),
        "gt_coverage": ag.sum().float() / gm.sum().clamp_min(1),
        "inversion_rate": ((gap <= 0) & mask).sum().float() / mask.sum().clamp_min(1),
        "margin_met": ((gap >= margin) & mask).sum().float() / mask.sum().clamp_min(1),
    }


def coverage_loss(score, geo, gm, union):
    """Equal-GT -log probability of at least one precise candidate region."""
    z = 2 * score.flatten(1).float()
    q = geo.all_iou.flatten(1, 2).float()
    logden = z.masked_fill(~union, -1e9).logsumexp(1)
    pos = union[..., None] & (q >= 0.7) & gm[:, None, :]
    num = z[..., None].expand_as(q).masked_fill(~pos, -1e9).logsumexp(1)
    active = pos.any(1) & gm
    pg = (logden[:, None] - num).clamp_min(0)
    pq = (pg * active).sum(1) / active.sum(1).clamp_min(1)
    return query_mean(pq, active.any(1)), {
        "gt_coverage": active.sum().float() / gm.sum().clamp_min(1),
        "probability_mass": (torch.exp(-pg) * active).sum() / active.sum().clamp_min(1),
    }


class CandidateHead(nn.Module):
    """Masked prefix pooling: valid outputs invariant to added batch padding."""

    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        self.net = nn.Sequential(
            nn.Linear((9 if kind == "support" else 5) * 64 + 4, 64),
            nn.GELU(),
            nn.Linear(64, 1 if kind == "support" else 2),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h, valid, token_valid):
        B, L, D = h.shape
        with torch.autocast(device_type=h.device.type, enabled=False):
            h = F.layer_norm(h.float(), (D,)).masked_fill(~token_valid[..., None], 0)
            prefix = F.pad(h.cumsum(1), (0, 0, 1, 0))
            counts = F.pad(token_valid.float().cumsum(1), (1, 0))
            lengths = token_valid.sum(1).clamp_min(1).float()[:, None]

            def pool(a, b):
                n = (counts[:, b] - counts[:, a]).clamp_min(1)
                return (prefix[:, b] - prefix[:, a]) / n[..., None]

            result = []
            for idx in torch.arange(L * L, device=h.device).split(256):
                a = idx // L
                b = torch.maximum(idx % L + 1, a + 1)
                w = b - a
                whole = pool(a, b)
                if self.kind == "support":
                    feats = [whole]
                    for bins in [2, 4]:
                        for i in range(bins):
                            u = a + w * i // bins
                            v = torch.maximum(a + w * (i + 1) // bins, u + 1).clamp_max(
                                L
                            )
                            feats.append(pool(u, v))
                    c = (w // 2).clamp_min(1)
                    feats += [
                        pool((a - c).clamp_min(0), a),
                        pool(b, (b + c).clamp_max(L)),
                    ]
                else:
                    prev = h[:, (a - 1).clamp_min(0)] * (a > 0)[None, :, None]
                    after = h[:, b.clamp_max(L - 1)] * (b < L)[None, :, None]
                    feats = [
                        h[:, a],
                        h[:, b - 1],
                        whole,
                        h[:, a] - prev,
                        after - h[:, b - 1],
                    ]
                geom = torch.stack(
                    [
                        w[None] / lengths,
                        a[None] / lengths,
                        b[None] / lengths,
                        torch.log1p(w.float())[None] / torch.log1p(lengths),
                    ],
                    -1,
                )
                result.append(self.net(torch.cat(feats + [geom], -1)))
            v = (
                torch.cat(result, 1)
                .reshape(B, L, L, -1)
                .masked_fill(~valid[..., None], 0)
            )
            return v[..., 0] if self.kind == "support" else v


def directional_boundary_loss(field, geo, gm):
    """Separate start/end one-sided interventions; never train S with this term."""
    q = geo.all_iou.flatten(1, 2)
    L = field.score.shape[1]
    valid = geo.valid.flatten(1)
    good = q.masked_fill(~valid[..., None], -1).argmax(1)
    start = good // L
    end = good % L
    width = end - start + 1
    pq = q.gather(1, good[:, None, :]).squeeze(1)
    maxq = geo.max_iou.flatten(1)
    stats = {}
    terms = []
    for name, value in [
        ("start", field.transition_start),
        ("end", field.transition_end),
    ]:
        z = value.flatten(1).float()
        p = z.gather(1, good)
        ls = []
        ms = []
        gs = []
        margins = []
        for step in [torch.ones_like(width), (width // 4).clamp_min(1)]:
            for sign in [-1, 1]:
                a = start + sign * step if name == "start" else start
                b = end + sign * step if name == "end" else end
                index = a.clamp(0, L - 1) * L + b.clamp(0, L - 1)
                mask = (
                    (a >= 0)
                    & (b < L)
                    & (a <= b)
                    & gm
                    & valid.gather(1, index)
                    & ((pq - maxq.gather(1, index)) > 0.001)
                )
                gap = p - z.gather(1, index)
                margin = (step.float() / width).clamp(0.02, 0.25)
                ls.append(F.relu(margin - gap))
                ms.append(mask)
                gs.append(gap)
                margins.append(margin)
        mask = torch.stack(ms, -1)
        gap = torch.stack(gs, -1)
        margin = torch.stack(margins, -1)
        pergt = (torch.stack(ls, -1) * mask).sum(-1) / mask.sum(-1).clamp_min(1)
        ag = mask.any(-1) & gm
        term = query_mean((pergt * ag).sum(1) / ag.sum(1).clamp_min(1), ag.any(1))
        terms.append(term)
        stats[name + "_pairs"] = mask.sum().detach()
        stats[name + "_correct"] = (
            (gap > 0) & mask
        ).sum().float() / mask.sum().clamp_min(1)
        stats[name + "_margin_met"] = (
            (gap >= margin) & mask
        ).sum().float() / mask.sum().clamp_min(1)
    return sum(terms) / 2, stats


class BoundedMultiScale(candidate_pools.MultiScale):
    def forward(self, h, pad):
        raw = super().forward(h, pad)
        # Bounded relative residual, zero-initialized and masked, no free amplitude.
        scale = (
            h.detach().float().square().mean(-1, keepdim=True).sqrt().clamp_min(0.01)
        )
        return (0.1 * scale * torch.tanh(raw.float() / scale)).masked_fill(
            pad[..., None], 0
        )


class Selector(candidate_pools.R50Selector):
    def raw_from_state(self, state, valid, carrier, video_padding_mask=None):
        f = super().raw_from_state(state, valid, carrier, video_padding_mask)
        if hasattr(self, "r51_transition"):
            v = self.r51_transition(
                f.round31_z_t, valid, self._token_valid(valid, video_padding_mask)
            )
            f.raw_transition_start = f.raw_transition_start + 0.5 * v[..., 0]
            f.raw_transition_end = f.raw_transition_end + 0.5 * v[..., 1]
            f.transition_start = torch.tanh(f.raw_transition_start).masked_fill(
                ~valid, 0
            )
            f.transition_end = torch.tanh(f.raw_transition_end).masked_fill(~valid, 0)
            f.score = self.compose_score(f)
            f.round31_rank_score = f.score
            f.round31_score_with_support = f.score
            f.round31_score_without_support = f.score - f.support
        return f


class Model(candidate_pools.R50Model):
    def experiment_contract(self):
        c = super().experiment_contract()
        c.update(r51_spec=self.r51_spec, r51_inference_uses_gt=False)
        return c

    def r50_adjust_terms(self, terms, outputs, batch, epoch):
        base = super().r50_adjust_terms(terms, outputs, batch, epoch)
        f = outputs.trifield_output
        g = base.geometry
        _, gm = losses._target_spans(outputs, batch)
        pools = candidate_pools.candidate_pools(f, g, gm, True)
        union = torch.stack(list(pools.values())).any(0)
        values = {
            k: getattr(base, k)
            for k in ["rank", "evidence", "support", "transition", "endpoint"]
        }
        stats = dict(base.metrics)
        spec = self.r51_spec
        mode = spec.get("quality", "none")
        if mode != "none":
            q, diag = quality_loss(f.score, g, gm, pools, balanced=mode != "source")
            length_weight = (
                1
                + self.rank_length_reweight_gain
                * losses._query_length_ratio(outputs, batch)
            )
            bce = query_mean(q * length_weight, gm.any(1))
            restricted = dataclasses.replace(g, valid=union.reshape_as(g.valid))
            _, _, kq, active = losses.gt_balanced_kl_rank_loss(
                2 * f.score.float(), restricted, gm, return_queries=True
            )
            kl = query_mean(kq * length_weight, active)
            alpha = 1.0 if mode in ("balanced", "source") else 0.25
            # Preserve old pair and query weighting; isolate relative-vs-quality balance.
            values["rank"] = base.rank + alpha * (bce - kl)
            stats.update({"query_objective/quality/" + k: v for k, v in diag.items()})
            stats["query_objective/quality/alpha"] = alpha
        if spec.get("precision"):
            term, diag = precision_pair(f.score, g, gm, union)
            values["rank"] = values["rank"] + 0.25 * term
            stats.update({"query_objective/precision/" + k: v for k, v in diag.items()})
            stats["query_objective/precision/loss"] = term.detach()
        if spec.get("coverage"):
            term, diag = coverage_loss(f.score, g, gm, union)
            values["rank"] = values["rank"] + 0.1 * term
            stats.update({"query_objective/coverage/" + k: v for k, v in diag.items()})
            stats["query_objective/coverage/loss"] = term.detach()
        if spec.get("query_e"):
            wrong = self._last_wrong_query_field
            positive = g.valid & (g.max_iou >= 0.7)
            gap = ((f.evidence - wrong.evidence) * positive).flatten(1).sum(
                1
            ) / positive.flatten(1).sum(1).clamp_min(1)
            term = query_mean(F.relu(0.1 - gap), positive.flatten(1).any(1))
            values["evidence"] = 0.5 * base.evidence + 0.5 * term
            stats["query_objective/e_query/loss"] = term.detach()
            stats["query_objective/e_query/gap"] = gap.detach().mean()
        if spec.get("directional_t"):
            values["transition"], diag = directional_boundary_loss(f, g, gm)
            stats.update(
                {"query_objective/directional/" + k: v for k, v in diag.items()}
            )
        for name in ["e_interaction", "s_projection", "t_interaction"]:
            m = getattr(self.selector, name)
            if hasattr(m, "r50_delta_rms"):
                stats["query_objective/structure/" + name + "_delta_rms"] = (
                    m.r50_delta_rms
                )
        total = sum(self.loss_weights[k] * v for k, v in values.items())
        stats.update({k: v.detach() for k, v in values.items()})
        stats["total"] = total.detach()
        # Name the source that actually survives; inherited diagnostics are references.
        stats["query_objective/actual_objective"] = 1.0
        result = dataclasses.replace(base, **values, total=total, metrics=stats)
        self.r50_last_terms = (
            result if getattr(self, "r50_capture_terms", False) else None
        )
        return result

    def r48_loss_probe(self, terms, epoch):
        if (
            not self.training
            or not torch.is_grad_enabled()
            or getattr(self, "r51_probe_epoch", None) == epoch
        ):
            return {}
        self.r51_probe_epoch = epoch
        result = {
            "epoch": epoch,
            "scope": "first actual training batch, same-parameter weighted gradients",
            "groups": {},
        }
        for group in [
            "e_interaction",
            "s_projection",
            "t_interaction",
            "r50_support",
            "r51_transition",
        ]:
            params = [
                p
                for n, p in self.selector.named_parameters()
                if group in n and p.requires_grad
            ]
            if not params:
                continue
            vectors = {}
            entry = {}
            for name in ["rank", "evidence", "support", "transition", "endpoint"]:
                gs = torch.autograd.grad(
                    getattr(terms, name) * self.loss_weights[name],
                    params,
                    retain_graph=True,
                    allow_unused=True,
                )
                vectors[name] = torch.cat(
                    [
                        torch.zeros_like(p).flatten()
                        if v is None
                        else v.detach().float().flatten()
                        for p, v in zip(params, gs)
                    ]
                )
                entry[name] = {
                    "norm": float(vectors[name].norm()),
                    "none_fraction": sum(v is None for v in gs) / len(gs),
                }
            a = vectors["rank"]
            entry["rank_cosine"] = {
                n: float((a @ v) / (a.norm() * v.norm()).clamp_min(1e-12))
                for n, v in vectors.items()
                if n != "rank"
            }
            result["groups"][group] = entry
        dest = Path(self.r50_probe_dir)
        dest.mkdir(parents=True, exist_ok=True)
        (dest / f"gradient_e{epoch:03d}.json").write_text(json.dumps(result, indent=2))
        return {}


def build_model(
    config, *, r51_spec=None, r51_init_checkpoint=None, r51_init_sha256=None, **kwargs
):
    spec = dict(r51_spec or {})
    allowed = {
        "quality",
        "precision",
        "coverage",
        "query_e",
        "support",
        "split_t",
        "directional_t",
        "bounded_multiscale",
        "input_budget",
    }
    if set(spec) - allowed:
        raise ValueError(spec)
    if spec.get("directional_t") and not spec.get("split_t"):
        raise ValueError("directional T requires independent endpoints")
    kwargs.pop("r50_spec", None)
    if "input_budget" in spec:
        kwargs["max_update_ratio"] = float(spec["input_budget"])
    model = candidate_pools.build_model(
        config,
        r50_spec={
            "mixed": True,
            "support": bool(spec.get("support")),
            "support_loss": bool(spec.get("support")),
            "multiscale": bool(spec.get("bounded_multiscale")),
        },
        **kwargs,
    )
    model.__class__ = Model
    model.r51_spec = spec
    model.selector.__class__ = Selector
    model.r50_probe_dir = str(config.output_dir) + "/r51_gradient_probes"
    for key, kind, seed in [
        ("support", "support", 5101),
        ("split_t", "transition", 5102),
    ]:
        if spec.get(key):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                head = CandidateHead(kind)
            setattr(
                model.selector,
                "r50_support" if key == "support" else "r51_transition",
                head,
            )
    if spec.get("bounded_multiscale"):
        for name in ["e_interaction", "s_projection", "t_interaction"]:
            getattr(model.selector, name).r50_multiscale.__class__ = BoundedMultiScale
    if r51_init_checkpoint:
        path = Path(r51_init_checkpoint)
        if hashlib.sha256(path.read_bytes()).hexdigest() != r51_init_sha256:
            raise ValueError("initial checkpoint hash mismatch")
        state = torch.load(path, map_location="cpu")["model"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise ValueError({"missing": missing, "unexpected": unexpected})
    return model
