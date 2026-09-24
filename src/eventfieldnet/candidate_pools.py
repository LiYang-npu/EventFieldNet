"""R50 structural field models; all inference paths depend only on inputs."""

import dataclasses
import json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from . import score_calibration
from .model import losses


def top_mask(score, valid, k):
    ids = (
        score.detach()
        .masked_fill(~valid, -torch.inf)
        .topk(min(k, score.shape[1]), 1)
        .indices
    )
    return torch.zeros_like(valid).scatter_(1, ids, True) & valid


@torch.no_grad()
def nms_mask(score, valid, length, k=30):
    """Batched greedy NMS, inclusive clip windows interpreted as [i,j+1]."""
    device = score.device
    idx = torch.arange(length * length, device=device)
    starts = (idx // length).float()
    ends = (idx % length + 1).float()
    alive = valid.clone()
    chosen = torch.zeros_like(valid)
    for _ in range(min(k, score.shape[1])):
        any_valid = alive.any(1)
        pick = score.detach().masked_fill(~alive, -torch.inf).argmax(1)
        chosen |= torch.zeros_like(valid).scatter_(1, pick[:, None], any_valid[:, None])
        inter = (
            torch.minimum(ends[None], ends[pick, None])
            - torch.maximum(starts[None], starts[pick, None])
        ).clamp_min(0)
        union = (ends - starts)[None] + (ends[pick] - starts[pick])[:, None] - inter
        overlap = inter / union.clamp_min(1e-6)
        alive &= (overlap <= 0.5) & any_valid[:, None]
    return chosen & valid


def candidate_pools(field, geometry, gt_mask, mixed):
    score = field.score.float().flatten(1)
    valid = geometry.valid.flatten(1).bool()
    L = field.score.shape[1]
    raw = top_mask(score, valid, 30)
    if not mixed:
        return {"raw": raw}
    pools = {"raw": raw, "nms": nms_mask(score, valid, L)}
    idx = torch.arange(L * L, device=score.device)
    width = idx % L - idx // L + 1
    # Deterministic coverage across normalized duration bins, independent of score.
    # No RNG changes: same initialization/data RNG as controls.
    strat = torch.zeros_like(valid)
    for lo, hi in [(0, 0.1), (0.1, 0.3), (0.3, 1.01)]:
        eligible = valid & (width[None] > lo * L) & (width[None] <= hi * L)
        counts = eligible.sum(1)
        order = eligible.long().cumsum(1) - 1
        for t in range(8):
            target = ((counts - 1).clamp_min(0) * t // 7)[:, None]
            strat |= eligible & (order == target)
    pools["stratified"] = strat
    quality = geometry.all_iou.float().flatten(1, 2).masked_fill(~valid[..., None], -1)
    ids = quality.topk(min(4, quality.shape[1]), dim=1).indices
    positive = torch.zeros_like(valid, dtype=torch.long)
    positive.scatter_add_(
        1, ids.flatten(1), gt_mask[:, None, :].expand_as(ids).flatten(1).long()
    )
    pools["gt_neighborhood"] = positive.gt(0) & valid
    return pools


def pool_mean(h, prefix, start, end):
    # start/end are candidate vectors, [start,end), clamped to sequence length.
    return (prefix[:, end] - prefix[:, start]) / (end - start).clamp_min(1)[
        None, :, None
    ]


class CandidateHead(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        dims = 9 * 64 + 4 if kind == "support" else 5 * 64 + 4
        self.net = nn.Sequential(nn.Linear(dims, 64), nn.GELU(), nn.Linear(64, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h, valid, token_valid):
        B, L, D = h.shape
        with torch.autocast(device_type=h.device.type, enabled=False):
            h = F.layer_norm(h.float(), (D,)).masked_fill(~token_valid[..., None], 0)
            prefix = F.pad(h.cumsum(1), (0, 0, 1, 0))
            idx = torch.arange(L * L, device=h.device)
            outputs = []
            for chunk in idx.split(256):
                start = chunk // L
                end = chunk % L + 1
                width = (end - start).clamp_min(1)
                end = torch.maximum(end, start + 1)
                whole = pool_mean(h, prefix, start, end)
                if self.kind == "support":
                    feats = [whole]
                    for bins in [2, 4]:
                        for part in range(bins):
                            a = start + width * part // bins
                            b = start + width * (part + 1) // bins
                            b = torch.maximum(b, a + 1).clamp_max(L)
                            feats.append(pool_mean(h, prefix, a, b))
                    context = (width // 2).clamp_min(1)
                    feats += [
                        pool_mean(h, prefix, (start - context).clamp_min(0), start),
                        pool_mean(h, prefix, end, (end + context).clamp_max(L)),
                    ]
                else:
                    left = h[:, start]
                    right = h[:, end - 1]
                    feats = [left, right, whole, left * right, right - left]
                geom = torch.stack(
                    [
                        width.float() / L,
                        start.float() / L,
                        end.float() / L,
                        torch.log1p(width.float()) / torch.log(h.new_tensor(L + 1.0)),
                    ],
                    -1,
                )
                z = torch.cat(feats + [geom[None].expand(B, -1, -1)], -1)
                outputs.append(self.net(z).squeeze(-1))
            return torch.cat(outputs, 1).reshape(B, L, L).masked_fill(~valid, 0)


class MultiScale(nn.Module):
    def __init__(self):
        super().__init__()
        self.branches = nn.ModuleList(
            [nn.Conv1d(64, 64, 3, padding=d, dilation=d, groups=64) for d in [1, 4, 12]]
        )
        self.mix = nn.Linear(64 * 3, 64)
        nn.init.zeros_(self.mix.weight)
        nn.init.zeros_(self.mix.bias)

    def forward(self, h, pad):
        x = h.masked_fill(pad[..., None], 0).transpose(1, 2)
        return self.mix(
            torch.cat([F.gelu(m(x)).transpose(1, 2) for m in self.branches], -1)
        ).masked_fill(pad[..., None], 0)


class R50Encode:
    def encode(self, video, text, video_pad, text_pad):
        h = super().encode(video, text, video_pad, text_pad)
        if hasattr(self, "r50_multiscale"):
            delta = self.r50_multiscale(h, video_pad)
            self.r50_delta_rms = delta.detach().float().square().mean().sqrt()
            h = h + delta
        return h


class R50Interaction(R50Encode, score_calibration.R48Interaction):
    pass


class R50SProjection(R50Encode, score_calibration.R48SProjection):
    pass


class R50Selector(score_calibration.R48Selector):
    def raw_from_state(self, state, valid, carrier, video_padding_mask=None):
        f = super().raw_from_state(
            state, valid, carrier, video_padding_mask=video_padding_mask
        )
        token_valid = self._token_valid(valid, video_padding_mask)
        if hasattr(self, "r50_support"):
            residual = self.r50_support(f.round31_z_s, valid, token_valid)
            f.r50_s_residual = residual
            f.support = (
                f.support + 0.5 * (1 - f.support.abs()) * torch.tanh(residual)
            ).masked_fill(~valid, 0)
            f.round31_deployed_support = f.support
        if hasattr(self, "r50_transition"):
            residual = self.r50_transition(f.round31_z_t, valid, token_valid)
            f.r50_t_residual = residual
            f.raw_transition_start = f.raw_transition_start + 0.5 * residual
            f.raw_transition_end = f.raw_transition_end + 0.5 * residual
            f.transition_start = torch.tanh(f.raw_transition_start).masked_fill(
                ~valid, 0
            )
            f.transition_end = torch.tanh(f.raw_transition_end).masked_fill(~valid, 0)
        f.score = self.compose_score(f)
        f.round31_rank_score = f.score
        f.round31_score_with_support = f.score
        f.round31_score_without_support = f.score - f.support
        return f


def structural_pair_loss(value, geometry, gtmask, kind):
    """Per-GT real-window interventions, filtered against *all* annotated GTs."""
    B, L, _ = value.shape
    L * L
    gtmask.shape[1]
    valid = geometry.valid.flatten(1)
    allq = geometry.all_iou.float().flatten(1, 2)
    quality = geometry.max_iou.float().flatten(1)
    best = allq.masked_fill(~valid[..., None], -1).argmax(1)
    start = best // L
    end = best % L
    width = end - start + 1
    step = (width // 4).clamp_min(1)
    perturb = [
        (start + step, end),
        (start, end - step),
        (start - step, end),
        (start, end + step),
        (start - step, end - step),
        (start + step, end + step),
    ]
    if kind == "transition":
        perturb += [(start, end.roll(1, 1)), (start.roll(1, 1), end)]
    pos = value.flatten(1).gather(1, best)
    pq = allq.gather(1, best[:, None, :]).squeeze(1)
    per = []
    masks = []
    gaps = []
    for a, b in perturb:
        inbounds = (a >= 0) & (b < L) & (a <= b) & gtmask
        ind = a.clamp(0, L - 1) * L + b.clamp(0, L - 1)
        neg = value.flatten(1).gather(1, ind)
        nq = quality.gather(1, ind)
        delta = (pq - nq).detach()
        mask = inbounds & valid.gather(1, ind) & (pq >= 0.7) & (delta >= 0.15)
        gap = pos - neg
        per.append(F.softplus(delta.clamp(0, 0.5) - gap))
        masks.append(mask)
        gaps.append(gap)
    mask = torch.stack(masks, -1)
    values = torch.stack(per, -1)
    gap = torch.stack(gaps, -1)
    pergt = (values * mask).sum(-1) / mask.sum(-1).clamp_min(1)
    active = mask.any(-1)
    perquery = (pergt * active).sum(1) / active.sum(1).clamp_min(1)
    aq = active.any(1)
    loss = perquery[aq].mean() if aq.any() else value.sum() * 0
    stats = {
        "pairs": mask.sum().detach(),
        "gt_coverage": (active.sum() / gtmask.sum().clamp_min(1)).detach(),
        "positive_gap_rate": ((gap > 0) & mask).sum().float() / mask.sum().clamp_min(1),
        "gap_mean": (gap.detach() * mask).sum() / mask.sum().clamp_min(1),
    }
    return loss, stats


class R50Model(score_calibration.R48Model):
    def experiment_contract(self):
        c = dict(super().experiment_contract())
        c.update(
            r50_spec=self.r50_spec,
            r50_inference_uses_gt=False,
            r50_loss_probe="actual final adjusted objective; legacy probe reports explicitly reference-only",
        )
        if self.r50_spec.get("support"):
            c["support"] = "edge S plus bounded candidate pyramid/context residual"
        if self.r50_spec.get("joint_t"):
            c["transition"] = (
                "coupled start/end residual conditioned on both endpoints and interior"
            )
        return c

    def r50_adjust_terms(self, terms, outputs, batch, epoch):
        spec = self.r50_spec
        field = outputs.trifield_output
        geo = terms.geometry
        _, gm = losses._target_spans(outputs, batch)
        metrics = dict(terms.metrics)
        values = {
            k: getattr(terms, k)
            for k in ["rank", "evidence", "support", "transition", "endpoint"]
        }
        pools = (
            candidate_pools(field, geo, gm, spec.get("mixed", False))
            if (spec.get("mixed") or spec.get("quality"))
            else None
        )
        if pools:
            union = torch.stack(list(pools.values())).any(0)
            metrics["candidate_pools/pool/count"] = union.sum(1).float().mean().detach()
            for k, m in pools.items():
                metrics["candidate_pools/pool/" + k + "_count"] = (
                    m.sum(1).float().mean().detach()
                )
            if spec.get("quality"):
                y = geo.max_iou.float().flatten(1).detach()
                z = 2 * field.score.float().flatten(1)
                bce = F.binary_cross_entropy_with_logits(z, y, reduction="none")
                # Equal source contribution; duplicates may appear across sources by design.
                query = torch.stack(
                    [(bce * m).sum(1) / m.sum(1).clamp_min(1) for m in pools.values()]
                ).mean(0)
                active = gm.any(1)
                new = query[active].mean() if active.any() else z.sum() * 0
                alpha = min(1.0, max(0.0, epoch / 5.0))
                values["rank"] = (1 - alpha) * terms.rank + alpha * new
                metrics.update(
                    {
                        "candidate_pools/quality/loss": new.detach(),
                        "candidate_pools/quality/mix": alpha,
                        "candidate_pools/quality/brier": (
                            ((z.sigmoid() - y).square() * union).sum()
                            / union.sum().clamp_min(1)
                        ).detach(),
                        "candidate_pools/quality/target_mean": (
                            (y * union).sum() / union.sum().clamp_min(1)
                        ).detach(),
                    }
                )
            else:
                restricted = dataclasses.replace(geo, valid=union.reshape_as(geo.valid))
                _, _, q, active = losses.gt_balanced_kl_rank_loss(
                    field.score.float() * 2, restricted, gm, return_queries=True
                )
                q = q * (
                    1
                    + self.rank_length_reweight_gain
                    * losses._query_length_ratio(outputs, batch)
                )
                kl = q[active].mean() if active.any() else q.sum() * 0
                old = field.round31_candidate_pair_probe["rank"]["kl_component"]
                values["rank"] = terms.rank - old + kl
                metrics["candidate_pools/mixed_kl"] = kl.detach()
            full = geo.all_iou.flatten(1, 2)
            used = full.masked_fill(~union[..., None], -1).amax(1)
            metrics["candidate_pools/pool/gt_recall07"] = (
                (used >= 0.7) & gm
            ).sum().float() / gm.sum().clamp_min(1)
        for key, kind in [("support_loss", "support"), ("joint_t_loss", "transition")]:
            if spec.get(key):
                value = (
                    field.support
                    if kind == "support"
                    else 0.5 * (field.transition_start + field.transition_end)
                )
                values[kind], stats = structural_pair_loss(value, geo, gm, kind)
                metrics.update(
                    {"candidate_pools/" + kind + "/" + k: v for k, v in stats.items()}
                )
        total = sum(self.loss_weights[k] * v for k, v in values.items())
        for k in ["rank", "support", "transition"]:
            if values[k] is not getattr(terms, k):
                for old in list(metrics):
                    if old.startswith(k + "/") or old.startswith(
                        "candidate_" + k + "/"
                    ):
                        metrics["legacy_reference/" + old] = metrics.pop(old)
        metrics.update({k: v.detach() for k, v in values.items()})
        metrics["total"] = total.detach()
        metrics["candidate_pools/recomposition_error"] = (
            (total - sum(self.loss_weights[k] * v for k, v in values.items()))
            .abs()
            .detach()
        )
        for attr in ["r50_s_residual", "r50_t_residual"]:
            if hasattr(field, attr):
                metrics["candidate_pools/" + attr + "/abs_mean"] = (
                    getattr(field, attr).detach().abs()[geo.valid].mean()
                )
        adjusted = dataclasses.replace(terms, **values, total=total, metrics=metrics)
        self.r50_last_terms = (
            adjusted if getattr(self, "r50_capture_terms", False) else None
        )
        return adjusted

    def r48_loss_probe(self, terms, epoch):
        if (
            not self.training
            or not torch.is_grad_enabled()
            or getattr(self, "r50_probe_epoch", None) == epoch
        ):
            return {}
        self.r50_probe_epoch = epoch
        groups = {
            k: [
                p
                for n, p in self.selector.named_parameters()
                if p.requires_grad and k in n
            ]
            for k in [
                "e_interaction",
                "s_projection",
                "t_interaction",
                "r50_support",
                "r50_transition",
            ]
        }
        data = {
            "scope": "first training batch, actual weighted objective",
            "epoch": epoch,
            "groups": {},
        }
        for group, params in groups.items():
            if not params:
                continue
            data["groups"][group] = {}
            for term in ["rank", "evidence", "support", "transition", "endpoint"]:
                grads = torch.autograd.grad(
                    getattr(terms, term) * self.loss_weights[term],
                    params,
                    retain_graph=True,
                    allow_unused=True,
                )
                norm = sum(
                    g.detach().float().square().sum() for g in grads if g is not None
                )
                data["groups"][group][term] = {
                    "norm": float(torch.sqrt(norm))
                    if isinstance(norm, torch.Tensor)
                    else 0.0,
                    "none_fraction": sum(g is None for g in grads) / len(params),
                }
        if self.r50_probe_dir:
            p = Path(self.r50_probe_dir)
            p.mkdir(parents=True, exist_ok=True)
            (p / f"gradient_e{epoch:03d}.json").write_text(json.dumps(data, indent=2))
        return {}


def build_model(config, *, r50_spec=None, **kwargs):
    spec = dict(r50_spec or {})
    allowed = {
        "mixed",
        "quality",
        "support",
        "support_loss",
        "joint_t",
        "joint_t_loss",
        "multiscale",
    }
    if set(spec) - allowed:
        raise ValueError(spec)
    if spec.get("support_loss") and not spec.get("support"):
        raise ValueError("support_loss requires complete support")
    if spec.get("joint_t_loss") and not spec.get("joint_t"):
        raise ValueError("joint T loss requires joint T")
    model = score_calibration.build_model(config, r48_spec={}, **kwargs)
    model.__class__ = R50Model
    model.r50_spec = spec
    model.r50_probe_dir = str(config.output_dir) + "/r50_gradient_probes"
    model.selector.__class__ = R50Selector
    for key, kind, seed in [
        ("support", "support", 5001),
        ("joint_t", "transition", 5002),
    ]:
        if spec.get(key):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                setattr(model.selector, "r50_" + kind, CandidateHead(kind))
    for i, name in enumerate(["e_interaction", "s_projection", "t_interaction"]):
        m = getattr(model.selector, name)
        m.__class__ = R50SProjection if name == "s_projection" else R50Interaction
        if spec.get("multiscale"):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(5010 + i)
                m.r50_multiscale = MultiScale()
    return model
