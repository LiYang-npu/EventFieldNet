"""R53 orthogonal readout/structure experiments; unchanged F03 objective."""

import dataclasses, json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from . import query_objective, candidate_objective


def zero_mlp(d, out=1):
    m = nn.Sequential(nn.Linear(d, 32), nn.GELU(), nn.Linear(32, out))
    nn.init.zeros_(m[-1].weight)
    nn.init.zeros_(m[-1].bias)
    return m


class Readout(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        self.net = zero_mlp(
            {"ring": 386, "boundary": 387, "mass": 5, "attention": 66}[kind],
            2 if kind == "boundary" else 1,
        )
        if kind in ["mass", "attention"]:
            self.token_gate = nn.Linear(64, 1)
            nn.init.zeros_(self.token_gate.weight)
            nn.init.zeros_(self.token_gate.bias)

    def forward(self, h, valid, token_valid):
        B, L, D = h.shape
        with torch.autocast(device_type=h.device.type, enabled=False):
            h = F.layer_norm(h.float(), (D,)).masked_fill(~token_valid[..., None], 0)
            count = F.pad(token_valid.float().cumsum(1), (1, 0))
            prefix = F.pad(h.cumsum(1), (0, 0, 1, 0))
            length = token_valid.sum(1).clamp_min(1)[:, None].float()

            def pool(a, b):
                return (prefix[:, b] - prefix[:, a]) / (
                    count[:, b] - count[:, a]
                ).clamp_min(1)[..., None]

            if self.kind in ["mass", "attention"]:
                logits = self.token_gate(h).squeeze(-1).clamp(-8, 8)
                mass = torch.exp(logits) * token_valid
                cp = F.pad(mass.cumsum(1), (1, 0))
                hp = F.pad((h * mass[..., None]).cumsum(1), (0, 0, 1, 0))
            out = []
            for idx in torch.arange(L * L, device=h.device).split(256):
                a = idx // L
                b = torch.maximum(idx % L + 1, a + 1)
                w = b - a
                dur = (count[:, b] - count[:, a]) / length
                if self.kind == "ring":
                    radius = (w // 4).clamp_min(1)
                    x = pool(a, b)
                    left = pool((a - radius).clamp_min(0), a)
                    right = pool(b, (b + radius).clamp_max(L))
                    feats = torch.cat(
                        [
                            x,
                            left,
                            right,
                            x - left,
                            x - right,
                            left - right,
                            dur[..., None],
                            torch.log1p(dur)[..., None],
                        ],
                        -1,
                    )
                elif self.kind == "boundary":
                    ls = []
                    rs = []
                    for k in [1, 3]:
                        inside_left = pool(a, torch.minimum(a + k, b))
                        outside_left = pool((a - k).clamp_min(0), a)
                        inside_right = pool(torch.maximum(b - k, a), b)
                        outside_right = pool(b, (b + k).clamp_max(L))
                        ls.append(inside_left - outside_left)
                        rs.append(inside_right - outside_right)
                    feats = torch.cat(
                        [
                            ls[0],
                            ls[1],
                            rs[0],
                            rs[1],
                            ls[0] * rs[0],
                            ls[1] * rs[1],
                            dur[..., None],
                            (a[None] / length)[..., None],
                            (b[None] / length)[..., None],
                        ],
                        -1,
                    )
                elif self.kind == "mass":
                    total = cp[:, -1:].clamp_min(1e-8)
                    inside = (cp[:, b] - cp[:, a]) / total
                    before = cp[:, a] / total
                    after = (cp[:, -1:] - cp[:, b]) / total
                    # Global query-conditioned evidence mass, not GT event coverage.
                    density = inside / dur.clamp_min(1 / length)
                    feats = torch.stack(
                        [inside, density.log1p(), before, after, dur], -1
                    )
                else:
                    den = (cp[:, b] - cp[:, a]).clamp_min(1e-8)
                    x = (hp[:, b] - hp[:, a]) / den[..., None]
                    feats = torch.cat(
                        [
                            x,
                            dur[..., None],
                            torch.log1p(den / (count[:, b] - count[:, a]).clamp_min(1))[
                                ..., None
                            ],
                        ],
                        -1,
                    )
                out.append(self.net(feats))
            z = torch.cat(out, 1).reshape(B, L, L, -1).masked_fill(~valid[..., None], 0)
            return z if self.kind == "boundary" else z[..., 0]


class Selector(query_objective.Selector):
    def compose_score(self, field, overrides=None, **kwargs):
        score = super().compose_score(field, overrides, **kwargs)
        if not hasattr(self, "r53_fusion"):
            return score
        get = lambda k: (
            overrides.get(k, getattr(field, k)) if overrides else getattr(field, k)
        )
        e, s, ts, te = [
            get(k).float()
            for k in ["evidence", "support", "transition_start", "transition_end"]
        ]
        x = torch.stack([e, s, ts, te, (ts + te) * 0.5, (ts - te).abs()], -1)
        with torch.autocast(device_type=e.device.type, enabled=False):
            delta = 0.25 * torch.tanh(self.r53_fusion(x))
        # Carrier unchanged; zero-initialized weights preserve exact baseline output.
        return (
            score
            + delta[..., 0] * e
            + delta[..., 1] * s
            + delta[..., 2] * (ts + te) * 0.5
        ).masked_fill(~field.valid, 0)

    def raw_from_state(self, state, valid, carrier, video_padding_mask=None):
        f = super().raw_from_state(state, valid, carrier, video_padding_mask)
        tv = self._token_valid(valid, video_padding_mask)
        self.r53_stats = {}
        for attr, kind, zname, fieldnames in [
            ("r53_support", "s", "round31_z_s", ["support"]),
            (
                "r53_boundary",
                "t",
                "round31_z_t",
                ["transition_start", "transition_end"],
            ),
            ("r53_evidence", "e", "round31_z_e", ["evidence"]),
        ]:
            if not hasattr(self, attr):
                continue
            out = getattr(self, attr)(getattr(f, zname), valid, tv)
            for i, name in enumerate(fieldnames):
                before = getattr(f, name)
                raw = out[..., i] if len(fieldnames) > 1 else out
                delta = 0.25 * (1 - before.abs()).clamp_min(0) * torch.tanh(raw)
                after = (before + delta).masked_fill(~valid, 0)
                setattr(f, name, after)
                self.r53_stats[name + "_delta_abs"] = delta.detach()[valid].abs().mean()
                # Keep raw transition views consistent with modified deployed tensors.
                if name.startswith("transition"):
                    setattr(
                        f,
                        "raw_" + name,
                        torch.atanh(after.float().clamp(-0.999999, 0.999999)),
                    )
        f.score = self.compose_score(f)
        f.round31_rank_score = f.score
        f.round31_score_with_support = f.score
        f.round31_score_without_support = self.compose_score(
            f, {"support": torch.zeros_like(f.support)}
        )
        f.round31_deployed_support = f.support
        if hasattr(self, "r53_fusion"):
            with (
                torch.no_grad(),
                torch.autocast(device_type=f.score.device.type, enabled=False),
            ):
                e, s, ts, te = [
                    getattr(f, k).float()
                    for k in [
                        "evidence",
                        "support",
                        "transition_start",
                        "transition_end",
                    ]
                ]
                weights = 1 + 0.25 * torch.tanh(
                    self.r53_fusion(
                        torch.stack(
                            [e, s, ts, te, 0.5 * (ts + te), (ts - te).abs()], -1
                        )
                    )
                )
                for i, k in enumerate(["E", "S", "T"]):
                    self.r53_stats[k + "_weight_mean"] = weights[..., i][valid].mean()
                    self.r53_stats[k + "_weight_std"] = weights[..., i][valid].std()
        f.r53_stats = dict(self.r53_stats)
        return f


class Model(candidate_objective.Model):
    def experiment_contract(self):
        c = super().experiment_contract()
        c.update(
            r53_spec=self.r53_spec,
            r53_inference_uses_gt=False,
            r53_new_head_lr_multiplier=20,
        )
        return c

    def parameter_groups(self):
        groups = super().parameter_groups()
        ids = {
            id(p)
            for n, p in self.selector.named_parameters()
            if n.startswith("r53_") and p.requires_grad
        }
        out = []
        for g in groups:
            fresh = [p for p in g.params if id(p) in ids]
            old = [p for p in g.params if id(p) not in ids]
            if old:
                out.append(dataclasses.replace(g, params=old))
            if fresh:
                out.append(
                    dataclasses.replace(
                        g, name="r53_structure", params=fresh, lr=g.lr * 20
                    )
                )
        return out

    def r50_adjust_terms(self, terms, outputs, batch, epoch):
        t = super().r50_adjust_terms(terms, outputs, batch, epoch)
        t.metrics.update(
            {
                "support_objective/structure/" + k: v.detach()
                for k, v in outputs.trifield_output.r53_stats.items()
            }
        )
        if self.r53_spec.get("fusion"):
            for k in list(t.metrics):
                if k.startswith("candidate_objective/overlap/remove_"):
                    t.metrics["legacy_additive_reference/" + k] = t.metrics.pop(k)
        return t

    def r48_loss_probe(self, terms, epoch):
        if (
            not self.training
            or not torch.is_grad_enabled()
            or getattr(self, "r53_probe_epoch", None) == epoch
        ):
            return {}
        self.r53_probe_epoch = epoch
        result = {
            "epoch": epoch,
            "scope": "first real batch; weighted actual losses",
            "groups": {},
        }
        for group in [
            "e_interaction",
            "s_projection",
            "t_interaction",
            "r53_support",
            "r53_boundary",
            "r53_evidence",
            "r53_fusion",
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
                gs = (
                    torch.autograd.grad(
                        getattr(terms, name) * self.loss_weights[name],
                        params,
                        retain_graph=True,
                        allow_unused=True,
                    )
                    if getattr(terms, name).requires_grad
                    else [None] * len(params)
                )
                vectors[name] = torch.cat(
                    [
                        torch.zeros_like(p).flatten()
                        if g is None
                        else g.detach().float().flatten()
                        for p, g in zip(params, gs)
                    ]
                )
                entry[name] = {
                    "norm": float(vectors[name].norm()),
                    "none_fraction": sum(g is None for g in gs) / len(gs),
                }
            a = vectors["rank"]
            entry["rank_cosine"] = {
                n: float((a @ v) / (a.norm() * v.norm()).clamp_min(1e-12))
                for n, v in vectors.items()
                if n != "rank"
            }
            result["groups"][group] = entry
        p = Path(self.r50_probe_dir)
        p.mkdir(parents=True, exist_ok=True)
        (p / f"gradient_e{epoch:03d}.json").write_text(json.dumps(result, indent=2))
        return {}


def build_model(config, *, r53_spec=None, **kwargs):
    spec = dict(r53_spec or {})
    assert not set(spec) - {"support", "boundary", "evidence", "fusion"}
    assert spec.get("support") in (None, "ring", "mass")
    model = candidate_objective.build_model(config, r52_spec={}, **kwargs)
    model.__class__ = Model
    model.r53_spec = spec
    model.selector.__class__ = Selector
    model.r50_probe_dir = str(config.output_dir) + "/r53_gradient_probes"
    for name, kind, seed in [
        ("r53_support", spec.get("support"), 5301),
        ("r53_boundary", "boundary" if spec.get("boundary") else None, 5302),
        ("r53_evidence", "attention" if spec.get("evidence") else None, 5303),
        ("r53_fusion", "fusion" if spec.get("fusion") else None, 5304),
    ]:
        if kind:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                head = zero_mlp(6, 3) if kind == "fusion" else Readout(kind)
            setattr(model.selector, name, head)
    return model
