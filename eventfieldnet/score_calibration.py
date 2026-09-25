"""Field-encoder extensions and candidate score calibration."""

import dataclasses
import torch
from torch import nn
from .model import adapter
from .model.interaction import RawInteraction
from .model.candidate_primitives import IndependentSProjectionEncoder
from .model.selector import ThreeFieldScoreHeads

VARIANTS = {
    "control": {},
    "rank_stop_s": {"rank_stop_s": True},
    "aux_detach": {"aux_detach_input": True},
    "soft_target": {"rank_target_exponent": 2.0},
    "no_pair": {"rank_candidate_pair": False},
    "e_lme": {"e_span_lme": True},
}
for name, overrides in VARIANTS.items():
    adapter.ROUND31_VARIANTS["r48_" + name] = dict(
        adapter.ROUND31_VARIANTS["y21"], **overrides
    )


class R48EncodingMixin:
    def encode(self, video, text, video_pad, text_pad):
        if self.training and getattr(self, "r48_channel_dropout", 0):
            probability = self.r48_channel_dropout
            mask = getattr(self, "r48_channel_mask", None)
            if mask is None or mask.shape[0] != video.shape[0]:
                mask = (
                    torch.rand((video.shape[0], 1, video.shape[-1]), device="cpu")
                    >= probability
                ).to(video.device)
                self.r48_channel_mask = mask
            video = video * mask
            self.r48_last_drop = float((~mask).float().mean())
        else:
            self.r48_last_drop = 0.0
        h = super().encode(video, text, video_pad, text_pad)
        if hasattr(self, "r48_temporal"):
            delta = self.r48_temporal(h.transpose(1, 2)).transpose(1, 2)
            delta = delta.masked_fill(video_pad[..., None].bool(), 0)
            self.r48_last_delta = delta.detach().square().mean().sqrt()
            h = h + delta
        return h


class R48Interaction(R48EncodingMixin, RawInteraction):
    pass


class R48SProjection(R48EncodingMixin, IndependentSProjectionEncoder):
    pass


class R48Selector(ThreeFieldScoreHeads):
    def _edge_field(self, h, token_valid, valid):
        values = list(super()._edge_field(h, token_valid, valid))
        if getattr(self, "r48_singleton", False):
            prepared = values[8]
            with torch.autocast(device_type=h.device.type, enabled=False):
                raw = self._edge_raw_components_from_prepared_pairs(prepared, prepared)[
                    "raw"
                ].float()
            diag = valid.diagonal(dim1=1, dim2=2) & token_valid
            raw = raw.masked_fill(~diag, 0)
            values[2] = values[2] + torch.diag_embed(raw.tanh())
            values[3] = values[3] | torch.diag_embed(diag)
            values[7] = values[7] + torch.diag_embed(raw)
        return tuple(values)


class R48Model(adapter.ThreeFieldModel):
    def forward(self, *args, **kwargs):
        # Reuse the same row/channel mask for clean and auxiliary counterfactual
        # encodings within this outer forward/loss; sample anew for the next batch.
        for name in ["e_interaction", "s_projection", "t_interaction"]:
            getattr(self.selector, name).r48_channel_mask = None
        return super().forward(*args, **kwargs)

    def r48_loss_probe(self, terms, epoch):
        if not self.training or not torch.is_grad_enabled():
            return {}
        if getattr(self, "r48_probe_epoch", None) == epoch:
            return {}
        self.r48_probe_epoch = epoch
        names = []
        params = []
        for name, p in self.selector.named_parameters():
            if p.requires_grad and any(
                x in name
                for x in [
                    "e_interaction.",
                    "s_projection.",
                    "edge_head.",
                    "t_interaction.",
                ]
            ):
                names.append(name)
                params.append(p)
        values = {}
        vectors = {}
        for term in ["rank", "evidence", "support", "transition", "endpoint"]:
            loss = getattr(terms, term) * self.loss_weights[term]
            grads = torch.autograd.grad(
                loss, params, retain_graph=True, allow_unused=True
            )
            vectors[term] = torch.cat(
                [
                    (
                        g.detach().float().flatten()
                        if g is not None
                        else torch.zeros_like(p).float().flatten()
                    )
                    for p, g in zip(params, grads)
                ]
            )
            values["score_calibration/gradient/" + term + "/weighted_norm"] = vectors[
                term
            ].norm()
            values["score_calibration/gradient/" + term + "/none_fraction"] = sum(
                g is None for g in grads
            ) / len(grads)
        for term in ["evidence", "support", "transition", "endpoint"]:
            a, b = vectors["rank"], vectors[term]
            values["score_calibration/gradient/rank_vs_" + term + "/cosine"] = (
                a @ b
            ) / (a.norm() * b.norm()).clamp_min(1e-12)
        # Separate artifact: these are first-batch probes, not epoch-average gradients.
        import json
        from pathlib import Path

        if self.r48_probe_dir:
            p = Path(self.r48_probe_dir)
            p.mkdir(parents=True, exist_ok=True)
            (p / f"gradient_e{int(epoch):03d}.json").write_text(
                json.dumps({k: float(v) for k, v in values.items()}, indent=2)
            )
        return values

    def parameter_groups(self):
        groups = super().parameter_groups()
        scale = self.r48_spec.get("parent_lr_scale", 1.0)
        return [
            dataclasses.replace(g, lr=g.lr * scale) if g.name != "trifield_heads" else g
            for g in groups
        ]

    def compute_loss(self, outputs, batch, teacher_outputs, epoch):
        result = super().compute_loss(outputs, batch, teacher_outputs, epoch)
        f = outputs.trifield_output
        valid = outputs.span_valid_mask.bool()
        for name in [
            "evidence",
            "support",
            "transition_start",
            "transition_end",
            "carrier",
        ]:
            t = getattr(f, name).detach().float()[valid]
            result.metrics["score_calibration/field/" + name + "/abs_mean"] = (
                t.abs().mean()
            )
            result.metrics["score_calibration/field/" + name + "/saturation_rate"] = (
                (t.abs() > 0.95).float().mean()
            )
        diag = valid.diagonal(dim1=1, dim2=2)
        result.metrics["score_calibration/singleton_support_abs"] = (
            f.support.diagonal(dim1=1, dim2=2)[diag].detach().abs().mean()
        )
        result.metrics["score_calibration/singleton_enabled"] = float(
            self.r48_spec.get("singleton", False)
        )
        result.metrics["score_calibration/parent_lr_scale"] = self.r48_spec.get(
            "parent_lr_scale", 1.0
        )
        for name in ["e_interaction", "s_projection", "t_interaction"]:
            m = getattr(self.selector, name)
            result.metrics["score_calibration/" + name + "/channel_drop_rate"] = (
                getattr(m, "r48_last_drop", 0.0)
            )
            result.metrics["score_calibration/" + name + "/temporal_delta_rms"] = (
                getattr(m, "r48_last_delta", 0.0)
            )
        return result


def build_model(config, *, r48_spec=None, **kwargs):
    spec = dict(r48_spec or {})
    allowed = {
        "variant",
        "parent_lr_scale",
        "protect_gt",
        "support_margin_scale",
        "singleton",
        "temporal",
        "channel_dropout",
    }
    if set(spec) - allowed:
        raise ValueError("Unknown R48 setting")
    name = spec.get("variant", "control")
    if name not in VARIANTS:
        raise ValueError(name)
    kwargs["variant"] = "r48_" + name
    kwargs.update(VARIANTS[name])
    model = adapter.build_trifield_round31_model(config, **kwargs)
    model.__class__ = R48Model
    model.r48_spec = spec
    model.r48_probe_dir = getattr(config, "output_dir", None)
    if model.r48_probe_dir:
        model.r48_probe_dir = str(model.r48_probe_dir) + "/r48_gradient_probes"
    model.r48_protect_gt = bool(spec.get("protect_gt", False))
    model.r48_support_margin_scale = float(spec.get("support_margin_scale", 1.0))
    model.selector.__class__ = R48Selector
    model.selector.r48_singleton = bool(spec.get("singleton", False))
    for i, name in enumerate(["e_interaction", "s_projection", "t_interaction"]):
        m = getattr(model.selector, name)
        m.__class__ = R48SProjection if name == "s_projection" else R48Interaction
        m.r48_channel_dropout = float(spec.get("channel_dropout", 0.0))
        if spec.get("temporal", False):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(4800 + i)
                m.r48_temporal = nn.Conv1d(64, 64, 3, padding=1, groups=64, bias=False)
                nn.init.zeros_(m.r48_temporal.weight)
    return model
