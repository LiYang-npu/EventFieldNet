"""Stage50 Event Atom Composition Field with a frozen Stage32 anchor."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping, Optional
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from training.contracts import LossResult, ParameterGroup
from training.probability import masked_softmax
from sg_components.model.event_field.framework_boundary_preserving_v2 import (
    EventFieldNetBoundaryPreserving,
)
from sg_components.model.event_field.framework_field_structured import (
    FieldStructuredOutput,
)

ATOM_MODES = ("a0_fixed", "a1_atoms", "a2_support", "a3_query")


def _diff(x, pad):
    d1 = x - torch.cat((x[:, :1], x[:, :-1]), 1)
    d2 = d1 - torch.cat((d1[:, :1], d1[:, :-1]), 1)
    return d1.masked_fill(pad[..., None], 0), d2.masked_fill(pad[..., None], 0)


def _fp(x, pad):
    v = (~pad).to(x.dtype)[..., None]
    n = v.sum((1, 2)).clamp_min(1)
    mean = (x * v).sum((1, 2)) / n
    rms = ((x * x * v).sum((1, 2)) / n).sqrt()
    delta = (x[:, 1:] - x[:, :-1]).abs().mean((1, 2)) if x.shape[1] > 1 else mean * 0
    return torch.stack((mean, rms, delta), -1)


@dataclass
class AtomCompositionOutput(FieldStructuredOutput):
    span_probs: Tensor
    anchor_span_logits: Tensor
    ranking_residual: Tensor
    atom_energy: Tensor
    branch_energies: Tensor
    atom_assignment: Tensor
    fixed_assignment: Tensor
    phase_coordinate: Tensor
    transition_start_logits: Tensor
    transition_end_logits: Tensor
    evidence_token_logits: Tensor
    evidence_atom_logits: Tensor
    support_atom_logits: Tensor
    composition_chart: Tensor
    atom_mass: Tensor
    query_composition_increment: Tensor
    field_input_fingerprints: Tensor
    anchor_logit_std: Tensor
    residual_cap_ratio: Tensor
    used_anchor: bool


class EventAtomCompositionField(nn.Module):
    """Eight ordered soft atoms; chart O(K^3), span projection O(L^2 K)."""

    def __init__(self, h, r=48, w=96, k=8, drop=0.1):
        super().__init__()
        self.k = int(k)
        self.transition = nn.Sequential(
            nn.LayerNorm(3 * h),
            nn.Linear(3 * h, w),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(w, r),
            nn.GELU(),
        )
        self.tstart = nn.Linear(r, 1)
        self.tend = nn.Linear(r, 1)
        self.tphase = nn.Linear(r, 1)
        self.pre = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, r))
        self.ctx = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, r))
        self.query = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, r))
        self.evidence = nn.Sequential(
            nn.LayerNorm(3 * r),
            nn.Linear(3 * r, w),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(w, 1),
        )
        self.atom_evidence = nn.Sequential(
            nn.LayerNorm(3 * r), nn.Linear(3 * r, w), nn.GELU(), nn.Linear(w, 1)
        )
        self.relation = nn.Sequential(
            nn.LayerNorm(7 * r + 2),
            nn.Linear(7 * r + 2, w),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(w, r),
            nn.GELU(),
        )
        self.support = nn.Linear(r, 1)
        self.query_support = nn.Sequential(
            nn.LayerNorm(3 * r), nn.Linear(3 * r, w), nn.GELU(), nn.Linear(w, 1)
        )
        self.depth = nn.Sequential(
            nn.LayerNorm(r), nn.Linear(r, w), nn.GELU(), nn.Linear(w, k)
        )
        self.gate = nn.Parameter(torch.zeros(()))
        self.register_buffer("cap", torch.zeros(()))

    @staticmethod
    def complexity(l, k=8):
        return {"chart_ops": k**3, "span_map_ops": l * l * k, "total": k**3 + l * l * k}

    def set_epoch(self, e):
        self.cap.fill_(0.0 if e <= 5 else 0.25 * min(max((e - 5) / 7.0, 0.0), 1.0))

    def _assign(self, z, pad, fixed):
        b, l = z.shape
        valid = ~pad
        if fixed:
            idx = torch.arange(l, device=z.device, dtype=z.dtype)[None]
            den = (valid.sum(1) - 1).clamp_min(1).to(z.dtype)[:, None]
            phase = idx / den * (self.k - 1)
            phase = torch.where(
                (valid.sum(1) <= 1)[:, None], torch.zeros_like(phase), phase
            )
        else:
            strength = (F.softplus(z.float()) + 1e-4) * valid.float()
            phase = (
                (strength.cumsum(1) - 0.5 * strength)
                / strength.sum(1, keepdim=True).clamp_min(1e-6)
                * (self.k - 1)
            )
            phase = phase.to(z.dtype)
        centers = torch.arange(self.k, device=z.device, dtype=z.dtype)
        a = (
            torch.softmax(-((phase[..., None] - centers) / 0.45).square(), -1)
            * valid[..., None]
        )
        return a, phase.masked_fill(pad, 0)

    @staticmethod
    def _agg(a, x):
        mass = a.sum(1).clamp_min(1e-6)
        return torch.einsum("blk,blr->bkr", a, x) / mass[..., None], mass

    def _components(self, a, xpre, xctx, etok, q_evidence, q_composition):
        ap, mass = self._agg(a, self.pre(xpre))
        ac, _ = self._agg(a, self.ctx(xctx))
        qq = q_evidence[:, None].expand_as(ac)
        leaf = (
            self.atom_evidence(torch.cat((ac, qq, ac * qq), -1)).squeeze(-1)
            + torch.einsum("blk,bl->bk", a, etok) / mass
        )
        lp, rp = ap[:, :-1], ap[:, 1:]
        lc, rc = ac[:, :-1], ac[:, 1:]
        mp = torch.stack((mass[:, :-1], mass[:, 1:]), -1)
        rel = self.relation(
            torch.cat((lp, rp, rp - lp, lc, rc, rc - lc, lc * rc, mp), -1)
        )
        base = self.support(rel).squeeze(-1)
        qr = q_composition[:, None].expand_as(rel)
        qdelta = self.query_support(torch.cat((rel, qr, rel * qr), -1)).squeeze(-1)
        return leaf, base, qdelta, self.depth(q_composition), rel, mass

    def _chart(self, leaf, support, qdelta, depth, level):
        cells = [[None] * self.k for _ in range(self.k)]
        attach = 0.0 * (support.sum(1) + qdelta.sum(1) + depth.sum(1))
        for i in range(self.k):
            cells[i][i] = leaf[:, i] + attach
        for width in range(2, self.k + 1):
            for i in range(self.k - width + 1):
                j = i + width - 1
                choices = []
                for m in range(i, j):
                    rel = attach
                    if level >= 2:
                        rel = rel + support[:, m]
                    if level >= 3:
                        rel = rel + qdelta[:, m] + depth[:, width - 1]
                    choices.append(cells[i][m] + cells[m + 1][j] + rel)
                value = torch.stack(choices, -1)
                cells[i][j] = (
                    torch.logsumexp(value, -1)
                    - value.new_tensor(float(len(choices))).log()
                )
        invalid = leaf.new_full((leaf.shape[0],), -1e4)
        return torch.stack(
            [
                torch.stack(
                    [
                        cells[i][j] if cells[i][j] is not None else invalid
                        for j in range(self.k)
                    ],
                    -1,
                )
                for i in range(self.k)
            ],
            1,
        )

    @staticmethod
    def _z(x):
        return (x - x.mean(1, keepdim=True)) / x.std(
            1, keepdim=True, unbiased=False
        ).clamp_min(1e-4)

    def _map(self, chart, a, valid):
        starts = []
        ends = []
        covers = []
        for atom in range(self.k):
            starts.append(torch.logsumexp(chart[:, atom, atom:], -1))
            ends.append(torch.logsumexp(chart[:, : atom + 1, atom], -1))
            covers.append(
                torch.logsumexp(
                    torch.stack(
                        [
                            chart[:, i, j]
                            for i in range(atom + 1)
                            for j in range(atom, self.k)
                        ],
                        -1,
                    ),
                    -1,
                )
            )
        starts = self._z(torch.stack(starts, -1))
        ends = self._z(torch.stack(ends, -1))
        covers = self._z(torch.stack(covers, -1))
        prefix = F.pad(a.cumsum(1), (0, 0, 1, 0))
        coverage = prefix[:, None, 1:] - prefix[:, :-1, None]
        fraction = (coverage / a.sum(1).clamp_min(1e-6)[:, None, None]).clamp(0, 1)
        interior = (fraction * covers[:, None, None]).sum(-1) / fraction.sum(
            -1
        ).clamp_min(1e-6)
        score = (
            0.25 * torch.einsum("blk,bk->bl", a, starts)[:, :, None]
            + 0.25 * torch.einsum("blk,bk->bl", a, ends)[:, None, :]
            + 0.5 * interior
        )
        return score.masked_fill(~valid, 0)

    def _branch(self, a, xpre, xctx, etok, q_evidence, q_composition, valid, level):
        leaf, support, qdelta, depth, rel, mass = self._components(
            a, xpre, xctx, etok, q_evidence, q_composition
        )
        chart = self._chart(leaf, support, qdelta, depth, level)
        return (
            self._map(chart, a, valid),
            chart,
            leaf,
            support + (qdelta if level >= 3 else 0 * qdelta),
            rel,
            mass,
        )

    def forward(self, xpre, xctx, q, pad, valid, anchor, mode, composition_query=None):
        if mode not in ATOM_MODES:
            raise ValueError(mode)
        d1, d2 = _diff(xpre, pad)
        transition_input = torch.cat((xpre, d1, d2), -1)
        hidden = self.transition(transition_input)
        ts = self.tstart(hidden).squeeze(-1).masked_fill(pad, 0)
        te = self.tend(hidden).squeeze(-1).masked_fill(pad, 0)
        tp = self.tphase(hidden).squeeze(-1).masked_fill(pad, 0)
        learned, phase = self._assign(tp, pad, False)
        fixed, _ = self._assign(tp, pad, True)
        ctx = self.ctx(xctx)
        q = self.query(q)
        composition_q = (
            q if composition_query is None else self.query(composition_query)
        )
        qt = q[:, None].expand_as(ctx)
        etok = (
            self.evidence(torch.cat((ctx, qt, ctx * qt), -1))
            .squeeze(-1)
            .masked_fill(pad, 0)
        )
        metas = [
            self._branch(fixed, xpre, xctx, etok, q, composition_q, valid, 0),
            self._branch(learned, xpre, xctx, etok, q, composition_q, valid, 1),
            self._branch(learned, xpre, xctx, etok, q, composition_q, valid, 2),
            self._branch(learned, xpre, xctx, etok, q, composition_q, valid, 3),
        ]
        branches = torch.stack([item[0] for item in metas], -1)
        index = ATOM_MODES.index(mode)
        energy = branches[..., index] + 0.0 * branches.sum(-1)
        legal = valid.to(anchor.dtype)
        count = legal.flatten(1).sum(1).clamp_min(1)
        mean = (anchor * legal).flatten(1).sum(1) / count
        std = (
            (
                ((anchor - mean[:, None, None]) * legal).square().flatten(1).sum(1)
                / count
            )
            .clamp_min(1e-8)
            .sqrt()
        )
        residual = (
            self.cap * std[:, None, None] * torch.tanh(self.gate) * torch.tanh(energy)
        )
        residual = residual.masked_fill(~valid, 0)
        assignment = fixed if index == 0 else learned
        meta = metas[index]
        support_pad = torch.zeros(
            meta[4].shape[:2], dtype=torch.bool, device=pad.device
        )
        fps = torch.stack(
            (_fp(transition_input, pad), _fp(xctx, pad), _fp(meta[4], support_pad)), 1
        )
        return {
            "span_logits": (anchor + residual).masked_fill(~valid, -1e4),
            "residual": residual,
            "energy": energy,
            "branches": branches,
            "assignment": assignment,
            "fixed": fixed,
            "phase": phase,
            "ts": ts,
            "te": te,
            "etok": etok,
            "eatom": meta[2],
            "support": meta[3],
            "chart": meta[1],
            "mass": meta[5],
            "query_increment": branches[..., 3] - branches[..., 2],
            "fingerprints": fps,
            "std": std,
        }


class SemanticAnchoredAtomGraph(EventFieldNetBoundaryPreserving):
    def __init__(
        self,
        *args: Any,
        mode="a3_query",
        atom_lr=1e-4,
        atom_rank=48,
        atom_hidden=96,
        num_atoms=8,
        iou_temperature=0.1,
        dropout=0.1,
        **kwargs: Any,
    ):
        if mode not in ATOM_MODES:
            raise ValueError(mode)
        super().__init__(*args, dropout=dropout, **kwargs)
        self.atom_mode = mode
        self.atom_lr = float(atom_lr)
        self.iou_temperature = float(iou_temperature)
        self.feature_bridge = EventAtomCompositionField(
            self.hidden_dim, atom_rank, atom_hidden, num_atoms, dropout
        )
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for parameter in self.feature_bridge.parameters():
            parameter.requires_grad_(True)
        self.__dict__["_stage50_pre"] = None
        self.__dict__["_stage50_hook_calls"] = 0
        self.__dict__["_stage50_hook_handle"] = (
            self._last_context().register_forward_pre_hook(self._capture_pre)
        )

    def _last_context(self):
        layers = getattr(
            getattr(getattr(self, "stem", None), "temporal_encoder", None),
            "layers",
            None,
        )
        if layers is None or not len(layers):
            raise RuntimeError("Stage50 requires stem.temporal_encoder.layers")
        return layers[-1]

    def _capture_pre(self, layer, args):
        value = args[0]
        if not bool(getattr(layer.self_attn, "batch_first", True)):
            value = value.transpose(0, 1)
        self.__dict__["_stage50_pre"] = value.detach()
        self.__dict__["_stage50_hook_calls"] += 1

    def train(self, mode=True):
        super().train(False)
        self.feature_bridge.train(mode)
        return self

    def set_epoch(self, epoch, training):
        self.feature_bridge.set_epoch(epoch)
        return {
            "mode": self.atom_mode,
            "stage32_frozen": True,
            "residual_cap": float(self.feature_bridge.cap),
        }

    @staticmethod
    def _parse(inputs, q, vm, qm):
        if isinstance(inputs, Mapping):
            leaked = {
                "gt_spans",
                "gt_span_mask",
                "span_labels",
                "targets",
            }.intersection(inputs)
            if leaked:
                raise AssertionError(
                    f"ground truth leaked into forward: {sorted(leaked)}"
                )
            return (
                inputs["src_vid"],
                inputs["src_txt"],
                inputs.get("video_padding_mask"),
                inputs.get("query_padding_mask"),
            )
        if q is None:
            raise TypeError("query_features required")
        return inputs, q, vm, qm

    def _latents(self, v, q, vm, qm):
        self.__dict__["_stage50_pre"] = None
        self.__dict__["_stage50_hook_calls"] = 0
        with torch.no_grad():
            ctx, query_tokens, pad, safe = self.stem.encode(v, q, vm, qm)
            summary = self.query_summary_norm(
                self._masked_query_summary(query_tokens, safe)
            )
        if (
            self.__dict__["_stage50_pre"] is None
            or self.__dict__["_stage50_hook_calls"] != 1
        ):
            raise RuntimeError("pre-context hook capture failed")
        return (
            torch.nan_to_num(self.__dict__["_stage50_pre"]).detach(),
            torch.nan_to_num(ctx).detach(),
            torch.nan_to_num(summary).detach(),
            pad,
        )

    def forward(
        self,
        inputs,
        query_features=None,
        video_padding_mask=None,
        query_padding_mask=None,
    ):
        v, q, vm, qm = self._parse(
            inputs, query_features, video_padding_mask, query_padding_mask
        )
        with torch.no_grad():
            anchor = super().forward(v, q, vm, qm)
        xpre, xctx, summary, pad = self._latents(v, q, vm, qm)
        valid = anchor.span_valid_mask.bool()
        anchor_logits = torch.nan_to_num(anchor.span_logits).detach()
        atom = self.feature_bridge(
            xpre, xctx, summary, pad, valid, anchor_logits, self.atom_mode
        )
        probs = masked_softmax(
            atom["span_logits"].flatten(1).float(), valid.flatten(1)
        ).reshape_as(anchor_logits)
        return AtomCompositionOutput(
            evidence_logits=torch.nan_to_num(anchor.evidence_logits).detach(),
            support_logits=torch.nan_to_num(anchor.support_logits).detach(),
            start_transition_logits=torch.nan_to_num(
                anchor.start_transition_logits
            ).detach(),
            end_transition_logits=torch.nan_to_num(
                anchor.end_transition_logits
            ).detach(),
            span_logits=atom["span_logits"],
            span_valid_mask=valid,
            base_span_logits=torch.nan_to_num(anchor.base_span_logits).detach(),
            quality_logits=torch.nan_to_num(anchor.quality_logits).detach(),
            span_probs=probs,
            anchor_span_logits=anchor_logits,
            ranking_residual=atom["residual"],
            atom_energy=atom["energy"],
            branch_energies=atom["branches"],
            atom_assignment=atom["assignment"],
            fixed_assignment=atom["fixed"],
            phase_coordinate=atom["phase"],
            transition_start_logits=atom["ts"],
            transition_end_logits=atom["te"],
            evidence_token_logits=atom["etok"],
            evidence_atom_logits=atom["eatom"],
            support_atom_logits=atom["support"],
            composition_chart=atom["chart"],
            atom_mass=atom["mass"],
            query_composition_increment=atom["query_increment"],
            field_input_fingerprints=atom["fingerprints"],
            anchor_logit_std=atom["std"],
            residual_cap_ratio=self.feature_bridge.cap.expand(anchor_logits.shape[0]),
            used_anchor=True,
        )

    @staticmethod
    def _targets(outputs, batch):
        spans = batch.targets["gt_spans"].to(outputs.span_logits.dtype)
        mask = batch.targets["gt_span_mask"].bool()
        pad = batch.inputs["video_padding_mask"].bool()
        length = outputs.span_logits.shape[-1]
        count = (~pad).sum(1).clamp_min(1).to(spans.dtype)
        idx = torch.arange(length, device=spans.device, dtype=spans.dtype)
        center = (idx[None] + 0.5) / count[:, None]
        gs = torch.minimum(spans[..., 0], spans[..., 1])
        ge = torch.maximum(spans[..., 0], spans[..., 1])
        foreground = (
            (
                (center[..., None] >= gs[:, None])
                & (center[..., None] <= ge[:, None])
                & mask[:, None]
            )
            .any(-1)
            .to(spans.dtype)
        )
        start_target = (
            torch.exp(
                -0.5
                * (
                    (center[..., None] - gs[:, None]) * count[:, None, None] / 1.5
                ).square()
            )
            .masked_fill(~mask[:, None], 0)
            .amax(-1)
        )
        end_target = (
            torch.exp(
                -0.5
                * (
                    (center[..., None] - ge[:, None]) * count[:, None, None] / 1.5
                ).square()
            )
            .masked_fill(~mask[:, None], 0)
            .amax(-1)
        )
        start = idx[None, :, None] / count[:, None, None]
        end = (idx[None, None, :] + 1) / count[:, None, None]
        inter = (
            torch.minimum(end[..., None], ge[:, None, None])
            - torch.maximum(start[..., None], gs[:, None, None])
        ).clamp_min(0)
        iou = inter / (
            (end - start).clamp_min(0)[..., None] + (ge - gs)[:, None, None] - inter
        ).clamp_min(1e-6)
        iou = (
            iou.masked_fill(~mask[:, None, None], 0)
            .amax(-1)
            .masked_fill(~outputs.span_valid_mask, 0)
        )
        return (
            foreground.masked_fill(pad, 0),
            start_target.masked_fill(pad, 0),
            end_target.masked_fill(pad, 0),
            iou,
        )

    def compute_loss(self, outputs, batch, teacher_outputs: Optional[Any], epoch):
        if teacher_outputs is not None:
            raise AssertionError("teacher span KD forbidden")
        foreground, start_target, end_target, iou = self._targets(outputs, batch)
        valid_tokens = ~batch.inputs["video_padding_mask"].bool()
        valid = outputs.span_valid_mask
        target = masked_softmax(
            (iou / self.iou_temperature).flatten(1), valid.flatten(1)
        ).reshape_as(iou)
        final_loss = (
            -(target * torch.log(outputs.span_probs.clamp_min(1e-8)))
            .flatten(1)
            .sum(1)
            .mean()
        )
        atom_probs = masked_softmax(
            outputs.atom_energy.flatten(1), valid.flatten(1)
        ).reshape_as(iou)
        atom_loss = (
            -(target * torch.log(atom_probs.clamp_min(1e-8))).flatten(1).sum(1).mean()
        )
        evidence = F.binary_cross_entropy_with_logits(
            outputs.evidence_token_logits[valid_tokens], foreground[valid_tokens]
        )
        transition = 0.5 * (
            F.binary_cross_entropy_with_logits(
                outputs.transition_start_logits[valid_tokens],
                start_target[valid_tokens],
            )
            + F.binary_cross_entropy_with_logits(
                outputs.transition_end_logits[valid_tokens], end_target[valid_tokens]
            )
        )
        atom_fg = torch.einsum(
            "blk,bl->bk", outputs.atom_assignment, foreground
        ) / outputs.atom_mass.clamp_min(1e-6)
        support = F.binary_cross_entropy_with_logits(
            outputs.support_atom_logits,
            torch.minimum(atom_fg[:, :-1], atom_fg[:, 1:]).detach(),
        )
        distribution = outputs.atom_mass / outputs.atom_mass.sum(
            1, keepdim=True
        ).clamp_min(1e-6)
        balance = (
            (
                distribution
                * torch.log((distribution * distribution.shape[1]).clamp_min(1e-8))
            )
            .sum(1)
            .mean()
        )
        loss = (
            final_loss
            + 0.3 * atom_loss
            + 0.2 * evidence
            + 0.2 * transition
            + 0.1 * support
            + 0.02 * balance
        )
        return LossResult(
            loss,
            {
                "final_listwise": final_loss.detach(),
                "atom_listwise": atom_loss.detach(),
                "evidence": evidence.detach(),
                "transition": transition.detach(),
                "support": support.detach(),
                "atom_balance": balance.detach(),
                "teacher_span_kd": 0.0,
            },
        )

    def decode(self, outputs, inputs):
        return outputs.span_probs.flatten(1).argmax(1)

    def diagnostics(self, outputs, batch):
        valid = outputs.span_valid_mask
        ratio = (
            outputs.ranking_residual.abs()
            / outputs.anchor_logit_std[:, None, None].clamp_min(1e-8)
        )[valid].mean()
        expected = (
            outputs.atom_assignment
            * torch.arange(outputs.atom_assignment.shape[-1], device=valid.device)
        ).sum(-1)
        return {
            "probability_error": (outputs.span_probs.flatten(1).sum(1) - 1).abs().max(),
            "mean_abs_residual_over_anchor_std": ratio,
            "residual_cap": self.feature_bridge.cap.detach(),
            "residual_gate": torch.tanh(self.feature_bridge.gate).detach(),
            "phase_monotonic_violation": F.relu(
                expected[:, :-1] - expected[:, 1:]
            ).mean(),
            "mean_atoms": (outputs.atom_mass > 0.1).float().sum(1).mean(),
            "used_anchor": 1.0,
        }

    def parameter_groups(self):
        return [
            ParameterGroup(
                "feature_bridge", list(self.feature_bridge.parameters()), self.atom_lr
            )
        ]


__all__ = [
    "ATOM_MODES",
    "AtomCompositionOutput",
    "EventAtomCompositionField",
    "SemanticAnchoredAtomGraph",
]
