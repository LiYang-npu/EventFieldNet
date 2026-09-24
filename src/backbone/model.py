"""Stage55: Stage32-compatible backbone causal matrix.

The implementation keeps the original coordinate system and official output
contract. C3-C5 inherit the Stage32 representation and add four zero-gated
cross-modal residual blocks. C5 uses the same-size candidate head as C4, but
adds the full-span IoU listwise and hard-negative ranking objective.
"""

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

MODES = ("C1_frozen", "C2_random", "C3_identity", "C4_stage32_head", "C5_contrast")
INHERITED_MODES = ("C3_identity", "C4_stage32_head", "C5_contrast")
STAGE32_KWARGS = dict(
    video_dim=514,
    query_dim=512,
    hidden_dim=384,
    num_heads=8,
    interaction_layers=3,
    feedforward_dim=1536,
    pair_dim=96,
    dropout=0.2,
    min_span_clips=2,
    context_rank=96,
    context_scales=(1,),
    context_residual_bound=1.0,
    context_boundary_radius=4,
    use_context_quality=True,
    boundary_rank=96,
    boundary_scales=(1, 3, 7),
    boundary_residual_bound=1.0,
    use_boundary_gate=True,
    field_rank=64,
    use_evidence_modulation=True,
    use_support_pyramid=True,
    use_derived_transition=True,
    use_quality_energy=True,
    bounded_field_energy=True,
    detach_quality_features=True,
    anchor_preserving_span=True,
    structured_energy_bound=0.5,
    quality_energy_bound=2.0,
)


def _masked_mean(x: Tensor, pad: Tensor) -> Tensor:
    valid = (~pad).to(x.dtype).unsqueeze(-1)
    return (x * valid).sum(1) / valid.sum(1).clamp_min(1.0)


def _safe_query(stem: nn.Module, query: Tensor, pad: Tensor) -> Tensor:
    projection = getattr(stem, "query_projection", None)
    if projection is not None:
        query = projection(query)
    return _masked_mean(query, pad)


class ZeroGatedCrossModalBlock(nn.Module):
    """A query-conditioned temporal residual block with an explicit zero gate."""

    def __init__(self, hidden: int, heads: int = 8, dropout: float = 0.2):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.attn = nn.MultiheadAttention(
            hidden, heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 4 * hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden, hidden),
        )
        self.query = nn.Linear(hidden, 2 * hidden)
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, x: Tensor, query: Tensor, pad: Tensor) -> Tensor:
        q = self.query(query).unsqueeze(1)
        scale, bias = q.chunk(2, -1)
        h = self.norm(x) * (1.0 + 0.1 * torch.tanh(scale)) + 0.1 * torch.tanh(bias)
        attn, _ = self.attn(h, h, h, key_padding_mask=pad, need_weights=False)
        update = self.ffn(h + attn)
        return x + torch.tanh(self.gate) * update.masked_fill(pad.unsqueeze(-1), 0.0)


class SharedSpanHead(nn.Module):
    """Shared endpoint/interval/query candidate head for C1-C3."""

    def __init__(self, hidden: int, min_span_clips: int = 2):
        super().__init__()
        self.min_span_clips = int(min_span_clips)
        self.start = nn.Linear(hidden, 1)
        self.end = nn.Linear(hidden, 1)
        self.query = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden))
        self.score = nn.Sequential(
            nn.LayerNorm(3 * hidden + 1),
            nn.Linear(3 * hidden + 1, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self, tokens: Tensor, query: Tensor, pad: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        b, length, hidden = tokens.shape
        valid_token = ~pad
        idx = torch.arange(length, device=tokens.device)
        starts, ends = idx[:, None], idx[None, :]
        width = ends - starts + 1
        valid = (
            (ends >= starts + self.min_span_clips - 1)
            & valid_token[:, :, None]
            & valid_token[:, None, :]
        )
        prefix = F.pad(tokens.cumsum(1), (0, 0, 1, 0))
        inside = (prefix[:, ends + 1] - prefix[:, starts]) / width.clamp_min(1)[
            None, :, :, None
        ]
        st = tokens[:, :, None, :].expand(-1, -1, length, -1)
        en = tokens[:, None, :, :].expand(-1, length, -1, -1)
        qu = self.query(query)[:, None, None, :].expand_as(inside)
        duration = (
            width.to(tokens.dtype)[None, :, :, None]
            / valid_token.sum(1).clamp_min(1)[:, None, None, None]
        )
        score = self.score(
            torch.cat((st, en, inside + inside * qu, duration), -1)
        ).squeeze(-1)
        start_logits = self.start(tokens).squeeze(-1).masked_fill(~valid_token, -1e4)
        end_logits = self.end(tokens).squeeze(-1).masked_fill(~valid_token, -1e4)
        logits = (
            start_logits[:, :, None] + end_logits[:, None, :] + score
        ).masked_fill(~valid, -1e4)
        probs = masked_softmax(logits.flatten(1).float(), valid.flatten(1)).reshape_as(
            logits
        )
        return logits, probs, valid, start_logits, end_logits


class MatchedFullSpanHead(nn.Module):
    """C4/C5 matched-size full-span head.

    Both modes execute the same segment representation and have the same
    state dictionary. C5 differs only in the loss used for that representation.
    """

    def __init__(self, hidden: int, rank: int = 96, min_span_clips: int = 2):
        super().__init__()
        self.min_span_clips = int(min_span_clips)
        self.start = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, rank))
        self.end = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, rank))
        self.interval = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, rank))
        self.query = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, rank))
        self.length = nn.Sequential(
            nn.Linear(1, rank), nn.GELU(), nn.Linear(rank, rank)
        )
        self.segment = nn.Sequential(
            nn.LayerNorm(5 * rank),
            nn.Linear(5 * rank, 2 * rank),
            nn.GELU(),
            nn.Linear(2 * rank, rank),
            nn.GELU(),
            nn.Linear(rank, 1),
        )
        self.contrast = nn.Sequential(
            nn.LayerNorm(5 * rank),
            nn.Linear(5 * rank, rank),
            nn.GELU(),
            nn.Linear(rank, 1),
        )

    def forward(
        self, tokens: Tensor, query: Tensor, pad: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        b, length, _ = tokens.shape
        valid_token = ~pad
        idx = torch.arange(length, device=tokens.device)
        starts, ends = idx[:, None], idx[None, :]
        width = ends - starts + 1
        valid = (
            (ends >= starts + self.min_span_clips - 1)
            & valid_token[:, :, None]
            & valid_token[:, None, :]
        )
        prefix = F.pad(tokens.cumsum(1), (0, 0, 1, 0))
        pooled = (prefix[:, ends + 1] - prefix[:, starts]) / width.clamp_min(1)[
            None, :, :, None
        ]
        st = self.start(tokens)[:, :, None, :].expand(-1, -1, length, -1)
        en = self.end(tokens)[:, None, :, :].expand(-1, length, -1, -1)
        inside = self.interval(pooled)
        qu = self.query(query)[:, None, None, :].expand_as(inside)
        length_feature = self.length(
            (
                width.to(tokens.dtype)[None, :, :, None]
                / valid_token.sum(1).clamp_min(1)[:, None, None, None]
            )
        )
        rep = torch.cat((st, en, inside, qu, length_feature.expand_as(inside)), -1)
        logits = self.segment(rep).squeeze(-1).masked_fill(~valid, -1e4)
        contrast = self.contrast(rep).squeeze(-1).masked_fill(~valid, 0.0)
        probs = masked_softmax(logits.flatten(1).float(), valid.flatten(1)).reshape_as(
            logits
        )
        start_logits = st.mean(-1).masked_fill(~valid_token[:, :, None], -1e4).amax(-1)
        end_logits = en.mean(-1).masked_fill(~valid_token[:, None, :], -1e4).amax(-2)
        return logits, probs, valid, start_logits, end_logits, contrast


@dataclass
class Stage55Output:
    span_logits: Tensor
    span_probs: Tensor
    span_valid_mask: Tensor
    start_logits: Tensor
    end_logits: Tensor
    evidence_logits: Tensor
    token_features: Tensor
    query_features: Tensor
    contrast_scores: Tensor
    inherited_span_logits: Tensor


class Stage55Model(EventFieldNetBoundaryPreserving):
    """Causal matrix model with one training Stage32-compatible implementation."""

    def __init__(
        self,
        mode: str,
        init_checkpoint: Optional[str] = None,
        bottom_lr: float = 2e-5,
        middle_lr: float = 5e-5,
        top_lr: float = 1e-4,
        head_lr: float = 1e-4,
        extra_lr: float = 1e-4,
        rank: int = 96,
        num_extra_blocks: int = 4,
        dropout: float = 0.2,
        **kwargs: Any,
    ):
        if mode not in MODES:
            raise ValueError(mode)
        super().__init__(dropout=dropout, **kwargs)
        self.mode = mode
        self.bottom_lr, self.middle_lr, self.top_lr = (
            float(bottom_lr),
            float(middle_lr),
            float(top_lr),
        )
        self.head_lr, self.extra_lr = float(head_lr), float(extra_lr)
        hidden = int(getattr(self, "hidden_dim", 384))
        self.extra_blocks = nn.ModuleList(
            [
                ZeroGatedCrossModalBlock(hidden, 8, dropout)
                for _ in range(num_extra_blocks)
            ]
        )
        self.shared_head = SharedSpanHead(hidden)
        self.matched_head = MatchedFullSpanHead(hidden, rank)
        self.c5_temperature = 0.1
        self._query_state: Optional[Tensor] = None
        self._pad_state: Optional[Tensor] = None
        self._post_tokens: Optional[Tensor] = None
        self._hook_calls = 0
        self._disable_extra = False
        layer = self._last_temporal_layer()
        self._pre_handle = layer.register_forward_pre_hook(self._inject_extra)
        self._post_handle = layer.register_forward_hook(self._capture_extra)
        if mode == "C1_frozen":
            for p in self.parameters():
                p.requires_grad_(False)
            for p in self.shared_head.parameters():
                p.requires_grad_(True)
        if mode in ("C1_frozen", "C2_random", "C3_identity", "C4_stage32_head"):
            for p in self.matched_head.parameters():
                p.requires_grad_(False)
        if mode in ("C4_stage32_head", "C5_contrast"):
            for p in self.shared_head.parameters():
                p.requires_grad_(False)
        if mode in ("C2_random", "C3_identity", "C5_contrast"):
            active_prefixes = (
                "stem.",
                "extra_blocks.",
                "shared_head.",
                "matched_head.",
            )
            inactive_stem_heads = (
                "stem.evidence_readout.",
                "stem.support_readout.",
                "stem.start_transition_readout.",
                "stem.end_transition_readout.",
            )
            for name, parameter in self.named_parameters():
                if not name.startswith(active_prefixes) or name.startswith(
                    inactive_stem_heads
                ):
                    parameter.requires_grad_(False)
        self.initialization_audit = {
            "checkpoint": init_checkpoint,
            "stage32_prediction_fusion": False,
            "teacher_kd": False,
            "state_loaded": False,
        }

    def _last_temporal_layer(self) -> nn.Module:
        layers = getattr(
            getattr(getattr(self, "stem", None), "temporal_encoder", None),
            "layers",
            None,
        )
        if layers is None or not len(layers):
            raise RuntimeError("Stage55 requires stem.temporal_encoder.layers")
        return layers[-1]

    def _inject_extra(self, layer: nn.Module, args: tuple[Any, ...]):
        if (
            self._query_state is None
            or self._pad_state is None
            or self.mode == "C1_frozen"
            or self._disable_extra
        ):
            return args
        tokens = args[0]
        batch_first = bool(
            getattr(getattr(layer, "self_attn", None), "batch_first", True)
        )
        x = tokens if batch_first else tokens.transpose(0, 1)
        for block in self.extra_blocks:
            x = block(x, self._query_state, self._pad_state)
        x = x if batch_first else x.transpose(0, 1)
        return (x,) + tuple(args[1:])

    def _capture_extra(self, layer: nn.Module, args: tuple[Any, ...], output: Any):
        del layer, args
        value = output[0] if isinstance(output, tuple) else output
        batch_first = bool(
            getattr(
                getattr(self._last_temporal_layer(), "self_attn", None),
                "batch_first",
                True,
            )
        )
        self._post_tokens = value if batch_first else value.transpose(0, 1)
        self._hook_calls += 1

    def _parse(
        self,
        inputs: Any,
        query: Optional[Tensor],
        video_pad: Optional[Tensor],
        query_pad: Optional[Tensor],
    ):
        if isinstance(inputs, Mapping):
            bad = {"gt_spans", "gt_span_mask", "span_labels", "targets"}.intersection(
                inputs
            )
            if bad:
                raise AssertionError(f"ground truth leaked into forward: {sorted(bad)}")
            return (
                inputs["src_vid"],
                inputs["src_txt"],
                inputs.get("video_padding_mask"),
                inputs.get("query_padding_mask"),
            )
        if query is None:
            raise TypeError("query tensor required")
        return inputs, query, video_pad, query_pad

    def _query_summary(self, query: Tensor, query_pad: Optional[Tensor]) -> Tensor:
        pad = (
            torch.zeros(query.shape[:2], dtype=torch.bool, device=query.device)
            if query_pad is None
            else query_pad.bool()
        )
        return _safe_query(self.stem, query, pad)

    def forward(
        self,
        inputs: Any,
        query_features: Optional[Tensor] = None,
        video_padding_mask: Optional[Tensor] = None,
        query_padding_mask: Optional[Tensor] = None,
    ) -> Stage55Output:
        video, query, video_pad, query_pad = self._parse(
            inputs, query_features, video_padding_mask, query_padding_mask
        )
        video_pad = (
            torch.zeros(video.shape[:2], dtype=torch.bool, device=video.device)
            if video_pad is None
            else video_pad.bool()
        )
        self._query_state = self._query_summary(query, query_pad)
        self._pad_state = video_pad
        self._post_tokens = None
        self._hook_calls = 0
        self._disable_extra = False
        inherited = super().forward(video, query, video_pad, query_pad)
        tokens = self._post_tokens
        if tokens is None or self._hook_calls != 1:
            raise RuntimeError("Stage55 last-layer capture must execute exactly once")
        valid = inherited.span_valid_mask.bool()
        if self.mode == "C4_stage32_head":
            # C4 is the inherited-head control. Keep the matched head for the
            # equal-parameter contract, but never let it determine C4 output.
            logits = torch.nan_to_num(inherited.span_logits)
            valid = inherited.span_valid_mask.bool()
            inherited_probs = getattr(inherited, "span_probs", None)
            probs = (
                inherited_probs
                if inherited_probs is not None
                else masked_softmax(
                    logits.flatten(1).float(), valid.flatten(1)
                ).reshape_as(logits)
            ).to(logits.dtype)
            evidence = torch.nan_to_num(
                getattr(inherited, "evidence_logits", torch.zeros_like(logits[..., 0]))
            )
            start = torch.nan_to_num(getattr(inherited, "start_logits", evidence))
            end = torch.nan_to_num(getattr(inherited, "end_logits", evidence))
            contrast = torch.zeros_like(logits)
        elif self.mode == "C5_contrast":
            logits, probs, valid, start, end, contrast = self.matched_head(
                tokens, self._query_state, video_pad
            )
            evidence = 0.5 * (start + end)
        else:
            logits, probs, valid, start, end = self.shared_head(
                tokens, self._query_state, video_pad
            )
            contrast = torch.zeros_like(logits)
            evidence = 0.5 * (start + end)
        return Stage55Output(
            span_logits=logits,
            span_probs=probs,
            span_valid_mask=valid,
            start_logits=start,
            end_logits=end,
            evidence_logits=torch.nan_to_num(evidence),
            token_features=tokens,
            query_features=self._query_state,
            contrast_scores=contrast,
            inherited_span_logits=torch.nan_to_num(inherited.span_logits),
        )

    @staticmethod
    def _iou(outputs: Stage55Output, batch: Any) -> Tensor:
        spans = batch.targets["gt_spans"].to(outputs.span_logits.dtype)
        mask = batch.targets["gt_span_mask"].bool()
        pad = batch.inputs["video_padding_mask"].bool()
        length = outputs.span_logits.shape[-1]
        count = (~pad).sum(1).clamp_min(1).to(spans.dtype)
        idx = torch.arange(length, device=spans.device, dtype=spans.dtype)
        ps = idx[None, :, None] / count[:, None, None]
        pe = (idx[None, None, :] + 1.0) / count[:, None, None]
        gs = torch.minimum(spans[..., 0], spans[..., 1])[:, None, None]
        ge = torch.maximum(spans[..., 0], spans[..., 1])[:, None, None]
        inter = (
            torch.minimum(pe[..., None], ge) - torch.maximum(ps[..., None], gs)
        ).clamp_min(0.0)
        iou = inter / (
            (pe - ps)[..., None].clamp_min(0.0) + (ge - gs).clamp_min(0.0) - inter
        ).clamp_min(1e-6)
        return (
            iou.masked_fill(~mask[:, None, None], 0.0)
            .amax(-1)
            .masked_fill(~outputs.span_valid_mask, 0.0)
        )

    def compute_loss(
        self, outputs: Stage55Output, batch: Any, teacher_outputs: Any, epoch: int
    ) -> LossResult:
        if teacher_outputs is not None:
            raise AssertionError("Stage55 forbids Stage32 prediction/KD fusion")
        iou = self._iou(outputs, batch)
        valid = outputs.span_valid_mask
        target = masked_softmax(
            (iou / self.c5_temperature).flatten(1), valid.flatten(1)
        ).reshape_as(iou)
        listwise = (
            -(target * torch.log(outputs.span_probs.clamp_min(1e-8)))
            .flatten(1)
            .sum(1)
            .mean()
        )
        inherited_probs = masked_softmax(
            outputs.inherited_span_logits.flatten(1).float(), valid.flatten(1)
        ).reshape_as(outputs.inherited_span_logits)
        anchor_listwise = (
            -(target * torch.log(inherited_probs.clamp_min(1e-8)))
            .flatten(1)
            .sum(1)
            .mean()
        )
        pad = batch.inputs["video_padding_mask"].bool()
        center = (
            torch.arange(
                outputs.start_logits.shape[-1], device=iou.device, dtype=iou.dtype
            )[None, :]
            + 0.5
        ) / (~pad).sum(1).clamp_min(1).to(iou.dtype)[:, None]
        spans = batch.targets["gt_spans"].to(iou.dtype)
        mask = batch.targets["gt_span_mask"].bool()
        gs = torch.minimum(spans[..., 0], spans[..., 1])
        ge = torch.maximum(spans[..., 0], spans[..., 1])
        st = (
            torch.exp(
                -0.5
                * (
                    (
                        (center[..., None] - gs[:, None])
                        * (~pad).sum(1)[:, None, None]
                        / 1.5
                    )
                    ** 2
                )
            )
            .masked_fill(~mask[:, None], 0.0)
            .amax(-1)
            .masked_fill(pad, 0.0)
        )
        et = (
            torch.exp(
                -0.5
                * (
                    (
                        (center[..., None] - ge[:, None])
                        * (~pad).sum(1)[:, None, None]
                        / 1.5
                    )
                    ** 2
                )
            )
            .masked_fill(~mask[:, None], 0.0)
            .amax(-1)
            .masked_fill(pad, 0.0)
        )
        endpoint = 0.5 * (
            F.binary_cross_entropy_with_logits(outputs.start_logits[~pad], st[~pad])
            + F.binary_cross_entropy_with_logits(outputs.end_logits[~pad], et[~pad])
        )
        contrast = outputs.span_logits.flatten(1)
        target_flat, valid_flat = iou.flatten(1), valid.flatten(1)
        pos = (
            contrast.masked_fill(~valid_flat, -1e4)
            .gather(1, target_flat.argmax(1)[:, None])
            .squeeze(1)
        )
        negative = contrast.masked_fill(~valid_flat | (target_flat > 0.3), -1e4).amax(1)
        hard = F.relu(0.2 - pos + negative).mean()
        calibration = (
            F.smooth_l1_loss(torch.sigmoid(outputs.contrast_scores[valid]), iou[valid])
            if self.mode == "C5_contrast"
            else listwise * 0.0
        )
        loss = (
            listwise
            + (
                0.05 * anchor_listwise
                if self.mode != "C1_frozen"
                else 0.0 * anchor_listwise
            )
            + 0.15 * endpoint
            + (
                0.25 * hard + 0.1 * calibration
                if self.mode == "C5_contrast"
                else 0.0 * hard
            )
        )
        return LossResult(
            loss,
            {
                "listwise": listwise.detach(),
                "endpoint": endpoint.detach(),
                "hard_negative": hard.detach(),
                "contrastive": calibration.detach(),
                "anchor_listwise": anchor_listwise.detach(),
                "epoch": float(epoch),
            },
        )

    def diagnostics(self, outputs: Stage55Output, batch: Any) -> Mapping[str, Tensor]:
        del batch
        valid = outputs.span_valid_mask
        return {
            "probability_error": (outputs.span_probs.flatten(1).sum(1) - 1.0)
            .abs()
            .max(),
            "zero_gate_mean": torch.stack(
                [torch.tanh(b.gate) for b in self.extra_blocks]
            ).mean(),
            "token_norm": outputs.token_features.float().norm(dim=-1).mean(),
            "inherited_logit_delta": (
                outputs.span_logits - outputs.inherited_span_logits
            )
            .abs()[valid]
            .mean(),
        }

    def set_epoch(self, epoch: int, training: bool) -> Mapping[str, Any]:
        return {
            "mode": self.mode,
            "epoch": int(epoch),
            "training": bool(training),
            "fixed_final": True,
            "stage32_prediction_fusion": False,
            "teacher_kd": False,
        }

    def parameter_groups(self) -> list[ParameterGroup]:
        groups: list[ParameterGroup] = []
        used: set[int] = set()
        named = list(self.named_parameters())

        def add(name: str, params: list[nn.Parameter], lr: float):
            params = [p for p in params if p.requires_grad and id(p) not in used]
            if params:
                groups.append(ParameterGroup(name, params, float(lr)))
                used.update(id(p) for p in params)

        base = [
            (n, p)
            for n, p in named
            if not (
                n.startswith("extra_blocks")
                or n.startswith("shared_head")
                or n.startswith("matched_head")
            )
        ]
        if self.mode == "C1_frozen":
            add("stage32_frozen", [p for _, p in base], 0.0)
        else:
            add(
                "inherited_bottom",
                [
                    p
                    for n, p in base
                    if "temporal_encoder.layers.0" in n
                    or "temporal_encoder.layers.1" in n
                ],
                self.bottom_lr,
            )
            add(
                "inherited_middle",
                [p for n, p in base if "temporal_encoder.layers.2" in n],
                self.middle_lr,
            )
            add("inherited_top", [p for n, p in base], self.top_lr)
        add("extra_blocks", list(self.extra_blocks.parameters()), self.extra_lr)
        add("shared_head", list(self.shared_head.parameters()), self.head_lr)
        add("matched_head", list(self.matched_head.parameters()), self.head_lr)
        missing = [n for n, p in named if p.requires_grad and id(p) not in used]
        if missing:
            raise RuntimeError(f"Stage55 parameter group omission: {missing}")
        return groups


__all__ = [
    "MODES",
    "INHERITED_MODES",
    "STAGE32_KWARGS",
    "Stage55Model",
    "Stage55Output",
    "SharedSpanHead",
    "MatchedFullSpanHead",
]
