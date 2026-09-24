"""Evidence-incremental dense full-span tri-field ranker.

This second-generation model addresses three failures measured in the first
verifier pilots: sparse hardest-pair gradients, unconstrained score scale and
role redundancy.  Evidence uses the inside semantic carrier, Support uses
inside-minus-outside low-frequency context, and Transition uses candidate-
relative entering/leaving phase.  Support and Transition can be made
incremental to Evidence by candidate-wise Gram-Schmidt residualization.

The candidate ranker is trained with a dense IoU-listwise target plus top-k
near/far hard negatives.  Its score is standardized per video before entering
the end-to-end span logits through a bounded positive gate.  Coordinates are
never moved and no checkpoint or old prediction is read.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import log, sqrt
from typing import Any, Mapping, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from training.config import RunnerConfig
from training.contracts import LossResult
from training.probability import masked_softmax
from backbone.model import STAGE32_KWARGS

from .scratch_plugin import ScratchE2EModel, initialize_scratch_e2e
from .semantic_carrier_plugin import build_repository_data_with_saliency


def _average_tokens(tokens: Tensor, kernel: int, padding_mask: Tensor) -> Tensor:
    values = F.avg_pool1d(
        tokens.transpose(1, 2), kernel_size=kernel, stride=1, padding=kernel // 2
    ).transpose(1, 2)
    return values.masked_fill(padding_mask.unsqueeze(-1), 0.0)


def _span_mean_vector(values: Tensor) -> Tensor:
    length = values.shape[1]
    prefix = F.pad(values.cumsum(1), (0, 0, 1, 0))
    index = torch.arange(length, device=values.device)
    starts = index[:, None]
    ends = index[None, :]
    width = ends - starts + 1
    return (prefix[:, ends + 1] - prefix[:, starts]) / width.clamp_min(1)[
        None, :, :, None
    ]


def _context_contrast_vector(values: Tensor, radius: int) -> Tensor:
    _, length, _ = values.shape
    prefix = F.pad(values.cumsum(1), (0, 0, 1, 0))
    index = torch.arange(length, device=values.device)
    starts = index[:, None]
    ends = index[None, :]
    inside = _span_mean_vector(values)
    left_begin = (starts - radius).clamp_min(0)
    right_begin = ends + 1
    right_end = (ends + 1 + radius).clamp_max(length)
    left_sum = prefix[:, starts] - prefix[:, left_begin]
    right_sum = prefix[:, right_end] - prefix[:, right_begin]
    count = (starts - left_begin) + (right_end - right_begin)
    context = (left_sum + right_sum) / count.clamp_min(1)[None, :, :, None]
    global_mean = values.mean(1)[:, None, None, :]
    context = torch.where(count[None, :, :, None] > 0, context, global_mean)
    return inside - context


def _boundary_phase_vectors(
    start_values: Tensor,
    end_values: Tensor,
    padding_mask: Tensor,
    radius: int,
) -> tuple[Tensor, Tensor]:
    batch, length, _ = start_values.shape
    device = start_values.device
    valid_length = (~padding_mask).sum(1).clamp_min(1)
    index = torch.arange(length, device=device)
    starts = index[None, :, None].expand(batch, length, length)
    ends = index[None, None, :].expand(batch, length, length)
    limit = valid_length[:, None, None]
    batch_index = torch.arange(batch, device=device)[:, None, None]

    def segment_mean(values: Tensor, begin: Tensor, finish: Tensor):
        prefix = F.pad(values.cumsum(1), (0, 0, 1, 0))
        begin = begin.clamp_min(0).minimum(limit)
        finish = finish.clamp_min(0).minimum(limit)
        count = (finish - begin).clamp_min(0)
        total = prefix[batch_index, finish] - prefix[batch_index, begin]
        mean = total / count.clamp_min(1)[..., None].to(values.dtype)
        return mean, count

    start_inside, _ = segment_mean(
        start_values, starts, torch.minimum(ends + 1, starts + radius)
    )
    start_outside, start_count = segment_mean(start_values, starts - radius, starts)
    end_inside, _ = segment_mean(
        end_values, torch.maximum(starts, ends - radius + 1), ends + 1
    )
    end_outside, end_count = segment_mean(end_values, ends + 1, ends + 1 + radius)
    start_phase = torch.where(
        start_count[..., None] > 0,
        start_inside - start_outside,
        torch.zeros_like(start_inside),
    )
    end_phase = torch.where(
        end_count[..., None] > 0,
        end_inside - end_outside,
        torch.zeros_like(end_inside),
    )
    return start_phase, end_phase


def _orthogonal_residual(value: Tensor, anchor: Tensor) -> Tensor:
    coefficient = (value * anchor).sum(-1, keepdim=True) / anchor.square().sum(
        -1, keepdim=True
    ).clamp_min(1.0e-6)
    return value - coefficient * anchor


def _score_correlation(left: Tensor, right: Tensor, valid: Tensor) -> Tensor:
    x = left[valid].float()
    y = right[valid].float()
    x = x - x.mean()
    y = y - y.mean()
    return (x * y).mean() / (
        x.square().mean().sqrt() * y.square().mean().sqrt()
    ).clamp_min(1.0e-6)


def _masked_mean_std(score: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
    weight = valid.to(score.dtype)
    count = weight.flatten(1).sum(1).clamp_min(1.0)
    mean = (score * weight).flatten(1).sum(1) / count
    centered = (score - mean[:, None, None]) * weight
    variance = centered.square().flatten(1).sum(1) / count
    return mean, variance.clamp_min(1.0e-4).sqrt()


@dataclass
class DenseTriFieldState:
    evidence_clip: Tensor
    support_clip: Tensor
    transition_start: Tensor
    transition_end: Tensor
    evidence_score: Tensor
    support_score: Tensor
    transition_score: Tensor
    rank_score: Tensor
    standardized_score: Tensor
    logit_delta: Tensor
    effective_gate: Tensor
    rank_score_std: Tensor
    evidence_support_cosine: Tensor
    evidence_transition_cosine: Tensor
    support_transition_cosine: Tensor
    support_residual_ratio: Tensor
    transition_residual_ratio: Tensor


class DenseIncrementalTriFieldRanker(nn.Module):
    def __init__(
        self,
        hidden: int = 384,
        rank: int = 64,
        context_radius: int = 3,
        orthogonalize_fields: bool = True,
        integrate_score: bool = True,
        initial_gate: float = 0.05,
        maximum_gate: float = 0.5,
        logit_scale: float = 10.0,
    ) -> None:
        super().__init__()
        if not 0.0 < initial_gate < maximum_gate:
            raise ValueError("initial gate must be inside (0, maximum_gate)")
        self.context_radius = int(context_radius)
        self.orthogonalize_fields = bool(orthogonalize_fields)
        self.integrate_score = bool(integrate_score)
        self.maximum_gate = float(maximum_gate)

        def projection():
            return nn.Sequential(
                nn.LayerNorm(hidden), nn.Linear(hidden, rank), nn.GELU()
            )

        self.evidence_tokens = projection()
        self.support_tokens = projection()
        self.transition_start_tokens = projection()
        self.transition_end_tokens = projection()
        self.evidence_query = projection()
        self.support_query = projection()
        self.transition_query = projection()
        self.rank_query = projection()
        self.evidence_clip_head = nn.Linear(rank, 1)
        self.support_clip_head = nn.Linear(rank, 1)
        self.transition_start_head = nn.Linear(rank, 1)
        self.transition_end_head = nn.Linear(rank, 1)
        self.duration = nn.Sequential(
            nn.Linear(1, rank), nn.GELU(), nn.Linear(rank, rank)
        )
        self.candidate_encoder = nn.Sequential(
            nn.LayerNorm(5 * rank),
            nn.Linear(5 * rank, 2 * rank),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2 * rank, rank),
            nn.LayerNorm(rank),
        )
        self.register_buffer("fixed_logit_scale", torch.tensor(float(logit_scale)))
        gate_probability = initial_gate / maximum_gate
        self.raw_gate = nn.Parameter(
            torch.tensor(log(gate_probability / (1.0 - gate_probability)))
        )
        if not self.integrate_score:
            self.raw_gate.requires_grad_(False)

    def reset_output_layers(self, initial_gate: float) -> None:
        gate_probability = initial_gate / self.maximum_gate
        with torch.no_grad():
            for module in (
                self.evidence_clip_head,
                self.support_clip_head,
                self.transition_start_head,
                self.transition_end_head,
            ):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            self.raw_gate.fill_(log(gate_probability / (1.0 - gate_probability)))

    def forward(
        self,
        tokens: Tensor,
        query: Tensor,
        padding_mask: Tensor,
        span_valid_mask: Tensor,
    ) -> DenseTriFieldState:
        semantic = tokens.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        support_input = _average_tokens(semantic, 3, padding_mask)
        qe = torch.tanh(self.evidence_query(query))[:, None, :]
        qs = torch.tanh(self.support_query(query))[:, None, :]
        qt = torch.tanh(self.transition_query(query))[:, None, :]
        evidence_tokens = self.evidence_tokens(semantic) * (1.0 + 0.25 * qe)
        support_tokens = self.support_tokens(support_input) * (1.0 + 0.25 * qs)
        transition_start_tokens = self.transition_start_tokens(semantic) * (
            1.0 + 0.25 * qt
        )
        transition_end_tokens = self.transition_end_tokens(semantic) * (1.0 + 0.25 * qt)
        token_mask = padding_mask.unsqueeze(-1)
        evidence_tokens = evidence_tokens.masked_fill(token_mask, 0.0)
        support_tokens = support_tokens.masked_fill(token_mask, 0.0)
        transition_start_tokens = transition_start_tokens.masked_fill(token_mask, 0.0)
        transition_end_tokens = transition_end_tokens.masked_fill(token_mask, 0.0)

        evidence = _span_mean_vector(evidence_tokens)
        support_raw = _context_contrast_vector(support_tokens, self.context_radius)
        transition_start, transition_end = _boundary_phase_vectors(
            transition_start_tokens,
            transition_end_tokens,
            padding_mask,
            self.context_radius,
        )
        transition_raw = (transition_start + transition_end) / sqrt(2.0)
        support = support_raw
        transition = transition_raw
        if self.orthogonalize_fields:
            support = _orthogonal_residual(support, evidence)
            transition = _orthogonal_residual(transition, evidence)
            transition = _orthogonal_residual(transition, support)

        mask = ~span_valid_mask[..., None]
        evidence = evidence.masked_fill(mask, 0.0)
        support_raw = support_raw.masked_fill(mask, 0.0)
        transition_raw = transition_raw.masked_fill(mask, 0.0)
        support = support.masked_fill(mask, 0.0)
        transition = transition.masked_fill(mask, 0.0)

        batch, length, _, rank = evidence.shape
        valid_tokens = (~padding_mask).sum(1).clamp_min(1)
        index = torch.arange(length, device=tokens.device)
        width = index[None, :] - index[:, None] + 1
        duration = width.to(tokens.dtype)[None, :, :, None] / valid_tokens[
            :, None, None, None
        ].to(tokens.dtype)
        duration_vector = self.duration(duration).expand(batch, -1, -1, -1)
        representation = torch.cat(
            (
                evidence,
                evidence * support,
                evidence * transition,
                support * transition,
                duration_vector,
            ),
            dim=-1,
        )
        candidate = F.normalize(self.candidate_encoder(representation).float(), dim=-1)
        rank_query = F.normalize(self.rank_query(query).float(), dim=-1)
        rank_score = self.fixed_logit_scale.float() * (
            candidate * rank_query[:, None, None, :]
        ).sum(-1)
        rank_score = rank_score.masked_fill(~span_valid_mask, 0.0)
        mean, score_std = _masked_mean_std(rank_score, span_valid_mask)
        standardized = (rank_score - mean[:, None, None]) / score_std[:, None, None]
        standardized = 3.0 * torch.tanh(standardized / 3.0)
        standardized = standardized.masked_fill(~span_valid_mask, 0.0)
        effective_gate = (
            self.maximum_gate * torch.sigmoid(self.raw_gate)
            if self.integrate_score
            else self.raw_gate.detach() * 0.0
        )
        logit_delta = effective_gate * standardized

        q = rank_query[:, None, None, :]
        evidence_score = self.fixed_logit_scale * (
            F.normalize(evidence.float(), dim=-1) * q
        ).sum(-1)
        support_score = self.fixed_logit_scale * (
            F.normalize(support.float(), dim=-1) * q
        ).sum(-1)
        transition_score = self.fixed_logit_scale * (
            F.normalize(transition.float(), dim=-1) * q
        ).sum(-1)
        evidence_score = evidence_score.masked_fill(~span_valid_mask, 0.0)
        support_score = support_score.masked_fill(~span_valid_mask, 0.0)
        transition_score = transition_score.masked_fill(~span_valid_mask, 0.0)

        def cosine(left: Tensor, right: Tensor) -> Tensor:
            value = F.cosine_similarity(left.float(), right.float(), dim=-1)
            return value[span_valid_mask].mean()

        def residual_ratio(value: Tensor, raw: Tensor) -> Tensor:
            numerator = value.float().norm(dim=-1)
            denominator = raw.float().norm(dim=-1).clamp_min(1.0e-6)
            return (numerator / denominator)[span_valid_mask].mean()

        return DenseTriFieldState(
            evidence_clip=self.evidence_clip_head(evidence_tokens).squeeze(-1),
            support_clip=self.support_clip_head(support_tokens).squeeze(-1),
            transition_start=self.transition_start_head(
                transition_start_tokens
            ).squeeze(-1),
            transition_end=self.transition_end_head(transition_end_tokens).squeeze(-1),
            evidence_score=evidence_score,
            support_score=support_score,
            transition_score=transition_score,
            rank_score=rank_score,
            standardized_score=standardized,
            logit_delta=logit_delta,
            effective_gate=effective_gate,
            rank_score_std=score_std.mean(),
            evidence_support_cosine=cosine(evidence, support),
            evidence_transition_cosine=cosine(evidence, transition),
            support_transition_cosine=cosine(support, transition),
            support_residual_ratio=residual_ratio(support, support_raw),
            transition_residual_ratio=residual_ratio(transition, transition_raw),
        )


class DenseTriFieldScratchModel(ScratchE2EModel):
    def __init__(
        self,
        *args: Any,
        orthogonalize_fields: bool = True,
        integrate_rank_score: bool = True,
        ranker_rank: int = 64,
        context_radius: int = 3,
        ranker_initial_gate: float = 0.05,
        ranker_maximum_gate: float = 0.5,
        dense_rank_weight: float = 0.5,
        role_proxy_weight: float = 0.1,
        iou_temperature: float = 0.1,
        required_margin: float = 0.2,
        hard_negative_topk: int = 8,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.orthogonalize_fields = bool(orthogonalize_fields)
        self.integrate_rank_score = bool(integrate_rank_score)
        self.dense_rank_weight = float(dense_rank_weight)
        self.role_proxy_weight = float(role_proxy_weight)
        self.iou_temperature = float(iou_temperature)
        self.required_margin = float(required_margin)
        self.hard_negative_topk = int(hard_negative_topk)
        self.tri_field_ranker = DenseIncrementalTriFieldRanker(
            hidden=384,
            rank=ranker_rank,
            context_radius=context_radius,
            orthogonalize_fields=orthogonalize_fields,
            integrate_score=integrate_rank_score,
            initial_gate=ranker_initial_gate,
            maximum_gate=ranker_maximum_gate,
        )

    def forward(
        self,
        inputs: Any,
        query_features: Optional[Tensor] = None,
        video_padding_mask: Optional[Tensor] = None,
        query_padding_mask: Optional[Tensor] = None,
    ):
        output = super().forward(
            inputs, query_features, video_padding_mask, query_padding_mask
        )
        pad = (
            video_padding_mask.bool()
            if video_padding_mask is not None
            else self._video_pad(inputs, output)
        )
        state = self.tri_field_ranker(
            output.token_features,
            output.query_features,
            pad,
            output.span_valid_mask,
        )
        if output.extension_output is None:
            raise RuntimeError("identity extension output missing")
        output.extension_output.state = state
        logits = output.span_logits.float() + state.logit_delta.float()
        probs = masked_softmax(
            logits.flatten(1), output.span_valid_mask.flatten(1)
        ).reshape_as(logits)
        return replace(
            output,
            span_logits=logits.to(output.span_logits.dtype),
            span_probs=probs.to(output.span_probs.dtype),
            evidence_logits=state.evidence_clip.to(output.evidence_logits.dtype),
        )

    @staticmethod
    def _clip_occupancy(batch: Any, length: int, dtype: torch.dtype) -> Tensor:
        pad = batch.inputs["video_padding_mask"].bool()
        count = (~pad).sum(1).clamp_min(1).to(dtype)
        center = (
            torch.arange(length, device=pad.device, dtype=dtype)[None] + 0.5
        ) / count[:, None]
        spans = batch.targets["gt_spans"].to(dtype)
        mask = batch.targets["gt_span_mask"].bool()
        start = torch.minimum(spans[..., 0], spans[..., 1])
        end = torch.maximum(spans[..., 0], spans[..., 1])
        occupancy = (
            (center[..., None] >= start[:, None])
            & (center[..., None] <= end[:, None])
            & mask[:, None]
        ).any(-1)
        return occupancy.to(dtype).masked_fill(pad, 0.0)

    @staticmethod
    def _boundary_targets(batch: Any, length: int, dtype: torch.dtype):
        pad = batch.inputs["video_padding_mask"].bool()
        count = (~pad).sum(1).clamp_min(1)
        center = (
            torch.arange(length, device=pad.device, dtype=dtype)[None] + 0.5
        ) / count.to(dtype)[:, None]
        spans = batch.targets["gt_spans"].to(dtype)
        mask = batch.targets["gt_span_mask"].bool()
        start = torch.minimum(spans[..., 0], spans[..., 1])
        end = torch.maximum(spans[..., 0], spans[..., 1])
        scale = count[:, None, None]
        start_target = (
            torch.exp(
                -0.5 * (((center[..., None] - start[:, None]) * scale / 1.5) ** 2)
            )
            .masked_fill(~mask[:, None], 0.0)
            .amax(-1)
            .masked_fill(pad, 0.0)
        )
        end_target = (
            torch.exp(-0.5 * (((center[..., None] - end[:, None]) * scale / 1.5) ** 2))
            .masked_fill(~mask[:, None], 0.0)
            .amax(-1)
            .masked_fill(pad, 0.0)
        )
        return start_target, end_target

    @staticmethod
    def _evidence_proxy(evidence: Tensor, batch: Any) -> Tensor:
        pad = batch.inputs["video_padding_mask"].bool()
        labels = batch.targets["saliency_all_labels"][:, : evidence.shape[1]].clamp(
            0, 1
        )
        bce = F.binary_cross_entropy_with_logits(evidence[~pad], labels[~pad])
        positive_index = batch.targets["saliency_pos_labels"].clamp_max(
            evidence.shape[1] - 1
        )
        negative_index = batch.targets["saliency_neg_labels"].clamp_max(
            evidence.shape[1] - 1
        )
        positive = evidence.gather(1, positive_index)
        negative = evidence.gather(1, negative_index)
        return (
            bce
            + 2.0
            * F.relu(0.15 + torch.sigmoid(negative) - torch.sigmoid(positive)).mean()
        )

    def _dense_listwise(self, score: Tensor, iou: Tensor, valid: Tensor) -> Tensor:
        flat_score = score.float().flatten(1)
        flat_iou = iou.float().flatten(1)
        flat_valid = valid.flatten(1)
        target_logits = (flat_iou / self.iou_temperature).masked_fill(
            ~flat_valid, -1.0e4
        )
        target = torch.softmax(target_logits, dim=1)
        log_probability = torch.log_softmax(
            flat_score.masked_fill(~flat_valid, -1.0e4), dim=1
        )
        cross_entropy = -(target * log_probability).sum(1)
        normalizer = flat_valid.sum(1).clamp_min(2).float().log()
        return (cross_entropy / normalizer).mean()

    def _topk_hard_rank(
        self, score: Tensor, iou: Tensor, valid: Tensor, low: float, high: float
    ) -> tuple[Tensor, Tensor, Tensor]:
        flat_score = score.float().flatten(1)
        flat_iou = iou.float().flatten(1)
        flat_valid = valid.flatten(1)
        positive_index = flat_iou.argmax(1, keepdim=True)
        positive = flat_score.gather(1, positive_index).squeeze(1)
        negative_mask = flat_valid & flat_iou.ge(low) & flat_iou.lt(high)
        # The discretized best candidate can itself have IoU below ``high``.
        # It is the positive by definition and must never also be a negative.
        negative_mask = negative_mask.scatter(1, positive_index, False)
        count = negative_mask.sum(1)
        has = count.gt(0)
        if not bool(has.any()):
            zero = score.float().sum() * 0.0
            return zero, zero.detach(), zero.detach()
        k = min(self.hard_negative_topk, flat_score.shape[1])
        negative = flat_score.masked_fill(~negative_mask, -1.0e4).topk(k, dim=1).values
        rank = torch.arange(k, device=score.device)[None]
        selected = rank < count[:, None].clamp_max(k)
        losses = F.softplus(self.required_margin - (positive[:, None] - negative))
        loss = losses[selected].mean()
        hardest = negative[:, 0]
        margin = positive[has] - hardest[has]
        return loss, margin.mean().detach(), margin.gt(0).float().mean().detach()

    def compute_loss(
        self, outputs: Any, batch: Any, teacher_outputs: Any, epoch: int
    ) -> LossResult:
        base = super().compute_loss(outputs, batch, teacher_outputs, epoch)
        state = outputs.extension_output.state if outputs.extension_output else None
        if not isinstance(state, DenseTriFieldState):
            raise RuntimeError("dense tri-field state missing before loss")
        iou = self._iou(outputs, batch)
        valid = outputs.span_valid_mask.bool()
        flat_iou = iou.float().flatten(1)
        best_iou = flat_iou.amax(1)
        near_self_collision_without_exclusion = (
            (best_iou.ge(0.30) & best_iou.lt(0.95)).float().mean()
        )
        listwise = self._dense_listwise(state.rank_score, iou, valid)
        far_rank, far_margin, far_positive_rate = self._topk_hard_rank(
            state.rank_score, iou, valid, 0.0, 0.30
        )
        near_rank, near_margin, near_positive_rate = self._topk_hard_rank(
            state.rank_score, iou, valid, 0.30, 0.95
        )
        dense_rank = (listwise + far_rank + near_rank) / 3.0
        _, final_far_margin, final_far_rate = self._topk_hard_rank(
            outputs.span_logits, iou, valid, 0.0, 0.30
        )
        _, final_near_margin, final_near_rate = self._topk_hard_rank(
            outputs.span_logits, iou, valid, 0.30, 0.95
        )

        pad = batch.inputs["video_padding_mask"].bool()
        occupancy = self._clip_occupancy(
            batch, state.support_clip.shape[1], state.support_clip.dtype
        )
        support_proxy = F.binary_cross_entropy_with_logits(
            state.support_clip[~pad], occupancy[~pad]
        )
        start_target, end_target = self._boundary_targets(
            batch, state.transition_start.shape[1], state.transition_start.dtype
        )
        transition_proxy = 0.5 * (
            F.binary_cross_entropy_with_logits(
                state.transition_start[~pad], start_target[~pad]
            )
            + F.binary_cross_entropy_with_logits(
                state.transition_end[~pad], end_target[~pad]
            )
        )
        evidence_proxy = self._evidence_proxy(state.evidence_clip, batch)
        role_proxy = (evidence_proxy + support_proxy + transition_proxy) / 3.0
        loss = (
            base.loss
            + self.dense_rank_weight * dense_rank
            + self.role_proxy_weight * role_proxy
        )
        metrics = dict(base.metrics)
        metrics.update(
            {
                "dense_trifield/rank": dense_rank.detach(),
                "dense_trifield/listwise": listwise.detach(),
                "dense_trifield/far_rank": far_rank.detach(),
                "dense_trifield/near_rank": near_rank.detach(),
                "dense_trifield/far_margin": far_margin,
                "dense_trifield/near_margin": near_margin,
                "dense_trifield/far_positive_rate": far_positive_rate,
                "dense_trifield/near_positive_rate": near_positive_rate,
                "dense_trifield/near_self_collision_without_exclusion": (
                    near_self_collision_without_exclusion.detach()
                ),
                "dense_trifield/final_far_margin": final_far_margin,
                "dense_trifield/final_near_margin": final_near_margin,
                "dense_trifield/final_far_positive_rate": final_far_rate,
                "dense_trifield/final_near_positive_rate": final_near_rate,
                "dense_trifield/evidence_proxy": evidence_proxy.detach(),
                "dense_trifield/support_proxy": support_proxy.detach(),
                "dense_trifield/transition_proxy": transition_proxy.detach(),
                "dense_trifield/role_proxy": role_proxy.detach(),
                "dense_trifield/rank_weight": self.dense_rank_weight,
                "dense_trifield/role_proxy_weight": self.role_proxy_weight,
            }
        )
        return LossResult(loss, metrics)

    def diagnostics(self, outputs: Any, batch: Any) -> Mapping[str, Tensor | float]:
        result = dict(super().diagnostics(outputs, batch))
        state = outputs.extension_output.state if outputs.extension_output else None
        if isinstance(state, DenseTriFieldState):
            valid = outputs.span_valid_mask.bool()
            carrier_logits = outputs.span_logits.float() - state.logit_delta.float()
            delta_to_carrier_std = state.logit_delta[valid].float().std(
                unbiased=False
            ) / carrier_logits[valid].float().std(unbiased=False).clamp_min(1.0e-6)
            result.update(
                {
                    "dense_trifield/effective_gate": state.effective_gate,
                    "dense_trifield/logit_delta_abs_mean": state.logit_delta[valid]
                    .float()
                    .abs()
                    .mean(),
                    "dense_trifield/rank_score_std": state.rank_score_std,
                    "dense_trifield/delta_to_carrier_std": delta_to_carrier_std,
                    "dense_trifield/evidence_support_score_corr": _score_correlation(
                        state.evidence_score, state.support_score, valid
                    ),
                    "dense_trifield/evidence_transition_score_corr": _score_correlation(
                        state.evidence_score, state.transition_score, valid
                    ),
                    "dense_trifield/support_transition_score_corr": _score_correlation(
                        state.support_score, state.transition_score, valid
                    ),
                    "dense_trifield/evidence_support_cosine": state.evidence_support_cosine,
                    "dense_trifield/evidence_transition_cosine": state.evidence_transition_cosine,
                    "dense_trifield/support_transition_cosine": state.support_transition_cosine,
                    "dense_trifield/support_residual_ratio": state.support_residual_ratio,
                    "dense_trifield/transition_residual_ratio": state.transition_residual_ratio,
                }
            )
        return result

    def experiment_contract(self) -> Mapping[str, Any]:
        contract = dict(super().experiment_contract())
        contract.update(
            {
                "dense_incremental_tri_field": True,
                "field_inputs": {
                    "Evidence": "inside_query_conditioned_semantic_mean",
                    "Support": "low_pass_inside_minus_outside_context",
                    "Transition": "combined_candidate_relative_enter_leave_phase",
                },
                "orthogonalize_fields": self.orthogonalize_fields,
                "rank_objective": "normalized_iou_listwise_plus_top8_near_far_contrastive",
                "rank_score_scale": "fixed_cosine_scale_10",
                "integration": "per_video_standardized_score_bounded_positive_gate",
                "integrate_rank_score": self.integrate_rank_score,
                "coordinate_movement": False,
                "old_prediction_score_fusion": False,
                "fixed_quota": False,
                "posthoc_reranking": False,
                "boundary_gate": False,
            }
        )
        return contract


def _freeze_boundary_gate_only_branch(model: DenseTriFieldScratchModel) -> list[str]:
    frozen = []
    if model.boundary_preserving.use_boundary_gate:
        raise AssertionError("dense tri-field baseline requires boundary gate off")
    for name, parameter in model.named_parameters():
        if name == "boundary_preserving.boundary_strength" or name.startswith(
            "boundary_preserving.boundary_head."
        ):
            parameter.requires_grad_(False)
            frozen.append(name)
    return frozen


def build_dense_trifield_model(
    config: Optional[RunnerConfig],
    orthogonalize_fields: bool = True,
    integrate_rank_score: bool = True,
    ranker_rank: int = 64,
    context_radius: int = 3,
    ranker_initial_gate: float = 0.05,
    ranker_maximum_gate: float = 0.5,
    dense_rank_weight: float = 0.5,
    role_proxy_weight: float = 0.1,
    iou_temperature: float = 0.1,
    required_margin: float = 0.2,
    hard_negative_topk: int = 8,
    extension_lr: float = 1.0e-4,
    scratch_total_epochs: int = 420,
    scratch_residual_gate: float = 0.05,
    **kwargs: Any,
) -> DenseTriFieldScratchModel:
    del config
    inherited = dict(STAGE32_KWARGS)
    inherited.update(kwargs)
    inherited["use_boundary_gate"] = False
    model = DenseTriFieldScratchModel(
        extension_lr=extension_lr,
        scratch_total_epochs=scratch_total_epochs,
        orthogonalize_fields=orthogonalize_fields,
        integrate_rank_score=integrate_rank_score,
        ranker_rank=ranker_rank,
        context_radius=context_radius,
        ranker_initial_gate=ranker_initial_gate,
        ranker_maximum_gate=ranker_maximum_gate,
        dense_rank_weight=dense_rank_weight,
        role_proxy_weight=role_proxy_weight,
        iou_temperature=iou_temperature,
        required_margin=required_margin,
        hard_negative_topk=hard_negative_topk,
        **inherited,
    )
    frozen = _freeze_boundary_gate_only_branch(model)
    model.initialization_audit = initialize_scratch_e2e(
        model, residual_gate=scratch_residual_gate
    )
    model.tri_field_ranker.reset_output_layers(ranker_initial_gate)
    model.initialization_audit["dense_tri_field_ranker"] = {
        "checkpoint": None,
        "boundary_gate": False,
        "inactive_boundary_parameters": frozen,
        "orthogonalize_fields": bool(orthogonalize_fields),
        "integrate_rank_score": bool(integrate_rank_score),
        "initial_effective_gate": (
            ranker_initial_gate if integrate_rank_score else 0.0
        ),
        "maximum_gate": ranker_maximum_gate,
        "dense_rank_weight": dense_rank_weight,
        "role_proxy_weight": role_proxy_weight,
    }
    return model


__all__ = [
    "DenseTriFieldState",
    "DenseTriFieldScratchModel",
    "build_repository_data_with_saliency",
    "build_dense_trifield_model",
]
