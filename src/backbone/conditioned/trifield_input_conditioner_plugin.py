"""End-to-end E/S/T conditioning before the EventFieldNet backbone.

Evidence is a channel-wise cross-modal agreement field. Support is an
occupancy-weighted temporal diffusion field. Transition is a signed local
high-frequency field and, in the anisotropic route, also acts as the barrier
that prevents Support from diffusing across semantic boundaries. The combined
raw-feature update is orthogonal, norm bounded and exactly norm preserving.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atanh, sqrt
from typing import Any, Mapping, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from backbone.model import STAGE32_KWARGS
from training.config import RunnerConfig
from training.contracts import LossResult

from .scratch_plugin import ScratchE2EModel, initialize_scratch_e2e
from .trifield_dense_ranker_plugin import (
    _freeze_boundary_gate_only_branch,
    _score_correlation,
    build_repository_data_with_saliency,
)


def _masked_query(text: Tensor, padding: Tensor) -> Tensor:
    weight = (~padding).to(text.dtype).unsqueeze(-1)
    return (text * weight).sum(1) / weight.sum(1).clamp_min(1.0)


def _masked_mean_std(value: Tensor, padding: Tensor) -> tuple[Tensor, Tensor]:
    weight = (~padding).to(value.dtype)
    count = weight.sum(1).clamp_min(1.0)
    mean = (value * weight).sum(1) / count
    variance = ((value - mean[:, None]).square() * weight).sum(1) / count
    return mean, variance.clamp_min(1.0e-6).sqrt()


def _left_right(value: Tensor) -> tuple[Tensor, Tensor]:
    return (
        torch.cat((value[:, :1], value[:, :-1]), dim=1),
        torch.cat((value[:, 1:], value[:, -1:]), dim=1),
    )


def _span_mean(value: Tensor) -> Tensor:
    length = value.shape[1]
    prefix = F.pad(value.cumsum(1), (1, 0))
    index = torch.arange(length, device=value.device)
    start, end = index[:, None], index[None, :]
    width = (end - start + 1).clamp_min(1).to(value.dtype)
    return (prefix[:, end + 1] - prefix[:, start]) / width[None]


@dataclass
class InputTriFieldState:
    evidence_score: Tensor
    support_score: Tensor
    enter_score: Tensor
    leave_score: Tensor
    role_updates: Tensor
    transition_start_input: Tensor
    transition_end_input: Tensor
    raw_update: Tensor
    protected_update: Tensor
    input_semantic: Tensor
    conditioned_semantic: Tensor
    effective_gates: Tensor
    transition_barrier: bool
    mean_conductance: Tensor
    max_update_ratio: float
    support_hops: int
    support_hop_mode: str
    support_second_hop_mix: Tensor
    transition_input_mode: str
    evidence_input_mode: str
    support_input_mode: str
    support_conditioning_position: str
    transition_conditioning_position: str
    query_identity: Tensor


class TriFieldInputConditioner(nn.Module):
    """Low-rank differentiated conditioning of the original 512-D video input."""

    def __init__(
        self,
        field_rank: int = 64,
        transition_barrier: bool = True,
        initial_field_gate: float = 0.05,
        output_gain: float = 0.10,
        max_update_ratio: float = 0.05,
        support_hops: int = 1,
        support_hop_mode: str = "fixed",
        transition_input_mode: str = "feature_delta",
        evidence_input_mode: str = "mean_query_product",
        support_input_mode: str = "diffusion_residual",
        support_conditioning_position: str = "pre_encoder",
        transition_conditioning_position: str = "pre_encoder",
    ) -> None:
        super().__init__()
        if field_rank < 8:
            raise ValueError("field_rank must be at least eight")
        if not 0.0 < initial_field_gate < 1.0:
            raise ValueError("initial_field_gate must be in (0, 1)")
        if not 0.0 < max_update_ratio < 0.5:
            raise ValueError("max_update_ratio must be in (0, 0.5)")
        if support_hops not in (1, 2):
            raise ValueError("support_hops must be one or two")
        if support_hop_mode not in ("fixed", "learned_global", "learned_local"):
            raise ValueError("unsupported support_hop_mode")
        if support_hops == 1 and support_hop_mode != "fixed":
            raise ValueError("learned Support hop mixing requires support_hops=2")
        if transition_input_mode not in (
            "feature_delta",
            "query_agreement_delta",
        ):
            raise ValueError("unsupported transition_input_mode")
        if evidence_input_mode not in (
            "mean_query_product",
            "token_attention_product",
        ):
            raise ValueError("unsupported evidence_input_mode")
        if support_input_mode not in (
            "diffusion_residual",
            "query_seed_transport",
            "query_agreement_transport",
        ):
            raise ValueError("unsupported support_input_mode")
        if support_conditioning_position not in ("pre_encoder", "post_encoder"):
            raise ValueError("unsupported support_conditioning_position")
        if transition_conditioning_position not in ("pre_encoder", "post_encoder"):
            raise ValueError("unsupported transition_conditioning_position")
        # Post-encoder conditioning is valid for both the video carrier and the
        # query-agreement carrier.  The latter is the deliberate dual-cue case:
        # vector direction carries query identity through LayerNorm, while the
        # post-normalization scalar retains local relevance/boundary strength.
        self.field_rank = int(field_rank)
        self.transition_barrier = bool(transition_barrier)
        self.max_update_ratio = float(max_update_ratio)
        self.support_hops = int(support_hops)
        self.support_hop_mode = str(support_hop_mode)
        self.transition_input_mode = str(transition_input_mode)
        self.evidence_input_mode = str(evidence_input_mode)
        self.support_input_mode = str(support_input_mode)
        self.support_conditioning_position = str(support_conditioning_position)
        self.transition_conditioning_position = str(transition_conditioning_position)
        self.field_encoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(512),
                    nn.Linear(512, field_rank),
                    nn.GELU(),
                    nn.LayerNorm(field_rank),
                )
                for _ in range(3)
            ]
        )
        self.role_to_semantic = nn.ModuleList(
            [nn.Linear(field_rank, 512, bias=False) for _ in range(3)]
        )
        self.raw_field_gates = nn.Parameter(torch.empty(3))
        self.reset_output(initial_field_gate, output_gain)
        # Additional hop mixers are initialized after every shared tensor and
        # inside an isolated RNG scope.  A treatment therefore preserves the
        # exact Support2 initialization and the caller's RNG stream.
        self.raw_support_second_hop_mix: Optional[nn.Parameter] = None
        self.support_hop_router: Optional[nn.Linear] = None
        if self.support_hop_mode == "learned_global":
            self.raw_support_second_hop_mix = nn.Parameter(torch.zeros(()))
        elif self.support_hop_mode == "learned_local":
            cpu_rng_state = torch.random.get_rng_state()
            try:
                self.support_hop_router = nn.Linear(3, 1)
                nn.init.zeros_(self.support_hop_router.weight)
                nn.init.zeros_(self.support_hop_router.bias)
            finally:
                torch.random.set_rng_state(cpu_rng_state)

    def reset_output(self, initial_field_gate: float, output_gain: float) -> None:
        with torch.no_grad():
            for layer in self.role_to_semantic:
                nn.init.xavier_uniform_(layer.weight, gain=output_gain)
            self.raw_field_gates.fill_(atanh(initial_field_gate))

    def forward(
        self,
        raw_video: Tensor,
        raw_text: Tensor,
        video_padding_mask: Tensor,
        query_padding_mask: Tensor,
    ) -> tuple[Tensor, InputTriFieldState]:
        semantic = raw_video[..., :512].float()
        normalized_video = F.normalize(semantic, dim=-1, eps=1.0e-6)
        if self.evidence_input_mode == "token_attention_product":
            normalized_text = F.normalize(raw_text.float(), dim=-1, eps=1.0e-6)
            token_logits = torch.einsum(
                "bld,bqd->blq", normalized_video, normalized_text
            ) * sqrt(float(semantic.shape[-1]))
            token_logits = token_logits.masked_fill(query_padding_mask[:, None], -1.0e4)
            token_weight = token_logits.softmax(-1).masked_fill(
                query_padding_mask[:, None], 0.0
            )
            token_weight = token_weight / token_weight.sum(-1, keepdim=True).clamp_min(
                1.0e-6
            )
            aligned_query = torch.einsum("blq,bqd->bld", token_weight, normalized_text)
            agreement = normalized_video * aligned_query
        else:
            query = F.normalize(
                _masked_query(raw_text.float(), query_padding_mask),
                dim=-1,
                eps=1.0e-6,
            )
            agreement = normalized_video * query[:, None]
        similarity = agreement.sum(-1).masked_fill(video_padding_mask, 0.0)
        mean, std = _masked_mean_std(similarity, video_padding_mask)
        occupancy = F.softplus((similarity - mean[:, None]) / std[:, None]).masked_fill(
            video_padding_mask, 0.0
        )
        standardized = (similarity - mean[:, None]) / std[:, None]

        left_video, right_video = _left_right(normalized_video)
        left_similarity, right_similarity = _left_right(similarity[..., None])
        left_similarity = left_similarity.squeeze(-1)
        right_similarity = right_similarity.squeeze(-1)
        enter_score = similarity - left_similarity
        leave_score = similarity - right_similarity
        left_change = (normalized_video - left_video).norm(dim=-1)
        right_change = (normalized_video - right_video).norm(dim=-1)
        left_strength = left_change + 2.0 * enter_score.abs()
        right_strength = right_change + 2.0 * leave_score.abs()
        if self.transition_barrier:
            left_conductance = torch.exp(-2.0 * left_strength)
            right_conductance = torch.exp(-2.0 * right_strength)
        else:
            left_conductance = torch.ones_like(left_strength)
            right_conductance = torch.ones_like(right_strength)
        valid = ~video_padding_mask
        edge_pad = torch.zeros_like(valid[:, :1])
        left_valid = torch.cat((edge_pad, valid[:, :-1]), dim=1)
        right_valid = torch.cat((valid[:, 1:], edge_pad), dim=1)
        left_conductance = left_conductance * left_valid.float()
        right_conductance = right_conductance * right_valid.float()
        denominator = (1.0 + left_conductance + right_conductance).clamp_min(1.0)

        def diffuse(value: Tensor) -> Tensor:
            left_value, right_value = _left_right(value)
            return (
                value
                + left_conductance[..., None] * left_value
                + right_conductance[..., None] * right_value
            ) / denominator[..., None]

        support_second_hop_mix = semantic.new_zeros(())

        def mix_support_hops(first: Tensor, second: Tensor) -> Tensor:
            nonlocal support_second_hop_mix
            if self.support_hop_mode == "fixed":
                mix = semantic.new_tensor(0.5)
            elif self.support_hop_mode == "learned_global":
                if self.raw_support_second_hop_mix is None:
                    raise RuntimeError("global Support hop mixer is missing")
                mix = torch.sigmoid(self.raw_support_second_hop_mix.float())
            else:
                if self.support_hop_router is None:
                    raise RuntimeError("local Support hop router is missing")
                router_input = torch.stack(
                    (
                        standardized.clamp(-4.0, 4.0),
                        torch.log1p(occupancy),
                        torch.tanh(0.5 * (left_strength + right_strength)),
                    ),
                    dim=-1,
                )
                mix = torch.sigmoid(self.support_hop_router(router_input.float()))
            if mix.ndim == 0:
                support_second_hop_mix = mix
                return first * (1.0 - mix) + second * mix
            valid_weight = valid.float()
            support_second_hop_mix = (
                mix.squeeze(-1) * valid_weight
            ).sum() / valid_weight.sum().clamp_min(1.0)
            return first * (1.0 - mix) + second * mix

        support_residual = None
        if self.support_input_mode == "query_agreement_transport":
            # Preserve query information in vector direction. Earlier support
            # modes encoded most query dependence as a positive scalar on a
            # video-only vector; the following LayerNorm largely removed that
            # amplitude signal. Diffusing the token-aligned agreement itself
            # instead gives Support a low-frequency, query-conditioned carrier
            # distinct from local Evidence and high-frequency Transition.
            first_diffused = diffuse(agreement)
            if self.support_hops == 2:
                support_input = mix_support_hops(
                    first_diffused, diffuse(first_diffused)
                )
            else:
                support_input = first_diffused
        elif self.support_input_mode == "query_seed_transport":
            support_source = normalized_video * torch.sigmoid(standardized)[..., None]
            first_diffused = diffuse(support_source)
            if self.support_hops == 2:
                support_input = mix_support_hops(
                    first_diffused, diffuse(first_diffused)
                )
            else:
                support_input = first_diffused
        else:
            first_diffused = diffuse(normalized_video)
            support_residual = first_diffused - normalized_video
            if self.support_hops == 2:
                support_residual = mix_support_hops(
                    support_residual,
                    diffuse(first_diffused) - normalized_video,
                )
            support_input = support_residual * occupancy[..., None]
        if self.transition_input_mode == "query_agreement_delta":
            left_transition, right_transition = _left_right(agreement)
            transition_carrier = agreement
        else:
            left_transition, right_transition = left_video, right_video
            transition_carrier = normalized_video
        transition_start_input = (transition_carrier - left_transition) * torch.tanh(
            4.0 * enter_score
        )[..., None]
        transition_end_input = (transition_carrier - right_transition) * torch.tanh(
            4.0 * leave_score
        )[..., None]
        transition_input = 0.5 * (transition_start_input + transition_end_input)
        fields = (agreement, support_input, transition_input)
        if (
            self.support_conditioning_position == "post_encoder"
            or self.transition_conditioning_position == "post_encoder"
        ):
            evidence_latent = self.field_encoders[0](agreement)
            support_base = support_input
            if (
                self.support_conditioning_position == "post_encoder"
                and self.support_input_mode == "diffusion_residual"
            ):
                support_base = support_residual
            support_latent = self.field_encoders[1](support_base)
            if self.support_conditioning_position == "post_encoder":
                support_latent = support_latent * torch.sigmoid(standardized)[..., None]
            if self.transition_conditioning_position == "post_encoder":
                left_latent = self.field_encoders[2](
                    transition_carrier - left_transition
                )
                right_latent = self.field_encoders[2](
                    transition_carrier - right_transition
                )
                transition_latent = 0.5 * (
                    left_latent * torch.tanh(4.0 * enter_score)[..., None]
                    + right_latent * torch.tanh(4.0 * leave_score)[..., None]
                )
            else:
                transition_latent = self.field_encoders[2](transition_input)
            role_updates = torch.stack(
                [
                    output(latent)
                    for output, latent in zip(
                        self.role_to_semantic,
                        (evidence_latent, support_latent, transition_latent),
                    )
                ],
                dim=-2,
            ).float()
        else:
            role_updates = torch.stack(
                [
                    output(encoder(value))
                    for encoder, output, value in zip(
                        self.field_encoders, self.role_to_semantic, fields
                    )
                ],
                dim=-2,
            ).float()
        gates = torch.tanh(self.raw_field_gates.float())
        raw_update = (role_updates * gates[None, None, :, None]).sum(-2)
        raw_update = raw_update.masked_fill(video_padding_mask[..., None], 0.0)

        carrier_square = semantic.square().sum(-1, keepdim=True)
        parallel = (raw_update * semantic).sum(
            -1, keepdim=True
        ) / carrier_square.clamp_min(1.0e-12)
        orthogonal = raw_update - parallel * semantic
        carrier_norm = carrier_square.sqrt()
        update_norm = orthogonal.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
        limit = self.max_update_ratio * carrier_norm
        bounded_norm = limit * torch.tanh(update_norm / limit.clamp_min(1.0e-8))
        bounded = orthogonal * (bounded_norm / update_norm)
        mixed = semantic + bounded
        conditioned = mixed * (
            carrier_norm / mixed.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
        )
        conditioned = torch.where(video_padding_mask[..., None], semantic, conditioned)
        protected_update = (conditioned - semantic).masked_fill(
            video_padding_mask[..., None], 0.0
        )
        output_video = torch.cat(
            (conditioned.to(raw_video.dtype), raw_video[..., 512:]), dim=-1
        )
        conductance_count = (
            (left_valid.float() + right_valid.float()).sum().clamp_min(1.0)
        )
        mean_conductance = (
            left_conductance.sum() + right_conductance.sum()
        ) / conductance_count
        state = InputTriFieldState(
            evidence_score=similarity,
            support_score=occupancy,
            enter_score=enter_score.masked_fill(video_padding_mask, 0.0),
            leave_score=leave_score.masked_fill(video_padding_mask, 0.0),
            role_updates=role_updates,
            transition_start_input=transition_start_input,
            transition_end_input=transition_end_input,
            raw_update=raw_update,
            protected_update=protected_update,
            input_semantic=semantic,
            conditioned_semantic=conditioned,
            effective_gates=gates,
            transition_barrier=self.transition_barrier,
            mean_conductance=mean_conductance.detach(),
            max_update_ratio=self.max_update_ratio,
            support_hops=self.support_hops,
            support_hop_mode=self.support_hop_mode,
            support_second_hop_mix=support_second_hop_mix,
            transition_input_mode=self.transition_input_mode,
            evidence_input_mode=self.evidence_input_mode,
            support_input_mode=self.support_input_mode,
            support_conditioning_position=self.support_conditioning_position,
            transition_conditioning_position=self.transition_conditioning_position,
            query_identity=F.normalize(
                _masked_query(raw_text.float(), query_padding_mask),
                dim=-1,
                eps=1.0e-6,
            ),
        )
        return output_video, state


class TriFieldInputConditionedModel(ScratchE2EModel):
    def __init__(
        self,
        *args: Any,
        input_field_rank: int = 64,
        transition_barrier: bool = True,
        initial_field_gate: float = 0.05,
        conditioner_output_gain: float = 0.10,
        max_update_ratio: float = 0.05,
        support_hops: int = 1,
        support_hop_mode: str = "fixed",
        transition_input_mode: str = "feature_delta",
        evidence_input_mode: str = "mean_query_product",
        support_input_mode: str = "diffusion_residual",
        support_conditioning_position: str = "pre_encoder",
        transition_conditioning_position: str = "pre_encoder",
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        # Keep the shared scratch initialization stream identical to the base.
        cpu_rng_state = torch.random.get_rng_state()
        try:
            self.input_conditioner = TriFieldInputConditioner(
                field_rank=input_field_rank,
                transition_barrier=transition_barrier,
                initial_field_gate=initial_field_gate,
                output_gain=conditioner_output_gain,
                max_update_ratio=max_update_ratio,
                support_hops=support_hops,
                support_hop_mode=support_hop_mode,
                transition_input_mode=transition_input_mode,
                evidence_input_mode=evidence_input_mode,
                support_input_mode=support_input_mode,
                support_conditioning_position=support_conditioning_position,
                transition_conditioning_position=transition_conditioning_position,
            )
        finally:
            torch.random.set_rng_state(cpu_rng_state)

    def forward(
        self,
        inputs: Any,
        query_features: Optional[Tensor] = None,
        video_padding_mask: Optional[Tensor] = None,
        query_padding_mask: Optional[Tensor] = None,
    ):
        if isinstance(inputs, Mapping):
            raw_video, raw_text = inputs["src_vid"], inputs["src_txt"]
            video_pad = inputs["video_padding_mask"].bool()
            query_pad = inputs.get("query_padding_mask")
            if query_pad is None:
                query_pad = torch.zeros(
                    raw_text.shape[:2], dtype=torch.bool, device=raw_text.device
                )
            conditioned, state = self.input_conditioner(
                raw_video, raw_text, video_pad, query_pad.bool()
            )
            conditioned_inputs = dict(inputs)
            conditioned_inputs["src_vid"] = conditioned
            output = ScratchE2EModel.forward(self, conditioned_inputs)
        else:
            if not isinstance(inputs, Tensor) or not isinstance(query_features, Tensor):
                raise TypeError(
                    "input-conditioned model requires video and text tensors"
                )
            video_pad = (
                torch.zeros(inputs.shape[:2], dtype=torch.bool, device=inputs.device)
                if video_padding_mask is None
                else video_padding_mask.bool()
            )
            query_pad = (
                torch.zeros(
                    query_features.shape[:2],
                    dtype=torch.bool,
                    device=query_features.device,
                )
                if query_padding_mask is None
                else query_padding_mask.bool()
            )
            conditioned, state = self.input_conditioner(
                inputs, query_features, video_pad, query_pad
            )
            output = ScratchE2EModel.forward(
                self, conditioned, query_features, video_pad, query_pad
            )
        if output.extension_output is None:
            raise RuntimeError("identity extension output missing")
        output.extension_output.state = state
        return output

    @staticmethod
    def _hard_margin(
        score: Tensor, iou: Tensor, valid: Tensor
    ) -> tuple[Tensor, Tensor]:
        flat_score, flat_iou, flat_valid = (
            score.float().flatten(1),
            iou.float().flatten(1),
            valid.flatten(1),
        )
        positive_index = flat_iou.argmax(1, keepdim=True)
        positive = flat_score.gather(1, positive_index).squeeze(1)
        negative = flat_valid & flat_iou.ge(0.50) & flat_iou.lt(0.95)
        negative = negative.scatter(1, positive_index, False)
        has = negative.any(1)
        hardest = flat_score.masked_fill(~negative, -1.0e4).amax(1)
        margin = positive[has] - hardest[has]
        zero = score.float().sum() * 0.0
        if not margin.numel():
            return zero.detach(), zero.detach()
        return margin.mean().detach(), margin.gt(0.0).float().mean().detach()

    def compute_loss(
        self, outputs: Any, batch: Any, teacher_outputs: Any, epoch: int
    ) -> LossResult:
        base = ScratchE2EModel.compute_loss(
            self, outputs, batch, teacher_outputs, epoch
        )
        margin, rate = self._hard_margin(
            outputs.span_logits,
            self._iou(outputs, batch),
            outputs.span_valid_mask.bool(),
        )
        metrics = dict(base.metrics)
        metrics.update(
            {
                "input_trifield/final_hard_margin": margin,
                "input_trifield/final_hard_positive_rate": rate,
                "input_trifield/auxiliary_loss_weight": 0.0,
            }
        )
        return LossResult(base.loss, metrics)

    def diagnostics(self, outputs: Any, batch: Any) -> Mapping[str, Tensor | float]:
        result = dict(super().diagnostics(outputs, batch))
        state = outputs.extension_output.state if outputs.extension_output else None
        if not isinstance(state, InputTriFieldState):
            return result
        valid = outputs.span_valid_mask.bool()
        iou = self._iou(outputs, batch)
        evidence = _span_mean(state.evidence_score).masked_fill(~valid, 0.0)
        length = evidence.shape[-1]
        index = torch.arange(length, device=evidence.device)
        width = (index[None, :] - index[:, None] + 1).clamp_min(1).to(evidence.dtype)
        token_count = (
            (~batch.inputs["video_padding_mask"].bool())
            .sum(1)
            .clamp_min(1)
            .to(evidence.dtype)
        )
        support = (
            _span_mean(state.support_score) * width[None] / token_count[:, None, None]
        ).masked_fill(~valid, 0.0)
        transition = 0.5 * (
            state.enter_score[:, :, None] + state.leave_score[:, None, :]
        )
        transition = transition.masked_fill(~valid, 0.0)
        input_std = state.input_semantic.float().std(unbiased=False).clamp_min(1.0e-6)
        input_norm = state.input_semantic.float().norm(dim=-1)
        norm_error = (
            state.conditioned_semantic.float().norm(dim=-1) - input_norm
        ).abs() / input_norm.clamp_min(1.0e-6)
        result.update(
            {
                "input_trifield/evidence_iou_corr": _score_correlation(
                    evidence, iou, valid
                ),
                "input_trifield/support_iou_corr": _score_correlation(
                    support, iou, valid
                ),
                "input_trifield/transition_iou_corr": _score_correlation(
                    transition, iou, valid
                ),
                "input_trifield/evidence_support_corr": _score_correlation(
                    evidence, support, valid
                ),
                "input_trifield/evidence_transition_corr": _score_correlation(
                    evidence, transition, valid
                ),
                "input_trifield/support_transition_corr": _score_correlation(
                    support, transition, valid
                ),
                "input_trifield/raw_update_to_input_std": state.raw_update.float().std(
                    unbiased=False
                )
                / input_std,
                "input_trifield/protected_update_to_input_std": state.protected_update.float().std(
                    unbiased=False
                )
                / input_std,
                "input_trifield/carrier_norm_relative_error": norm_error.masked_fill(
                    batch.inputs["video_padding_mask"].bool(), 0.0
                ).mean(),
                "input_trifield/mean_conductance": state.mean_conductance,
                "input_trifield/evidence_gate": state.effective_gates[0],
                "input_trifield/support_gate": state.effective_gates[1],
                "input_trifield/transition_gate": state.effective_gates[2],
                "input_trifield/support_second_hop_mix": state.support_second_hop_mix,
            }
        )
        return result

    def experiment_contract(self) -> Mapping[str, Any]:
        contract = dict(super().experiment_contract())
        conditioner = self.input_conditioner
        contract.update(
            {
                "input_trifield": True,
                "injection_point": "raw_video_semantics_before_backbone",
                "tri_field_inputs": {
                    "Evidence": "raw_channelwise_video_query_agreement",
                    "Support": "positive_occupancy_weighted_temporal_diffusion",
                    "Transition": "signed_enter_leave_high_frequency_and_optional_diffusion_barrier",
                },
                "transition_barrier": conditioner.transition_barrier,
                "carrier_protection": "orthogonal_bounded_exact_raw_feature_norm",
                "max_update_ratio": conditioner.max_update_ratio,
                "support_hops": conditioner.support_hops,
                "support_hop_mode": conditioner.support_hop_mode,
                "transition_input_mode": conditioner.transition_input_mode,
                "evidence_input_mode": conditioner.evidence_input_mode,
                "support_input_mode": conditioner.support_input_mode,
                "support_conditioning_position": conditioner.support_conditioning_position,
                "transition_conditioning_position": conditioner.transition_conditioning_position,
                "auxiliary_ranking_loss": False,
                "auxiliary_contrastive_loss": False,
                "coordinate_movement": False,
                "old_prediction_score_fusion": False,
                "posthoc_reranking": False,
            }
        )
        return contract


def build_input_conditioned_trifield_model(
    config: Optional[RunnerConfig],
    input_field_rank: int = 64,
    transition_barrier: bool = True,
    initial_field_gate: float = 0.05,
    conditioner_output_gain: float = 0.10,
    max_update_ratio: float = 0.05,
    support_hops: int = 1,
    support_hop_mode: str = "fixed",
    transition_input_mode: str = "feature_delta",
    evidence_input_mode: str = "mean_query_product",
    support_input_mode: str = "diffusion_residual",
    support_conditioning_position: str = "pre_encoder",
    transition_conditioning_position: str = "pre_encoder",
    extension_lr: float = 1.0e-4,
    scratch_total_epochs: int = 420,
    scratch_residual_gate: float = 0.05,
    **kwargs: Any,
) -> TriFieldInputConditionedModel:
    del config
    inherited = dict(STAGE32_KWARGS)
    inherited.update(kwargs)
    inherited["use_boundary_gate"] = False
    model = TriFieldInputConditionedModel(
        extension_lr=extension_lr,
        scratch_total_epochs=scratch_total_epochs,
        input_field_rank=input_field_rank,
        transition_barrier=transition_barrier,
        initial_field_gate=initial_field_gate,
        conditioner_output_gain=conditioner_output_gain,
        max_update_ratio=max_update_ratio,
        support_hops=support_hops,
        support_hop_mode=support_hop_mode,
        transition_input_mode=transition_input_mode,
        evidence_input_mode=evidence_input_mode,
        support_input_mode=support_input_mode,
        support_conditioning_position=support_conditioning_position,
        transition_conditioning_position=transition_conditioning_position,
        **inherited,
    )
    frozen = _freeze_boundary_gate_only_branch(model)
    # The baseline scratch initializer performs special head initialization after
    # traversing every registered module.  Temporarily detach the private input
    # conditioner so its extra Linear layers cannot shift that shared RNG stream.
    # The conditioner's constructor is itself RNG-isolated in __init__, so its
    # parameters remain paired without changing the baseline's final RNG state.
    conditioner = model._modules.pop("input_conditioner")
    try:
        model.initialization_audit = initialize_scratch_e2e(
            model, residual_gate=scratch_residual_gate
        )
    finally:
        model.add_module("input_conditioner", conditioner)
    model.initialization_audit["input_trifield"] = {
        "checkpoint": None,
        "boundary_gate": False,
        "inactive_boundary_parameters": frozen,
        "input_field_rank": input_field_rank,
        "transition_barrier": transition_barrier,
        "initial_field_gate": initial_field_gate,
        "conditioner_output_gain": conditioner_output_gain,
        "max_update_ratio": max_update_ratio,
        "support_hops": support_hops,
        "transition_input_mode": transition_input_mode,
        "evidence_input_mode": evidence_input_mode,
        "support_input_mode": support_input_mode,
        "support_conditioning_position": support_conditioning_position,
        "transition_conditioning_position": transition_conditioning_position,
        "auxiliary_loss_weight": 0.0,
        "genuinely_distinct_raw_inputs": True,
        "shared_initialization_rng_isolation": True,
    }
    return model


__all__ = [
    "InputTriFieldState",
    "TriFieldInputConditionedModel",
    "TriFieldInputConditioner",
    "build_input_conditioned_trifield_model",
    "build_repository_data_with_saliency",
]
