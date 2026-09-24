"""Standalone EventFieldNet SpanSet model on the non-DETR C3 span grid."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

import torch
from torch import Tensor

from backbone.model import STAGE32_KWARGS
from training.config import RunnerConfig
from training.contracts import LossResult
from training.probability import masked_softmax
from span_fields.trifield_span_set import TriFieldSequence
from span_fields.trifield_span_set_objective import TriFieldSpanSetObjective

from .scratch_plugin import initialize_scratch_e2e
from .trifield_final_hardrank_plugin import FinalHardRankInputModel
from .trifield_input_conditioner_plugin import (
    InputTriFieldState,
    _freeze_boundary_gate_only_branch,
    build_repository_data_with_saliency,
)
from .trifield_spanset_c3_adapter import TriFieldC3SpanSetAdapter, dense_grid_view


MODES = ("objective_only", "persistent_trifield")


def _fields(state: InputTriFieldState, valid: Tensor) -> TriFieldSequence:
    return TriFieldSequence(
        evidence=state.evidence_score,
        support=state.support_score,
        transition_start=state.enter_score,
        transition_end=state.leave_score,
        valid_mask=valid.bool(),
        role_features=state.role_updates,
        query_identity=state.query_identity,
    )


class TriFieldSpanSetC3Model(FinalHardRankInputModel):
    """EventFieldNet main model with C3 used only as a dense candidate backend."""

    def __init__(
        self,
        *args: Any,
        span_set_mode: str = "persistent_trifield",
        span_set_listwise_weight: float = 0.25,
        span_set_margin_weight: float = 0.50,
        span_set_wrong_query_weight: float = 0.25,
        wrong_query_shift: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if span_set_mode not in MODES:
            raise ValueError(f"span_set_mode must be one of {MODES}")
        self.span_set_mode = str(span_set_mode)
        self.span_set_listwise_weight = float(span_set_listwise_weight)
        self.span_set_margin_weight = float(span_set_margin_weight)
        self.span_set_wrong_query_weight = float(span_set_wrong_query_weight)
        self.wrong_query_shift = int(wrong_query_shift)
        for name, value in (
            ("span_set_listwise_weight", span_set_listwise_weight),
            ("span_set_margin_weight", span_set_margin_weight),
            ("span_set_wrong_query_weight", span_set_wrong_query_weight),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        self.span_set_objective = TriFieldSpanSetObjective(
            semantic_score_weight=1.0,
            quality_score_weight=0.0,
            strict_positive_iou=0.70,
            strict_negative_iou=0.30,
            ranking_margin=0.20,
            wrong_query_margin=0.20,
        )
        self.span_set_adapter = (
            TriFieldC3SpanSetAdapter(objective=self.span_set_objective)
            if self.span_set_mode == "persistent_trifield"
            else None
        )

    @staticmethod
    def _padding_from_inputs(
        inputs: Any,
        query_features: Optional[Tensor],
        video_padding_mask: Optional[Tensor],
        query_padding_mask: Optional[Tensor],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if isinstance(inputs, Mapping):
            raw_video = inputs["src_vid"]
            raw_text = inputs["src_txt"]
            video_pad = inputs["video_padding_mask"].bool()
            query_pad = inputs.get("query_padding_mask")
            if query_pad is None:
                query_pad = torch.zeros(
                    raw_text.shape[:2], dtype=torch.bool, device=raw_text.device
                )
            return raw_video, raw_text, video_pad, query_pad.bool()
        if not isinstance(inputs, Tensor):
            raise TypeError("tensor or mapping inputs required")
        if not isinstance(query_features, Tensor):
            raise TypeError("tensor input requires tensor query_features")
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
        return inputs, query_features, video_pad, query_pad

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
        if output.extension_output is None:
            raise RuntimeError("tri-field state missing from C3 output")
        state = output.extension_output.state
        if not isinstance(state, InputTriFieldState):
            raise RuntimeError("unexpected tri-field state type")
        raw_video, raw_text, video_pad, query_pad = self._padding_from_inputs(
            inputs, query_features, video_padding_mask, query_padding_mask
        )
        fields = _fields(state, ~video_pad)
        wrong_fields = None
        if (
            self.span_set_mode == "persistent_trifield"
            and self.training
            and raw_text.shape[0] > 1
            and self.wrong_query_shift
        ):
            wrong_query_indices = self._wrong_query_indices(raw_text, query_pad)
            _, wrong_state = self.input_conditioner(
                raw_video,
                raw_text.index_select(0, wrong_query_indices),
                video_pad,
                query_pad.index_select(0, wrong_query_indices),
            )
            wrong_fields = _fields(wrong_state, ~video_pad)

        if self.span_set_adapter is None:
            view = dense_grid_view(
                output.span_logits, output.span_valid_mask, ~video_pad
            )
            adapted: Dict[str, Any] = {
                "span_logits": output.span_logits,
                "view": view,
                "trifield_span_set": {},
            }
        else:
            adapted = self.span_set_adapter(
                span_logits=output.span_logits,
                span_valid_mask=output.span_valid_mask,
                fields=fields,
                wrong_query_fields=wrong_fields,
            )
            output.span_logits = adapted["span_logits"]
            output.span_probs = (
                masked_softmax(
                    output.span_logits.flatten(1).float(),
                    output.span_valid_mask.flatten(1),
                )
                .reshape_as(output.span_logits)
                .to(output.span_probs.dtype)
            )
        # Expose the backend-independent frame contract to additive experiment
        # modules.  This avoids rerunning the full input conditioner and keeps
        # correct/wrong-query fields tied to the exact same forward pass.
        adapted["fields"] = fields
        adapted["wrong_query_fields"] = wrong_fields
        output._trifield_span_set_adapted = adapted
        return output

    @staticmethod
    def _targets_xx(batch: Any) -> Sequence[Tensor]:
        spans = batch.targets["gt_spans"].float()
        mask = batch.targets["gt_span_mask"].bool()
        result = []
        for row, row_mask in zip(spans, mask):
            row = row[row_mask]
            result.append(
                torch.stack(
                    (
                        torch.minimum(row[:, 0], row[:, 1]),
                        torch.maximum(row[:, 0], row[:, 1]),
                    ),
                    -1,
                )
            )
        return result

    def _span_set_result(self, outputs: Any, batch: Any):
        adapted = getattr(outputs, "_trifield_span_set_adapted", None)
        if adapted is None:
            raise RuntimeError("SpanSet adaptation missing before objective")
        view = adapted["view"]
        rank = adapted["trifield_span_set"]
        return self.span_set_objective(
            spans_xx=view.spans_xx,
            semantic_logits=outputs.span_logits.flatten(1),
            targets_xx=self._targets_xx(batch),
            candidate_valid=view.valid,
            evidence_score=rank.get("evidence_score"),
            wrong_query_evidence_score=rank.get("wrong_query_evidence_score"),
        )

    def _wrong_query_indices(
        self, raw_text: Tensor, query_padding_mask: Tensor
    ) -> Tensor:
        """Return a deterministic non-self query permutation for training.

        Subclasses may replace the selection policy without duplicating the
        complete forward path. The default exactly reproduces the historical
        cyclic shift.
        """

        del query_padding_mask
        return torch.arange(raw_text.shape[0], device=raw_text.device).roll(
            self.wrong_query_shift
        )

    def compute_loss(
        self, outputs: Any, batch: Any, teacher_outputs: Any, epoch: int
    ) -> LossResult:
        base = super().compute_loss(outputs, batch, teacher_outputs, epoch)
        result = self._span_set_result(outputs, batch)
        wrong_weight = (
            self.span_set_wrong_query_weight
            if self.span_set_mode == "persistent_trifield"
            else 0.0
        )
        added = (
            self.span_set_listwise_weight * result.losses["loss_candidate_listwise"]
            + self.span_set_margin_weight * result.losses["loss_candidate_margin"]
            + wrong_weight * result.losses["loss_wrong_query_field"]
        )
        metrics = dict(base.metrics)
        metrics.update(
            {f"spanset/{name}": value for name, value in result.diagnostics.items()}
        )
        metrics.update(
            {
                "spanset/added_loss": added.detach(),
                "spanset/listwise_weight": self.span_set_listwise_weight,
                "spanset/margin_weight": self.span_set_margin_weight,
                "spanset/wrong_query_weight": wrong_weight,
            }
        )
        outputs._trifield_span_set_diagnostics = result.diagnostics
        return LossResult(base.loss + added, metrics)

    def diagnostics(self, outputs: Any, batch: Any) -> Mapping[str, Tensor | float]:
        result = dict(super().diagnostics(outputs, batch))
        objective = self._span_set_result(outputs, batch)
        result.update(
            {f"spanset/{name}": value for name, value in objective.diagnostics.items()}
        )
        rank = outputs._trifield_span_set_adapted["trifield_span_set"]
        for name in (
            "semantic_residual",
            "quality_residual",
            "evidence_score",
            "support_score",
            "transition_start_score",
            "transition_end_score",
        ):
            if name in rank:
                value = rank[name].detach().float()
                result[f"spanset/{name}_abs_mean"] = value.abs().mean()
        return result

    def experiment_contract(self) -> Mapping[str, Any]:
        contract = dict(super().experiment_contract())
        contract.update(
            {
                "method": "EventFieldNet_tri_field_SpanSet",
                "method_is_detr_extension": False,
                "candidate_backend": "C3_dense_start_end_grid",
                "candidate_backend_replaceable": True,
                "span_set_mode": self.span_set_mode,
                "evidence_route": "semantic_residual_only",
                "support_transition_route": "quality_residual_only",
                "complete_span_objective": True,
                "wrong_query_high_iou_objective": (
                    self.span_set_mode == "persistent_trifield"
                ),
                "coordinate_movement": False,
                "old_prediction_score_fusion": False,
                "posthoc_reranking": False,
            }
        )
        return contract


def build_trifield_spanset_c3_model(
    config: Optional[RunnerConfig],
    span_set_mode: str = "persistent_trifield",
    span_set_listwise_weight: float = 0.25,
    span_set_margin_weight: float = 0.50,
    span_set_wrong_query_weight: float = 0.25,
    semantic_preservation_weight: float = 0.10,
    final_hard_rank_weight: float = 0.10,
    final_hard_rank_margin: float = 0.20,
    final_hard_rank_topk: int = 8,
    input_field_rank: int = 64,
    initial_field_gate: float = 0.05,
    conditioner_output_gain: float = 0.10,
    max_update_ratio: float = 0.05,
    extension_lr: float = 1.0e-4,
    scratch_total_epochs: int = 420,
    scratch_residual_gate: float = 0.05,
    **kwargs: Any,
) -> TriFieldSpanSetC3Model:
    del config
    # The strongest dual-cue parent already carries these role settings in its
    # config. Consume each exactly once, while keeping safe defaults for direct
    # construction and smoke tests.
    support_hop_mode = kwargs.pop("support_hop_mode", None)
    role_options = {
        "transition_barrier": kwargs.pop("transition_barrier", True),
        "support_hops": kwargs.pop("support_hops", 1),
        "transition_input_mode": kwargs.pop(
            "transition_input_mode", "query_agreement_delta"
        ),
        "evidence_input_mode": kwargs.pop(
            "evidence_input_mode", "token_attention_product"
        ),
        "support_input_mode": kwargs.pop(
            "support_input_mode", "query_agreement_transport"
        ),
        "support_conditioning_position": kwargs.pop(
            "support_conditioning_position", "post_encoder"
        ),
        "transition_conditioning_position": kwargs.pop(
            "transition_conditioning_position", "pre_encoder"
        ),
    }
    # Older, still valid C3 parents (including the data81 copy) predate the
    # optional hop-mode switch.  Do not leak a default-only keyword through the
    # cooperative constructor chain; pass it only when an experiment explicitly
    # requests the newer interface.
    if support_hop_mode is not None:
        role_options["support_hop_mode"] = support_hop_mode
    inherited = dict(STAGE32_KWARGS)
    inherited.update(kwargs)
    inherited["use_boundary_gate"] = False
    model = TriFieldSpanSetC3Model(
        span_set_mode=span_set_mode,
        span_set_listwise_weight=span_set_listwise_weight,
        span_set_margin_weight=span_set_margin_weight,
        span_set_wrong_query_weight=span_set_wrong_query_weight,
        semantic_preservation_weight=semantic_preservation_weight,
        final_hard_rank_weight=final_hard_rank_weight,
        final_hard_rank_margin=final_hard_rank_margin,
        final_hard_rank_topk=final_hard_rank_topk,
        input_field_rank=input_field_rank,
        initial_field_gate=initial_field_gate,
        conditioner_output_gain=conditioner_output_gain,
        max_update_ratio=max_update_ratio,
        **role_options,
        extension_lr=extension_lr,
        scratch_total_epochs=scratch_total_epochs,
        **inherited,
    )
    frozen = _freeze_boundary_gate_only_branch(model)
    detached_modules: Dict[str, Any] = {}
    for name in ("input_conditioner", "span_set_adapter", "span_set_objective"):
        if name in model._modules:
            detached_modules[name] = model._modules.pop(name)
    try:
        model.initialization_audit = initialize_scratch_e2e(
            model, residual_gate=scratch_residual_gate
        )
    finally:
        for name, module in detached_modules.items():
            model.add_module(name, module)
    model.initialization_audit["trifield_spanset"] = {
        "checkpoint": None,
        "method_is_detr_extension": False,
        "candidate_backend": "C3_dense_start_end_grid",
        "span_set_mode": span_set_mode,
        "boundary_gate": False,
        "inactive_boundary_parameters": frozen,
        "shared_initialization_rng_isolation": True,
        "coordinate_movement": False,
    }
    return model


__all__ = [
    "MODES",
    "TriFieldSpanSetC3Model",
    "build_repository_data_with_saliency",
    "build_trifield_spanset_c3_model",
]
