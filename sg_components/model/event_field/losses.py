"""Losses for the standalone Event Field F0 module."""

from typing import Dict, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as functional

from .model import EventFieldOutput
from .targets import EventFieldTargets


class EventFieldLoss(nn.Module):
    """Masked BCE objective for the four Event Field readouts."""

    def __init__(
        self,
        evidence_weight: float = 1.0,
        support_weight: float = 0.10,
        transition_weight: float = 0.25,
    ) -> None:
        super().__init__()
        self.evidence_weight = evidence_weight
        self.support_weight = support_weight
        self.transition_weight = transition_weight

    def forward(
        self,
        outputs: EventFieldOutput,
        targets: EventFieldTargets,
        padding_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Return total and unweighted component losses."""
        reference = outputs.evidence_logits
        expected_shape = reference.shape
        if reference.ndim != 2:
            raise ValueError("Event Field logits must have shape [B, L]")

        output_tensors = (
            outputs.support_logits,
            outputs.start_transition_logits,
            outputs.end_transition_logits,
        )
        target_tensors = (
            targets.evidence,
            targets.support,
            targets.start_transition,
            targets.end_transition,
        )
        if any(
            tensor.shape != expected_shape
            for tensor in (*output_tensors, *target_tensors)
        ):
            raise ValueError(
                "all Event Field outputs and targets must share shape [B, L]"
            )
        if targets.evidence_mask.shape != expected_shape:
            raise ValueError(
                "evidence_mask must share shape [B, L] with Event Field outputs"
            )

        if padding_mask is None:
            valid_positions = torch.ones_like(reference)
        else:
            if padding_mask.shape != expected_shape:
                raise ValueError(f"padding_mask must have shape {expected_shape}")
            valid_positions = (
                ~padding_mask.to(device=reference.device, dtype=torch.bool)
            ).to(reference.dtype)

        evidence_positions = valid_positions * targets.evidence_mask.to(
            device=reference.device,
            dtype=reference.dtype,
        )
        evidence_loss = self._masked_bce(
            reference, targets.evidence, evidence_positions
        )
        support_loss = self._masked_bce(
            outputs.support_logits, targets.support, valid_positions
        )
        start_loss = self._masked_bce(
            outputs.start_transition_logits,
            targets.start_transition,
            valid_positions,
        )
        end_loss = self._masked_bce(
            outputs.end_transition_logits,
            targets.end_transition,
            valid_positions,
        )
        transition_loss = 0.5 * (start_loss + end_loss)
        total_loss = (
            self.evidence_weight * evidence_loss
            + self.support_weight * support_loss
            + self.transition_weight * transition_loss
        )
        return {
            "loss": total_loss,
            "loss_evidence": evidence_loss,
            "loss_support": support_loss,
            "loss_transition": transition_loss,
            "loss_start_transition": start_loss,
            "loss_end_transition": end_loss,
        }

    @staticmethod
    def _masked_bce(logits: Tensor, targets: Tensor, valid_positions: Tensor) -> Tensor:
        targets = targets.to(device=logits.device, dtype=logits.dtype)
        elementwise_loss = functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
        )
        return (
            elementwise_loss * valid_positions
        ).sum() / valid_positions.sum().clamp_min(1.0)
