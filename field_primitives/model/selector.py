"""Round1 E/S/T heads built on the V1 selector contract.

The default support_mode=linear is the V1 implementation at the score
equation level. The optional nonlinear mode adds a small residual head after
the support span aggregation; it does not alter the candidate grid or any
other field.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch
from torch import Tensor, nn

from field_core.selector import (
    FieldScoreOutput,
    TriFieldScoreHeads,
    span_readout,
)


class Round1FieldScoreHeads(TriFieldScoreHeads):
    """V1 field heads with an optional post-aggregation nonlinear S residual."""

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 64,
        *,
        support_mode: str = "linear",
    ) -> None:
        super().__init__(input_dim=input_dim, hidden_dim=hidden_dim)
        support_mode = str(support_mode).lower()
        if support_mode not in {"linear", "nonlinear"}:
            raise ValueError("support_mode must be 'linear' or 'nonlinear'")
        self.support_mode = support_mode
        if support_mode == "nonlinear":
            nonlinear_hidden = max(4, int(hidden_dim) // 2)
            self.support_nonlinear = nn.Sequential(
                nn.Linear(int(hidden_dim) + 1, nonlinear_hidden),
                nn.GELU(),
                nn.Linear(nonlinear_hidden, 1),
            )
        else:
            self.support_nonlinear = None

    def raw_from_state(
        self,
        state: Any,
        valid: Tensor,
        carrier: Tensor,
        video_padding_mask: Tensor | None = None,
    ) -> FieldScoreOutput:
        base = super().raw_from_state(
            state,
            valid,
            carrier,
            video_padding_mask=video_padding_mask,
        )
        if self.support_nonlinear is None:
            return base

        role_updates = self._state_role_updates(state)
        support_token = self.support.encode(role_updates[:, :, 1, :])
        # Reconstruct the parent's normalized inclusive width so the new
        # head can model length effects after aggregation. The V1 support
        # readout remains active as a stable linear baseline.
        if video_padding_mask is None:
            token_valid = valid.any(-1) | valid.any(-2)
            count = token_valid.sum(-1).clamp_min(1).float()
        else:
            token_valid = ~video_padding_mask.bool()
            count = token_valid.sum(-1).clamp_min(1).float()
        length = carrier.shape[-1]
        index = torch.arange(length, device=carrier.device, dtype=torch.float32)
        width = (index[None, None, :] - index[None, :, None] + 1.0).clamp_min(
            1.0
        ) / count[:, None, None]
        # Retain scalar span scores, recomputing the large hidden grid only
        # when its backward pass needs it.
        delta = span_readout(
            support_token, self.support_nonlinear, width
        ).squeeze(-1)
        raw_support = base.raw_support + delta
        support = torch.tanh(raw_support.float()).masked_fill(~base.valid, 0.0)
        score = (base.score - base.support + support).masked_fill(~base.valid, 0.0)
        return replace(
            base,
            raw_support=raw_support,
            support=support,
            score=score,
        )


__all__ = ["Round1FieldScoreHeads"]
