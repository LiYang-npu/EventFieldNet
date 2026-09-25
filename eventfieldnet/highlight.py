"""Clip-level highlight head and the fixed hard-negative training objective."""

import torch
from torch import nn
from torch.nn import functional as F


class HighlightHead(nn.Module):
    """Predict a highlight logit from shared (384) and evidence (64) features."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(448),
            nn.Linear(448, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, features):
        return self.net(features).squeeze(-1)


def highlight_loss(logits, targets, mask):
    """Balanced soft BCE, with an equal mixture of all and hardest negatives.

    Each target is the fraction of three annotators assigning VeryGood (4).
    Positives have target > 0; negatives have target == 0. Each nonempty
    positive/negative stratum contributes equally within a query, and queries
    contribute equally. Padded clips contribute neither values nor gradients.
    """
    targets = targets.detach()
    binary = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    positive = mask & (targets > 0)
    negative = mask & (targets == 0)
    positive_count = positive.sum(1)
    negative_count = negative.sum(1)

    positive_mean = (binary * positive).sum(1) / positive_count.clamp_min(1)
    negative_mean = (binary * negative).sum(1) / negative_count.clamp_min(1)
    hardest = binary.masked_fill(~negative, float("-inf")).topk(
        min(16, binary.shape[1]), dim=1
    ).values.clamp_min(0)
    hard_mean = hardest.sum(1) / negative_count.clamp(min=1, max=16)
    negative_mean = 0.5 * negative_mean + 0.5 * hard_mean

    strata = (positive_count > 0).float() + (negative_count > 0).float()
    per_query = (
        positive_mean * (positive_count > 0)
        + negative_mean * (negative_count > 0)
    ) / strata.clamp_min(1)
    active = mask.any(1)
    loss = per_query[active].mean() if active.any() else logits.sum() * 0
    return loss, {
        "bce": loss,
        "positive_clips": positive.sum(),
        "negative_clips": negative.sum(),
        "valid_clips": mask.sum(),
    }
