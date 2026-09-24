"""Candidate ordinal local evidence loss; explicit ratings only.

The low member is relatively weaker evidence, not a semantic negative label.
No IoU or GT-span quality threshold is used. Unrated zeros have no meaning.
"""

import torch


def explicit_local_ordinal_loss(evidence, ratings, rated, token_valid, margin=0.2):
    if evidence.ndim != 2 or any(
        x.shape != evidence.shape for x in (ratings, rated, token_valid)
    ):
        raise ValueError("all inputs must be B x L")
    eligible = rated.bool() & token_valid.bool()
    if not torch.isfinite(ratings[eligible]).all():
        raise ValueError("explicit valid ratings must be finite")
    if not torch.isfinite(evidence[token_valid.bool()]).all():
        raise ValueError("valid evidence must be finite")
    # Detach human ordinal labels and never use their absolute scale as targets.
    quality = ratings.detach().masked_fill(~eligible, 0)
    pairs = (
        eligible[:, :, None]
        & eligible[:, None, :]
        & (quality[:, :, None] > quality[:, None, :])
    )
    safe_evidence = evidence.masked_fill(~token_valid.bool(), 0)
    gap = safe_evidence[:, :, None] - safe_evidence[:, None, :]
    counts = pairs.sum((1, 2))
    each = torch.relu(margin - gap).masked_fill(~pairs, 0).sum(
        (1, 2)
    ) / counts.clamp_min(1)
    active = counts > 0
    value = each[active].mean() if active.any() else safe_evidence.sum() * 0
    observed_gap = gap.detach()[pairs]
    diagnostics = {
        "pair_margin_mean": observed_gap.mean()
        if observed_gap.numel()
        else value.detach().new_zeros(()),
        "pair_margin_met_rate": (observed_gap >= margin).float().mean()
        if observed_gap.numel()
        else value.detach().new_zeros(()),
    }
    return value, {
        **diagnostics,
        "pair_counts": counts.detach(),
        "active_queries": active.sum().detach(),
        "eligible_clips": eligible.sum(1).detach(),
        "term_source": "explicit_local_ordinal"
        if active.any()
        else "zero_no_explicit_pairs",
    }
