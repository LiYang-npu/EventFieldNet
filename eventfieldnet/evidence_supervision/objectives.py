"""GT-bag counterfactual E objectives; no saliency or candidate-IoU input.

This release enables C02 only. Rows contain the same video's E tokens under its correct and paired query.
Pairs must be reciprocal.  C00/C02 use the normalized GT overlap weights;
C01/C03 reweight within each GT using detached correct-query E.  Every GT
has equal weight within its query and every active query has equal weight.
The caller supplies deterministic, differentiable E-only scores and applies
the external E coefficient.  This module never changes model/RNG state.
"""

import math

import torch


ARMS = ("C02",)


def objective(clean, wrong, gt_weights, pairs, valid, arm, margin=0.2, temperature=0.2):
    """Public model interface; temperature is the local pool tau."""
    return counterfactual_evidence_loss(
        clean, wrong, gt_weights, pairs, valid, arm=arm, margin=margin, tau=temperature
    )

def _require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def counterfactual_evidence_loss(
    clean, wrong, gt_weights, pairs, valid, arm, margin=0.2, tau=0.2
):
    """Return an unscaled scalar loss and detached scalar diagnostic tensors.

    Args:
        clean, wrong: Floating [B,L] E scores, matched token positions.
        gt_weights: Nonnegative [B,G,L] metadata. Each nonempty GT sums to
            one, empty/padded GTs are all zero. Padding must already be zero.
            GT geometry must come from the dataset's actual time mapping.
        pairs: Integer [B], reciprocal on valid rows; -1 allowed if invalid.
        valid: Boolean [B] accepted semantic-pair mask. Both sides must be
            accepted. A pair missing a GT bag on either side is skipped.
        arm: C02 only: mean pooling and row/column hinges.

    A local weight is proportional to gt_weight*exp(clean.detach()/tau).
    The exact same weight pools clean and wrong, with no gradient through
    the selection weights. Missing GTs/queries are never counted as zeros
    in the denominator. Empty accepted sets return a graph-connected zero.
    """
    _require(arm in ARMS, f"unknown arm {arm!r}")
    _require(math.isfinite(float(margin)) and margin > 0, "margin must be positive finite")
    _require(math.isfinite(float(tau)) and tau > 0, "tau must be positive finite")
    _require(clean.ndim == 2 and wrong.shape == clean.shape, "clean/wrong must match [B,L]")
    _require(clean.is_floating_point() and wrong.dtype == clean.dtype, "score dtypes must match and be floating")
    batch, length = clean.shape
    _require(gt_weights.ndim == 3 and gt_weights.shape[0] == batch and gt_weights.shape[2] == length,
             "gt_weights must be [B,G,L]")
    _require(gt_weights.is_floating_point(), "GT weights must be floating")
    _require(pairs.shape == (batch,) and pairs.dtype in (torch.int8, torch.int16, torch.int32, torch.int64),
             "pairs must be integer [B]")
    _require(valid.shape == (batch,) and valid.dtype == torch.bool, "valid must be bool [B]")
    _require(all(t.device == clean.device for t in (wrong, gt_weights, pairs, valid)), "all inputs must share device")
    _require(torch.isfinite(clean).all() and torch.isfinite(wrong).all(), "nonfinite E scores")
    _require(torch.isfinite(gt_weights).all() and (gt_weights >= 0).all(), "GT weights must be finite nonnegative")
    _require(((pairs >= -1) & (pairs < batch)).all(), "pair index outside [-1,B)")
    if batch:
        _require((pairs[valid] >= 0).all(), "valid query has no pair")
        ids = torch.arange(batch, device=clean.device)
        selected = pairs[valid].long()
        _require(valid[selected].all(), "pair acceptance must be symmetric")
        _require((pairs[selected] == ids[valid]).all(), "valid pairs must be reciprocal")
        _require((selected != ids[valid]).all(), "self-pairs are not counterfactuals")

    base = gt_weights.detach().to(dtype=clean.dtype)
    mass = base.sum(-1)
    gt_mask = mass > 0
    _require(torch.isclose(mass[gt_mask], torch.ones_like(mass[gt_mask]), atol=1e-5, rtol=1e-5).all(),
             "each nonempty GT weight must sum to one")
    safe_pairs = pairs.clamp_min(0).long()
    has_gt = gt_mask.any(-1)
    active = valid & has_gt
    if batch:
        active = active & has_gt[safe_pairs]
    graph_zero = (clean.sum() + wrong.sum()) * 0.0
    count = active.to(clean.dtype).sum()
    gt_count = gt_mask.to(clean.dtype).sum(-1)
    is_local = False

    weights = base

    def query_gt_mean(value):
        return (value * gt_mask.to(value.dtype)).sum(-1) / gt_count.clamp_min(1)

    def active_mean(value):
        return (value * active.to(value.dtype)).sum() / count.clamp_min(1)

    a = query_gt_mean((weights * clean[:, None, :]).sum(-1))
    b = query_gt_mean((weights * wrong[:, None, :]).sum(-1))
    row_gap = a - b
    col_gap = a - b[safe_pairs] if batch else a - b
    row_hinge = (float(margin) - row_gap).relu()
    col_hinge = (float(margin) - col_gap).relu()
    combined = 0.5 * (row_hinge + col_hinge)
    loss = active_mean(combined) + graph_zero

    # Entropy is measured per GT, then averaged by GT/query exactly as loss.
    entropy = -(weights * weights.clamp_min(torch.finfo(weights.dtype).tiny).log()).sum(-1)
    effective = entropy.exp() * gt_mask.to(entropy.dtype)
    max_weight = weights.amax(-1) if length else mass * 0.0
    row_active = row_hinge > 0
    col_active = col_hinge > 0
    all_active = row_active & col_active
    if batch:
        all_active = all_active & all_active[safe_pairs]
    metrics = {
        "active_queries": count,
        "active_pairs": count / 2.0,
        "active_gt": (gt_count * active.to(gt_count.dtype)).sum(),
        "available_gt": gt_mask.to(clean.dtype).sum(),
        "accepted_queries": valid.to(clean.dtype).sum(),
        "skipped_missing_gt_queries": (valid & ~active).to(clean.dtype).sum(),
        "clean_score": active_mean(a),
        "wrong_score": active_mean(b),
        "row_gap": active_mean(row_gap),
        "column_gap": active_mean(col_gap),
        "row_hinge": active_mean(row_hinge),
        "column_hinge": active_mean(col_hinge),
        "row_violation_fraction": active_mean(row_active.to(clean.dtype)),
        "column_violation_fraction": active_mean(col_active.to(clean.dtype)),
        "column_only_violation_fraction": active_mean((col_active & ~row_active).to(clean.dtype)),
        "row_only_violation_fraction": active_mean((row_active & ~col_active).to(clean.dtype)),
        "all_four_hinges_active_fraction": active_mean(all_active.to(clean.dtype)),
        "column_excess_hinge": active_mean((col_hinge - row_hinge).relu()),
        "loss_minus_row_only": loss - active_mean(row_hinge),
        "did_gap": active_mean(row_gap + row_gap[safe_pairs]) if batch else graph_zero,
        "pool_entropy": active_mean(query_gt_mean(entropy)),
        "pool_effective_clips": active_mean(query_gt_mean(effective)),
        "pool_max_weight": active_mean(query_gt_mean(max_weight)),
        "support_clips": active_mean(query_gt_mean((base > 0).to(clean.dtype).sum(-1))),
        "clean_saturation_fraction": active_mean(query_gt_mean((base * (clean.abs()[:, None, :] >= 0.95)).sum(-1))),
        "wrong_saturation_fraction": active_mean(query_gt_mean((base * (wrong.abs()[:, None, :] >= 0.95)).sum(-1))),
        "local_pool": clean.new_tensor(float(is_local)),
        "row_and_column": clean.new_tensor(float(arm in ("C02", "C03"))),
        "loss": loss,
    }
    return loss, {key: value.detach() for key, value in metrics.items()}
