"""Candidate A loss primitive: anchored query-relative ordinal supervision."""

import torch


def query_relative_ordinal(
    clean, wrong, ratings, rated, token_valid, wrong_valid, margin=0.2
):
    if clean.ndim != 2 or any(
        x.shape != clean.shape for x in (ratings, rated, token_valid)
    ):
        raise ValueError("Expected aligned B x L tensors")
    if wrong_valid.shape != (clean.shape[0],):
        raise ValueError("Expected per-query wrong validity")
    valid = token_valid.bool()
    eligible = rated.bool() & valid
    paired = wrong_valid.bool()
    if (
        not torch.isfinite(clean[valid]).all()
        or not torch.isfinite(ratings[eligible]).all()
    ):
        raise ValueError("Nonfinite valid input")
    if wrong is None:
        if paired.any():
            raise ValueError("Missing wrong field for valid counterfactual")
        wrong = torch.zeros_like(clean)
    if wrong.shape != clean.shape:
        raise ValueError("Wrong field shape mismatch")
    if not torch.isfinite(wrong[valid & paired[:, None]]).all():
        raise ValueError("Nonfinite wrong input")
    c = clean.float().masked_fill(~valid, 0)
    w = wrong.float().masked_fill(~valid | ~paired[:, None], 0)
    labels = ratings.detach().masked_fill(~eligible, 0)
    pairs = (
        eligible[:, :, None]
        & eligible[:, None, :]
        & (labels[:, :, None] > labels[:, None, :])
    )
    count = pairs.sum((1, 2))
    active = count > 0
    relative = active & paired
    fallback = active & ~paired
    d = c[:, :, None] - c[:, None, :]
    dw = w[:, :, None] - w[:, None, :]
    clean_hinge = torch.relu(margin - d)
    difference_hinge = torch.relu(margin - (d - dw))
    c_each = clean_hinge.masked_fill(~pairs, 0).sum((1, 2)) / count.clamp_min(1)
    q_each = difference_hinge.masked_fill(~pairs, 0).sum((1, 2)) / count.clamp_min(1)
    each = torch.where(paired, 0.5 * (c_each + q_each), c_each)
    loss = each[active].mean() if active.any() else c.sum() * 0

    def mean_or_zero(value, mask):
        values = value.detach()[mask]
        return values.mean() if values.numel() else c.detach().new_zeros(())

    rp = pairs & paired[:, None, None]
    detail = dict(
        pair_counts=count.detach(),
        active_queries=active.sum().detach(),
        eligible_clips=eligible.sum(1).detach(),
        pair_margin_mean=mean_or_zero(d, pairs),
        pair_margin_met_rate=mean_or_zero((d >= margin).float(), pairs),
        relative_queries=relative.sum().detach(),
        fallback_queries=fallback.sum().detach(),
        no_rated_pair_queries=(~active).sum().detach(),
        relative_pair_count=rp.sum().detach(),
        clean_hinge_query_mean=mean_or_zero(c_each, active),
        difference_hinge_query_mean=mean_or_zero(q_each, relative),
        clean_gap_pair_mean=mean_or_zero(d, pairs),
        wrong_gap_pair_mean=mean_or_zero(dw, rp),
        difference_gap_pair_mean=mean_or_zero(d - dw, rp),
        clean_margin_met_pair_rate=mean_or_zero((d >= margin).float(), pairs),
        difference_margin_met_pair_rate=mean_or_zero((d - dw >= margin).float(), rp),
    )
    detail["per_query"] = {
        "active": active.detach(),
        "relative": relative.detach(),
        "clean_hinge": c_each.detach(),
        "difference_hinge": q_each.detach(),
        "clean_gap": (
            d.masked_fill(~pairs, 0).sum((1, 2)) / count.clamp_min(1)
        ).detach(),
        "wrong_gap": (
            dw.masked_fill(~pairs, 0).sum((1, 2)) / count.clamp_min(1)
        ).detach(),
        "difference_gap": (
            (d - dw).masked_fill(~pairs, 0).sum((1, 2)) / count.clamp_min(1)
        ).detach(),
    }
    return loss, detail


def explicit_query_validity(metadata, batch_size, device, base_pair=None):
    """Only filter the new A term; never change existing task pair construction."""
    from field_core.adapter import _metadata_rows, _valid_wrong_query_pairs

    rows = _metadata_rows(metadata, batch_size)
    permutation, pair = _valid_wrong_query_pairs(metadata, batch_size)
    if base_pair is not None:
        pair &= base_pair.detach().cpu().bool()
    valid = pair.clone()
    missing = identical = 0
    bins = [0, 0, 0]
    similarities = []
    for i in range(batch_size):
        if not pair[i]:
            similarities.append(None)
            continue
        a = (rows[i] or {}).get("query")
        b = (rows[int(permutation[i])] or {}).get("query")
        a = " ".join(a.split()).casefold() if isinstance(a, str) else ""
        b = " ".join(b.split()).casefold() if isinstance(b, str) else ""
        if not a or not b:
            valid[i] = False
            missing += 1
            similarities.append(None)
            continue
        aa = set(a.split())
        bb = set(b.split())
        sim = len(aa & bb) / len(aa | bb)
        similarities.append(sim)
        if a == b:
            valid[i] = False
            identical += 1
            continue
        bins[0 if sim < 0.25 else 1 if sim < 0.5 else 2] += 1
    return (
        valid.to(device),
        dict(
            base_wrong_queries=int(pair.sum()),
            missing_query_text=missing,
            identical_query_text=identical,
            valid_wrong_queries=int(valid.sum()),
            jaccard_lt025_queries=bins[0],
            jaccard_025_05_queries=bins[1],
            jaccard_ge05_queries=bins[2],
        ),
        similarities,
    )
