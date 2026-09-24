"""Real-grid geometric competition, and GT-free same-length S centering."""

from __future__ import annotations
import torch
from torch import Tensor
from torch.nn import functional as F


def center_support_by_length(score: Tensor, valid: Tensor):
    """Differentiable group mean over actual legal candidates, O(B L^2)."""
    b, length, width = score.shape
    if length != width or valid.shape != score.shape:
        raise ValueError("square scores and matching validity are required")
    index = torch.arange(length, device=score.device)
    groups = (index[None, :] - index[:, None]).clamp_min(0).flatten()
    groups = groups[None].expand(b, -1)
    mask = valid.bool().flatten(1)
    values = score.float().flatten(1).masked_fill(~mask, 0.0)
    sums = values.new_zeros((b, length)).scatter_add(1, groups, values)
    counts = values.new_zeros((b, length)).scatter_add(1, groups, mask.float())
    means = sums / counts.clamp_min(1.0)
    centered = (values - means.gather(1, groups)).masked_fill(~mask, 0.0)
    return centered.reshape_as(score), {"means": means, "counts": counts}


NEGATIVE_SELECTION_MODES = (
    "hardest",
    "semihard",
    "soft",
    "stratified",
    "counterfactual",
    "uniform",
    "topk",
    "adaptive_margin",
)


def _cpu_noise(shape, device, dtype=torch.float32):
    """Uniform(0,1) noise drawn from the ambient CPU RNG stream, moved to
    ``device``. Never touches the CUDA RNG: the parent runtime owns that
    stream (see model/_rng.py), and any model-internal branch that samples
    on a CUDA generator would make its consumption order depend on this
    code path, which the runtime's capture/restore machinery does not
    expect. The global CPU stream is already captured/restored by the
    runtime around each step, so drawing from it here composes correctly
    with checkpoint-resume determinism.
    """
    return torch.rand(shape, dtype=dtype).to(device)


def _gumbel_noise(shape, device):
    u = _cpu_noise(shape, device).clamp_(1e-6, 1 - 1e-6)
    return -torch.log(-torch.log(u))


def select_candidate_pairs(
    score: Tensor, geometry, gt_mask: Tensor, *, mode: str = "hardest", epoch: int = 0
):
    """One detached clean-F negative per GT/direction; ties use flat index.

    IoUs come directly from candidate_geometry. No GT rounding, surrogate
    coordinate reconstruction, or extra tolerance alters pool membership.

    ``mode`` controls how the negative is picked from the eligible pool
    (round32/round34 extension; every mode other than "hardest" is opt-in
    and does not change behavior for y0..y9 or any config that does not set
    it). "hardest" is byte-identical to the original always-argmax
    selection. See agent_workspace/claude/work/round34_negative_selection_design.md
    for the literature each mode is grounded in.
    """
    if mode not in NEGATIVE_SELECTION_MODES:
        raise ValueError(f"unknown negative_selection_mode {mode!r}")
    with torch.no_grad():
        b, length, _ = score.shape
        valid = geometry.valid.bool().flatten(1)
        iou = geometry.all_iou.float().flatten(1, 2)
        gm = gt_mask.bool()
        if iou.shape[:2] != valid.shape or iou.shape[2] != gm.shape[1]:
            raise ValueError("candidate/GT dimensions disagree")
        positive_quality, p = iou.masked_fill(~valid[..., None], -1.0).max(1)
        positive_valid = gm & positive_quality.ge(0.7) & valid.any(1)[:, None]
        quality = iou.masked_fill(~gm[:, None], -1.0).max(-1).values
        flat = torch.arange(length * length, device=score.device)
        starts, ends = flat // length, flat % length
        ps, pe = p // length, p % length
        same = flat[None, :, None].eq(p[:, None])
        short = (
            (starts[None, :, None] >= ps[:, None])
            & (ends[None, :, None] <= pe[:, None])
            & ~same
        )
        wide = (
            (starts[None, :, None] <= ps[:, None])
            & (ends[None, :, None] >= pe[:, None])
            & ~same
        )

        # eligibility gap: fixed at .2 for every mode except "adaptive_margin",
        # which anneals it from an easier .4 down to the original .2 over the
        # first 20 epochs (collapse-prevention literature: start with an
        # easier margin, tighten as training stabilizes).
        if mode == "adaptive_margin":
            eligibility_gap = max(0.2, 0.4 - 0.01 * float(epoch))
        else:
            eligibility_gap = 0.2
        training = (
            valid[..., None]
            & positive_valid[:, None]
            & (quality[..., None] <= positive_quality[:, None] - eligibility_gap)
        )

        detached_f = score.detach().float().flatten(1)
        positive_score = detached_f.gather(1, p)  # [B, G]
        width = ends - starts  # [L*L]
        pwidth = pe - ps  # [B, G]

        selected, available, margins, pool_counts = [], [], [], []
        for direction in (short, wide):
            eligible = training & direction  # [B, L*L, G]
            scores_bc = detached_f[..., None].expand_as(iou)  # [B, L*L, G]
            masked_scores = scores_bc.masked_fill(~eligible, -torch.inf)
            hardest_n = masked_scores.argmax(1)

            if mode in ("hardest", "adaptive_margin"):
                n = hardest_n
            elif mode == "semihard":
                # FaceNet-style: prefer the *least* difficult negative that
                # still violates the hinge margin, instead of the hardest.
                # Falls back to "hardest" wherever nothing violates (hinge
                # is 0 there regardless of which eligible index is picked).
                implied_margin = (
                    positive_quality[:, None, :] - quality[:, :, None]
                )  # [B,L*L,G]
                gap = positive_score[:, None, :] - scores_bc
                violation = implied_margin - gap
                violates = eligible & violation.gt(0)
                semihard_n = violation.masked_fill(~violates, torch.inf).argmin(1)
                n = torch.where(violates.any(1), semihard_n, hardest_n)
            elif mode == "soft":
                # Temperature-softened score-weighted sampling (Gumbel-max
                # trick) instead of a deterministic argmax over live scores.
                temperature = 2.0
                noisy = masked_scores / temperature + _gumbel_noise(
                    masked_scores.shape, score.device
                )
                n = noisy.masked_fill(~eligible, -torch.inf).argmax(1)
                n = torch.where(eligible.any(1), n, hardest_n)
            elif mode == "stratified":
                # Two IoU-quality strata (near-threshold vs far-below);
                # coin-flip which stratum to draw from, then pick uniformly
                # at random within it -- avoids always drawing from the
                # single hardest stratum.
                near = eligible & (
                    quality[:, :, None] > positive_quality[:, None, :] - 0.4
                )
                far = eligible & ~near
                coin = _cpu_noise((b, iou.shape[-1]), score.device).lt(0.5)
                use_near = torch.where(
                    near.any(1), coin, torch.zeros_like(coin)
                ) | ~far.any(1)
                pool = torch.where(use_near[:, None, :], near, far)
                pool = torch.where(pool.any(1, keepdim=True), pool, eligible)
                noise = _cpu_noise(masked_scores.shape, score.device)
                n = noise.masked_fill(~pool, -torch.inf).argmax(1)
                n = torch.where(eligible.any(1), n, hardest_n)
            elif mode == "counterfactual":
                # Geometry-only choice: the eligible candidate whose width is
                # closest to the positive's width (smallest structural
                # perturbation that still qualifies), independent of the
                # model's current score -- same spirit as the local
                # counterfactual replacement pairs already used for E/S/T.
                width_gap = (width[None, :, None] - pwidth[:, None, :]).abs().float()
                n = width_gap.masked_fill(~eligible, torch.inf).argmin(1)
                n = torch.where(eligible.any(1), n, hardest_n)
            elif mode == "uniform":
                # Pure random draw among all eligible candidates, no score
                # weighting at all -- the cleanest control for "does hard
                # mining help at all" versus picking blind.
                noise = _cpu_noise(masked_scores.shape, score.device)
                n = noise.masked_fill(~eligible, -torch.inf).argmax(1)
                n = torch.where(eligible.any(1), n, hardest_n)
            elif mode == "topk":
                # Bounded top-K by score, then uniform-random within that K
                # (does not chase whichever single candidate is currently
                # hardest; K is fixed and does not grow as training sharpens
                # the score distribution).
                k = min(8, iou.shape[1])
                topk_scores, topk_idx = masked_scores.topk(k, dim=1)  # [B,K,G]
                pool_size = eligible.sum(1).clamp(max=k)  # [B,G]
                pick = (
                    _cpu_noise(pool_size.shape, score.device)
                    * pool_size.clamp_min(1).float()
                ).long()
                pick = pick.clamp(max=k - 1)
                n = topk_idx.gather(1, pick[:, None, :]).squeeze(1)
                n = torch.where(eligible.any(1), n, hardest_n)
            else:  # pragma: no cover - guarded by the earlier ValueError
                raise AssertionError(mode)

            active = eligible.any(1)
            margin = positive_quality - quality.gather(1, n)
            selected.append(n)
            available.append(active)
            margins.append(margin.masked_fill(~active, 0.0))
            pool_counts.append(eligible.sum(1))
        return dict(
            positive=p,
            positive_quality=positive_quality,
            positive_valid=positive_valid,
            negative=torch.stack(selected, -1),
            available=torch.stack(available, -1),
            margin=torch.stack(margins, -1),
            pool_counts=torch.stack(pool_counts, -1),
            gt_mask=gm,
            directions=("short", "wide"),
            negative_selection_mode=mode,
        )


def pair_query_loss(score: Tensor, pairs, *, wide_only: bool):
    """Directions equally within GT, active GT equally within query."""
    flat = score.float().flatten(1)
    p = flat.gather(1, pairs["positive"])
    n = flat.gather(1, pairs["negative"].flatten(1)).reshape_as(pairs["negative"])
    gaps = p[..., None] - n
    available = pairs["available"].clone()
    if wide_only:
        available[..., 0] = False
    hinge = F.relu(pairs["margin"] - gaps).masked_fill(~available, 0.0)
    gt_active = available.any(-1)
    per_gt = hinge.sum(-1) / available.sum(-1).clamp_min(1)
    query = per_gt.sum(-1) / gt_active.sum(-1).clamp_min(1)
    active = gt_active.any(-1)
    return (
        query,
        active,
        dict(
            gaps=gaps,
            hinge=hinge,
            available=available,
            per_gt=per_gt,
            gt_active=gt_active,
        ),
    )


def active_mean(value: Tensor, active: Tensor):
    return value[active].mean() if bool(active.any()) else value.sum() * 0.0


def pair_metrics(pairs, detail):
    available = detail["available"]
    gap = detail["gaps"][available].detach()
    margin = pairs["margin"][available]
    zero = detail["gaps"].detach().new_zeros(())
    return dict(
        active_query_count=detail["gt_active"].any(-1).sum().float(),
        active_gt_count=detail["gt_active"].sum().float(),
        pair_count=available.sum().float(),
        short_pair_count=available[..., 0].sum().float(),
        wide_pair_count=available[..., 1].sum().float(),
        gap_mean=gap.mean() if gap.numel() else zero,
        margin_mean=margin.mean() if margin.numel() else zero,
        margin_met_rate=(gap >= margin).float().mean() if gap.numel() else zero,
        positive_eligible_gt_count=pairs["positive_valid"].sum().float(),
        total_gt_count=pairs["gt_mask"].sum().float(),
    )
