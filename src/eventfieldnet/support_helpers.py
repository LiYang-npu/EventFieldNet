"""Two bounded S replacements; inference consumes no temporal targets.

Install before optimizer construction. Call apply_support after R59 forward.
The inherited edge-S readout is a recorded reference, not the active S head.
Anchors explicitly exclude that replaced head, preventing a stale-head route.
"""

import torch
from torch import nn
from torch.nn import functional as F

from . import support_geometry

MODES = ("content_completeness", "direct_quality")


def masked_mean(value, mask):
    return (value * mask).sum() / mask.sum().clamp_min(1)


class DirectQuality(nn.Module):
    """Four geometry/content features, five zero-init trainable scalars.

    Constructing nn.Parameter(torch.zeros(...)) consumes no random numbers.
    The features are nonzero at the uniform gate, activating this readout on
    the first update and the upstream token gate on subsequent updates.
    """

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(4))
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, details):
        features = torch.stack(
            [
                details[name].float()
                for name in ("coverage", "density_ratio", "quality", "uniform")
            ],
            -1,
        )
        with torch.autocast(device_type=features.device.type, enabled=False):
            raw = F.linear(
                features, self.weight[None].float(), self.bias[None].float()
            ).squeeze(-1)
        return torch.tanh(raw).masked_fill(~details["eligible"], 0.0), features


def prepare_selector(selector, mode):
    if mode not in MODES:
        raise ValueError("unknown R65 S replacement " + str(mode))
    if not hasattr(selector, "r59_local_support"):
        raise ValueError("S replacements require the registered R59 local token gate")
    selector.r65_support_mode = mode
    if mode == "direct_quality":
        selector.r65_quality_readout = DirectQuality()


def apply_support(selector, field, valid, video_padding_mask=None):
    mode = selector.r65_support_mode
    legacy = field.support
    # Exact GT-free, S-independent reference. Deleting S at inference does
    # not reselect anchors, and obsolete edge-S cannot determine the region.
    anchor_score = (
        field.carrier.float()
        + field.evidence.float()
        + 0.5 * (field.transition_start.float() + field.transition_end.float())
    ).masked_fill(~valid, 0.0)
    anchors = support_geometry.select_anchors(anchor_score, valid)
    token_valid = selector._token_valid(valid, video_padding_mask)
    _, details, local_stats = selector.r59_local_support(
        field.round31_z_s, torch.zeros_like(legacy), valid, token_valid, anchors
    )
    if mode == "content_completeness":
        # Replace the old edge mean directly: no inherited-edge attenuation.
        support = torch.tanh(2.0 * (details["quality"] - details["uniform"]))
        support = support.masked_fill(~details["eligible"], 0.0)
        features = None
    else:
        support, features = selector.r65_quality_readout(details)
    field.r65_legacy_support = legacy
    field.r65_s_details = details
    field.r65_s_features = features
    field.r65_s_anchor_score = anchor_score.detach()
    field.r65_s_anchors = anchors
    field.support = support.masked_fill(~valid, 0.0)
    # Keep the public R59 structural cache consistent with actual deployment.
    field.r59_base_score = anchor_score
    field.r59_anchors = anchors
    field.r59_old_support = legacy
    field.r59_support_details = dict(details, delta=field.support - legacy)
    actual_local_stats = dict(local_stats)
    actual_local_stats["delta_abs"] = masked_mean((field.support - legacy).abs(), valid)
    actual_local_stats["delta_signed"] = masked_mean(field.support - legacy, valid)
    field.r59_stats = {
        "anchor_count": anchors["active"].sum(1).float().mean().detach(),
        "anchor_best_iou": masked_mean(anchors["best_iou"], valid.flatten(1)).detach(),
        **{"s/" + k: v.detach() for k, v in actual_local_stats.items()},
    }
    field.score = selector.compose_score(field)
    field.round31_rank_score = field.score
    field.round31_score_with_support = field.score
    field.round31_score_without_support = selector.compose_score(
        field, {"support": torch.zeros_like(field.support)}
    )
    field.round31_deployed_support = field.support
    stats = {
        "eligible_fraction": masked_mean(details["eligible"].float(), valid),
        "deployed_abs": masked_mean(field.support.abs(), valid),
        "legacy_abs": masked_mean(legacy.abs(), valid),
        "replacement_abs": masked_mean((field.support - legacy).abs(), valid),
        "saturation_fraction": masked_mean((field.support.abs() > 0.95).float(), valid),
        "coverage": masked_mean(details["coverage"], details["eligible"]),
        "purity": masked_mean(details["density_ratio"], details["eligible"]),
        "unattenuated_residual_abs": masked_mean(
            (details["quality"] - details["uniform"]).abs(), valid
        ),
        "gate_weight_norm": selector.r59_local_support.weight.float().norm(),
    }
    if mode == "direct_quality":
        stats["readout_weight_norm"] = (
            selector.r65_quality_readout.weight.float().norm()
        )
        stats["readout_bias"] = selector.r65_quality_readout.bias.float()
    field.r65_s_stats = {k: v.detach() for k, v in stats.items()}
    return field


def direct_quality_objective(field, geometry):
    """One Brier objective, equal occupied IoU bins and equal queries.

    Labels are used only here. Ineligible locations are excluded: the S
    deployment is structurally neutral there and must not dilute gradients.
    Existing coverage loss may be retained as a *reference* probe only.
    """
    valid = geometry.valid.bool() & field.r65_s_details["eligible"]
    target = geometry.max_iou.detach().float()
    prediction = (field.support.float() + 1.0) * 0.5
    error = (prediction - target).square()
    buckets = (target < 0.3, (target >= 0.3) & (target < 0.7), target >= 0.7)
    values, active, stats = [], [], {}
    for index, bucket in enumerate(buckets):
        mask = valid & bucket
        count = mask.sum((1, 2))
        values.append((error * mask).sum((1, 2)) / count.clamp_min(1))
        active.append(count > 0)
        stats[f"q{index}_count"] = count.float().mean()
        stats[f"q{index}_brier"] = masked_mean(error, mask)
    values, active = torch.stack(values, 1), torch.stack(active, 1)
    per_query = (values * active).sum(1) / active.sum(1).clamp_min(1)
    loss = masked_mean(per_query, active.any(1))
    stats.update(
        loss=loss,
        eligible_fraction=masked_mean(valid.float(), geometry.valid),
        active_queries=active.any(1).sum(),
        prediction_mean=masked_mean(prediction, valid),
        target_mean=masked_mean(target, valid),
        brier=masked_mean(error, valid),
    )
    return loss, {k: v.detach() for k, v in stats.items()}
