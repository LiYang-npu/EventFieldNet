"""R65 bounded carrier transforms and train-only quality intercept.

Call ``carrier_input`` BEFORE the inherited selector forward: R59 chooses its
local-support anchors from the composed score inside that forward. Mutating a
finished field would leave the anchors and cached scores inconsistent.

No GT, validation labels, parameters, RNG or checkpoints enter carrier_input.
The output is an INPUT to the existing tanh carrier readout, not its final value.
"""

import math

import torch
from torch import nn


def carrier_input(raw, valid, mode="centered_rms"):
    """Center valid scores, softly scale their variance, preserve graph paths.

    centered_rms is the same mathematical transform as R60. The optional
    centered_rms_smoothsign composes asinh with the inherited tanh, producing
    z/sqrt(1+z*z), a smooth bounded transform with polynomial rather than
    exponential tail slope. Neither mode guarantees improved MR.
    """
    if raw.ndim != 3 or raw.shape != valid.shape:
        raise ValueError("carrier input expects matching [B,L,L] tensors")
    if mode not in ("centered_rms", "centered_rms_smoothsign"):
        raise ValueError(mode)
    valid = valid.bool()
    with torch.autocast(device_type=raw.device.type, enabled=False):
        x = raw.float().masked_fill(~valid, 0.0)
        count = valid.sum((1, 2))
        n = count.clamp_min(1).float()
        mean = x.sum((1, 2)) / n
        centered = (x - mean[:, None, None]).masked_fill(~valid, 0.0)
        variance = centered.square().sum((1, 2)) / n
        denominator = (1.0 + variance).sqrt()
        z = (centered / denominator[:, None, None]).masked_fill(~valid, 0.0)
        transformed = torch.asinh(z) if mode.endswith("smoothsign") else z
        transformed = transformed.masked_fill(~valid, 0.0)
        deployed = transformed.tanh().masked_fill(~valid, 0.0)
        # These are local slopes w.r.t. z; the moment-normalization Jacobian
        # is not included. Use autograd(raw) probes for the actual total path.
        local_slope = (
            (1.0 + z.square()).pow(-1.5)
            if mode.endswith("smoothsign")
            else 1.0 - deployed.square()
        ).masked_fill(~valid, 0.0)
        stats = {
            "count": count,
            "raw_mean": mean,
            "raw_variance": variance,
            "denominator": denominator,
            "normalized_mean": z.sum((1, 2)) / n,
            "normalized_rms": (z.square().sum((1, 2)) / n).sqrt(),
            "deployed_mean": deployed.sum((1, 2)) / n,
            "local_bounding_slope": local_slope.sum((1, 2)) / n,
            "negative_saturation": ((deployed < -0.95) & valid).sum((1, 2)) / n,
            "positive_saturation": ((deployed > 0.95) & valid).sum((1, 2)) / n,
        }
    return transformed, {name: value.detach() for name, value in stats.items()}


def install_quality_bias(model, bias_init=0.0):
    """Install only R58's existing quality-loss intercept after R59 build.

    R59's builder enforces an unbiased reference. A new wrapper can first
    construct that identical reference, then call this helper explicitly.
    R58.r50_adjust_terms already consumes selector.r58_calibration_bias ONLY
    in the added quality BCE, and its parameter_groups places this scalar in
    its own LR=1e-3, weight_decay=0 group. Forward/ranking scores do not use it.
    """
    if not math.isfinite(float(bias_init)):
        raise ValueError("non-finite quality intercept")
    if getattr(model, "r58_spec", None) != {"mode": "kl_plus_quality", "bias": False}:
        raise ValueError("expected exact unbiased R58 reference before install")
    if hasattr(model.selector, "r58_calibration_bias"):
        raise ValueError("quality intercept already installed")
    model.selector.r58_calibration_bias = nn.Parameter(torch.tensor(float(bias_init)))
    model.r58_spec = {"mode": "kl_plus_quality", "bias": True}
    model.r58_bias_init = float(bias_init)
    return model.selector.r58_calibration_bias


def fit_quality_intercept(logits, targets, weights):
    """Optional CPU scalar fit to an explicitly TRAIN-only calibration panel.

    Caller must assert panel QIDs belong to training and use actual R58 source
    weights, query-length factors, and GT-active masks. Never derive this from
    validation metrics. Arrays are detached and the model is not updated here.
    """
    z, y, w = [
        x.detach().double().cpu().reshape(-1) for x in (logits, targets, weights)
    ]
    if not (z.shape == y.shape == w.shape):
        raise ValueError("calibration arrays must have identical shape")
    if not all(torch.isfinite(x).all() for x in (z, y, w)) or (w < 0).any():
        raise ValueError("invalid calibration arrays")
    active = w > 0
    if not active.any() or ((y[active] < 0) | (y[active] > 1)).any():
        raise ValueError("invalid calibration targets or empty panel")
    z, y, w = z[active], y[active], w[active]
    w = w / w.sum()
    target = float((y * w).sum())
    if not 0.0 < target < 1.0:
        raise ValueError("intercept fit requires a non-degenerate target mean")
    probability = lambda b: float((torch.sigmoid(z + b) * w).sum())
    lo, hi = -16.0, 16.0
    while probability(lo) >= target and lo > -1024.0:
        lo *= 2.0
    while probability(hi) <= target and hi < 1024.0:
        hi *= 2.0
    if not probability(lo) < target < probability(hi):
        raise ValueError("intercept fit bracket failed")
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if probability(mid) < target:
            lo = mid
        else:
            hi = mid
    bias = (lo + hi) / 2.0
    return bias, {
        "weighted_target_mean": target,
        "initial_probability_mean": probability(0.0),
        "calibrated_probability_mean": probability(bias),
        "stationarity_error": abs(probability(bias) - target),
        "candidate_count": int(z.numel()),
    }
