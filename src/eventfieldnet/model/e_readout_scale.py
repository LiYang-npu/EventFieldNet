"""Parameter-free input scaling used only by the Round31 E readout.

The carrier, S input, and T input deliberately do not use this module.  The
operator is kept in its own file so the deployed E input and its diagnostics
share exactly the same arithmetic.
"""

from __future__ import annotations

from typing import Any

import torch


EPSILON = 1.0e-6
FIXED_GAIN = 974.43609777058
E_READOUT_SCALES = ("raw", "fixed_init", "query_rms", "token_rms")


def _check_input(
    z: torch.Tensor, valid_tokens: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(z, torch.Tensor) or z.ndim != 3:
        raise ValueError("E readout input must have shape [B,T,C]")
    if z.shape[-1] != 64:
        raise ValueError(f"E readout input must have C=64, got {z.shape[-1]}")
    if not isinstance(valid_tokens, torch.Tensor) or valid_tokens.shape != z.shape[:2]:
        raise ValueError("valid_tokens must have shape [B,T]")
    valid = valid_tokens.to(dtype=torch.bool)
    if (~valid).all(dim=1).any():
        raise ValueError("an all-padding query cannot be scaled")
    # Validation is performed on the actual valid values.  Padding is allowed
    # to be arbitrary in an upstream diagnostic, but never enters a norm.
    z32 = z if z.dtype == torch.float32 else z.float()
    if not torch.isfinite(z32.masked_select(valid.unsqueeze(-1))).all():
        raise ValueError("valid E readout input contains NaN/Inf")
    return z32, valid


def scale_e_readout_input(
    z: torch.Tensor,
    valid_tokens: torch.Tensor,
    mode: str,
    *,
    gain: float = FIXED_GAIN,
    epsilon: float = EPSILON,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return the E-readout input and graph-connected scale telemetry.

    ``raw`` intentionally returns the original tensor object: this preserves
    the Round25 raw graph and its floating point operation order.  Adaptive
    modes use FP32 norms with the denominator left in the autograd graph.
    ``gain_tensor`` is shaped ``[B,T,1]`` and is broadcast by the readout.
    """
    if mode not in E_READOUT_SCALES:
        raise ValueError(
            f"unknown e_readout_scale {mode!r}; expected one of {E_READOUT_SCALES}"
        )
    if (
        isinstance(epsilon, bool)
        or not isinstance(epsilon, (float, int))
        or float(epsilon) != EPSILON
    ):
        raise ValueError("Round31 fixes epsilon=1e-6")
    if (
        isinstance(gain, bool)
        or not isinstance(gain, (float, int))
        or float(gain) != FIXED_GAIN
    ):
        raise ValueError("Round31 fixes initialization gain=974.43609777058")
    z32, valid = _check_input(z, valid_tokens)
    bsz, steps, _ = z.shape

    if mode == "raw":
        # Do not mask, cast, or otherwise reorder this branch.  RawInteraction
        # already supplies the deployment mask and Round25 parity depends on
        # retaining its exact graph.
        gain_tensor = z.new_ones((bsz, steps, 1))
        denominator = z.new_ones((bsz, 1, 1))
        return z, {
            "mode": mode,
            "gain_tensor": gain_tensor,
            "denominator": denominator,
            "valid_tokens": valid,
            "epsilon": float(epsilon),
        }

    z_valid = z32.masked_fill(~valid.unsqueeze(-1), 0.0)
    if mode == "fixed_init":
        denominator = z32.new_ones((bsz, steps, 1))
        gain_tensor = denominator * float(gain)
    elif mode == "query_rms":
        # One RMS per query; the denominator is intentionally differentiable.
        count = valid.sum(dim=1, keepdim=True).to(dtype=z32.dtype).unsqueeze(-1)
        mean_sq = z_valid.square().sum(dim=(1, 2), keepdim=True) / (
            count * z32.shape[-1]
        )
        denominator = (mean_sq + float(epsilon)).sqrt()
        gain_tensor = (denominator.reciprocal()).expand(-1, steps, -1)
    else:  # token_rms
        mean_sq = z_valid.square().mean(dim=-1, keepdim=True)
        denominator = (mean_sq + float(epsilon)).sqrt()
        gain_tensor = denominator.reciprocal()

    gain_tensor = gain_tensor.masked_fill(~valid.unsqueeze(-1), 0.0)
    transformed = z_valid * gain_tensor
    if not torch.isfinite(transformed).all():
        raise ValueError("scaled E readout input contains NaN/Inf")
    return transformed, {
        "mode": mode,
        "gain_tensor": gain_tensor,
        "denominator": denominator,
        "valid_tokens": valid,
        "epsilon": float(epsilon),
    }


def summarize_e_readout_scale(
    z: torch.Tensor,
    transformed: torch.Tensor,
    telemetry: dict[str, Any],
) -> dict[str, Any]:
    """Small detached telemetry record for field probes."""
    valid = telemetry["valid_tokens"]
    gain = telemetry["gain_tensor"]
    denom = telemetry["denominator"]
    valid3 = valid.unsqueeze(-1)

    def _finite_quantiles(x: torch.Tensor) -> list[float]:
        values = x.detach().float().masked_select(valid3).flatten()
        if values.numel() == 0:
            return []
        return [
            float(q)
            for q in torch.quantile(
                values, torch.tensor((0.1, 0.5, 0.9), device=values.device)
            )
        ]

    return {
        "mode": telemetry["mode"],
        "epsilon": telemetry["epsilon"],
        "input_rms": float(
            torch.sqrt(
                z.detach().float().masked_select(valid3).square().mean() + 0.0
            ).item()
        ),
        "output_rms": float(
            torch.sqrt(
                transformed.detach().float().masked_select(valid3).square().mean() + 0.0
            ).item()
        ),
        "gain_quantiles": _finite_quantiles(gain),
        "denominator_quantiles": _finite_quantiles(denom.expand_as(gain)),
        "valid_count": int(valid.sum().item()),
        "padding_zero": bool(
            (transformed.detach().masked_select(~valid3) == 0).all().item()
        ),
        "requires_grad": bool(transformed.requires_grad),
    }


def _self_test() -> None:
    torch.manual_seed(2600)
    z = torch.randn(2, 5, 64, dtype=torch.float32, requires_grad=True)
    valid = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]], dtype=torch.bool)
    raw, raw_t = scale_e_readout_input(z, valid, "raw")
    assert raw is z
    assert torch.equal(raw_t["gain_tensor"], torch.ones(2, 5, 1))
    fixed, fixed_t = scale_e_readout_input(z, valid, "fixed_init")
    assert torch.allclose(fixed[valid], z[valid] * FIXED_GAIN)
    for mode in ("query_rms", "token_rms"):
        out, telemetry = scale_e_readout_input(z, valid, mode)
        assert torch.isfinite(out).all()
        assert torch.all(out[~valid] == 0)
        assert torch.all(telemetry["gain_tensor"][valid] > 0)
        # Adaptive modes have no hidden fixed gain multiplier.
        if mode == "token_rms":
            assert torch.allclose(
                out[valid] * telemetry["denominator"][valid],
                z[valid],
                atol=2e-5,
                rtol=2e-5,
            )
        out.sum().backward(retain_graph=True)
        assert z.grad is not None and torch.isfinite(z.grad).all()


if __name__ == "__main__":
    _self_test()
    print("round31 e-readout scale self-test: PASS")
