"""Shared diagnostics and trainability helpers for the input conditioner."""
from __future__ import annotations
import torch
from torch import Tensor, nn

def _score_correlation(left: Tensor, right: Tensor, valid: Tensor) -> Tensor:
    x = left[valid].float()
    y = right[valid].float()
    x = x - x.mean()
    y = y - y.mean()
    return (x * y).mean() / (
        x.square().mean().sqrt() * y.square().mean().sqrt()
    ).clamp_min(1.0e-6)

def _freeze_boundary_gate_only_branch(model: nn.Module) -> list[str]:
    frozen = []
    if model.boundary_preserving.use_boundary_gate:
        raise AssertionError("dense tri-field baseline requires boundary gate off")
    for name, parameter in model.named_parameters():
        if name == "boundary_preserving.boundary_strength" or name.startswith(
            "boundary_preserving.boundary_head."
        ):
            parameter.requires_grad_(False)
            frozen.append(name)
    return frozen
