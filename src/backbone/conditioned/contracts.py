"""Versioned extension contracts around the strongest Stage55 C3 carrier."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Protocol, runtime_checkable

from torch import Tensor, nn

from training.contracts import LossResult


C3_EXTENSION_API_VERSION = "c3_extension_v1"


@dataclass(frozen=True)
class C3Context:
    """GT-free tensors exposed to an extension during forward.

    The temporal coordinate system is the original C3 triangular span grid.
    Targets never appear here. Extensions may score existing complete spans,
    but coordinate-changing proposals require a separate decoder adapter.
    """

    token_features: Tensor
    query_features: Tensor
    video_padding_mask: Tensor
    span_valid_mask: Tensor
    base_span_logits: Tensor
    base_span_probs: Tensor
    start_logits: Tensor
    end_logits: Tensor


@dataclass
class C3ExtensionOutput:
    """Optional additive outputs from a C3 extension."""

    logit_delta: Optional[Tensor] = None
    state: Any = None
    diagnostics: Mapping[str, Tensor | float] = field(default_factory=dict)


@runtime_checkable
class C3Extension(Protocol):
    """Minimal interface future experiments implement instead of forking C3."""

    api_version: str

    def train(self, mode: bool = True) -> Any: ...

    def parameters(self, recurse: bool = True): ...

    def forward(self, context: C3Context) -> C3ExtensionOutput: ...

    def compute_loss(
        self,
        output: C3ExtensionOutput,
        model_output: Any,
        batch: Any,
        epoch: int,
    ) -> Optional[LossResult]: ...

    def set_epoch(self, epoch: int, training: bool) -> Mapping[str, Any] | None: ...

    def contract(self) -> Mapping[str, Any]: ...


class C3ExtensionModule(nn.Module):
    """Convenience base class with safe no-op loss and phase hooks."""

    api_version = C3_EXTENSION_API_VERSION

    def compute_loss(
        self,
        output: C3ExtensionOutput,
        model_output: Any,
        batch: Any,
        epoch: int,
    ) -> Optional[LossResult]:
        del output, model_output, batch, epoch
        return None

    def set_epoch(self, epoch: int, training: bool) -> Mapping[str, Any] | None:
        return {"epoch": int(epoch), "training": bool(training)}

    def contract(self) -> Mapping[str, Any]:
        return {
            "api_version": self.api_version,
            "coordinate_movement": False,
            "prediction_fusion": False,
            "ground_truth_in_forward": False,
        }


def validate_extension(extension: nn.Module) -> None:
    version = getattr(extension, "api_version", None)
    if version != C3_EXTENSION_API_VERSION:
        raise TypeError(
            f"extension API mismatch: expected {C3_EXTENSION_API_VERSION}, got {version!r}"
        )
    for method in ("forward", "compute_loss", "set_epoch", "contract"):
        if not callable(getattr(extension, method, None)):
            raise TypeError(f"C3 extension is missing callable {method}()")
    contract = dict(extension.contract())
    forbidden = {
        "coordinate_movement": True,
        "prediction_fusion": True,
        "ground_truth_in_forward": True,
    }
    violations = [
        name for name, value in forbidden.items() if contract.get(name) == value
    ]
    if violations:
        raise ValueError(f"C3 v1 extension violates base contract: {violations}")


__all__ = [
    "C3Context",
    "C3Extension",
    "C3ExtensionModule",
    "C3ExtensionOutput",
    "C3_EXTENSION_API_VERSION",
    "validate_extension",
]
