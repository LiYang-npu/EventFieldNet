"""Built-in C3 extensions; identity is the exact base-control implementation."""

from __future__ import annotations

from typing import Any, Mapping

from .contracts import C3Context, C3ExtensionModule, C3ExtensionOutput


class IdentityExtension(C3ExtensionModule):
    """No parameters, no logit change, no extra loss: exact C3 control."""

    def forward(self, context: C3Context) -> C3ExtensionOutput:
        del context
        return C3ExtensionOutput(
            logit_delta=None,
            diagnostics={"identity": 1.0, "logit_delta_abs_mean": 0.0},
        )

    def contract(self) -> Mapping[str, Any]:
        return {
            **super().contract(),
            "name": "identity",
            "trainable_parameters": 0,
            "base_equivalent": True,
        }


def build_identity_extension(**kwargs: Any) -> IdentityExtension:
    if kwargs:
        raise ValueError(f"identity extension accepts no kwargs: {sorted(kwargs)}")
    return IdentityExtension()


__all__ = ["IdentityExtension", "build_identity_extension"]
