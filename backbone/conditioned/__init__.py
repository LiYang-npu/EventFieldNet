"""Conditioned span-backbone models and extension interfaces."""

from .contracts import C3Context, C3Extension, C3ExtensionOutput
from .extensions import IdentityExtension
from .model import ConditionedSpanModel, ConditionedSpanOutput

__all__ = [
    "C3Context",
    'ConditionedSpanModel',
    'ConditionedSpanOutput',
    "C3Extension",
    "C3ExtensionOutput",
    "IdentityExtension",
]
