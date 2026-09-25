"""Scratch extension construction; no checkpoint initialization factories."""
from __future__ import annotations
import importlib
from pathlib import Path
from typing import Any, Mapping
import torch
from torch import nn
from eventfieldnet.dataset import build_repository_data
from .contracts import C3ExtensionModule, validate_extension

def resolve_factory(target: str):
    if ":" not in target:
        raise ValueError(f"factory target must be module:symbol, got {target!r}")
    module_name, symbol = target.split(":", 1)
    module = importlib.import_module(module_name)
    try:
        return getattr(module, symbol)
    except AttributeError as exc:
        raise ImportError(f"{module_name!r} has no {symbol!r}") from exc

def build_extension(spec: Mapping[str, Any] | str) -> C3ExtensionModule:
    if isinstance(spec, str):
        target, kwargs = spec, {}
    else:
        target = str(spec["target"])
        kwargs = dict(spec.get("kwargs", {}))
    extension = resolve_factory(target)(**kwargs)
    if not isinstance(extension, nn.Module):
        raise TypeError("C3 extension factory must return torch.nn.Module")
    validate_extension(extension)
    return extension

def checkpoint_payload(path: str | Path) -> Mapping[str, Any]:
    checkpoint = Path(path)
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError(f"C3 base checkpoint payload must be mapping: {checkpoint}")
    return payload

def checkpoint_state(payload: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    state = payload.get("model", payload)
    if not isinstance(state, Mapping):
        raise TypeError("C3 base checkpoint has no model state")
    if state and all(str(key).startswith("module.") for key in state):
        return {str(key)[7:]: value for key, value in state.items()}
    return state
