"""Factories and strict C3-best initialization for modular experiments."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
from torch import nn

from training.config import RunnerConfig
from feature_bridge.plugin import build_repository_data
from backbone.evaluator import SegmentOfficialEvalAdapter
from backbone.model import STAGE32_KWARGS

from .contracts import C3ExtensionModule, validate_extension
from .model import C3ExperimentModel


INITIALIZATION_MODES = ("c3_best", "stage14_best", "random")
STAGE14_NEW_PREFIXES = (
    "boundary_preserving.",
    "extra_blocks.",
    "matched_head.",
    "shared_head.",
    "extension.",
)


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


def _checkpoint_mode(payload: Mapping[str, Any]) -> Optional[str]:
    config = payload.get("config", {})
    if not isinstance(config, Mapping):
        return None
    model_factory = config.get("model_factory", {})
    if not isinstance(model_factory, Mapping):
        return None
    kwargs = model_factory.get("kwargs", {})
    return (
        str(kwargs.get("mode"))
        if isinstance(kwargs, Mapping) and kwargs.get("mode")
        else None
    )


def initialize_from_c3_best(
    model: C3ExperimentModel,
    path: str | Path,
    expected_epoch: Optional[int] = None,
) -> dict[str, Any]:
    checkpoint = Path(path).resolve()
    if checkpoint.name != "best_val.pt":
        raise RuntimeError(
            f"C3 base must be selected best_val.pt, got {checkpoint.name}"
        )
    payload = checkpoint_payload(checkpoint)
    state = checkpoint_state(payload)
    expected_missing = set(model.state_dict()) - set(state)
    illegal_missing = sorted(
        name for name in expected_missing if not name.startswith("extension.")
    )
    if illegal_missing:
        raise RuntimeError(
            f"non-extension parameters missing from C3 best: {illegal_missing}"
        )
    incompatible = model.load_state_dict(state, strict=False)
    if (
        set(incompatible.missing_keys) != expected_missing
        or incompatible.unexpected_keys
    ):
        raise RuntimeError(f"strict C3-best initialization mismatch: {incompatible}")
    epoch = payload.get("epoch")
    if (
        expected_epoch is not None
        and epoch is not None
        and int(epoch) != int(expected_epoch)
    ):
        raise RuntimeError(f"C3 best epoch {epoch} != expected {expected_epoch}")
    mode = _checkpoint_mode(payload)
    if mode not in (None, "C3_identity"):
        raise RuntimeError(f"C3 base checkpoint mode must be C3_identity, got {mode!r}")
    return {
        "checkpoint": str(checkpoint),
        "source_epoch": None if epoch is None else int(epoch),
        "source_mode": mode,
        "source_selection": "best_official_validation",
        "strict_base_load": True,
        "missing_extension_keys": sorted(expected_missing),
        "unexpected_keys": [],
        "source_prediction_outputs_read": False,
        "prediction_fusion": False,
        "coordinate_movement": False,
        "extension_contract": dict(model.extension.contract()),
    }


def initialize_from_stage14_best(
    model: C3ExperimentModel,
    path: str | Path,
    expected_epoch: Optional[int] = None,
) -> dict[str, Any]:
    """Load Stage14's shared carrier and initialize only later-stage modules.

    Every tensor shared by Stage14 and C3 must exist with an identical shape.
    Missing tensors are accepted only for modules introduced after Stage14.
    """
    checkpoint = Path(path).resolve()
    if checkpoint.name != "best_val.pt":
        raise RuntimeError(
            f"Stage14 base must be archived as selected best_val.pt, got {checkpoint.name}"
        )
    payload = checkpoint_payload(checkpoint)
    state = checkpoint_state(payload)
    target = model.state_dict()
    expected_missing = set(target) - set(state)
    illegal_missing = sorted(
        name for name in expected_missing if not name.startswith(STAGE14_NEW_PREFIXES)
    )
    unexpected = sorted(set(state) - set(target))
    shape_mismatch = sorted(
        name
        for name in set(state) & set(target)
        if state[name].shape != target[name].shape
    )
    if illegal_missing or unexpected or shape_mismatch:
        raise RuntimeError(
            "strict Stage14 initialization mismatch: "
            f"illegal_missing={illegal_missing}, unexpected={unexpected}, "
            f"shape_mismatch={shape_mismatch}"
        )
    incompatible = model.load_state_dict(state, strict=False)
    if (
        set(incompatible.missing_keys) != expected_missing
        or incompatible.unexpected_keys
    ):
        raise RuntimeError(f"strict Stage14 initialization mismatch: {incompatible}")
    epoch = payload.get("epoch")
    if (
        expected_epoch is not None
        and epoch is not None
        and int(epoch) != int(expected_epoch)
    ):
        raise RuntimeError(f"Stage14 best epoch {epoch} != expected {expected_epoch}")
    return {
        "checkpoint": str(checkpoint),
        "source_epoch": None if epoch is None else int(epoch),
        "source_mode": "fscc_local",
        "source_selection": "stage14_best_official_validation",
        "strict_base_load": True,
        "missing_later_stage_keys": sorted(expected_missing),
        "unexpected_keys": [],
        "source_prediction_outputs_read": False,
        "prediction_fusion": False,
        "coordinate_movement": False,
        "extension_contract": dict(model.extension.contract()),
    }


def build_c3_experiment_model(
    config: Optional[RunnerConfig],
    base_checkpoint: Optional[str] = None,
    initialization: str = "c3_best",
    extension: Mapping[str, Any]
    | str = "backbone.conditioned.extensions:build_identity_extension",
    expected_base_epoch: Optional[int] = None,
    extension_lr: float = 1.0e-4,
    **kwargs: Any,
) -> C3ExperimentModel:
    del config
    if initialization not in INITIALIZATION_MODES:
        raise ValueError(f"initialization must be one of {INITIALIZATION_MODES}")
    model = C3ExperimentModel(
        extension=build_extension(extension),
        extension_lr=extension_lr,
        **STAGE32_KWARGS,
        **kwargs,
    )
    if initialization == "c3_best":
        if not base_checkpoint:
            raise ValueError("base_checkpoint is required for c3_best initialization")
        model.initialization_audit = initialize_from_c3_best(
            model, base_checkpoint, expected_base_epoch
        )
    elif initialization == "stage14_best":
        if not base_checkpoint:
            raise ValueError(
                "base_checkpoint is required for stage14_best initialization"
            )
        model.initialization_audit = initialize_from_stage14_best(
            model, base_checkpoint, expected_base_epoch
        )
    else:
        if base_checkpoint:
            raise ValueError("random initialization forbids base_checkpoint")
        if expected_base_epoch is not None:
            raise ValueError("random initialization forbids expected_base_epoch")
        model.initialization_audit = {
            "checkpoint": None,
            "source_epoch": None,
            "source_mode": None,
            "source_selection": "random_initialization",
            "strict_base_load": False,
            "missing_extension_keys": [],
            "unexpected_keys": [],
            "source_prediction_outputs_read": False,
            "prediction_fusion": False,
            "coordinate_movement": False,
            "extension_contract": dict(model.extension.contract()),
        }
    return model


def build_official_evaluator(
    config: RunnerConfig,
    root: str,
    project_root: str,
    wrapper: Optional[str] = None,
):
    del config
    path = (
        Path(wrapper)
        if wrapper
        else Path(root) / "stage55" / "backbone" / "conditioned" / "official_eval.py"
    )
    return SegmentOfficialEvalAdapter(
        wrapper=str(path), project_root=str(project_root), root=str(root)
    )


__all__ = [
    "build_c3_experiment_model",
    "build_extension",
    "build_official_evaluator",
    "build_repository_data",
    "checkpoint_payload",
    "checkpoint_state",
    "initialize_from_c3_best",
    "initialize_from_stage14_best",
    "STAGE14_NEW_PREFIXES",
    "INITIALIZATION_MODES",
]
