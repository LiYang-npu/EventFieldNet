"""Low-overhead, exact optimizer-step telemetry for round31.

The official stage50 runner owns the training loop.  Round31 keeps that loop's
semantics and calls this recorder at the same three boundaries as the runner:
after finite-gradient checks, immediately after clipping, and immediately
after ``optimizer.step``.  Only the first real optimizer step of an epoch is
sampled, so telemetry cannot turn into a second per-batch probe.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Optional, Sequence

import torch


SCHEMA = "eventfieldnet_trifield_round31_step_telemetry_v1"


class StepTelemetryError(RuntimeError):
    """The requested optimizer-step telemetry contract is invalid."""


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise StepTelemetryError(
                f"{name} must be scalar, got shape {tuple(value.shape)}"
            )
        value = value.detach().float().cpu().item()
    value = float(value)
    if not math.isfinite(value):
        raise StepTelemetryError(f"{name} is non-finite: {value!r}")
    return value


def _values(gradient: torch.Tensor) -> torch.Tensor:
    if gradient.is_sparse:
        return gradient.detach().coalesce().values().float()
    return gradient.detach().float()


def _global_grad_norm(parameters: Sequence[torch.nn.Parameter]) -> float:
    """Return the exact L2 norm over the same parameter list used by clip."""
    total: Optional[torch.Tensor] = None
    for parameter in parameters:
        gradient = parameter.grad
        if gradient is None:
            continue
        values = _values(gradient)
        term = torch.sum(values * values, dtype=torch.float64)
        total = term if total is None else total + term.to(total.device)
    if total is None:
        return 0.0
    return _finite_float(torch.sqrt(total), name="gradient_norm")


def _parameter_norm(values: Sequence[torch.Tensor]) -> float:
    total: Optional[torch.Tensor] = None
    for value in values:
        term = torch.sum(value.float() * value.float(), dtype=torch.float64)
        total = term if total is None else total + term.to(total.device)
    if total is None:
        return 0.0
    return _finite_float(torch.sqrt(total), name="parameter_norm")


def _round31_route_for_name(name: str) -> str:
    """Map a parameter name to a canonical R31 route bucket.

    ``parent_carrier`` is reserved for the actual parent model path;
    ``legacy_support`` is reserved for selector support residual parameters.
    The old ``legacy`` alias is emitted separately for compatibility.
    """
    if name.startswith(("selector.edge_head.", "edge_head.")):
        return "s_head"
    if name.startswith(("selector.s_projection.", "s_projection.")):
        return "s_projection"
    if name.startswith(("selector.e_interaction.", "e_interaction.")):
        return "e_interaction"
    if name.startswith(("selector.t_interaction.", "t_interaction.")):
        return "t_interaction"
    if name.startswith("parent_model."):
        return "parent_carrier"
    if "carrier" in name.lower() or name.endswith("carrier_bias"):
        return "carrier"
    if name.startswith(
        (
            "selector.support.",
            "selector.context_branch.",
            "selector.dispersion_branch.",
        )
    ):
        return "legacy_support"
    if name.startswith(("shared_head.", "score_head.")):
        return "legacy_support"
    return "other"


def _gradient_state(parameter: torch.nn.Parameter) -> dict[str, Any]:
    gradient = parameter.grad
    if gradient is None:
        return {
            "grad_is_none": True,
            "grad_is_zero": None,
            "grad_is_finite": None,
            "grad_l2": None,
        }
    values = _values(gradient)
    finite = bool(torch.isfinite(values).all()) if values.numel() else True
    zero = bool(values.eq(0).all()) if values.numel() else True
    l2 = _parameter_norm((values,))
    return {
        "grad_is_none": False,
        "grad_is_zero": zero,
        "grad_is_finite": finite,
        "grad_l2": l2,
    }


def _optimizer_state_snapshot(
    optimizer: Any, parameter: torch.nn.Parameter
) -> dict[str, Any]:
    state = getattr(optimizer, "state", {}).get(parameter, {})
    if not isinstance(state, Mapping) or not state:
        return {"state_present": False, "state_keys": [], "step": None}
    raw_step = state.get("step")
    if isinstance(raw_step, torch.Tensor):
        if raw_step.numel() != 1:
            step: Any = {"available": False, "reason": "non-scalar step"}
        else:
            step = float(raw_step.detach().float().cpu().item())
            if math.isfinite(step) and step.is_integer():
                step = int(step)
    elif raw_step is None:
        step = None
    else:
        try:
            step = float(raw_step)
            if math.isfinite(step) and step.is_integer():
                step = int(step)
        except (TypeError, ValueError):
            step = {"available": False, "reason": "non-numeric step"}
    return {
        "state_present": True,
        "state_keys": sorted(str(key) for key in state.keys()),
        "step": step,
    }


@dataclass
class _StepCapture:
    epoch: int
    batch_index: int
    local_optimizer_step: int
    global_optimizer_step: int
    max_norm: float
    grad_norm_preclip: float
    parameter_before: dict[int, torch.Tensor]
    parameter_before_norm: float
    learning_rates: list[float]
    group_names: list[str]
    grad_norm_postclip: Optional[float] = None
    clip_returned_norm: Optional[float] = None
    clip_coef: Optional[float] = None
    route_gradient_state: Optional[dict[str, dict[str, dict[str, Any]]]] = None
    optimizer_state_before: Optional[dict[str, dict[str, Any]]] = None


class StepTelemetryRecorder:
    """Record one exact training update per epoch.

    The recorder owns no optimizer state and never calls an optimizer method.
    The caller invokes :meth:`before_clip`, :meth:`after_clip`, and
    :meth:`after_step` around the already-authoritative operations.  Parameter
    snapshots are copied to CPU FP32 only for the selected update, which keeps
    GPU peak memory bounded while still measuring the actual post-step delta.
    """

    def __init__(
        self,
        optimizer: Any,
        parameters: Sequence[torch.nn.Parameter],
        *,
        enabled: bool = True,
        max_samples_per_epoch: int = 1,
    ) -> None:
        if max_samples_per_epoch != 1:
            raise StepTelemetryError(
                "round31 step telemetry supports at most one sample per epoch"
            )
        self.optimizer = optimizer
        unique: list[torch.nn.Parameter] = []
        seen: set[int] = set()
        for parameter in parameters:
            if not isinstance(parameter, torch.nn.Parameter):
                raise StepTelemetryError(
                    "telemetry parameters must be torch.nn.Parameter objects"
                )
            if not parameter.requires_grad or id(parameter) in seen:
                continue
            seen.add(id(parameter))
            unique.append(parameter)
        self.parameters = tuple(unique)
        self.enabled = bool(enabled)
        self.max_samples_per_epoch = 1
        self.global_optimizer_step = 0
        self.current_epoch: Optional[int] = None
        self._sampled = False
        self._capture: Optional[_StepCapture] = None
        self._last_result: Optional[dict[str, Any]] = None
        self._names_by_parameter: dict[int, str] = {}
        self._group_by_parameter: dict[int, int] = {}
        for group_index, group in enumerate(getattr(optimizer, "param_groups", ())):
            for parameter in group.get("params", ()):
                self._group_by_parameter.setdefault(id(parameter), group_index)

    @classmethod
    def from_config(
        cls,
        optimizer: Any,
        model: Any,
        raw_value: Any = None,
    ) -> "StepTelemetryRecorder":
        value = raw_value if isinstance(raw_value, Mapping) else {}
        enabled = bool(value.get("enabled", True))
        sample_count = int(value.get("max_samples_per_epoch", 1))
        parameters = [p for p in model.parameters() if p.requires_grad]
        recorder = cls(
            optimizer,
            parameters,
            enabled=enabled,
            max_samples_per_epoch=sample_count,
        )
        recorder._names_by_parameter = {id(p): n for n, p in model.named_parameters()}
        return recorder

    def begin_epoch(self, epoch: int) -> None:
        if self._capture is not None:
            raise StepTelemetryError(
                "cannot begin an epoch with an unfinished telemetry sample"
            )
        self.current_epoch = int(epoch)
        self._sampled = False
        self._last_result = None

    def before_clip(
        self,
        *,
        epoch: int,
        batch_index: int,
        local_optimizer_step: int,
        max_norm: float,
    ) -> Optional[_StepCapture]:
        if self.current_epoch != int(epoch):
            raise StepTelemetryError(
                f"telemetry epoch mismatch: active={self.current_epoch}, received={epoch}"
            )
        self.global_optimizer_step += 1
        if not self.enabled or self._sampled:
            return None
        self._sampled = True
        max_norm = _finite_float(max_norm, name="grad_clip")
        if max_norm <= 0.0:
            raise StepTelemetryError("grad_clip must be positive")
        before = {
            id(parameter): parameter.detach().float().cpu().clone()
            for parameter in self.parameters
        }
        group_names: list[str] = []
        learning_rates: list[float] = []
        for index, group in enumerate(getattr(self.optimizer, "param_groups", ())):
            group_names.append(str(group.get("name", index)))
            learning_rates.append(
                _finite_float(group.get("lr", 0.0), name=f"lr[{index}]")
            )
        route_gradient_state: dict[str, dict[str, dict[str, Any]]] = {}
        optimizer_state_before: dict[str, dict[str, Any]] = {}
        for parameter in self.parameters:
            name = self._names_by_parameter.get(
                id(parameter), f"<unnamed:{id(parameter)}>"
            )
            route = _round31_route_for_name(name)
            route_gradient_state.setdefault(route, {})[name] = _gradient_state(
                parameter
            )
            optimizer_state_before[name] = _optimizer_state_snapshot(
                self.optimizer, parameter
            )
        capture = _StepCapture(
            epoch=int(epoch),
            batch_index=int(batch_index),
            local_optimizer_step=int(local_optimizer_step),
            global_optimizer_step=int(self.global_optimizer_step),
            max_norm=max_norm,
            grad_norm_preclip=_global_grad_norm(self.parameters),
            parameter_before=before,
            parameter_before_norm=_parameter_norm(tuple(before.values())),
            learning_rates=learning_rates,
            group_names=group_names,
            route_gradient_state=route_gradient_state,
            optimizer_state_before=optimizer_state_before,
        )
        self._capture = capture
        return capture

    def after_clip(
        self,
        capture: Optional[_StepCapture],
        *,
        returned_norm: Any,
    ) -> None:
        if capture is None:
            return
        if capture is not self._capture:
            raise StepTelemetryError("unknown telemetry capture passed to after_clip")
        capture.clip_returned_norm = _finite_float(
            returned_norm, name="clip_returned_norm"
        )
        capture.grad_norm_postclip = _global_grad_norm(self.parameters)
        # torch.nn.utils.clip_grad_norm_ uses max_norm / (total_norm + 1e-6)
        # before clamping to one.  Use its returned total_norm so this audit
        # reports the same coefficient as the authoritative clipping call.
        clip_norm = capture.clip_returned_norm
        if clip_norm <= 0.0:
            capture.clip_coef = 1.0
        else:
            capture.clip_coef = min(1.0, capture.max_norm / (clip_norm + 1.0e-6))

    def after_step(self, capture: Optional[_StepCapture]) -> None:
        if capture is None:
            return
        if capture is not self._capture:
            raise StepTelemetryError("unknown telemetry capture passed to after_step")
        if capture.grad_norm_postclip is None or capture.clip_coef is None:
            raise StepTelemetryError("after_step called before after_clip")

        group_sq: dict[int, float] = {}
        group_max: dict[int, float] = {}
        group_counts: dict[int, int] = {}
        update_sq = 0.0
        update_max = 0.0
        parameter_after_norm_values: list[torch.Tensor] = []
        e_readout_steps = {}
        projection_steps = {}
        route_steps = {
            "s_head": {},
            "s_projection": {},
            "e_interaction": {},
            "t_interaction": {},
            "carrier": {},
            "parent_carrier": {},
            "legacy_support": {},
            "legacy": {},
            "other": {},
        }
        for parameter in self.parameters:
            before = capture.parameter_before[id(parameter)]
            after = parameter.detach().float().cpu()
            if tuple(after.shape) != tuple(before.shape):
                raise StepTelemetryError(
                    "parameter shape changed across optimizer.step"
                )
            delta = after - before
            delta_sq = float(torch.sum(delta * delta, dtype=torch.float64).item())
            update_sq += delta_sq
            delta_max = (
                _finite_float(delta.abs().max(), name="update_max_abs")
                if delta.numel()
                else 0.0
            )
            update_max = max(update_max, delta_max)
            parameter_after_norm_values.append(after)
            group = self._group_by_parameter.get(id(parameter), -1)
            group_sq[group] = group_sq.get(group, 0.0) + delta_sq
            group_max[group] = max(group_max.get(group, 0.0), delta_max)
            group_counts[group] = group_counts.get(group, 0) + int(parameter.numel())
            name = self._names_by_parameter.get(id(parameter), "")
            if name.startswith("selector.e_interaction.readout."):
                e_readout_steps[name] = {
                    "parameter_l2_before": _parameter_norm((before,)),
                    "parameter_l2_after": _parameter_norm((after,)),
                    "actual_step_l2": math.sqrt(max(0.0, delta_sq)),
                    "actual_step_max_abs": delta_max,
                    "finite": bool(
                        torch.isfinite(after).all() and torch.isfinite(delta).all()
                    ),
                }
            if (
                name.startswith(("selector.e_interaction.", "selector.s_projection."))
                and ".readout." not in name
            ):
                projection_steps[name] = {
                    "parameter_l2_before": _parameter_norm((before,)),
                    "parameter_l2_after": _parameter_norm((after,)),
                    "actual_step_l2": math.sqrt(max(0.0, delta_sq)),
                    "actual_step_max_abs": delta_max,
                    "optimizer_group": group,
                    "finite": bool(
                        torch.isfinite(after).all() and torch.isfinite(delta).all()
                    ),
                }
            # Keep an explicit route map for the R31 gradient-path audit.  A
            # missing name is placed in ``other`` rather than inferred as a
            # zero update; this preserves the distinction between no path and
            # a connected parameter whose AdamW step happens to be zero.
            route = _round31_route_for_name(name)
            route_steps[route][name] = {
                "parameter_l2_before": _parameter_norm((before,)),
                "parameter_l2_after": _parameter_norm((after,)),
                "actual_step_l2": math.sqrt(max(0.0, delta_sq)),
                "actual_step_max_abs": delta_max,
                "optimizer_group": group,
                "finite": bool(
                    torch.isfinite(after).all() and torch.isfinite(delta).all()
                ),
                "gradient_before_step": (
                    capture.route_gradient_state.get(route, {}).get(name, {})
                    if capture.route_gradient_state is not None
                    else {}
                ),
                "optimizer_state_before": (
                    capture.optimizer_state_before.get(name, {})
                    if capture.optimizer_state_before is not None
                    else {}
                ),
                "optimizer_state_after": _optimizer_state_snapshot(
                    self.optimizer, parameter
                ),
            }

        groups: list[dict[str, Any]] = []
        for group_index, name in enumerate(capture.group_names):
            groups.append(
                {
                    "index": group_index,
                    "name": name,
                    "lr": capture.learning_rates[group_index],
                    "update_l2": math.sqrt(max(0.0, group_sq.get(group_index, 0.0))),
                    "update_max_abs": float(group_max.get(group_index, 0.0)),
                    "parameter_numel": int(group_counts.get(group_index, 0)),
                }
            )
        if -1 in group_sq:
            groups.append(
                {
                    "index": -1,
                    "name": "unassigned",
                    "lr": 0.0,
                    "update_l2": math.sqrt(max(0.0, group_sq[-1])),
                    "update_max_abs": float(group_max.get(-1, 0.0)),
                    "parameter_numel": int(group_counts.get(-1, 0)),
                }
            )
        update_l2 = math.sqrt(max(0.0, update_sq))
        self._last_result = {
            "sampled": True,
            "epoch": capture.epoch,
            "batch_index": capture.batch_index,
            "local_optimizer_step": capture.local_optimizer_step,
            "global_optimizer_step": capture.global_optimizer_step,
            "grad_norm_preclip": capture.grad_norm_preclip,
            "grad_norm_postclip": capture.grad_norm_postclip,
            "grad_norm_before_clip": capture.grad_norm_preclip,
            "grad_norm_after_clip": capture.grad_norm_postclip,
            "clip_returned_norm": capture.clip_returned_norm,
            "grad_clip_max_norm": capture.max_norm,
            "clip_coef": float(capture.clip_coef),
            "clip_factor": float(capture.clip_coef),
            "actual_grad_norm_ratio": (
                float(capture.grad_norm_postclip) / capture.grad_norm_preclip
                if capture.grad_norm_preclip > 0.0
                else 1.0
            ),
            "clip_applied": bool(capture.clip_coef < 1.0),
            "parameter_l2_before": capture.parameter_before_norm,
            "parameter_l2_after": _parameter_norm(tuple(parameter_after_norm_values)),
            "update_l2": update_l2,
            "parameter_update_l2": update_l2,
            "actual_parameter_step_l2": update_l2,
            "update_max_abs": update_max,
            "relative_update_l2": update_l2
            / max(capture.parameter_before_norm, 1.0e-12),
            "learning_rates": {
                name: lr
                for name, lr in zip(capture.group_names, capture.learning_rates)
            },
            "groups": groups,
            "e_readout_parameters": e_readout_steps,
            "field_projection_parameters": projection_steps,
            "round31_route_parameter_steps": route_steps,
            "round31_route_step_schema": {
                "groups": [
                    "s_head",
                    "s_projection",
                    "e_interaction",
                    "t_interaction",
                    "carrier",
                    "parent_carrier",
                    "legacy_support",
                    "legacy",
                    "other",
                ],
                "scope": "actual optimizer step; gradients captured after backward before clipping; parameter deltas and optimizer state measured before/after optimizer.step",
                "gradient_fields": [
                    "grad_is_none",
                    "grad_is_zero",
                    "grad_is_finite",
                    "grad_l2",
                ],
                "optimizer_state_fields": ["state_present", "state_keys", "step"],
                "legacy_alias": "legacy mirrors selector support compatibility parameters; parent_model parameters are parent_carrier",
            },
        }
        self._capture = None

    def end_epoch(self, epoch: int) -> dict[str, Any]:
        if self._capture is not None:
            raise StepTelemetryError(
                "cannot finish epoch with an unfinished telemetry sample"
            )
        if self.current_epoch != int(epoch):
            raise StepTelemetryError(
                f"telemetry epoch mismatch at end: active={self.current_epoch}, received={epoch}"
            )
        result: dict[str, Any] = {
            "schema": SCHEMA,
            "epoch": int(epoch),
            "enabled": self.enabled,
            "max_samples_per_epoch": 1,
            "global_optimizer_step": int(self.global_optimizer_step),
            "sampled": self._last_result is not None,
        }
        if self._last_result is not None:
            result["sample"] = dict(self._last_result)
        return result

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "enabled": self.enabled,
            "max_samples_per_epoch": 1,
            "global_optimizer_step": int(self.global_optimizer_step),
            "last_epoch": self.current_epoch,
        }

    def load_state_dict(self, state: Any) -> None:
        if state in (None, {}):
            return
        if not isinstance(state, Mapping) or state.get("schema") != SCHEMA:
            raise StepTelemetryError("invalid round31 step telemetry checkpoint state")
        saved_enabled = bool(state.get("enabled", self.enabled))
        if saved_enabled != self.enabled:
            raise StepTelemetryError(
                "checkpoint telemetry enabled flag disagrees with config"
            )
        self.global_optimizer_step = int(state.get("global_optimizer_step", 0))
        if self.global_optimizer_step < 0:
            raise StepTelemetryError(
                "checkpoint global_optimizer_step must be non-negative"
            )
        saved_max = int(state.get("max_samples_per_epoch", 1))
        if saved_max != 1:
            raise StepTelemetryError(
                "checkpoint exceeds one telemetry sample per epoch"
            )


__all__ = ["SCHEMA", "StepTelemetryError", "StepTelemetryRecorder"]
