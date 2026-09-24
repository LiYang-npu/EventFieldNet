"""Phase-bounded, resumable runtime for trifield_round31.

The runner deliberately keeps orchestration separate from the validated
field_core implementation.  It owns phase budgets, durable state,
checkpoint recovery, official validation, and bounded resource release.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import sys
import time
from dataclasses import dataclass, field
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import torch

from .epoch_runner import run_epoch_with_step_telemetry
from .step_telemetry import StepTelemetryRecorder
from .budget_ledger import recovered_seconds, ledger_payload


SCHEMA = "eventfieldnet_trifield_round31_runtime_v1"
SELECTION_METRICS = ("MR-mAP", "mAP@0.75", "R1@0.7")
DEFAULT_TARGET_MR = 57.0
DEFAULT_AUDIT_SNAPSHOT_LIMIT = 16


class Round31RuntimeError(RuntimeError):
    """Configuration or lifecycle error that should be reported to the caller."""


class OfficialMetricError(Round31RuntimeError):
    """The official evaluator did not return the required selection metrics."""


@dataclass(frozen=True)
class EarlyStopPolicy:
    enabled: bool = True
    warmup_epochs: int = 9
    target_metric: str = "MR-mAP"
    target_value: float = DEFAULT_TARGET_MR
    favorable_patience: int = 1
    poor_patience: int = 5
    poor_delta: float = 2.0
    no_improve_patience: int = 0
    no_improve_delta: float = 0.0
    # round32 extension (scheme B): stop when a val-side proxy metric (e.g.
    # candidate_rank/margin_met_rate) plateaus, instead of waiting for the
    # lagging official MR-mAP metric to already have dropped. Disabled by
    # default (plateau_metric=""); zero behavior change for existing configs
    # that don't set it.
    plateau_metric: str = ""
    plateau_source: str = "val"
    plateau_patience: int = 5
    plateau_min_relative_gain: float = 0.03

    @classmethod
    def from_value(cls, value: Any, *, default_warmup: int = 9) -> "EarlyStopPolicy":
        if value is None:
            return cls(warmup_epochs=default_warmup)
        if isinstance(value, bool):
            return cls(enabled=value, warmup_epochs=default_warmup)
        if not isinstance(value, Mapping):
            raise Round31RuntimeError("early_stop must be a mapping or boolean")
        return cls(
            enabled=bool(value.get("enabled", True)),
            warmup_epochs=max(0, int(value.get("warmup_epochs", default_warmup))),
            target_metric=str(value.get("target_metric", "MR-mAP")),
            target_value=float(value.get("target_value", DEFAULT_TARGET_MR)),
            favorable_patience=max(1, int(value.get("favorable_patience", 1))),
            poor_patience=max(1, int(value.get("poor_patience", 5))),
            poor_delta=max(0.0, float(value.get("poor_delta", 2.0))),
            no_improve_patience=max(0, int(value.get("no_improve_patience", 0))),
            no_improve_delta=max(0.0, float(value.get("no_improve_delta", 0.0))),
            plateau_metric=str(value.get("plateau_metric", "")),
            plateau_source=str(value.get("plateau_source", "val")),
            plateau_patience=max(1, int(value.get("plateau_patience", 5))),
            plateau_min_relative_gain=max(
                0.0, float(value.get("plateau_min_relative_gain", 0.03))
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "warmup_epochs": self.warmup_epochs,
            "target_metric": self.target_metric,
            "target_value": self.target_value,
            "favorable_patience": self.favorable_patience,
            "poor_patience": self.poor_patience,
            "poor_delta": self.poor_delta,
            "no_improve_patience": self.no_improve_patience,
            "no_improve_delta": self.no_improve_delta,
            "plateau_metric": self.plateau_metric,
            "plateau_source": self.plateau_source,
            "plateau_patience": self.plateau_patience,
            "plateau_min_relative_gain": self.plateau_min_relative_gain,
        }


@dataclass(frozen=True)
class PhaseBudget:
    name: str
    max_epochs: int
    wallclock_seconds: float
    start_epoch: int = 0
    schedule_epochs: Optional[int] = None
    warmup_epochs: int = 9
    audit_epochs: tuple[int, ...] = ()
    audit_snapshot_limit: int = DEFAULT_AUDIT_SNAPSHOT_LIMIT
    early_stop: EarlyStopPolicy = field(default_factory=EarlyStopPolicy)

    @classmethod
    def from_value(cls, value: Any, *, index: int = 0) -> "PhaseBudget":
        if not isinstance(value, Mapping):
            raise Round31RuntimeError(f"phase {index} must be a mapping")
        name = str(value.get("name", value.get("id", f"phase{index}")))
        start = int(value.get("start_epoch", value.get("resume_epoch", 0)))
        if "max_epochs" in value:
            count = int(value["max_epochs"])
        elif "epochs" in value:
            raw_end = int(value["epochs"])
            count = raw_end - start if raw_end > start else raw_end
        else:
            raise Round31RuntimeError(f"phase {name} is missing max_epochs")
        if count <= 0:
            raise Round31RuntimeError(f"phase {name} max_epochs must be positive")
        wall = float(value.get("wallclock_seconds", value.get("max_seconds", 0.0)))
        if wall < 0:
            raise Round31RuntimeError(
                f"phase {name} wallclock_seconds must be non-negative"
            )
        audit = tuple(sorted({int(x) for x in value.get("audit_epochs", ())}))
        limit = max(
            1, int(value.get("audit_snapshot_limit", DEFAULT_AUDIT_SNAPSHOT_LIMIT))
        )
        warmup = max(0, int(value.get("warmup_epochs", 9)))
        policy = EarlyStopPolicy.from_value(
            value.get("early_stop"), default_warmup=warmup
        )
        return cls(
            name=name,
            max_epochs=count,
            wallclock_seconds=wall,
            start_epoch=start,
            schedule_epochs=(
                None
                if value.get("schedule_epochs") is None
                else max(1, int(value["schedule_epochs"]))
            ),
            warmup_epochs=warmup,
            audit_epochs=audit,
            audit_snapshot_limit=limit,
            early_stop=policy,
        )

    @property
    def end_epoch(self) -> int:
        return self.start_epoch + self.max_epochs

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "max_epochs": self.max_epochs,
            "end_epoch": self.end_epoch,
            "wallclock_seconds": self.wallclock_seconds,
            "start_epoch": self.start_epoch,
            "schedule_epochs": self.schedule_epochs,
            "warmup_epochs": self.warmup_epochs,
            "audit_epochs": list(self.audit_epochs),
            "audit_snapshot_limit": self.audit_snapshot_limit,
            "early_stop": self.early_stop.to_dict(),
        }


@dataclass
class PhaseRuntime:
    """Already-constructed model/data/evaluator handles for one phase."""

    model: Any
    bundle: Any
    config: Any
    optimizer: Any
    scheduler: Any
    evaluator: Any
    device: Any
    registered_trainable_ids: Sequence[int] = ()
    run_epoch: Optional[Callable[..., Any]] = None
    request_factory: Optional[Callable[..., Any]] = None
    model_factory_target: str = ""
    data_factory_target: str = ""
    evaluator_factory_target: str = ""
    probe_suite: Any = None
    probe_runner: Optional[Callable[..., Any]] = None
    probe_epochs: Sequence[int] = ()
    fixed_panel: Any = None
    step_telemetry: Optional[StepTelemetryRecorder] = None


@dataclass(frozen=True)
class PhaseResult:
    status: str
    phase: str
    completed_epoch: int
    next_phase_index: Optional[int]
    reason: str
    best_epoch: Optional[int]
    best_metrics: Mapping[str, float]
    active_seconds: float
    output_dir: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "phase": self.phase,
            "completed_epoch": self.completed_epoch,
            "next_phase_index": self.next_phase_index,
            "reason": self.reason,
            "best_epoch": self.best_epoch,
            "best_metrics": dict(self.best_metrics),
            "active_seconds": self.active_seconds,
            "output_dir": self.output_dir,
        }


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return _json_safe(value.detach().cpu().item())
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(x) for x in value]
    return str(value)


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        _json_safe(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    temp.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temp, path)


def _append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_json_safe(value), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(value, temp)
    os.replace(temp, path)


def _load_torch(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _capture_rng() -> dict[str, Any]:
    state: dict[str, Any] = {
        "torch_cpu": torch.get_rng_state(),
        "python": random.getstate(),
    }
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except Exception:
        pass
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: Mapping[str, Any]) -> None:
    if "torch_cpu" in state:
        torch.set_rng_state(state["torch_cpu"])
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        try:
            import numpy as np

            np.random.set_state(state["numpy"])
        except Exception:
            pass
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _release_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        torch.cuda.empty_cache()


def _is_oom(exc: BaseException) -> bool:
    return (
        isinstance(exc, torch.cuda.OutOfMemoryError)
        or "out of memory" in str(exc).lower()
    )


def phase_budgets(raw: Mapping[str, Any]) -> list[PhaseBudget]:
    values = raw.get("phases")
    if values is None:
        value = raw.get("phase", raw)
        values = [value]
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise Round31RuntimeError("phases must be a non-empty list")
    phases = [PhaseBudget.from_value(value, index=i) for i, value in enumerate(values)]
    if not phases:
        raise Round31RuntimeError("at least one phase is required")
    return phases


def _safe_name(name: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("._")
    return value or "phase"


def _metric(metrics: Mapping[str, Any], name: str) -> float:
    aliases = {
        "MR-mAP": ("MR-mAP", "mr_map", "MR_mAP", "mR-mAP", "moment_retrieval_mAP"),
        "mAP@0.75": ("mAP@0.75", "mAP_0.75", "map75", "mAP75", "mAP@75"),
        "R1@0.7": ("R1@0.7", "R1_0.7", "r1@0.7", "R1@70", "R1_70"),
    }
    for key in aliases.get(name, (name,)):
        if key in metrics:
            value = float(metrics[key])
            if math.isfinite(value):
                return value
    raise OfficialMetricError(
        f"official evaluator missing finite metric {name!r}: {sorted(metrics)}"
    )


def _selection_key(
    metrics: Mapping[str, Any], epoch: int
) -> tuple[float, float, float, int]:
    return (
        _metric(metrics, "MR-mAP"),
        _metric(metrics, "mAP@0.75"),
        _metric(metrics, "R1@0.7"),
        -int(epoch),
    )


def _normalize_official(value: Any) -> dict[str, float]:
    candidate = value
    if isinstance(value, Mapping) and isinstance(value.get("metrics"), Mapping):
        candidate = value["metrics"]
    if not isinstance(candidate, Mapping):
        raise OfficialMetricError("official evaluator result must be a mapping")
    try:
        mod = importlib.import_module("training.evaluator")
        normalizer = getattr(mod, "normalize_official_metrics", None)
        if normalizer is not None:
            normalized = normalizer(candidate, require_all=True)
            if isinstance(normalized, Mapping):
                candidate = normalized
    except Exception:
        pass
    return {str(k): float(v) for k, v in candidate.items() if _is_finite_number(v)}


def _is_finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    result: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            result.append(json.loads(line))
    return result


def _early_stop_reason(
    records: Sequence[Mapping[str, Any]],
    policy: EarlyStopPolicy,
    *,
    phase_start_epoch: int,
    epoch: int,
) -> Optional[str]:
    """Return a stop reason only after warmup and a complete official record."""
    if not policy.enabled or not records:
        return None
    local_epoch = int(epoch) - int(phase_start_epoch)
    if local_epoch <= policy.warmup_epochs:
        return None
    values: list[float] = []
    for record in records:
        metrics = record.get("official", record)
        if isinstance(metrics, Mapping) and _is_finite_number(
            metrics.get(policy.target_metric)
        ):
            values.append(float(metrics[policy.target_metric]))
    if not values:
        return None
    latest = records[-1].get("official", records[-1])
    if not isinstance(latest, Mapping) or not _is_finite_number(
        latest.get(policy.target_metric)
    ):
        return None
    current = values[-1]
    if (
        current >= policy.target_value
        and len(values) >= policy.favorable_patience
        and all(v >= policy.target_value for v in values[-policy.favorable_patience :])
    ):
        return f"target_reached:{policy.target_metric}>={policy.target_value:g}"
    if len(values) >= policy.poor_patience:
        best = max(values[:-1]) if len(values) > 1 else values[0]
        if best - current >= policy.poor_delta and all(
            best - v >= policy.poor_delta for v in values[-policy.poor_patience :]
        ):
            return f"poor_progress:{policy.target_metric} dropped {best - current:g}"
    if policy.plateau_metric:
        pvalues: list[float] = []
        for record in records:
            source = record.get(policy.plateau_source, {})
            if isinstance(source, Mapping) and _is_finite_number(
                source.get(policy.plateau_metric)
            ):
                pvalues.append(float(source[policy.plateau_metric]))
        if len(pvalues) >= policy.plateau_patience:
            window = pvalues[-policy.plateau_patience :]
            base = window[0]
            best_in_window = max(window)
            gain = (
                (best_in_window - base) / abs(base)
                if base != 0.0
                else (best_in_window - base)
            )
            if gain < policy.plateau_min_relative_gain:
                return (
                    f"plateau:{policy.plateau_source}.{policy.plateau_metric} "
                    f"relative_gain={gain:g}<{policy.plateau_min_relative_gain:g}"
                )
    if policy.no_improve_patience:
        if len(values) > policy.no_improve_patience:
            best = max(values[: -policy.no_improve_patience])
            recent = values[-policy.no_improve_patience :]
            if max(recent) - best <= policy.no_improve_delta:
                return f"no_improve:{policy.target_metric}"
    return None


def _apply_loss_weight_schedule(model: Any, raw: Mapping[str, Any], epoch: int) -> None:
    """round32 extension (scheme A): optionally mutate model.loss_weights in
    place once epoch reaches a configured switch point.

    Opt-in via top-level config key "loss_weight_schedule":
        {"switch_epoch": 8, "after": {"rank": 0.25, "evidence": 0.2, ...}}
    Absent for any config that doesn't set this key -- zero behavior change.
    model.loss_weights is a plain mutable dict read fresh by compute_loss_terms
    on every step (see model/adapter.py), so this takes effect on the very
    next training step after the switch epoch; it does not touch the
    ROUND31_VARIANTS table or the one-time construction-time weight lock,
    which only validates the *initial* weights a config builds the model with.
    """
    schedule = raw.get("loss_weight_schedule")
    if not schedule:
        return
    if not isinstance(schedule, Mapping):
        raise Round31RuntimeError("loss_weight_schedule must be a mapping")
    if not hasattr(model, "loss_weights") or not isinstance(model.loss_weights, dict):
        raise Round31RuntimeError(
            "loss_weight_schedule requires a mutable model.loss_weights dict"
        )
    switch_epoch = int(schedule.get("switch_epoch", 0))
    after = schedule.get("after")
    if not isinstance(after, Mapping):
        raise Round31RuntimeError("loss_weight_schedule.after must be a mapping")
    if epoch < switch_epoch:
        return
    for key, value in after.items():
        if key not in model.loss_weights:
            raise Round31RuntimeError(
                f"loss_weight_schedule.after has unknown weight key {key!r}"
            )
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < 0.0:
            raise Round31RuntimeError(
                f"loss_weight_schedule.after[{key!r}] must be finite and non-negative"
            )
        model.loss_weights[key] = numeric


def _apply_negative_selection_mode(model: Any, raw: Mapping[str, Any]) -> None:
    """round34 extension: optionally set model.negative_selection_mode from
    an opt-in top-level config key "negative_selection_mode" (one of the
    modes in model/candidate_pair_loss.py::NEGATIVE_SELECTION_MODES).
    Absent for any config that doesn't set this key -- model stays on
    "hardest", the original always-argmax selection, zero behavior change.
    """
    value = raw.get("negative_selection_mode")
    if value is None:
        return
    if not hasattr(model, "negative_selection_mode"):
        raise Round31RuntimeError(
            "negative_selection_mode requires model.negative_selection_mode"
        )
    model.negative_selection_mode = str(value)


def _apply_length_bias_config(model: Any, raw: Mapping[str, Any]) -> None:
    """round36 length-bias countermeasures: optionally set
    model.rank_length_reweight_gain (H1) and model.duration_aux_weight (H4)
    from opt-in top-level config keys. Absent keys leave the model at its
    0.0 default -- zero behavior change for every config that doesn't set
    them.
    """
    gain = raw.get("rank_length_reweight_gain")
    if gain is not None:
        if not hasattr(model, "rank_length_reweight_gain"):
            raise Round31RuntimeError(
                "rank_length_reweight_gain requires model.rank_length_reweight_gain"
            )
        numeric = float(gain)
        if not math.isfinite(numeric) or numeric < 0.0:
            raise Round31RuntimeError(
                "rank_length_reweight_gain must be finite and non-negative"
            )
        model.rank_length_reweight_gain = numeric
    weight = raw.get("duration_aux_weight")
    if weight is not None:
        if not hasattr(model, "duration_aux_weight"):
            raise Round31RuntimeError(
                "duration_aux_weight requires model.duration_aux_weight"
            )
        numeric = float(weight)
        if not math.isfinite(numeric) or numeric < 0.0:
            raise Round31RuntimeError(
                "duration_aux_weight must be finite and non-negative"
            )
        model.duration_aux_weight = numeric
    length_gain = raw.get("length_conditional_gain")
    if length_gain is not None:
        if not hasattr(model, "length_conditional_gain"):
            raise Round31RuntimeError(
                "length_conditional_gain requires model.length_conditional_gain"
            )
        numeric = float(length_gain)
        if not math.isfinite(numeric) or numeric < 0.0:
            raise Round31RuntimeError(
                "length_conditional_gain must be finite and non-negative"
            )
        model.length_conditional_gain = numeric
    carrier_weight = raw.get("carrier_desaturation_weight")
    if carrier_weight is not None:
        if not hasattr(model, "carrier_desaturation_weight"):
            raise Round31RuntimeError(
                "carrier_desaturation_weight requires model.carrier_desaturation_weight"
            )
        numeric = float(carrier_weight)
        if not math.isfinite(numeric) or numeric < 0.0:
            raise Round31RuntimeError(
                "carrier_desaturation_weight must be finite and non-negative"
            )
        model.carrier_desaturation_weight = numeric
    topk_restrict = raw.get("rank_topk_restrict")
    if topk_restrict is not None:
        if not hasattr(model, "rank_topk_restrict"):
            raise Round31RuntimeError(
                "rank_topk_restrict requires model.rank_topk_restrict"
            )
        numeric = int(topk_restrict)
        if numeric < 0:
            raise Round31RuntimeError(
                "rank_topk_restrict must be a non-negative integer"
            )
        model.rank_topk_restrict = numeric
    short_relief = raw.get("short_relief_gain")
    if short_relief is not None:
        if not hasattr(model, "short_relief_gain"):
            raise Round31RuntimeError(
                "short_relief_gain requires model.short_relief_gain"
            )
        numeric = float(short_relief)
        if not math.isfinite(numeric) or numeric < 0.0:
            raise Round31RuntimeError(
                "short_relief_gain must be finite and non-negative"
            )
        model.short_relief_gain = numeric
    shift_fallback = raw.get("support_shift_relaxed_fallback")
    if shift_fallback is not None:
        if not hasattr(model, "support_shift_relaxed_fallback"):
            raise Round31RuntimeError(
                "support_shift_relaxed_fallback requires model.support_shift_relaxed_fallback"
            )
        if not isinstance(shift_fallback, bool):
            raise Round31RuntimeError(
                "support_shift_relaxed_fallback must be a boolean"
            )
        model.support_shift_relaxed_fallback = shift_fallback


def _phase_dir(output: Path, index: int, phase: PhaseBudget) -> Path:
    return output / f"phase_{index:02d}_{_safe_name(phase.name)}"


def _checkpoint_payload(
    runtime: PhaseRuntime,
    raw: Mapping[str, Any],
    config_hash: str,
    phase_index: int,
    phase: PhaseBudget,
    epoch: int,
    official: Mapping[str, Any],
    active_seconds: float,
    status: str,
) -> dict[str, Any]:
    model_state = runtime.model.state_dict()
    model_state = {
        str(k): v.detach().cpu() if isinstance(v, torch.Tensor) else v
        for k, v in model_state.items()
    }
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": status,
        "phase_index": phase_index,
        "phase": phase.to_dict(),
        "epoch": int(epoch),
        "config_hash": config_hash,
        "config": dict(raw),
        "model": model_state,
        "optimizer": runtime.optimizer.state_dict()
        if runtime.optimizer is not None
        else None,
        "scheduler": runtime.scheduler.state_dict()
        if runtime.scheduler is not None
        else None,
        "rng": {**_capture_rng(), "loaders": _capture_loader_rng(runtime.bundle)},
        "step_telemetry": (
            runtime.step_telemetry.state_dict()
            if runtime.step_telemetry is not None
            else None
        ),
        "official": dict(official),
        "active_seconds": float(active_seconds),
        "experiment_contract": {
            "model_factory": runtime.model_factory_target,
            "data_factory": runtime.data_factory_target,
            "evaluator_factory": runtime.evaluator_factory_target,
            "registered_trainable_ids": list(runtime.registered_trainable_ids),
            "precision": raw.get("precision", raw.get("runtime", {}).get("precision")),
        },
    }
    return payload


def _write_state(output: Path, state: Mapping[str, Any]) -> None:
    _atomic_json(output / "round31_state.json", state)


def _write_status(output: Path, state: Mapping[str, Any]) -> None:
    _atomic_json(output / "run_status.json", state)


def _ensure_output(output: Path, *, resume: bool) -> None:
    if output.exists() and output.is_symlink():
        raise Round31RuntimeError(f"refusing symlink output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if resume:
        return
    allowed = {".gitkeep"}
    entries = [p for p in output.iterdir() if p.name not in allowed]
    if entries:
        raise Round31RuntimeError(
            f"output directory is non-empty; pass --resume to continue: {output}"
        )


def _load_checkpoint(
    runtime: PhaseRuntime,
    path: Path,
    raw: Mapping[str, Any],
    config_hash: str,
    phase_index: int,
    phase: PhaseBudget,
) -> tuple[int, float, list[dict[str, Any]]]:
    payload = _load_torch(path)
    if not isinstance(payload, Mapping) or payload.get("schema") != SCHEMA:
        raise Round31RuntimeError(f"invalid runtime checkpoint: {path}")
    if str(payload.get("config_hash")) != config_hash:
        raise Round31RuntimeError(
            "checkpoint config hash does not match current config"
        )
    if int(payload.get("phase_index", -1)) != phase_index:
        raise Round31RuntimeError("checkpoint belongs to another phase")
    saved_phase = payload.get("phase")
    if isinstance(saved_phase, Mapping) and str(saved_phase.get("name")) != phase.name:
        raise Round31RuntimeError("checkpoint phase name does not match current phase")
    model_state = payload.get("model")
    if not isinstance(model_state, Mapping):
        raise Round31RuntimeError("checkpoint has no model state")
    runtime.model.load_state_dict(model_state, strict=True)
    if runtime.optimizer is not None and payload.get("optimizer") is not None:
        runtime.optimizer.load_state_dict(payload["optimizer"])
    if runtime.scheduler is not None and payload.get("scheduler") is not None:
        runtime.scheduler.load_state_dict(payload["scheduler"])
    if payload.get("rng") is not None:
        _restore_runtime_rng(runtime, payload["rng"])
    if runtime.step_telemetry is not None:
        runtime.step_telemetry.load_state_dict(payload.get("step_telemetry"))
    phase_dir = path.parent.parent
    records = _read_jsonl(phase_dir / "history.jsonl")
    epoch = int(payload.get("epoch", phase.start_epoch))
    records = [r for r in records if int(r.get("epoch", -1)) <= epoch]
    official = payload.get("official")
    if isinstance(official, Mapping) and not any(
        int(r.get("epoch", -1)) == epoch for r in records
    ):
        try:
            for metric_name in SELECTION_METRICS:
                _metric(official, metric_name)
        except OfficialMetricError:
            pass
        else:
            repaired = {
                "schema": SCHEMA,
                "phase_index": phase_index,
                "phase": phase.name,
                "epoch": epoch,
                "active_seconds": float(payload.get("active_seconds", 0.0)),
                "official": dict(official),
                "checkpoint_reconciled": True,
            }
            _append_jsonl(phase_dir / "history.jsonl", repaired)
            records.append(repaired)
    return epoch, float(payload.get("active_seconds", 0.0)), records


def _audit_checkpoint(path: Path, *, epoch: int, kind: str) -> dict[str, Any]:
    return {
        "kind": kind,
        "epoch": int(epoch),
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _file_hash(path),
    }


def _copy_verified(source: Path, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(destination.name + f".tmp.{os.getpid()}")
    shutil.copy2(source, temp)
    os.replace(temp, destination)
    return {
        "path": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": _file_hash(destination),
    }


def _save_audit_snapshot(
    last_path: Path,
    phase_dir: Path,
    *,
    epoch: int,
    audit_epochs: Sequence[int],
    limit: int,
) -> Optional[dict[str, Any]]:
    if int(epoch) not in set(int(x) for x in audit_epochs):
        return None
    target_dir = phase_dir / "audit_snapshots"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"epoch_{int(epoch):04d}.pt"
    record = _copy_verified(last_path, target)
    snapshots = sorted(target_dir.glob("epoch_*.pt"), key=lambda p: p.name)
    while len(snapshots) > limit:
        snapshots[0].unlink()
        snapshots.pop(0)
    manifest = [
        {
            "epoch": int(re.search(r"epoch_(\d+)", p.stem).group(1)),
            "path": str(p),
            "bytes": p.stat().st_size,
            "sha256": _file_hash(p),
        }
        for p in snapshots
    ]
    _atomic_json(
        phase_dir / "audit_manifest.json", {"limit": limit, "snapshots": manifest}
    )
    return record


def _run_one_epoch(
    runtime: PhaseRuntime, loader: Any, epoch: int, *, training: bool
) -> tuple[dict[str, Any], dict[str, Any]]:
    runner = runtime.run_epoch
    if runner is None:
        common_runner = importlib.import_module("training.runner")
        runner = getattr(common_runner, "_run_epoch", None)
    if not callable(runner):
        raise Round31RuntimeError("phase runtime has no run_epoch implementation")
    optimizer = runtime.optimizer if training else None
    scheduler = runtime.scheduler if training else None
    scheduler_step_before = scheduler.last_epoch if scheduler is not None else None
    runner_args = (
        runtime.model,
        runtime.bundle,
        loader,
        runtime.config,
        runtime.device,
        int(epoch),
        optimizer,
        scheduler,
        None,
        set(runtime.registered_trainable_ids),
    )
    if training and runtime.step_telemetry is not None:
        result = run_epoch_with_step_telemetry(
            *runner_args,
            step_observer=runtime.step_telemetry,
        )
    else:
        result = runner(*runner_args)
    if isinstance(result, tuple) and len(result) == 2:
        metrics, phase = result
    else:
        metrics, phase = result, {}
    if not isinstance(metrics, Mapping):
        raise Round31RuntimeError(
            f"run_epoch returned non-mapping metrics: {type(metrics).__name__}"
        )
    phase_info = dict(phase) if isinstance(phase, Mapping) else {"value": str(phase)}
    if training:
        phase_info["schedule_probe"] = probe_actual_schedule(
            runtime, scheduler_step_before, len(loader)
        )
    return dict(metrics), phase_info


def _evaluate_official(
    runtime: PhaseRuntime,
    *,
    checkpoint: Path,
    output_dir: Path,
    epoch: int,
) -> dict[str, float]:
    if runtime.evaluator is None:
        raise Round31RuntimeError("official evaluator is required")
    if runtime.request_factory is None:
        request = SimpleNamespace(
            checkpoint=checkpoint, output_dir=output_dir, epoch=int(epoch)
        )
    else:
        factory = runtime.request_factory
        try:
            request = factory(
                checkpoint=checkpoint, output_dir=output_dir, epoch=int(epoch)
            )
        except TypeError:
            request = factory(checkpoint, output_dir, int(epoch))
    evaluator = runtime.evaluator
    method = getattr(evaluator, "evaluate", None)
    value = method(request) if callable(method) else evaluator(request)
    metrics = _normalize_official(value)
    for name in SELECTION_METRICS:
        _metric(metrics, name)
    # round36 H7: equal-weight the three duration buckets instead of the
    # volume-weighted default "MR-mAP" (which is dominated by short/middle
    # query counts and can mask a real Long-bucket decline). Purely
    # additive -- a NEW key, nothing here reads or overwrites it unless a
    # config explicitly sets early_stop.target_metric to this name.
    bucket_names = ("MR-mAP-Long_Avg", "MR-mAP-Middle_Avg", "MR-mAP-Short_Avg")
    if all(name in metrics and math.isfinite(metrics[name]) for name in bucket_names):
        metrics["MR-mAP-LengthWeighted"] = sum(
            metrics[name] for name in bucket_names
        ) / len(bucket_names)
    return metrics


def _state_summary(
    *,
    status: str,
    phase_index: int,
    phase: PhaseBudget,
    completed_epoch: int,
    active_seconds: float,
    reason: str = "",
    best_epoch: Optional[int] = None,
    best_metrics: Optional[Mapping[str, Any]] = None,
    next_phase_index: Optional[int] = None,
    error: Optional[BaseException] = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema": SCHEMA,
        "status": status,
        "phase_index": int(phase_index),
        "phase": phase.to_dict(),
        "completed_epoch": int(completed_epoch),
        "active_seconds": float(active_seconds),
        "reason": reason,
        "best_epoch": best_epoch,
        "best_metrics": dict(best_metrics or {}),
        "next_phase_index": next_phase_index,
        "updated_at_unix": time.time(),
    }
    if error is not None:
        value["error_type"] = type(error).__name__
        value["error"] = str(error)
    return value


def _find_best(
    records: Sequence[Mapping[str, Any]],
) -> tuple[Optional[int], dict[str, float]]:
    best_key: Optional[tuple[float, float, float, int]] = None
    best_epoch: Optional[int] = None
    best_metrics: dict[str, float] = {}
    for record in records:
        metrics = record.get("official")
        if not isinstance(metrics, Mapping):
            continue
        epoch = int(record.get("epoch", -1))
        try:
            key = _selection_key(metrics, epoch)
        except OfficialMetricError:
            continue
        if best_key is None or key > best_key:
            best_key = key
            best_epoch = epoch
            best_metrics = {
                str(k): float(v) for k, v in metrics.items() if _is_finite_number(v)
            }
    return best_epoch, best_metrics


def _write_root_identity(
    output: Path, raw: Mapping[str, Any], config_hash: str
) -> None:
    identity_path = output / "run_identity.json"
    if identity_path.exists():
        prior = json.loads(identity_path.read_text(encoding="utf-8"))
        if str(prior.get("config_hash")) != config_hash:
            raise Round31RuntimeError("existing output belongs to another config")
        return
    _atomic_json(output / "config.json", raw)
    _atomic_json(
        identity_path,
        {
            "schema": SCHEMA,
            "pid": os.getpid(),
            "config_hash": config_hash,
            "config": dict(raw),
            "python": sys.version,
            "started_at_unix": time.time(),
            "hostname": os.environ.get("COMPUTERNAME", os.environ.get("HOSTNAME", "")),
        },
    )


def run_phase(
    runtime: PhaseRuntime,
    raw: Mapping[str, Any],
    output_dir: str | Path,
    *,
    phase_index: int = 0,
    resume: bool = False,
    config_hash: Optional[str] = None,
    audit: Optional[Mapping[str, Any]] = None,
) -> PhaseResult:
    """Run one bounded phase and return at an epoch boundary.

    A successful phase writes a verified last.pt and, when applicable,
    best_val.pt.  It never launches another phase.  Resuming loads the
    phase checkpoint plus optimizer, scheduler, and RNG state.
    """
    output = Path(output_dir).resolve()
    _ensure_output(output, resume=resume)
    phases = phase_budgets(raw)
    if phase_index < 0 or phase_index >= len(phases):
        raise Round31RuntimeError(
            f"phase_index {phase_index} outside {len(phases)} phases"
        )
    phase = phases[phase_index]
    cfg_hash = config_hash or _canonical_hash(raw)
    _write_root_identity(output, raw, cfg_hash)
    if audit is not None:
        _atomic_json(output / "optimizer_audit.json", audit)

    phase_dir = _phase_dir(output, phase_index, phase)
    phase_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = phase_dir / "checkpoints"
    official_dir = phase_dir / "official_by_epoch"
    checkpoints.mkdir(exist_ok=True)
    official_dir.mkdir(exist_ok=True)
    _atomic_json(
        phase_dir / "phase_config.json",
        {
            "schema": SCHEMA,
            "config_hash": cfg_hash,
            "phase_index": phase_index,
            "phase": phase.to_dict(),
        },
    )

    last_path = checkpoints / "last.pt"
    best_path = checkpoints / "best_val.pt"
    pending_path = checkpoints / "pending.pt"
    completed = phase.start_epoch
    active_before = 0.0
    records: list[dict[str, Any]] = []
    if resume and last_path.exists():
        completed, active_before, records = _load_checkpoint(
            runtime, last_path, raw, cfg_hash, phase_index, phase
        )
    elif resume:
        records = _read_jsonl(phase_dir / "history.jsonl")
        if records:
            raise Round31RuntimeError(
                "phase history exists without a durable last.pt checkpoint"
            )
    ledger_path = phase_dir / "budget_ledger.json"
    ledger_identity = {
        "config_hash": cfg_hash,
        "phase_index": phase_index,
        "phase": phase.to_dict(),
    }
    if resume:
        active_before = recovered_seconds(
            ledger_path, ledger_identity, active_before, phase.wallclock_seconds
        )
    elif ledger_path.exists():
        raise Round31RuntimeError(
            "phase budget ledger already exists; inspect before resuming"
        )
    if completed < phase.start_epoch or completed > phase.end_epoch:
        raise Round31RuntimeError(
            f"checkpoint epoch {completed} is outside phase {phase.start_epoch}..{phase.end_epoch}"
        )
    best_epoch, best_metrics = _find_best(records)
    if best_epoch is not None:
        if not best_path.exists():
            if best_epoch == completed and last_path.exists():
                _copy_verified(last_path, best_path)
            else:
                raise Round31RuntimeError(
                    f"best checkpoint is missing for historical best epoch {best_epoch}"
                )
        else:
            best_payload = _load_torch(best_path)
            best_file_epoch = (
                int(best_payload.get("epoch", -1))
                if isinstance(best_payload, Mapping)
                else -1
            )
            best_file_metrics = (
                best_payload.get("official", {})
                if isinstance(best_payload, Mapping)
                else {}
            )
            if best_file_epoch != best_epoch:
                if best_epoch == completed and last_path.exists():
                    _copy_verified(last_path, best_path)
                else:
                    raise Round31RuntimeError(
                        f"best checkpoint epoch {best_file_epoch} disagrees with history {best_epoch}"
                    )
            else:
                try:
                    if _selection_key(
                        best_file_metrics, best_file_epoch
                    ) != _selection_key(best_metrics, best_epoch):
                        raise OfficialMetricError(
                            "best checkpoint metrics disagree with history"
                        )
                except OfficialMetricError:
                    if best_epoch == completed and last_path.exists():
                        _copy_verified(last_path, best_path)
                    else:
                        raise Round31RuntimeError(
                            "best checkpoint metrics disagree with historical best"
                        )

    started = time.monotonic()
    status = _state_summary(
        status="running",
        phase_index=phase_index,
        phase=phase,
        completed_epoch=completed,
        active_seconds=active_before,
        best_epoch=best_epoch,
        best_metrics=best_metrics,
        next_phase_index=phase_index + 1 if phase_index + 1 < len(phases) else None,
    )
    _write_state(output, status)
    _write_status(output, status)

    def elapsed() -> float:
        return active_before + max(0.0, time.monotonic() - started)

    def persist_budget(in_flight=True):
        _atomic_json(ledger_path, ledger_payload(ledger_identity, elapsed(), in_flight))

    reason = ""
    final_status = "complete"
    persist_budget()
    _apply_negative_selection_mode(runtime.model, raw)
    _apply_length_bias_config(runtime.model, raw)
    try:
        if not resume and completed == phase.start_epoch and phase.start_epoch == 0:
            _run_probe(runtime, phase_dir, 0, phase="pre_update")
            persist_budget()
        while completed < phase.end_epoch:
            if phase.wallclock_seconds > 0 and elapsed() >= phase.wallclock_seconds:
                final_status = "budget_exhausted"
                reason = "wallclock_boundary"
                break
            epoch = completed + 1
            _apply_loss_weight_schedule(runtime.model, raw, epoch)
            train_start_lrs = {
                str(group.get("name", i)): float(group["lr"])
                for i, group in enumerate(
                    getattr(runtime.optimizer, "param_groups", [])
                )
            }
            train_metrics, train_phase = _run_one_epoch(
                runtime, getattr(runtime.bundle, "train_loader"), epoch, training=True
            )
            val_metrics, val_phase = _run_one_epoch(
                runtime, getattr(runtime.bundle, "val_loader"), epoch, training=False
            )
            pending_payload = _checkpoint_payload(
                runtime,
                raw,
                cfg_hash,
                phase_index,
                phase,
                epoch,
                {},
                elapsed(),
                "pending_official",
            )
            _atomic_torch_save(pending_path, pending_payload)
            official_epoch_dir = official_dir / f"epoch_{epoch:04d}"
            official_epoch_dir.mkdir(parents=True, exist_ok=True)
            try:
                official = _evaluate_official(
                    runtime,
                    checkpoint=pending_path,
                    output_dir=official_epoch_dir,
                    epoch=epoch,
                )
            finally:
                _restore_rng(pending_payload["rng"])
            persist_budget()
            pending_payload["active_seconds"] = elapsed()
            pending_payload["official"] = official
            pending_payload["status"] = "verified_official"
            _atomic_torch_save(pending_path, pending_payload)
            copied = _copy_verified(pending_path, last_path)
            pending_path.unlink(missing_ok=True)

            record: dict[str, Any] = {
                "schema": SCHEMA,
                "phase_index": phase_index,
                "phase": phase.name,
                "epoch": epoch,
                "active_seconds": elapsed(),
                "train": _json_safe(train_metrics),
                "train_phase": _json_safe(train_phase),
                "val": _json_safe(val_metrics),
                "val_phase": _json_safe(val_phase),
                "official": dict(official),
                "learning_rates_start": train_start_lrs,
                "learning_rates_end": {
                    str(group.get("name", i)): float(group["lr"])
                    for i, group in enumerate(
                        getattr(runtime.optimizer, "param_groups", [])
                    )
                },
                "checkpoint": copied,
            }
            _append_jsonl(phase_dir / "history.jsonl", record)
            _run_probe(runtime, phase_dir, epoch, phase="post_validation")
            persist_budget()
            _append_jsonl(
                phase_dir / "official_results.jsonl",
                {
                    "epoch": epoch,
                    "phase": phase.name,
                    "metrics": official,
                    "checkpoint": copied,
                },
            )
            current_key = _selection_key(official, epoch)
            previous_key = (
                _selection_key(best_metrics, best_epoch)
                if best_epoch is not None and best_metrics
                else None
            )
            best_copied = None
            if previous_key is None or current_key > previous_key:
                best_epoch = epoch
                best_metrics = dict(official)
                best_copied = _copy_verified(last_path, best_path)
            _append_jsonl(
                phase_dir / "checkpoint_audit.jsonl",
                {
                    "epoch": epoch,
                    "last": copied,
                    "best": best_copied,
                },
            )
            audit = _save_audit_snapshot(
                last_path,
                phase_dir,
                epoch=epoch,
                audit_epochs=phase.audit_epochs,
                limit=phase.audit_snapshot_limit,
            )
            if audit is not None:
                _append_jsonl(
                    phase_dir / "checkpoint_audit.jsonl",
                    {
                        "epoch": epoch,
                        "audit_snapshot": audit,
                    },
                )
            completed = epoch
            records.append(record)
            stop_reason = _early_stop_reason(
                records,
                phase.early_stop,
                phase_start_epoch=phase.start_epoch,
                epoch=epoch,
            )
            if stop_reason:
                final_status = "early_stop"
                reason = stop_reason
            elif completed >= phase.end_epoch:
                final_status = "complete"
                reason = "epoch_budget"
            elif phase.wallclock_seconds > 0 and elapsed() >= phase.wallclock_seconds:
                final_status = "budget_exhausted"
                reason = "wallclock_boundary"
            state = _state_summary(
                status=("running" if not reason else final_status),
                phase_index=phase_index,
                phase=phase,
                completed_epoch=completed,
                active_seconds=elapsed(),
                reason=reason,
                best_epoch=best_epoch,
                best_metrics=best_metrics,
                next_phase_index=(
                    phase_index + 1
                    if phase_index + 1 < len(phases) and not reason
                    else None
                ),
            )
            _write_state(output, state)
            _write_status(output, state)
            if reason:
                break
        if completed >= phase.end_epoch and not reason:
            final_status, reason = "complete", "epoch_budget"
        terminal_probe = None
        if completed > 0 and final_status in {
            "complete",
            "early_stop",
            "budget_exhausted",
        }:
            terminal_probe = _run_terminal_probe(runtime, phase_dir, completed)
        result = PhaseResult(
            status=final_status,
            phase=phase.name,
            completed_epoch=completed,
            next_phase_index=(
                phase_index + 1
                if phase_index + 1 < len(phases) and final_status == "complete"
                else None
            ),
            reason=reason or "epoch_budget",
            best_epoch=best_epoch,
            best_metrics=best_metrics,
            active_seconds=elapsed(),
            output_dir=str(phase_dir),
        )
        final_state = _state_summary(
            status=result.status,
            phase_index=phase_index,
            phase=phase,
            completed_epoch=completed,
            active_seconds=result.active_seconds,
            reason=result.reason,
            best_epoch=best_epoch,
            best_metrics=best_metrics,
            next_phase_index=result.next_phase_index,
        )
        if terminal_probe is not None:
            final_state["terminal_probe"] = terminal_probe
        _write_state(output, final_state)
        _write_status(output, final_state)
        _atomic_json(phase_dir / "phase_result.json", result.to_dict())
        return result
    except BaseException as exc:
        failed_status = "oom" if _is_oom(exc) else "failed"
        state = _state_summary(
            status=failed_status,
            phase_index=phase_index,
            phase=phase,
            completed_epoch=completed,
            active_seconds=elapsed(),
            reason="exception",
            best_epoch=best_epoch,
            best_metrics=best_metrics,
            error=exc,
        )
        state["pending_checkpoint"] = (
            str(pending_path) if pending_path.exists() else None
        )
        _write_state(output, state)
        _write_status(output, state)
        _atomic_json(phase_dir / "phase_result.json", state)
        raise
    finally:
        try:
            persist_budget(in_flight=False)
        finally:
            _release_cuda()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Round31RuntimeError(f"config must be a JSON object: {path}")
    return value


def _spec_target(spec: Any) -> str:
    if isinstance(spec, str):
        return spec
    if isinstance(spec, Mapping):
        return str(spec.get("target", ""))
    return str(getattr(spec, "target", ""))


def _install_roots(
    raw: Mapping[str, Any], remote_root: str | Path | None
) -> tuple[Path | None, tuple[Path, ...]]:
    try:
        child = importlib.import_module("field_core.child")
    except (ImportError, ModuleNotFoundError):
        child = None
    if child is not None and callable(getattr(child, "install_import_roots", None)):
        root, roots = child.install_import_roots(
            raw, remote_root=remote_root, strict=remote_root is not None
        )
    else:
        root = Path(str(remote_root)).resolve() if remote_root else None
        roots = tuple([root] if root else ())
    values = [str(p) for p in roots if p is not None]
    prior = os.environ.get("PYTHONPATH")
    if prior:
        values.append(prior)
    if values:
        os.environ["PYTHONPATH"] = os.pathsep.join(values)
        for value in reversed(values):
            if value and value not in sys.path:
                sys.path.insert(0, value)
    return root, tuple(Path(p) for p in values if p)


def _load_runner_config(config_path: Path) -> Any:
    module = importlib.import_module("training.config")
    cls = getattr(module, "RunnerConfig", None)
    if cls is None or not callable(getattr(cls, "load", None)):
        raise Round31RuntimeError("training.config.RunnerConfig.load is unavailable")
    return cls.load(config_path)


def _project_root(raw: Mapping[str, Any]) -> Path:
    spec = raw.get("evaluator_factory")
    kwargs = spec.get("kwargs", {}) if isinstance(spec, Mapping) else {}
    value = kwargs.get("project_root") if isinstance(kwargs, Mapping) else None
    value = raw.get("project_root") if value in (None, "") else value
    return Path(str(value)).resolve() if value not in (None, "") else Path.cwd()


def _official_request(
    config: Any,
    *,
    checkpoint: Path,
    output_dir: Path,
    config_path: Path,
    project_root: Path,
    epoch: int,
) -> Any:
    contracts = importlib.import_module("training.contracts")
    cls = getattr(contracts, "OfficialEvalRequest")
    policy = getattr(config, "checkpointing", None)
    device = (
        getattr(policy, "official_device", None) if policy else None
    ) or config.device
    batch = (
        getattr(policy, "official_batch_size", None) if policy else None
    ) or config.batch_size
    return cls(
        project_root=project_root,
        checkpoint=checkpoint,
        output_dir=output_dir,
        config_path=config_path,
        device=str(device),
        batch_size=int(batch),
        num_workers=int(config.num_workers),
        precision=str(config.precision),
    )


def _optimizer_lr_scale(raw: Mapping[str, Any], model: Any) -> float:
    value = raw.get("optimizer_lr_scale")
    if value is None and isinstance(raw.get("model_factory"), Mapping):
        kwargs = raw["model_factory"].get("kwargs", {})
        if isinstance(kwargs, Mapping):
            value = kwargs.get("optimizer_lr_scale")
    if value is None:
        options = getattr(model, "options", None)
        value = getattr(options, "optimizer_lr_scale", 1.0)
    scale = float(1.0 if value is None else value)
    if not math.isfinite(scale) or scale <= 0:
        raise Round31RuntimeError(
            f"optimizer_lr_scale must be finite and positive, got {scale}"
        )
    return scale


def _scale_optimizer_groups(optimizer: Any, scale: float) -> dict[str, Any]:
    groups = getattr(optimizer, "param_groups", None)
    if groups is None:
        raise Round31RuntimeError("optimizer has no param_groups")
    before: list[float] = []
    after: list[float] = []
    for index, group in enumerate(groups):
        if "lr" not in group:
            raise Round31RuntimeError(f"optimizer group {index} has no learning rate")
        old = float(group["lr"])
        new = old * scale
        if not math.isfinite(new) or new <= 0:
            raise Round31RuntimeError(
                f"invalid scaled learning rate in group {index}: {new}"
            )
        group["lr"] = new
        before.append(old)
        after.append(new)
    return {
        "scale": scale,
        "group_count": len(groups),
        "before": before,
        "after": after,
        "all_groups_scaled": True,
    }


def _create_scheduler_for_config(
    rt, optimizer, raw, *, schedule_epochs, warmup_epochs, updates
):
    """round32 extension: opt-in sg_detr-style milestone LR schedule (G0,
    the sg_detr-training-regime mimic group). Default (no lr_schedule_mode
    key, or "cosine") delegates straight to the unmodified
    field_core.runtime.create_scheduler -- zero behavior change for
    every existing config. "milestone" reproduces sg_detr's
    WarmupMultiStepLR shape (code/configs/optimizer/finetune.yaml):
    linear warmup for warmup_epochs, then a step decay by gamma at each
    epoch in milestones.
    """
    mode = str(raw.get("lr_schedule_mode", "cosine")).lower()
    if mode == "cosine":
        return rt.create_scheduler(
            optimizer,
            total_steps=schedule_epochs * updates,
            warmup_steps=warmup_epochs * updates,
        )
    if mode != "milestone":
        raise Round31RuntimeError(
            f"unknown lr_schedule_mode {mode!r}; expected cosine or milestone"
        )
    milestones = raw.get("lr_schedule_milestones")
    if not isinstance(milestones, list) or not milestones:
        raise Round31RuntimeError(
            "lr_schedule_mode=milestone requires a non-empty lr_schedule_milestones list of epoch numbers"
        )
    gamma = float(raw.get("lr_schedule_gamma", 0.5))
    if not (0.0 < gamma < 1.0):
        raise Round31RuntimeError("lr_schedule_gamma must be in (0, 1)")
    milestone_steps = sorted(int(m) * updates for m in milestones)
    warmup_steps = warmup_epochs * updates

    def factor(step: int) -> float:
        completed = step + 1
        if warmup_steps and completed <= warmup_steps:
            return completed / float(warmup_steps)
        decayed = sum(1 for m in milestone_steps if completed > m)
        return gamma**decayed

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _build_phase_runtime(
    config_path: Path,
    raw: Mapping[str, Any],
    *,
    remote_root: str | Path | None = None,
    output_dir: Path | None = None,
) -> tuple[PhaseRuntime, Any, tuple[Path | None, tuple[Path, ...]], dict[str, Any]]:
    root, roots = _install_roots(raw, remote_root)
    rt = importlib.import_module("field_core.runtime")
    config = _load_runner_config(config_path)
    _seed_everything(int(getattr(config, "seed", raw.get("seed", 0))))
    requested_seed = int(getattr(config, "seed", raw.get("seed", 0)))
    factory_cuda_states = None
    if str(config.device).startswith("cuda") and torch.cuda.is_available():
        # Factories construct CPU modules and must not change the training CUDA stream.
        factory_cuda_states = [x.clone() for x in torch.cuda.get_rng_state_all()]
        if int(torch.cuda.initial_seed()) != requested_seed:
            raise Round31RuntimeError(
                "seed_everything did not initialize the requested CUDA seed"
            )

    factory = getattr(rt, "instantiate_factory")
    bundle = factory(config.data_factory, config)
    model = factory(config.model_factory, config)
    if not isinstance(model, torch.nn.Module):
        raise Round31RuntimeError("model_factory must return torch.nn.Module")
    device = torch.device(str(config.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise Round31RuntimeError(f"CUDA unavailable: {device}")
    model.to(device)
    optimizer, optimizer_audit = rt.build_optimizer(
        model,
        lr=float(raw.get("learning_rate", 1.0e-4)),
        weight_decay=float(config.weight_decay),
    )
    scale = _optimizer_lr_scale(raw, model)
    scale_audit = _scale_optimizer_groups(optimizer, scale)
    step_telemetry = StepTelemetryRecorder.from_config(
        optimizer, model, raw.get("step_telemetry")
    )
    updates = len(getattr(bundle, "train_loader"))
    phases = phase_budgets(raw)
    max_schedule = max(
        int(getattr(config, "schedule_epochs", 0) or 0),
        max((int(p.schedule_epochs or 0) for p in phases), default=0),
        int(getattr(config, "epochs", 0) or 0),
    )
    schedule_epochs = int(raw.get("schedule_epochs", max_schedule))
    if schedule_epochs < 1:
        schedule_epochs = max_schedule
    warmup_epochs = int(
        raw.get("warmup_epochs", getattr(config, "warmup_epochs", 0) or 0)
    )
    if warmup_epochs < 0 or warmup_epochs > schedule_epochs:
        raise Round31RuntimeError(
            f"warmup_epochs={warmup_epochs} exceeds schedule_epochs={schedule_epochs}"
        )
    scheduler = _create_scheduler_for_config(
        rt,
        optimizer,
        raw,
        schedule_epochs=schedule_epochs,
        warmup_epochs=warmup_epochs,
        updates=max(1, updates),
    )
    evaluator = rt.evaluator_from_config(config)
    model_target = _spec_target(
        raw.get("model_factory", getattr(config, "model_factory", ""))
    )
    data_target = _spec_target(
        raw.get("data_factory", getattr(config, "data_factory", ""))
    )
    evaluator_target = _spec_target(
        raw.get("evaluator_factory", getattr(config, "evaluator_factory", ""))
    )
    cfg_for_request = config_path
    project_root = _project_root(raw)

    def request_factory(*, checkpoint: Path, output_dir: Path, epoch: int) -> Any:
        return _official_request(
            config,
            checkpoint=checkpoint,
            output_dir=output_dir,
            config_path=cfg_for_request,
            project_root=project_root,
            epoch=epoch,
        )

    runtime = PhaseRuntime(
        model=model,
        bundle=bundle,
        config=config,
        optimizer=optimizer,
        scheduler=scheduler,
        evaluator=evaluator,
        device=device,
        registered_trainable_ids=tuple(
            id(parameter) for parameter in model.parameters() if parameter.requires_grad
        ),
        run_epoch=getattr(
            importlib.import_module("training.runner"), "_run_epoch", None
        ),
        request_factory=request_factory,
        probe_suite=(
            factory(raw["probe_factory"], config)
            if raw.get("probe_factory") is not None
            else None
        ),
        # Round31ProbeSuite owns the fixed 4x64 panel and FP32 isolation.  Use
        # its direct .run(request) contract instead of the round1 child shell.
        probe_runner=None,
        probe_epochs=tuple(int(x) for x in raw.get("probe_epochs", (0, 1, 8, 18, 50))),
        fixed_panel=(
            getattr(importlib.import_module("field_core.child"), "_fixed_panel")(
                raw, config_path
            )
            if raw.get("fixed_panel_path") not in (None, "")
            else None
        ),
        step_telemetry=step_telemetry,
        model_factory_target=model_target,
        data_factory_target=data_target,
        evaluator_factory_target=evaluator_target,
    )
    runtime.schedule_probe_config = {
        "schedule_epochs": schedule_epochs,
        "warmup_epochs": warmup_epochs,
        "updates_per_epoch": updates,
        "mode": raw.get("lr_schedule_mode", "cosine"),
        "milestones": raw.get("lr_schedule_milestones"),
        "gamma": raw.get("lr_schedule_gamma", 0.5),
    }
    audit = {
        "schema": SCHEMA,
        "model_factory": model_target,
        "data_factory": data_target,
        "evaluator_factory": evaluator_target,
        "optimizer": optimizer_audit,
        "lr_scale": scale_audit,
        "schedule_epochs": schedule_epochs,
        "warmup_epochs": warmup_epochs,
        "updates_per_epoch": updates,
        "device": str(device),
        "precision": str(config.precision),
        "step_telemetry": step_telemetry.state_dict(),
    }
    if factory_cuda_states is not None:
        before_restore = torch.cuda.get_rng_state_all()
        changed = any(
            not torch.equal(a, b) for a, b in zip(factory_cuda_states, before_restore)
        )
        torch.cuda.set_rng_state_all(factory_cuda_states)
        restored = torch.cuda.get_rng_state_all()
        if len(restored) != len(factory_cuda_states) or any(
            not torch.equal(a, b) for a, b in zip(restored, factory_cuda_states)
        ):
            raise Round31RuntimeError("factory CUDA RNG restoration failed")
        audit["cuda_rng_factory_guard"] = {
            "requested_seed": requested_seed,
            "final_initial_seed": int(torch.cuda.initial_seed()),
            "device_count": len(restored),
            "factory_changed_state": changed,
            "restored_exact": True,
            "state_sha256": [
                hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest() for x in restored
            ],
        }
    else:
        audit["cuda_rng_factory_guard"] = {"scope": "CPU runtime, no CUDA RNG state"}
    return runtime, config, (root, roots), audit


def _restore_previous_phase(
    runtime: PhaseRuntime,
    raw: Mapping[str, Any],
    output: Path,
    *,
    phase_index: int,
    cfg_hash: str,
) -> Optional[Path]:
    if phase_index <= 0:
        return None
    phases = phase_budgets(raw)
    prior_dir = _phase_dir(output, phase_index - 1, phases[phase_index - 1])
    prior = prior_dir / "checkpoints" / "last.pt"
    if not prior.exists():
        raise Round31RuntimeError(
            f"phase {phase_index} requires previous checkpoint: {prior}"
        )
    payload = _load_torch(prior)
    if not isinstance(payload, Mapping) or payload.get("schema") != SCHEMA:
        raise Round31RuntimeError(f"invalid previous phase checkpoint: {prior}")
    if str(payload.get("config_hash")) != cfg_hash:
        raise Round31RuntimeError("previous phase checkpoint config hash mismatch")
    prior_phase = phases[phase_index - 1]
    prior_epoch = int(payload.get("epoch", -1))
    prior_result_path = prior_dir / "phase_result.json"
    prior_result = (
        json.loads(prior_result_path.read_text(encoding="utf-8"))
        if prior_result_path.exists()
        else {}
    )
    if (
        prior_epoch != prior_phase.end_epoch
        or int(prior_result.get("completed_epoch", -1)) != prior_phase.end_epoch
    ):
        raise Round31RuntimeError(
            f"cannot enter phase {phase_index}: prior phase ended at {prior_epoch}, "
            f"expected {prior_phase.end_epoch}"
        )
    if prior_result.get("status") not in {"complete", "early_stop"}:
        raise Round31RuntimeError(
            f"cannot enter phase {phase_index}: prior status {prior_result.get('status')!r}"
        )
    runtime.model.load_state_dict(payload["model"], strict=True)
    if runtime.optimizer is not None and payload.get("optimizer") is not None:
        runtime.optimizer.load_state_dict(payload["optimizer"])
    if runtime.scheduler is not None and payload.get("scheduler") is not None:
        runtime.scheduler.load_state_dict(payload["scheduler"])
    if payload.get("rng") is not None:
        _restore_runtime_rng(runtime, payload["rng"])
    if runtime.step_telemetry is not None:
        runtime.step_telemetry.load_state_dict(payload.get("step_telemetry"))
    return prior


def _phase_state_path(output: Path) -> Path:
    return output / "round31_state.json"


def _phase_index_from_state(output: Path, requested: Optional[int]) -> int:
    if requested is not None:
        return int(requested)
    path = _phase_state_path(output)
    if not path.exists():
        return 0
    value = json.loads(path.read_text(encoding="utf-8"))
    return int(value.get("phase_index", 0))


def run_config(
    config_path: str | Path,
    *,
    remote_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    resume: bool = False,
    phase_index: Optional[int] = None,
) -> PhaseResult:
    """Build a fresh process-local runtime and execute exactly one phase."""
    path = Path(config_path).resolve()
    raw = _load_json(path)
    phases = phase_budgets(raw)
    output_value = output_dir if output_dir is not None else raw.get("output_dir")
    if output_value in (None, ""):
        raise Round31RuntimeError("output_dir is required in config or --output-dir")
    output = Path(str(output_value)).resolve()
    selected = _phase_index_from_state(output, phase_index)
    if selected < 0 or selected >= len(phases):
        raise Round31RuntimeError(
            f"phase_index {selected} outside {len(phases)} phases"
        )
    cfg_hash = _canonical_hash(raw)
    phase_dir = _phase_dir(output, selected, phases[selected])
    entering_phase = (
        selected > 0 and not (phase_dir / "checkpoints" / "last.pt").exists()
    )
    if resume or selected > 0:
        _ensure_output(output, resume=True)
    runtime: Optional[PhaseRuntime] = None
    try:
        runtime, config, _, audit = _build_phase_runtime(
            path, raw, remote_root=remote_root, output_dir=output
        )
        audit["phase_index"] = selected
        if entering_phase:
            _restore_previous_phase(
                runtime, raw, output, phase_index=selected, cfg_hash=cfg_hash
            )
        result = run_phase(
            runtime,
            raw,
            output,
            phase_index=selected,
            resume=(resume or selected > 0),
            config_hash=cfg_hash,
            audit=audit,
        )
        return result
    except BaseException as exc:
        # Build failures happen before run_phase has a chance to create the
        # phase state.  Record them in the same top-level status schema.
        if runtime is None:
            _ensure_output(output, resume=True)
            _write_status(
                output,
                {
                    "schema": SCHEMA,
                    "status": "oom" if _is_oom(exc) else "failed",
                    "phase_index": selected,
                    "config_hash": cfg_hash,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "updated_at_unix": time.time(),
                },
            )
        raise
    finally:
        runtime = None
        _release_cuda()


def plan_queue(
    config_paths: Sequence[str | Path],
    output_dir: str | Path,
    *,
    max_concurrent: int = 1,
) -> dict[str, Any]:
    """Materialize an inert queue manifest; this function never starts jobs."""
    if max_concurrent < 1:
        raise Round31RuntimeError("max_concurrent must be positive")
    paths = [Path(value).resolve() for value in config_paths]
    if not paths:
        raise Round31RuntimeError("at least one config is required")
    jobs: list[dict[str, Any]] = []
    for path in paths:
        raw = _load_json(path)
        phases = phase_budgets(raw)
        jobs.append(
            {
                "job_id": f"{path.stem}:{_canonical_hash(raw)[:12]}",
                "config": str(path),
                "config_hash": _canonical_hash(raw),
                "seed": raw.get("seed"),
                "variant": (
                    raw.get("variant")
                    or (
                        raw.get("model_factory", {}).get("kwargs", {}).get("variant")
                        if isinstance(raw.get("model_factory"), Mapping)
                        else None
                    )
                ),
                "output_dir": raw.get("output_dir"),
                "phases": [phase.to_dict() for phase in phases],
                "status": "queued",
            }
        )
    output = Path(output_dir).resolve()
    _ensure_output(output, resume=False)
    manifest = {
        "schema": SCHEMA,
        "status": "planned",
        "created_at_unix": time.time(),
        "max_concurrent": int(max_concurrent),
        "auto_promote": False,
        "gpu_policy": {"requested": "parent-controlled", "max_workers": 4},
        "jobs": jobs,
    }
    _atomic_json(output / "queue_manifest.json", manifest)
    return manifest


def _cli_error(exc: BaseException) -> int:
    print(
        json.dumps(
            {
                "schema": SCHEMA,
                "status": "oom" if _is_oom(exc) else "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
            sort_keys=True,
        ),
        file=sys.stderr,
    )
    return 2


def cli_main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m eventfieldnet.runtime",
        description="bounded, resumable trifield round31 phase runner",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run exactly one configured phase")
    run.add_argument("--config", required=True, type=Path)
    run.add_argument("--remote-root", type=Path, default=None)
    run.add_argument("--output-dir", type=Path, default=None)
    run.add_argument("--phase-index", type=int, default=None)
    run.add_argument("--resume", action="store_true")
    plan = sub.add_parser("plan", help="write an inert queue manifest")
    plan.add_argument("--configs", required=True, nargs="+", type=Path)
    plan.add_argument("--output", required=True, type=Path)
    plan.add_argument("--max-concurrent", type=int, default=1)
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            value = plan_queue(
                args.configs, args.output, max_concurrent=args.max_concurrent
            )
            print(json.dumps(value, indent=2, sort_keys=True))
            return 0
        result = run_config(
            args.config,
            remote_root=args.remote_root,
            output_dir=args.output_dir,
            resume=args.resume,
            phase_index=args.phase_index,
        )
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        return 0
    except BaseException as exc:
        return _cli_error(exc)


main = cli_main


__all__ = [
    "SCHEMA",
    "SELECTION_METRICS",
    "EarlyStopPolicy",
    "PhaseBudget",
    "PhaseRuntime",
    "PhaseResult",
    "Round31RuntimeError",
    "StepTelemetryRecorder",
    "OfficialMetricError",
    "phase_budgets",
    "run_phase",
    "run_config",
    "plan_queue",
    "cli_main",
    "main",
]


def _run_probe(
    runtime: PhaseRuntime,
    phase_dir: Path,
    epoch: int,
    *,
    phase: str,
    force: bool = False,
) -> Any:
    """Run the repository probe suite in an explicitly disabled autocast context."""
    if runtime.probe_suite is None or (
        not force and int(epoch) not in {int(x) for x in runtime.probe_epochs}
    ):
        return None
    probe_dir = phase_dir / "probes"
    receipt = probe_dir / f"probe_e{int(epoch):03d}.json"
    if receipt.exists():
        return {"status": "already_present", "path": str(receipt)}
    callback = runtime.probe_runner
    if callback is None:
        callback = getattr(runtime.probe_suite, "run", None)
    if not callable(callback):
        callback = runtime.probe_suite if callable(runtime.probe_suite) else None
    if callback is None:
        raise Round31RuntimeError("probe_suite has no run callback")
    fp32 = nullcontext()
    if getattr(runtime.device, "type", "") in {"cuda", "cpu"}:
        fp32 = torch.autocast(device_type=runtime.device.type, enabled=False)
    with fp32:
        if runtime.probe_runner is not None:
            result = runtime.probe_runner(
                runtime.probe_suite,
                model=runtime.model,
                bundle=runtime.bundle,
                config=runtime.config,
                device=runtime.device,
                epoch=int(epoch),
                output_dir=probe_dir,
                fixed_panel=runtime.fixed_panel,
                phase=str(phase),
            )
        else:
            request = SimpleNamespace(
                model=runtime.model,
                bundle=runtime.bundle,
                config=runtime.config,
                device=runtime.device,
                epoch=int(epoch),
                output_dir=probe_dir,
                fixed_panel=runtime.fixed_panel,
                phase=str(phase),
            )
            result = callback(request)
    if result is None:
        result = {}
    if not isinstance(result, Mapping):
        raise Round31RuntimeError("probe callback must return a mapping")
    if runtime.probe_runner is not None:
        # An external probe runner may already validate and atomically write the
        # canonical receipt; do not wrap it in a second nested receipt.
        return result
    _atomic_json(
        receipt,
        {
            "schema": SCHEMA,
            "epoch": int(epoch),
            "phase": str(phase),
            "result": result,
        },
    )
    return result


def _seed_everything(seed: int) -> None:
    try:
        runner = importlib.import_module("training.runner")
        seed_fn = getattr(runner, "seed_everything", None)
        if callable(seed_fn):
            seed_fn(int(seed))
            return
    except (ImportError, ModuleNotFoundError):
        pass
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    try:
        import numpy as np

        np.random.seed(int(seed) & 0xFFFFFFFF)
    except Exception:
        pass


def _loader_rng_objects(bundle: Any) -> Iterable[tuple[str, torch.Generator]]:
    seen: set[int] = set()
    for loader_name in ("train_loader", "val_loader"):
        loader = getattr(bundle, loader_name, None)
        for owner_name, owner in (
            (loader_name, loader),
            (f"{loader_name}.sampler", getattr(loader, "sampler", None)),
            (f"{loader_name}.batch_sampler", getattr(loader, "batch_sampler", None)),
        ):
            generator = getattr(owner, "generator", None)
            if isinstance(generator, torch.Generator) and id(generator) not in seen:
                seen.add(id(generator))
                yield owner_name, generator


def _capture_loader_rng(bundle: Any) -> dict[str, torch.Tensor]:
    return {
        name: generator.get_state().clone()
        for name, generator in _loader_rng_objects(bundle)
    }


def _restore_loader_rng(bundle: Any, state: Any) -> None:
    if not isinstance(state, Mapping):
        return
    generators = dict(_loader_rng_objects(bundle))
    for name, value in state.items():
        generator = generators.get(str(name))
        if generator is not None and isinstance(value, torch.Tensor):
            generator.set_state(value)


def _restore_runtime_rng(runtime: PhaseRuntime, state: Mapping[str, Any]) -> None:
    _restore_rng(state)
    _restore_loader_rng(runtime.bundle, state.get("loaders"))


def probe_actual_schedule(runtime, before, expected_updates):
    cfg = runtime.schedule_probe_config
    scheduler = runtime.scheduler
    completed = scheduler.last_epoch + 1
    warmup = cfg["warmup_epochs"] * max(1, cfg["updates_per_epoch"])
    total = cfg["schedule_epochs"] * max(1, cfg["updates_per_epoch"])
    mode = cfg.get("mode", "cosine")
    if mode == "milestone":
        # round32 MR57 G0 group: mirrors _create_scheduler_for_config's
        # milestone factor exactly (warmup then step decay by gamma at
        # each configured epoch boundary).
        milestone_steps = sorted(
            int(m) * max(1, cfg["updates_per_epoch"])
            for m in (cfg.get("milestones") or [])
        )
        gamma = float(cfg.get("gamma", 0.5))
        if warmup and completed <= warmup:
            factor = completed / float(warmup)
        else:
            decayed = sum(1 for m in milestone_steps if completed > m)
            factor = gamma**decayed
    else:
        factor = (
            completed / float(warmup)
            if warmup and completed <= warmup
            else 0.1
            + 0.9
            * 0.5
            * (
                1
                + math.cos(
                    math.pi
                    * min(
                        1.0,
                        max(0.0, (completed - warmup) / float(max(1, total - warmup))),
                    )
                )
            )
        )
    groups = []
    for index, (group, base) in enumerate(
        zip(runtime.optimizer.param_groups, scheduler.base_lrs)
    ):
        expected = base * factor
        actual = float(group["lr"])
        groups.append(
            {
                "name": group.get("name", str(index)),
                "base_lr": base,
                "actual_lr": actual,
                "expected_lr": expected,
                "absolute_error": abs(actual - expected),
                "cumulative_actual_lr": group.get("round31_cumulative_lr"),
                "exposure_updates": group.get("round31_exposure_updates"),
            }
        )
    correct = len(groups) == len(runtime.optimizer.param_groups) and all(
        g["absolute_error"] < 1.0e-12 for g in groups
    )
    update_count = scheduler.last_epoch - before
    if not correct or update_count != expected_updates:
        raise Round31RuntimeError(
            "actual scheduler trace differs from configured curve or optimizer step count"
        )
    return dict(
        cfg,
        scheduler_last_epoch=scheduler.last_epoch,
        optimizer_updates=update_count,
        expected_updates=expected_updates,
        groups=groups,
        formula_match=correct,
    )


def _run_terminal_probe(runtime, phase_dir, epoch, max_seconds=120.0):
    """Bounded diagnostics after the last committed checkpoint, never training.

    The queue's unchanged 3300s process-group deadline remains authoritative.
    SIGALRM is a secondary limit; unsupported hosts explicitly skip this probe.
    """
    import signal
    import threading
    import time

    receipt = phase_dir / "probes" / f"probe_e{int(epoch):03d}.json"
    audit = phase_dir / "terminal_probe_status.json"
    result = {
        "epoch": int(epoch),
        "max_seconds": float(max_seconds),
        "optimizer_steps": 0,
    }
    if receipt.exists():
        result.update(status="existing_receipt", path=str(receipt))
    elif runtime.probe_suite is None:
        result.update(status="incomplete", reason="no_probe_suite")
    elif (
        not hasattr(signal, "setitimer")
        or threading.current_thread() is not threading.main_thread()
    ):
        result.update(status="incomplete", reason="bounded_signal_watchdog_unavailable")
    else:

        class TerminalProbeTimeout(TimeoutError):
            pass

        def alarm_handler(signum, frame):
            raise TerminalProbeTimeout("terminal probe exceeded its time budget")

        previous_handler = signal.getsignal(signal.SIGALRM)
        previous_timer = signal.getitimer(signal.ITIMER_REAL)
        if previous_timer[0] > 0:
            result.update(status="incomplete", reason="existing_alarm_preserved")
        else:
            start = time.monotonic()
            signal.signal(signal.SIGALRM, alarm_handler)
            signal.setitimer(signal.ITIMER_REAL, float(max_seconds))
            try:
                _run_probe(
                    runtime,
                    phase_dir,
                    epoch,
                    phase="terminal_committed_checkpoint",
                    force=True,
                )
                if not receipt.is_file():
                    raise RuntimeError("terminal probe did not write its receipt")
                result.update(status="completed", path=str(receipt))
            except TerminalProbeTimeout:
                result.update(status="incomplete", reason="terminal_probe_timeout")
            except Exception as exc:
                result.update(
                    status="failed", reason=type(exc).__name__ + ": " + str(exc)
                )
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, previous_handler)
            result["elapsed_seconds"] = time.monotonic() - start
    _atomic_json(audit, result)
    return result
