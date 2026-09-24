"""Runtime helpers for the field_core scratch experiment.

This module is intentionally a thin adapter around the repository's reviewed
training.runner._run_epoch and CheckpointLifecycle contracts.  It does not
contain a model or a loss implementation, and importing it has no side
effects.  In particular, no checkpoint is loaded and no training is started by
this module.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, is_dataclass
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import torch
from torch import nn


class RuntimeConfigError(RuntimeError):
    """Configuration or lifecycle contract failure before/during a run."""

    pass


EXPECTED_RUN_EPOCH_PARAMETERS = (
    "model",
    "bundle",
    "loader",
    "config",
    "device",
    "epoch",
    "optimizer",
    "scheduler",
    "teacher",
    "registered_trainable_ids",
)


def resolve_target(target: str) -> Any:
    """Resolve a strict module:symbol factory target."""
    if ":" not in str(target):
        raise ValueError(f"factory target must be 'module:symbol', got {target!r}")
    module_name, symbol = str(target).split(":", 1)
    module = importlib.import_module(module_name)
    try:
        return getattr(module, symbol)
    except AttributeError as exc:
        raise ImportError(f"{module_name!r} does not export {symbol!r}") from exc


def _spec_parts(spec: Any) -> tuple[str, dict[str, Any]]:
    if isinstance(spec, str):
        return spec, {}
    if isinstance(spec, Mapping):
        return str(spec["target"]), dict(spec.get("kwargs") or {})
    target = getattr(spec, "target", None)
    if target is None:
        raise TypeError(f"factory spec has no target: {type(spec).__name__}")
    return str(target), dict(getattr(spec, "kwargs", {}) or {})


def instantiate_factory(spec: Any, config: Any) -> Any:
    """Use the official config-first factory convention."""
    target, kwargs = _spec_parts(spec)
    factory = resolve_target(target)
    return factory(config=config, **kwargs)


def assert_scratch_config(raw: Mapping[str, Any]) -> None:
    """Reject checkpoint inheritance/resume for the base run."""
    checkpoint = raw.get("base_checkpoint")
    if checkpoint not in (None, ""):
        raise ValueError(
            f"field_core requires scratch; got base_checkpoint={checkpoint!r}"
        )
    if bool(raw.get("resume", False)):
        raise ValueError("field_core requires resume=false")


def _parameter_group_parts(
    group: Any, default_lr: float
) -> tuple[str, list[nn.Parameter], float, Optional[float]]:
    if isinstance(group, Mapping):
        name = str(group.get("name", ""))
        params = list(group.get("params") or [])
        lr = float(group.get("lr", default_lr))
        weight_decay = group.get("weight_decay")
    else:
        name = str(getattr(group, "name", ""))
        params = list(getattr(group, "params", ()) or ())
        lr = float(getattr(group, "lr", default_lr))
        weight_decay = getattr(group, "weight_decay", None)
    if not name:
        raise ValueError("optimizer parameter group needs a non-empty name")
    return name, params, lr, None if weight_decay is None else float(weight_decay)


def build_optimizer(
    model: nn.Module,
    *,
    lr: float = 1.0e-4,
    weight_decay: float = 1.0e-4,
) -> tuple[torch.optim.Optimizer, dict[str, Any]]:
    """Build AdamW and prove exact coverage of every trainable parameter.

    Models may expose the repository's named parameter_groups() method.  If
    absent, one explicit all-trainable group is used.  Any groups returned by a
    model are checked strictly; silently dropping a trainable branch is an
    error.
    """
    named = dict(model.named_parameters())
    owner = {id(parameter): name for name, parameter in named.items()}
    trainable_ids = {
        id(parameter) for parameter in named.values() if parameter.requires_grad
    }
    if not trainable_ids:
        raise ValueError("model has no trainable parameters")

    method = getattr(model, "parameter_groups", None)
    if callable(method):
        raw_groups = list(method())
        if not raw_groups:
            raise ValueError("model.parameter_groups() returned no groups")
    else:
        raw_groups = [
            {
                "name": "all_trainable",
                "params": [
                    parameter for parameter in named.values() if parameter.requires_grad
                ],
                "lr": float(lr),
            }
        ]

    payload: list[dict[str, Any]] = []
    audit_groups: dict[str, Any] = {}
    seen: list[int] = []
    names: set[str] = set()
    for raw_group in raw_groups:
        name, params, group_lr, group_decay = _parameter_group_parts(
            raw_group, float(lr)
        )
        if name in names:
            raise ValueError(f"duplicate optimizer group name: {name}")
        names.add(name)
        if not params:
            raise ValueError(f"empty optimizer group: {name}")
        unknown = [id(parameter) for parameter in params if id(parameter) not in owner]
        if unknown:
            raise ValueError(
                f"optimizer group {name} contains parameters outside model"
            )
        frozen = [
            owner[id(parameter)] for parameter in params if not parameter.requires_grad
        ]
        if frozen:
            raise ValueError(
                f"optimizer group {name} contains frozen parameters: {frozen[:5]}"
            )
        ids = [id(parameter) for parameter in params]
        if len(ids) != len(set(ids)):
            raise ValueError(f"optimizer group {name} contains duplicate parameters")
        seen.extend(ids)
        decay = float(weight_decay if group_decay is None else group_decay)
        payload.append(
            {
                "name": name,
                "params": params,
                "lr": group_lr,
                "weight_decay": decay,
            }
        )
        audit_groups[name] = {
            "base_lr": group_lr,
            "weight_decay": decay,
            "tensor_count": len(params),
            "numel": sum(int(parameter.numel()) for parameter in params),
            "names": [owner[id(parameter)] for parameter in params],
        }

    if len(seen) != len(set(seen)):
        raise ValueError("optimizer parameter IDs contain duplicates across groups")
    seen_ids = set(seen)
    if seen_ids != trainable_ids:
        missing = [
            name
            for name, parameter in named.items()
            if parameter.requires_grad and id(parameter) not in seen_ids
        ]
        extra = [
            owner.get(identifier, "<unknown>")
            for identifier in seen_ids - trainable_ids
        ]
        raise ValueError(
            f"optimizer coverage mismatch; missing={missing[:10]} extra={extra[:10]}"
        )

    optimizer = torch.optim.AdamW(payload)
    audit = {
        "groups": audit_groups,
        "optimizer_parameter_id_count": len(seen),
        "optimizer_unique_id_count": len(seen_ids),
        "optimizer_id_exact_cover": True,
        "total_trainable_numel": sum(
            int(parameter.numel())
            for parameter in named.values()
            if parameter.requires_grad
        ),
        "total_parameter_numel": sum(
            int(parameter.numel()) for parameter in named.values()
        ),
    }
    return optimizer, audit


def registered_trainable_ids(model: nn.Module) -> set[int]:
    return {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }


def assert_run_epoch_contract(run_epoch: Any) -> None:
    """Fail early if the imported repository runner is not the reviewed one."""
    if not callable(run_epoch):
        raise TypeError("training.runner._run_epoch is not callable")
    names = tuple(inspect.signature(run_epoch).parameters)
    if names != EXPECTED_RUN_EPOCH_PARAMETERS:
        raise RuntimeError(
            "training.runner._run_epoch signature drift: "
            f"{names} != {EXPECTED_RUN_EPOCH_PARAMETERS}"
        )


def run_official_epoch(
    *,
    model: nn.Module,
    bundle: Any,
    loader: Iterable[Any],
    config: Any,
    device: torch.device,
    epoch: int,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[Any],
    teacher: Optional[nn.Module],
    registered_ids: set[int],
) -> tuple[dict[str, float], dict[str, Any]]:
    """Delegate exactly one epoch to the repository training runner."""
    common_runner = importlib.import_module("training.runner")
    run_epoch = getattr(common_runner, "_run_epoch", None)
    assert_run_epoch_contract(run_epoch)
    return run_epoch(
        model,
        bundle,
        loader,
        config,
        device,
        int(epoch),
        optimizer,
        scheduler,
        teacher,
        registered_ids,
    )


def autocast_context(device: torch.device, precision: str = "bf16"):
    """Return the reviewed BF16 context without enabling it on CPU."""
    if precision == "bf16" and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if precision not in {"bf16", "fp32"}:
        raise ValueError(f"unsupported precision: {precision!r}")
    return nullcontext()


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_steps: int,
    cosine_floor: float = 0.1,
):
    """Repository-compatible warmup + cosine scheduler."""
    if total_steps < 1 or warmup_steps < 0 or warmup_steps > total_steps:
        raise ValueError("invalid total_steps/warmup_steps")
    if not 0.0 <= cosine_floor < 1.0:
        raise ValueError("cosine_floor must be in [0, 1)")

    def factor(step: int) -> float:
        completed = step + 1
        if warmup_steps and completed <= warmup_steps:
            return completed / float(warmup_steps)
        progress = min(
            1.0,
            max(
                0.0,
                (completed - warmup_steps) / float(max(1, total_steps - warmup_steps)),
            ),
        )
        return cosine_floor + (1.0 - cosine_floor) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def evaluator_from_config(config: Any) -> Any:
    spec = getattr(config, "evaluator_factory", None)
    if spec is None:
        raise ValueError("official evaluator_factory is required")
    return instantiate_factory(spec, config)


def config_dict(config: Any) -> dict[str, Any]:
    if isinstance(config, Mapping):
        return dict(config)
    if callable(getattr(config, "to_dict", None)):
        return dict(config.to_dict())
    if is_dataclass(config):
        return dict(asdict(config))
    raise TypeError(f"config cannot be converted to a mapping: {type(config).__name__}")


def checkpoint_payload(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    config: Any,
    val_metrics: Optional[Mapping[str, Any]],
) -> dict[str, Any]:
    """Create the training checkpoint schema without reading a checkpoint."""
    contract = getattr(model, "experiment_contract", None)
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else {},
        "epoch": int(epoch),
        "epochs_budget": int(config_dict(config).get("epochs", 0)),
        "selection": "fixed_final_with_independent_best",
        "selection_split": "official_validation",
        "config": config_dict(config),
        "val_metrics": dict(val_metrics or {}),
        "experiment_contract": dict(contract()) if callable(contract) else {},
    }


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if torch.is_tensor(value):
        if value.numel() != 1:
            return {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
        return value.detach().float().cpu().item()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return str(value)


def _shape_tree(value: Any, depth: int = 0) -> Any:
    if depth > 2:
        return str(type(value).__name__)
    if torch.is_tensor(value):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, Mapping):
        return {str(key): _shape_tree(item, depth + 1) for key, item in value.items()}
    if is_dataclass(value):
        return {
            field: _shape_tree(getattr(value, field), depth + 1)
            for field in value.__dataclass_fields__
        }
    if isinstance(value, (list, tuple)):
        return {"type": type(value).__name__, "length": len(value)}
    return {"type": type(value).__name__}


def _scalar_metrics(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, Any] = {}
    for key, item in value.items():
        if torch.is_tensor(item) and item.numel() == 1:
            result[str(key)] = float(item.detach().float().cpu())
        elif isinstance(item, (int, float, bool)):
            result[str(key)] = item
    return result


__all__ = [
    "RuntimeConfigError",
    "EXPECTED_RUN_EPOCH_PARAMETERS",
    "assert_run_epoch_contract",
    "assert_scratch_config",
    "autocast_context",
    "build_optimizer",
    "checkpoint_payload",
    "config_dict",
    "create_scheduler",
    "evaluator_from_config",
    "instantiate_factory",
    "registered_trainable_ids",
    "resolve_target",
    "run_official_epoch",
]


# Shared file helpers used by the executable child runtime.
def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(_json_value(value), indent=2, sort_keys=True, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(_json_value(value), sort_keys=True, ensure_ascii=False) + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def _load_json(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"JSON file unavailable: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    from .child import main as child_main

    return int(child_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
