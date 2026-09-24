"""Shared fixed-final training entry point for Stage47 plugins."""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import math
import random
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import torch
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_

from .audit import (
    assert_frozen_zero_drift,
    build_optimizer,
    current_learning_rates,
    hash_sources,
    parameter_drift,
    snapshot_parameters,
)
from .checkpointing import CheckpointLifecycle
from .config import FactoryConfig, RunnerConfig
from .contracts import DataBundle, LossResult, OfficialEvalRequest, PreparedBatch


def resolve_target(target: str) -> Any:
    if ":" not in target:
        raise ValueError(f"factory target must be 'module:symbol', got {target!r}")
    module_name, symbol = target.split(":", 1)
    module = importlib.import_module(module_name)
    try:
        return getattr(module, symbol)
    except AttributeError as exc:
        raise ImportError(f"{module_name!r} does not export {symbol!r}") from exc


def instantiate(spec: FactoryConfig, config: RunnerConfig) -> Any:
    factory = resolve_target(spec.target)
    return factory(config=config, **dict(spec.kwargs))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _finite_scalar(value: Tensor | float | int, name: str) -> float:
    result = float(value.detach()) if torch.is_tensor(value) else float(value)
    if not math.isfinite(result):
        raise FloatingPointError(f"non-finite metric {name}: {result}")
    return result


def _autocast(config: RunnerConfig, device: torch.device):
    if config.precision == "bf16" and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _assert_batch_contract(batch: PreparedBatch, target_keys: Iterable[str]) -> None:
    if not isinstance(batch.inputs, Mapping) or not isinstance(batch.targets, Mapping):
        raise TypeError("PreparedBatch inputs and targets must be mappings")
    overlap = set(batch.inputs).intersection(batch.targets)
    explicit = set(target_keys).intersection(batch.inputs)
    if overlap or explicit:
        raise AssertionError(
            f"ground-truth keys leaked into forward inputs: {sorted(overlap | explicit)}"
        )
    if batch.batch_size < 1:
        raise ValueError("PreparedBatch.batch_size must be positive")


def _teacher_forward(
    teacher: Optional[nn.Module], inputs: Mapping[str, Any], weight: float
) -> Optional[Any]:
    if teacher is None or weight == 0.0:
        return None
    teacher.eval()
    with torch.no_grad():
        return teacher(inputs)


def _phase_diagnostics(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, str):
        return {"phase": value}
    if not isinstance(value, Mapping):
        raise TypeError("set_epoch must return a mapping, string, or None")
    result: Dict[str, Any] = {}
    for key, item in value.items():
        if torch.is_tensor(item):
            if item.numel() != 1:
                raise TypeError(f"phase diagnostic {key!r} must be scalar")
            item = item.detach().item()
        if not isinstance(item, (str, int, float, bool, type(None))):
            raise TypeError(
                f"phase diagnostic {key!r} is not JSON-scalar: {type(item).__name__}"
            )
        result[str(key)] = item
    return result


def set_model_epoch(
    model: nn.Module,
    epoch: int,
    training: bool,
    registered_trainable_ids: set[int],
) -> Dict[str, Any]:
    """Invoke the phase callback and enforce stable optimizer membership."""
    callback = getattr(model, "set_epoch", None)
    if callback is None or not callable(callback):
        raise TypeError("Stage47 model must implement set_epoch(epoch, training)")
    diagnostics = _phase_diagnostics(callback(int(epoch), bool(training)))
    current_ids = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    if current_ids != registered_trainable_ids:
        raise RuntimeError(
            "set_epoch changed requires_grad/optimizer membership; use gradient or path gating instead"
        )
    diagnostics.setdefault("epoch", int(epoch))
    diagnostics.setdefault("training", bool(training))
    return diagnostics


def _create_scheduler(
    optimizer: torch.optim.Optimizer, total_steps: int, warmup_steps: int
):
    def factor(step: int) -> float:
        completed = step + 1
        if warmup_steps > 0 and completed <= warmup_steps:
            return completed / float(warmup_steps)
        remaining = max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, (completed - warmup_steps) / float(remaining)))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=factor)


def _run_epoch(
    model: nn.Module,
    bundle: DataBundle,
    loader: Iterable[Any],
    config: RunnerConfig,
    device: torch.device,
    epoch: int,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[Any],
    teacher: Optional[nn.Module],
    registered_trainable_ids: set[int],
) -> tuple[Dict[str, float], Dict[str, Any]]:
    training = optimizer is not None
    model.train(training)
    phase = set_model_epoch(model, epoch, training, registered_trainable_ids)
    teacher_weight = config.teacher.weight(epoch)
    if teacher is not None and teacher_weight > 0.0:
        teacher.train(False)
        teacher_callback = getattr(teacher, "set_epoch", None)
        if callable(teacher_callback):
            teacher_callback(int(epoch), False)
    totals: Dict[str, float] = {}
    examples = 0
    if training:
        optimizer.zero_grad(set_to_none=True)
    step_count = 0
    loader_length = len(loader) if hasattr(loader, "__len__") else None
    for batch_index, raw in enumerate(loader):
        prepared = bundle.prepare_batch(raw, device)
        _assert_batch_contract(prepared, config.target_keys)
        with torch.set_grad_enabled(training), _autocast(config, device):
            outputs = model(prepared.inputs)
            teacher_outputs = _teacher_forward(teacher, prepared.inputs, teacher_weight)
            result = model.compute_loss(outputs, prepared, teacher_outputs, epoch)
            if not isinstance(result, LossResult):
                raise TypeError(
                    "compute_loss must return training.contracts.LossResult"
                )
            if result.loss.ndim != 0 or not torch.isfinite(result.loss):
                raise FloatingPointError(
                    f"non-finite or non-scalar loss at epoch {epoch}, batch {batch_index}"
                )
            scaled = result.loss / float(config.grad_accumulation)
        if training:
            scaled.backward()
            is_last = loader_length is not None and batch_index + 1 == loader_length
            should_step = (batch_index + 1) % config.grad_accumulation == 0 or is_last
            if should_step:
                for name, parameter in model.named_parameters():
                    if (
                        parameter.grad is not None
                        and not torch.isfinite(parameter.grad).all()
                    ):
                        raise FloatingPointError(f"non-finite gradient in {name}")
                clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], config.grad_clip
                )
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step_count += 1
        metrics: Dict[str, Tensor | float] = {
            "loss": result.loss,
            **dict(result.metrics),
        }
        diagnostics = model.diagnostics(outputs, prepared)
        metrics.update({f"diag/{name}": value for name, value in diagnostics.items()})
        examples += prepared.batch_size
        for name, value in metrics.items():
            totals[name] = (
                totals.get(name, 0.0)
                + _finite_scalar(value, name) * prepared.batch_size
            )
    if examples == 0:
        raise RuntimeError("empty data loader")
    output = {name: value / examples for name, value in totals.items()}
    output["optimizer_steps"] = float(step_count)
    output["teacher_weight"] = float(config.teacher.weight(epoch))
    return output, phase


def _module_source(target: str) -> Optional[Path]:
    obj = resolve_target(target)
    source = inspect.getsourcefile(obj)
    return Path(source).resolve() if source else None


def train(config_path: str | Path) -> Dict[str, Any]:
    config_path = Path(config_path).resolve()
    config = RunnerConfig.load(config_path)
    output_dir = Path(config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(config.seed)
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {device}")

    bundle = instantiate(config.data_factory, config)
    if not isinstance(bundle, DataBundle):
        raise TypeError("data_factory must return training.contracts.DataBundle")
    model = instantiate(config.model_factory, config)
    if not isinstance(model, nn.Module):
        raise TypeError(
            "model_factory must return torch.nn.Module implementing Stage47Model"
        )
    model.to(device)
    teacher = (
        instantiate(config.teacher.factory, config) if config.teacher.factory else None
    )
    if teacher is not None:
        if not isinstance(teacher, nn.Module):
            raise TypeError("teacher factory must return torch.nn.Module")
        teacher.to(device).eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)

    groups = model.parameter_groups()
    optimizer, optimizer_audit = build_optimizer(model, groups, config.weight_decay)
    registered_trainable_ids = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    train_batches = len(bundle.train_loader)
    updates_per_epoch = math.ceil(train_batches / config.grad_accumulation)
    scheduler = _create_scheduler(
        optimizer,
        total_steps=config.epochs * updates_per_epoch,
        warmup_steps=config.warmup_epochs * updates_per_epoch,
    )
    before = snapshot_parameters(model)

    resolved_config = output_dir / "config.json"
    config.save(resolved_config)
    source_roots = [
        Path(__file__).resolve().parent,
        *[Path(p) for p in config.source_roots],
    ]
    for target in (config.model_factory.target, config.data_factory.target):
        source = _module_source(target)
        if source is not None:
            source_roots.append(source)
    hashes = hash_sources(source_roots)
    (output_dir / "code_hashes.json").write_text(
        json.dumps(hashes, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "optimizer_audit.json").write_text(
        json.dumps(optimizer_audit, indent=2) + "\n", encoding="utf-8"
    )

    lifecycle: Optional[CheckpointLifecycle] = None
    if config.checkpointing.enabled:
        if config.evaluator_factory is None:
            raise ValueError("checkpointing.enabled requires evaluator_factory")
        adapter = instantiate(config.evaluator_factory, config)
        official_device = config.checkpointing.official_device or config.device
        official_batch_size = (
            config.checkpointing.official_batch_size or config.batch_size
        )
        lifecycle = CheckpointLifecycle(
            output_dir=output_dir,
            policy=config.checkpointing,
            evaluator=adapter,
            request_template=OfficialEvalRequest(
                project_root=Path.cwd(),
                checkpoint=output_dir / "checkpoints" / "pending.pt",
                output_dir=output_dir / "official_by_epoch" / "pending",
                config_path=resolved_config,
                device=official_device,
                batch_size=official_batch_size,
                num_workers=config.num_workers,
                precision=config.precision,
            ),
        )
        lifecycle.initialize()

    history_path = output_dir / "history.jsonl"
    history_path.write_text("", encoding="utf-8")
    final_val: Optional[Dict[str, float]] = None
    for epoch in range(1, config.epochs + 1):
        start_lrs = current_learning_rates(optimizer)
        train_metrics, train_phase = _run_epoch(
            model,
            bundle,
            bundle.train_loader,
            config,
            device,
            epoch,
            optimizer,
            scheduler,
            teacher,
            registered_trainable_ids,
        )
        val_metrics, val_phase = _run_epoch(
            model,
            bundle,
            bundle.val_loader,
            config,
            device,
            epoch,
            None,
            None,
            teacher,
            registered_trainable_ids,
        )
        final_val = val_metrics
        record = {
            "epoch": epoch,
            "selection": "fixed-final",
            "learning_rates_start": start_lrs,
            "learning_rates_end": current_learning_rates(optimizer),
            "train_phase": train_phase,
            "val_phase": val_phase,
            "train": train_metrics,
            "val": val_metrics,
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "epochs_budget": config.epochs,
            "selection": "fixed-final",
            "config": config.to_dict(),
            "val_metrics": val_metrics,
        }
        if lifecycle is None:
            torch.save(checkpoint, output_dir / "last.pt")
        else:
            lifecycle.process_epoch(
                epoch=epoch,
                checkpoint_payload=checkpoint,
                metadata={
                    "train": train_metrics,
                    "val": val_metrics,
                    "learning_rates_start": start_lrs,
                    "learning_rates_end": current_learning_rates(optimizer),
                },
            )

    drift = parameter_drift(model, before)
    if config.fail_on_frozen_drift:
        assert_frozen_zero_drift(drift)
    (output_dir / "parameter_drift.json").write_text(
        json.dumps(drift, indent=2) + "\n", encoding="utf-8"
    )
    lifecycle_summary = (
        lifecycle.finalize(config.epochs) if lifecycle is not None else None
    )
    final = {
        "epoch": config.epochs,
        "epochs_budget": config.epochs,
        "selection": "fixed-final",
        "checkpoint": str(output_dir / "last.pt"),
        "val": final_val,
        "optimizer_audit": optimizer_audit,
    }
    if lifecycle_summary is not None:
        final["best_val"] = lifecycle_summary
        final["official"] = lifecycle_summary["last_official"]
    (output_dir / "final_metrics.json").write_text(
        json.dumps(final, indent=2) + "\n", encoding="utf-8"
    )

    if config.evaluator_factory is not None and lifecycle is None:
        adapter = instantiate(config.evaluator_factory, config)
        official_dir = output_dir / "official"
        official = adapter.evaluate(
            OfficialEvalRequest(
                project_root=Path.cwd(),
                checkpoint=output_dir / "last.pt",
                output_dir=official_dir,
                config_path=resolved_config,
                device=config.device,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
                precision=config.precision,
            )
        )
        final["official"] = dict(official)
        (official_dir / "normalized_metrics.json").write_text(
            json.dumps(official, indent=2) + "\n", encoding="utf-8"
        )
        (output_dir / "final_metrics.json").write_text(
            json.dumps(final, indent=2) + "\n", encoding="utf-8"
        )
    return final


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    train(args.config)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Stage47 runner failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
