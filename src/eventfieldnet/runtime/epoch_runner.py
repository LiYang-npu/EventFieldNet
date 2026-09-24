"""Round31 copy of the validated training epoch loop with one step observer.

This file intentionally mirrors ``training.runner._run_epoch``.  The only
semantic addition is an optional observer around the existing
``clip_grad_norm_``/``optimizer.step`` pair.  Keeping the observer inside the
loop is required: an outer wrapper cannot recover the preclip norm or the
actual parameter delta without changing optimizer behavior.
"""

from __future__ import annotations

import importlib
from typing import Any, Optional

import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_


def run_epoch_with_step_telemetry(
    model: Any,
    bundle: Any,
    loader: Any,
    config: Any,
    device: torch.device,
    epoch: int,
    optimizer: Optional[Any],
    scheduler: Optional[Any],
    teacher: Optional[Any],
    registered_trainable_ids: set[int],
    *,
    step_observer: Any = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Run one epoch using the official loop plus a bounded step observer."""
    common_runner = importlib.import_module("training.runner")
    contracts = importlib.import_module("training.contracts")
    loss_result_type = getattr(contracts, "LossResult")
    assert_batch_contract = getattr(common_runner, "_assert_batch_contract")
    autocast = getattr(common_runner, "_autocast")
    teacher_forward = getattr(common_runner, "_teacher_forward")
    finite_scalar = getattr(common_runner, "_finite_scalar")
    set_model_epoch = getattr(common_runner, "set_model_epoch")

    training = optimizer is not None
    model.train(training)
    phase = set_model_epoch(model, epoch, training, registered_trainable_ids)
    teacher_weight = config.teacher.weight(epoch)
    if teacher is not None and teacher_weight > 0.0:
        teacher.train(False)
        teacher_callback = getattr(teacher, "set_epoch", None)
        if callable(teacher_callback):
            teacher_callback(int(epoch), False)
    totals: dict[str, float] = {}
    examples = 0
    if training:
        optimizer.zero_grad(set_to_none=True)
        if step_observer is not None:
            step_observer.begin_epoch(int(epoch))
    step_count = 0
    loader_length = len(loader) if hasattr(loader, "__len__") else None
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    for batch_index, raw in enumerate(loader):
        prepared = bundle.prepare_batch(raw, device)
        assert_batch_contract(prepared, config.target_keys)
        with torch.set_grad_enabled(training), autocast(config, device):
            outputs = model(prepared.inputs)
            teacher_outputs = teacher_forward(teacher, prepared.inputs, teacher_weight)
            result = model.compute_loss(outputs, prepared, teacher_outputs, epoch)
            if not isinstance(result, loss_result_type):
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
                capture = None
                if step_observer is not None:
                    capture = step_observer.before_clip(
                        epoch=int(epoch),
                        batch_index=int(batch_index),
                        local_optimizer_step=int(step_count + 1),
                        max_norm=float(config.grad_clip),
                    )
                clipped_norm = clip_grad_norm_(parameters, config.grad_clip)
                if step_observer is not None:
                    step_observer.after_clip(capture, returned_norm=clipped_norm)
                used_lrs = [float(g["lr"]) for g in optimizer.param_groups]
                optimizer.step()
                for group, used_lr in zip(optimizer.param_groups, used_lrs):
                    group["round31_cumulative_lr"] = (
                        group.get("round31_cumulative_lr", 0.0) + used_lr
                    )
                    group["round31_exposure_updates"] = (
                        group.get("round31_exposure_updates", 0) + 1
                    )
                if step_observer is not None:
                    step_observer.after_step(capture)
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step_count += 1
        metrics: dict[str, Tensor | float] = {
            "loss": result.loss,
            **dict(result.metrics),
        }
        diagnostics = model.diagnostics(outputs, prepared)
        metrics.update({f"diag/{name}": value for name, value in diagnostics.items()})
        examples += prepared.batch_size
        for name, value in metrics.items():
            totals[name] = (
                totals.get(name, 0.0) + finite_scalar(value, name) * prepared.batch_size
            )
    if examples == 0:
        raise RuntimeError("empty data loader")
    output = {name: value / examples for name, value in totals.items()}
    output["optimizer_steps"] = float(step_count)
    output["teacher_weight"] = float(config.teacher.weight(epoch))
    if training and step_observer is not None:
        phase = dict(phase)
        phase["step_telemetry"] = step_observer.end_epoch(int(epoch))
    return output, phase


__all__ = ["run_epoch_with_step_telemetry"]
