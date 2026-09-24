"""Child CLI for the field_core scratch run."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib
import itertools
import json
import os
from pathlib import Path
import platform
import random
import sys
import time
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn

try:
    from . import runtime as rt
except ImportError:
    import field_core.runtime as rt


DEFAULT_PROBE_EPOCHS = (0, 1, 8, 18, 50)
DEFAULT_MODEL_TARGET = "field_core.adapter:build_trifield_base_model"
DEFAULT_EVALUATOR_TARGET = "backbone.conditioned.plugin:build_official_evaluator"


@dataclass(frozen=True)
class ProbeRequest:
    """The stable API passed to base_engineering's ProbeSuite.run method."""

    model: nn.Module
    bundle: Any
    config: Any
    device: torch.device
    epoch: int
    output_dir: Path
    fixed_panel: Mapping[str, Any] | None
    phase: str


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(
        json.dumps(rt._json_value(value), indent=2, sort_keys=True, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _load_json(path: Path) -> dict[str, Any]:
    return rt._load_json(path)


def _sha256(path: Path) -> str:
    return rt._sha256_file(path)


def _root(raw: Mapping[str, Any], remote_root: str | Path | None) -> Path:
    spec = raw.get("evaluator_factory")
    kwargs = spec.get("kwargs", {}) if isinstance(spec, Mapping) else {}
    value = (
        remote_root
        if remote_root is not None
        else (kwargs.get("root") if isinstance(kwargs, Mapping) else None)
    )
    value = raw.get("remote_root") if value in (None, "") else value
    if value in (None, ""):
        raise rt.RuntimeConfigError(
            "remote root required in evaluator kwargs or --remote-root"
        )
    path = Path(str(value))
    if not path.is_absolute() or ".." in path.parts:
        raise rt.RuntimeConfigError(f"remote root must be absolute: {path}")
    return path


def _project_root(raw: Mapping[str, Any]) -> Path:
    spec = raw.get("evaluator_factory")
    kwargs = spec.get("kwargs", {}) if isinstance(spec, Mapping) else {}
    value = kwargs.get("project_root") if isinstance(kwargs, Mapping) else None
    value = raw.get("project_root") if value in (None, "") else value
    return Path(str(value)).resolve() if value not in (None, "") else Path.cwd()


def install_import_roots(
    raw: Mapping[str, Any],
    *,
    remote_root: str | Path | None = None,
    strict: bool = True,
) -> tuple[Path, tuple[Path, ...]]:
    root = _root(raw, remote_root)
    initial = [root / "src", root]
    for path in reversed(initial):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    paths = [root / "src", root]
    configured = raw.get("source_roots")
    if isinstance(configured, list):
        paths.extend(Path(str(path)) for path in configured)
    paths.extend(initial)
    unique, seen = [], set()
    for path in paths:
        if str(path) not in seen:
            seen.add(str(path))
            unique.append(path)
    for path in reversed(unique):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return root, tuple(unique)


class _PythonPath:
    def __init__(self, paths: Sequence[Path]):
        self.paths = tuple(paths)
        self.old = os.environ.get("PYTHONPATH")

    def __enter__(self):
        values = [str(path) for path in self.paths]
        if self.old:
            values.append(self.old)
        os.environ["PYTHONPATH"] = os.pathsep.join(values)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.old is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = self.old
        return False


def _fixed_panel(raw: Mapping[str, Any], config_path: Path) -> Mapping[str, Any] | None:
    value = raw.get("fixed_panel_path")
    if value in (None, ""):
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = config_path.parent / path
    payload = _load_json(path)
    for name, section in payload.items():
        if isinstance(section, Mapping) and "qids" in section:
            qids = section["qids"]
            if not isinstance(qids, list) or len(qids) != 256:
                raise rt.RuntimeConfigError(f"fixed panel {name} must have 256 qids")
    return payload


def _probe_epochs(raw: Mapping[str, Any], config: Any) -> tuple[int, ...]:
    values = raw.get("probe_epochs", list(DEFAULT_PROBE_EPOCHS))
    if not isinstance(values, (list, tuple)):
        raise rt.RuntimeConfigError("probe_epochs must be a list")
    epochs = tuple(sorted({int(value) for value in values}))
    if not epochs or epochs[0] != 0 or min(epochs) < 0:
        raise rt.RuntimeConfigError("probe_epochs must include e0")
    if int(config.epochs) < max(epochs):
        raise rt.RuntimeConfigError(f"epochs={config.epochs} misses e{max(epochs)}")
    return epochs


def _probe_suite(raw: Mapping[str, Any], config: Any) -> Any:
    spec = raw.get("probe_factory")
    if spec is None:
        raise rt.RuntimeConfigError("formal base config requires probe_factory")
    suite = rt.instantiate_factory(spec, config)
    if not callable(getattr(suite, "run", None)) and not callable(suite):
        raise TypeError("probe_factory must return .run(request) or callable")
    return suite


class _ProbeGuard:
    """Restore mode, requires_grad, buffer, RNG, and grad state after a probe."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.training: dict[int, bool] = {}
        self.requires_grad: dict[int, bool] = {}
        self.versions: dict[int, int] = {}
        self.buffers: dict[str, Tensor] = {}
        self.cpu_rng: Tensor | None = None
        self.cuda_rng: list[Tensor] | None = None
        self.python_rng: object | None = None
        self.numpy_rng: object | None = None
        self.grads: dict[int, Tensor | None] = {}
        self.grad_enabled = True

    def __enter__(self):
        self.training = {
            id(module): bool(module.training) for module in self.model.modules()
        }
        self.requires_grad = {
            id(parameter): bool(parameter.requires_grad)
            for parameter in self.model.parameters()
        }
        self.versions = {
            id(parameter): int(getattr(parameter, "_version", 0))
            for parameter in self.model.parameters()
        }
        self.buffers = {
            name: buffer.detach().clone() for name, buffer in self.model.named_buffers()
        }
        self.cpu_rng = torch.random.get_rng_state()
        if torch.cuda.is_available():
            self.cuda_rng = [state.clone() for state in torch.cuda.get_rng_state_all()]
        self.grad_enabled = torch.is_grad_enabled()
        self.python_rng = random.getstate()
        try:
            import numpy as np

            self.numpy_rng = np.random.get_state()
        except (ImportError, ModuleNotFoundError):
            self.numpy_rng = None
        self.grads = {
            id(parameter): None
            if parameter.grad is None
            else parameter.grad.detach().clone()
            for parameter in self.model.parameters()
        }
        return self

    def __exit__(self, exc_type, exc, tb):
        changed = [
            name
            for name, parameter in self.model.named_parameters()
            if int(getattr(parameter, "_version", 0))
            != self.versions.get(id(parameter), 0)
        ]
        try:
            if changed and exc is None:
                raise RuntimeError(f"probe mutated parameters: {changed[:10]}")
        finally:
            modules = {id(module): module for module in self.model.modules()}
            for identifier, mode in self.training.items():
                if identifier in modules:
                    modules[identifier].training = mode
            parameters = {
                id(parameter): parameter for parameter in self.model.parameters()
            }
            for identifier, required in self.requires_grad.items():
                if identifier in parameters:
                    parameters[identifier].requires_grad_(required)
            for identifier, gradient in self.grads.items():
                if identifier in parameters:
                    parameters[identifier].grad = (
                        None if gradient is None else gradient.clone()
                    )
            buffers = dict(self.model.named_buffers())
            for name, before in self.buffers.items():
                current = buffers.get(name)
                if current is not None and current.shape == before.shape:
                    current.detach().copy_(before)
            if self.cpu_rng is not None:
                torch.random.set_rng_state(self.cpu_rng)
            if self.cuda_rng is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(self.cuda_rng)
            if self.python_rng is not None:
                random.setstate(self.python_rng)
            if self.numpy_rng is not None:
                try:
                    import numpy as np

                    np.random.set_state(self.numpy_rng)
                except (ImportError, ModuleNotFoundError):
                    pass
            torch.set_grad_enabled(self.grad_enabled)
        return False


def _validate_probe_result(result: Mapping[str, Any]) -> None:
    """Fail closed if the base probe callback did not cover its required panel."""
    if int(result.get("batch_count", 0)) != 4 or int(result.get("batch_size", 0)) != 64:
        raise rt.RuntimeConfigError(
            "probe suite must observe exactly four fixed batches of 64 rows"
        )
    panel = result.get("fixed_panel")
    if not isinstance(panel, Mapping) or not bool(panel.get("available", False)):
        raise rt.RuntimeConfigError("probe suite did not report the fixed QID panel")
    if int(panel.get("count", 0)) != 256:
        raise rt.RuntimeConfigError("probe suite fixed QID panel must contain 256 QIDs")
    probe = result.get("probe")
    if not isinstance(probe, Mapping):
        raise rt.RuntimeConfigError("probe suite result is missing base probes")
    losses = probe.get("loss_gradients")
    terms = losses.get("loss_terms") if isinstance(losses, Mapping) else None
    required_losses = {"rank", "evidence", "support", "transition", "endpoint"}
    if not isinstance(terms, Mapping) or not required_losses.issubset(terms):
        missing = sorted(required_losses - set(terms or {}))
        raise rt.RuntimeConfigError(f"probe suite missing loss probes: {missing}")
    for key in (
        "three_fields",
        "candidate_membership",
        "score_identity",
        "parameter_groups",
    ):
        if key not in probe:
            raise rt.RuntimeConfigError(f"probe suite missing structural probe: {key}")


def run_probe_suite(
    suite: Any,
    *,
    model: nn.Module,
    bundle: Any,
    config: Any,
    device: torch.device,
    epoch: int,
    output_dir: Path,
    fixed_panel: Mapping[str, Any] | None,
    phase: str,
) -> dict[str, Any]:
    """Invoke base_engineering's observer and write exactly one JSON receipt."""
    output_dir.mkdir(parents=True, exist_ok=True)
    receipt = output_dir / f"probe_e{int(epoch):03d}.json"
    if receipt.exists() or receipt.is_symlink():
        raise FileExistsError(f"refusing overwrite of probe receipt: {receipt}")
    request = ProbeRequest(
        model, bundle, config, device, int(epoch), output_dir, fixed_panel, str(phase)
    )
    with _ProbeGuard(model):
        callback = getattr(suite, "run", None)
        result = callback(request) if callable(callback) else suite(request)
    if result is None:
        result = {}
    if not isinstance(result, Mapping):
        raise TypeError("ProbeSuite.run(request) must return a mapping")
    payload = {
        "schema": "eventfieldnet_trifield_base_v1_probe_receipt_v2",
        "epoch": int(epoch),
        "phase": str(phase),
        "status": "observed",
        "result": dict(result),
    }
    _atomic_json(receipt, payload)
    return payload


def _identity(
    raw: Mapping[str, Any],
    config_path: Path,
    output: Path,
    mode: str,
    roots: Sequence[Path],
) -> dict[str, Any]:
    targets = []
    for key in ("model_factory", "data_factory", "evaluator_factory"):
        if raw.get(key) is not None:
            targets.append(rt._spec_parts(raw[key])[0])
    return {
        "schema": "eventfieldnet_trifield_base_v1_run_identity_v1",
        "status": "initialized",
        "mode": mode,
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "host": platform.node(),
        "python": sys.executable,
        "torch": torch.__version__,
        "started_unix": time.time(),
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "output_dir": str(output),
        "factories": targets,
        "source_roots": [str(path) for path in roots],
        "scratch": {"base_checkpoint": None, "resume": False, "teacher": None},
    }


def _fresh_output(path: Path) -> None:
    if path.is_symlink():
        raise rt.RuntimeConfigError(f"output directory is a symlink: {path}")
    path.mkdir(parents=True, exist_ok=True)
    items = [item for item in path.iterdir() if item.name != ".gitkeep"]
    if items:
        raise FileExistsError(f"refusing nonempty output directory: {path}")


def _start(
    path: Path,
    raw: Mapping[str, Any],
    config_path: Path,
    mode: str,
    roots: Sequence[Path],
) -> None:
    identity = _identity(raw, config_path, path, mode, roots)
    _atomic_json(path / "run_identity.json", identity)
    _atomic_json(path / "config.json", dict(raw))
    _status(path, "initialized", mode=mode, config_sha256=identity["config_sha256"])


def _status(path: Path, status: str, **fields: Any) -> None:
    _atomic_json(
        path / "run_status.json",
        {
            "schema": "eventfieldnet_trifield_base_v1_run_status_v1",
            "status": str(status),
            "pid": os.getpid(),
            "updated_unix": time.time(),
            **fields,
        },
    )


def _progress(path: Path, **fields: Any) -> None:
    payload = {
        "schema": "eventfieldnet_trifield_base_v1_progress_v1",
        "pid": os.getpid(),
        "updated_unix": time.time(),
        **fields,
    }
    _atomic_json(path / "progress.json", payload)
    print(json.dumps(rt._json_value(payload), sort_keys=True), flush=True)


class _StepTracker:
    def __init__(
        self, model: nn.Module, optimizer: torch.optim.Optimizer, output: Path
    ):
        self.model, self.optimizer, self.output = model, optimizer, output
        self.steps, self.epoch, self.loss, self.finite = 0, 0, None, True
        self.model_had = "compute_loss" in model.__dict__
        self.optimizer_had = "step" in optimizer.__dict__
        self.model_old = model.__dict__.get("compute_loss")
        self.optimizer_old = optimizer.__dict__.get("step")

    def install(self) -> None:
        original_compute, original_step = self.model.compute_loss, self.optimizer.step
        tracker = self

        def compute(*args: Any, **kwargs: Any) -> Any:
            result = original_compute(*args, **kwargs)
            value = getattr(result, "loss", None)
            if torch.is_tensor(value) and value.ndim == 0:
                tracker.loss = float(value.detach().float().cpu())
                tracker.finite = bool(torch.isfinite(value.detach()).item())
            return result

        def step(*args: Any, **kwargs: Any) -> Any:
            result = original_step(*args, **kwargs)
            tracker.steps += 1
            if tracker.steps == 1:
                _progress(
                    tracker.output,
                    status="first_optimizer_step",
                    epoch=tracker.epoch,
                    optimizer_step=tracker.steps,
                    loss=tracker.loss,
                    finite=tracker.finite,
                )
            return result

        self.model.__dict__["compute_loss"] = compute
        self.optimizer.__dict__["step"] = step

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def record(self, epoch: int, metrics: Mapping[str, Any]) -> None:
        self.epoch = int(epoch)
        _progress(
            self.output,
            status="epoch_complete",
            epoch=int(epoch),
            optimizer_step=self.steps,
            loss=self.loss,
            finite=self.finite,
            metrics=dict(metrics),
        )

    def uninstall(self) -> None:
        if self.model_had:
            self.model.__dict__["compute_loss"] = self.model_old
        else:
            self.model.__dict__.pop("compute_loss", None)
        if self.optimizer_had:
            self.optimizer.__dict__["step"] = self.optimizer_old
        else:
            self.optimizer.__dict__.pop("step", None)


def _bundle(value: Any) -> Any:
    cls = getattr(importlib.import_module("training.contracts"), "DataBundle")
    if not isinstance(value, cls):
        raise TypeError("data_factory must return training.contracts.DataBundle")
    return value


def _config(path: Path) -> Any:
    cls = getattr(importlib.import_module("training.config"), "RunnerConfig")
    return cls.load(path)


def _seed(seed: int) -> None:
    try:
        getattr(importlib.import_module("training.runner"), "seed_everything")(
            int(seed)
        )
    except (ImportError, ModuleNotFoundError, AttributeError):
        random.seed(int(seed))
        torch.manual_seed(int(seed))


def _device(config: Any) -> torch.device:
    device = torch.device(str(config.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise rt.RuntimeConfigError(f"CUDA unavailable: {device}")
    return device


def _request(
    config: Any, checkpoint: Path, output: Path, config_path: Path, project_root: Path
) -> Any:
    cls = getattr(importlib.import_module("training.contracts"), "OfficialEvalRequest")
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
        output_dir=output,
        config_path=config_path,
        device=str(device),
        batch_size=int(batch),
        num_workers=int(config.num_workers),
        precision=str(config.precision),
    )


def _make_lifecycle(
    config: Any, output: Path, evaluator: Any, config_path: Path, project_root: Path
) -> Any:
    policy = getattr(config, "checkpointing", None)
    if policy is None or not bool(policy.enabled):
        raise rt.RuntimeConfigError("formal base requires checkpointing.enabled=true")
    cls = getattr(
        importlib.import_module("training.checkpointing"), "CheckpointLifecycle"
    )
    lifecycle = cls(
        output_dir=output,
        policy=policy,
        evaluator=evaluator,
        request_template=_request(
            config,
            output / "checkpoints" / "pending.pt",
            output / "official_by_epoch" / "pending",
            config_path,
            project_root,
        ),
    )
    lifecycle.initialize()
    return lifecycle


def _scratch_marker(
    output: Path, model: nn.Module, audit: Mapping[str, Any], config: Any
) -> None:
    _atomic_json(
        output / "scratch_initialized.json",
        {
            "schema": "eventfieldnet_trifield_base_v1_scratch_v1",
            "status": "initialized",
            "base_checkpoint": None,
            "resume": False,
            "teacher": None,
            "parameter_count": sum(1 for _ in model.parameters()),
            "parameter_numel": sum(int(p.numel()) for p in model.parameters()),
            "trainable_numel": sum(
                int(p.numel()) for p in model.parameters() if p.requires_grad
            ),
            "optimizer_id_exact_cover": bool(
                audit.get("optimizer_id_exact_cover", False)
            ),
            "epochs": int(config.epochs),
            "precision": str(config.precision),
        },
    )


def _save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(dict(payload), tmp)
    os.replace(tmp, path)


def _lrs(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    try:
        fn = getattr(
            importlib.import_module("training.audit"), "current_learning_rates"
        )
        return dict(fn(optimizer))
    except (ImportError, ModuleNotFoundError, AttributeError):
        return {
            str(group.get("name", index)): float(group["lr"])
            for index, group in enumerate(optimizer.param_groups)
        }


def _drift(model: nn.Module, before: Mapping[str, Tensor]) -> Mapping[str, Any]:
    try:
        fn = getattr(importlib.import_module("training.audit"), "parameter_drift")
        return dict(fn(model, before))
    except (ImportError, ModuleNotFoundError, AttributeError):
        result = {}
        for name, value in model.named_parameters():
            delta = (value.detach().cpu() - before[name]).abs()
            result[name] = {
                "max_abs": float(delta.max()) if delta.numel() else 0.0,
                "l2": float(torch.linalg.vector_norm(delta.float()))
                if delta.numel()
                else 0.0,
                "trainable": bool(value.requires_grad),
            }
        return result


def _frozen(drift: Mapping[str, Mapping[str, Any]]) -> None:
    try:
        getattr(importlib.import_module("training.audit"), "assert_frozen_zero_drift")(
            drift
        )
    except (ImportError, ModuleNotFoundError, AttributeError):
        bad = [
            name
            for name, value in drift.items()
            if not value["trainable"] and value["max_abs"] != 0.0
        ]
        if bad:
            raise AssertionError(f"frozen parameters changed: {bad[:20]}")


def run_training(
    config_path: str | Path,
    *,
    remote_root: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    raw = _load_json(config_path)
    rt.assert_scratch_config(raw)
    if str(raw.get("precision", "bf16")) != "bf16":
        raise rt.RuntimeConfigError("formal base training requires precision=bf16")
    _, roots = install_import_roots(raw, remote_root=remote_root, strict=True)
    config = _config(config_path)
    probe_epochs = _probe_epochs(raw, config)
    panel = _fixed_panel(raw, config_path)
    output = (
        Path(output_dir).resolve()
        if output_dir is not None
        else Path(str(config.output_dir)).resolve()
    )
    _fresh_output(output)
    _start(output, raw, config_path, "train", roots)
    _seed(int(config.seed))
    device = _device(config)
    try:
        with _PythonPath(roots):
            bundle = _bundle(rt.instantiate_factory(config.data_factory, config))
            model = rt.instantiate_factory(config.model_factory, config)
            if not isinstance(model, nn.Module):
                raise TypeError("model_factory must return torch.nn.Module")
            model.to(device)
            optimizer, audit = rt.build_optimizer(
                model,
                lr=float(raw.get("learning_rate", 1e-4)),
                weight_decay=float(config.weight_decay),
            )
            registered = rt.registered_trainable_ids(model)
            updates = len(bundle.train_loader)
            schedule_epochs = int(raw.get("schedule_epochs", config.epochs))
            if schedule_epochs < 1 or schedule_epochs > int(config.epochs):
                raise rt.RuntimeConfigError(
                    f"schedule_epochs must be in [1, epochs], got {schedule_epochs}"
                )
            warmup_epochs = int(config.warmup_epochs)
            if warmup_epochs > schedule_epochs:
                raise rt.RuntimeConfigError(
                    f"warmup_epochs={warmup_epochs} exceeds schedule_epochs={schedule_epochs}"
                )
            scheduler = rt.create_scheduler(
                optimizer,
                total_steps=schedule_epochs * updates,
                warmup_steps=warmup_epochs * updates,
            )
            _atomic_json(output / "optimizer_audit.json", audit)
            _scratch_marker(output, model, audit, config)
            resolved = output / "config.json"
            evaluator = rt.evaluator_from_config(config)
            lifecycle = _make_lifecycle(
                config, output, evaluator, resolved, _project_root(raw)
            )
            suite = _probe_suite(raw, config)
            _status(
                output,
                "running",
                epoch=0,
                optimizer_step=0,
                schedule_epochs=schedule_epochs,
                warmup_epochs=warmup_epochs,
                config_sha256=_sha256(config_path),
            )
            run_probe_suite(
                suite,
                model=model,
                bundle=bundle,
                config=config,
                device=device,
                epoch=0,
                output_dir=output / "probes",
                fixed_panel=panel,
                phase="pre_update",
            )
            try:
                before = getattr(
                    importlib.import_module("training.audit"), "snapshot_parameters"
                )(model)
            except (ImportError, ModuleNotFoundError, AttributeError):
                before = {
                    name: value.detach().cpu().clone()
                    for name, value in model.named_parameters()
                }
            tracker = _StepTracker(model, optimizer, output)
            tracker.install()
            try:
                final_val = None
                for epoch in range(1, int(config.epochs) + 1):
                    tracker.set_epoch(epoch)
                    start = _lrs(optimizer)
                    train_metrics, train_phase = rt.run_official_epoch(
                        model=model,
                        bundle=bundle,
                        loader=bundle.train_loader,
                        config=config,
                        device=device,
                        epoch=epoch,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        teacher=None,
                        registered_ids=registered,
                    )
                    val_metrics, val_phase = rt.run_official_epoch(
                        model=model,
                        bundle=bundle,
                        loader=bundle.val_loader,
                        config=config,
                        device=device,
                        epoch=epoch,
                        optimizer=None,
                        scheduler=None,
                        teacher=None,
                        registered_ids=registered,
                    )
                    final_val = val_metrics
                    record = {
                        "schema": "eventfieldnet_trifield_base_v1_history_v1",
                        "epoch": epoch,
                        "selection": "fixed_final_with_independent_best",
                        "learning_rates_start": start,
                        "learning_rates_end": _lrs(optimizer),
                        "train_phase": train_phase,
                        "val_phase": val_phase,
                        "train": train_metrics,
                        "val": val_metrics,
                    }
                    rt._append_jsonl(output / "history.jsonl", record)
                    if epoch in probe_epochs:
                        run_probe_suite(
                            suite,
                            model=model,
                            bundle=bundle,
                            config=config,
                            device=device,
                            epoch=epoch,
                            output_dir=output / "probes",
                            fixed_panel=panel,
                            phase="post_validation",
                        )
                    official = lifecycle.process_epoch(
                        epoch=epoch,
                        checkpoint_payload=rt.checkpoint_payload(
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            epoch=epoch,
                            config=config,
                            val_metrics=val_metrics,
                        ),
                        metadata={
                            "train": train_metrics,
                            "val": val_metrics,
                            "learning_rates_start": start,
                            "learning_rates_end": _lrs(optimizer),
                        },
                    )
                    _atomic_json(
                        output / f"epoch_{epoch:03d}_receipt.json",
                        {**record, "official": official},
                    )
                    tracker.record(
                        epoch,
                        {
                            "train_loss": train_metrics.get("loss"),
                            "val_loss": val_metrics.get("loss"),
                            "official": official.get("metrics"),
                        },
                    )
                    _status(
                        output,
                        "running",
                        epoch=epoch,
                        optimizer_step=tracker.steps,
                        schedule_epochs=schedule_epochs,
                        warmup_epochs=warmup_epochs,
                        official=official.get("metrics"),
                    )
                try:
                    drift = _drift(model, before)
                    _frozen(drift)
                    _atomic_json(output / "parameter_drift.json", drift)
                except Exception:
                    raise
                summary = lifecycle.finalize(int(config.epochs))
                final = {
                    "schema": "eventfieldnet_trifield_base_v1_final_v1",
                    "status": "complete",
                    "epoch": int(config.epochs),
                    "epochs_budget": int(config.epochs),
                    "selection": "fixed_final_with_independent_best",
                    "checkpoint": str(output / "last.pt"),
                    "val": final_val,
                    "best_val": summary,
                    "official": summary.get("last_official"),
                    "optimizer_audit": audit,
                }
                _atomic_json(output / "final_metrics.json", final)
                _status(
                    output,
                    "complete",
                    epoch=int(config.epochs),
                    optimizer_step=tracker.steps,
                    best_epoch=summary.get("best_epoch"),
                    official=summary.get("last_official"),
                )
                return final
            finally:
                tracker.uninstall()
    except Exception as exc:
        _status(output, "failed", error_type=type(exc).__name__, error=str(exc))
        raise


def run_smoke(
    config_path: str | Path,
    *,
    remote_root: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Independent fresh two-step official-runner smoke plus one evaluator call."""
    config_path = Path(config_path).resolve()
    raw = _load_json(config_path)
    rt.assert_scratch_config(raw)
    _, roots = install_import_roots(raw, remote_root=remote_root, strict=True)
    config = _config(config_path)
    panel = _fixed_panel(raw, config_path)
    output = (
        Path(output_dir).resolve()
        if output_dir
        else Path(str(config.output_dir))
        .resolve()
        .with_name(Path(str(config.output_dir)).name + ".smoke")
    )
    _fresh_output(output)
    _start(output, raw, config_path, "smoke", roots)
    _seed(int(config.seed))
    device = _device(config)
    try:
        with _PythonPath(roots):
            bundle = _bundle(rt.instantiate_factory(config.data_factory, config))
            model = rt.instantiate_factory(config.model_factory, config)
            if not isinstance(model, nn.Module):
                raise TypeError("model_factory must return torch.nn.Module")
            model.to(device)
            optimizer, audit = rt.build_optimizer(
                model,
                lr=float(raw.get("learning_rate", 1e-4)),
                weight_decay=float(config.weight_decay),
            )
            registered = rt.registered_trainable_ids(model)
            scheduler = rt.create_scheduler(optimizer, total_steps=2, warmup_steps=0)
            _atomic_json(output / "optimizer_audit.json", audit)
            _scratch_marker(output, model, audit, config)
            if raw.get("probe_factory") is not None:
                run_probe_suite(
                    _probe_suite(raw, config),
                    model=model,
                    bundle=bundle,
                    config=config,
                    device=device,
                    epoch=0,
                    output_dir=output / "probes",
                    fixed_panel=panel,
                    phase="smoke_pre_update",
                )
            tracker = _StepTracker(model, optimizer, output)
            tracker.install()
            try:
                metrics, phase = rt.run_official_epoch(
                    model=model,
                    bundle=bundle,
                    loader=itertools.islice(bundle.train_loader, 2),
                    config=config,
                    device=device,
                    epoch=0,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    teacher=None,
                    registered_ids=registered,
                )
            finally:
                tracker.uninstall()
            steps = int(metrics.get("optimizer_steps", 0))
            if steps != 2:
                raise rt.RuntimeConfigError(
                    f"smoke expected 2 optimizer steps, got {steps}"
                )
            checkpoint = output / "checkpoints" / "smoke.pt"
            _save(
                rt.checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=0,
                    config=config,
                    val_metrics=metrics,
                ),
                checkpoint,
            )
            evaluator = rt.evaluator_from_config(config)
            official = dict(
                evaluator.evaluate(
                    _request(
                        config,
                        checkpoint,
                        output / "official",
                        output / "config.json",
                        _project_root(raw),
                    )
                )
            )
            result = {
                "schema": "eventfieldnet_trifield_base_v1_smoke_v1",
                "status": "pass",
                "model_factory": rt._spec_parts(raw["model_factory"])[0],
                "steps": steps,
                "finite": bool(torch.isfinite(torch.tensor(float(metrics["loss"])))),
                "train_metrics": metrics,
                "phase": phase,
                "official": official,
                "checkpoint": str(checkpoint),
                "formal_fresh_init": True,
            }
            _atomic_json(output / "smoke_result.json", result)
            _status(output, "complete", optimizer_step=steps, official=official)
            return result
    except Exception as exc:
        _status(output, "failed", error_type=type(exc).__name__, error=str(exc))
        raise


def materialize_config(
    template_path: str | Path,
    output_path: str | Path,
    *,
    model_target: str = DEFAULT_MODEL_TARGET,
    model_kwargs: Mapping[str, Any] | None = None,
    probe_target: str | None = "field_core.probes:build_probe_suite",
    probe_kwargs: Mapping[str, Any] | None = None,
    output_dir: str | None = None,
    root: str | None = None,
    project_root: str | None = None,
    wrapper: str | None = None,
    epochs: int = 50,
    batch_size: int | None = None,
    device: str | None = None,
    learning_rate: float = 1e-4,
    warmup_epochs: int | None = 9,
    schedule_epochs: int | None = 180,
    fixed_panel_path: str | None = None,
) -> Path:
    """Materialize only live runner/data/evaluator fields for the new base.

    Historical templates contain old experiment contracts and loss declarations.
    They are intentionally not copied into the new checkpoint config.
    """
    source = _load_json(Path(template_path).resolve())
    data_factory = source.get("data_factory")
    if not isinstance(data_factory, (str, Mapping)):
        raise rt.RuntimeConfigError("template must provide data_factory")
    source_output = output_dir if output_dir is not None else source.get("output_dir")
    if source_output in (None, ""):
        raise rt.RuntimeConfigError("output_dir is required")
    source_evaluator = source.get("evaluator_factory")
    evaluator = dict(source_evaluator) if isinstance(source_evaluator, Mapping) else {}
    evaluator["target"] = DEFAULT_EVALUATOR_TARGET
    evaluator_kwargs = dict(evaluator.get("kwargs") or {})
    if root is not None:
        evaluator_kwargs["root"] = str(root)
    if project_root is not None:
        evaluator_kwargs["project_root"] = str(project_root)
    if wrapper is not None:
        evaluator_kwargs["wrapper"] = str(wrapper)
    evaluator["kwargs"] = evaluator_kwargs
    if not evaluator_kwargs.get("root"):
        raise rt.RuntimeConfigError("evaluator root is required")
    effective_batch = int(
        batch_size if batch_size is not None else source.get("batch_size", 64)
    )
    effective_device = str(
        device if device is not None else source.get("device", "cuda:0")
    )
    candidate: dict[str, Any] = {
        "schema": "eventfieldnet_trifield_base_v1_server81_formal_config_v1",
        "model_factory": {
            "target": str(model_target),
            "kwargs": dict(model_kwargs or {}),
        },
        "data_factory": json.loads(json.dumps(data_factory)),
        "output_dir": str(source_output),
        "seed": int(source.get("seed", 2026)),
        "epochs": int(epochs),
        "batch_size": effective_batch,
        "grad_accumulation": 1,
        "num_workers": int(source.get("num_workers", 2)),
        "device": effective_device,
        "precision": "bf16",
        "warmup_epochs": int(9 if warmup_epochs is None else warmup_epochs),
        "schedule_epochs": int(180 if schedule_epochs is None else schedule_epochs),
        "grad_clip": float(source.get("grad_clip", 1.0)),
        "weight_decay": float(source.get("weight_decay", 1.0e-4)),
        "learning_rate": float(learning_rate),
        "teacher": {"factory": None, "end_epoch": 0},
        "evaluator_factory": evaluator,
        "target_keys": list(source.get("target_keys") or ["gt_spans", "gt_span_mask"]),
        "source_roots": [],
        "fixed_final": True,
        "resume": False,
        "fail_on_frozen_drift": bool(source.get("fail_on_frozen_drift", True)),
        "base_checkpoint": None,
        "checkpointing": {
            "enabled": True,
            "keep_epoch_checkpoints": False,
            "stop_on_evaluation_failure": True,
            "official_device": effective_device,
            "official_batch_size": effective_batch,
            "evaluator_version": "repository_official_validation_trifield_base_v1",
        },
        "probe_epochs": list(DEFAULT_PROBE_EPOCHS),
    }
    panel = (
        fixed_panel_path
        if fixed_panel_path is not None
        else source.get("fixed_panel_path")
    )
    if panel not in (None, ""):
        candidate["fixed_panel_path"] = str(panel)
    if probe_target is not None:
        candidate["probe_factory"] = {
            "target": str(probe_target),
            "kwargs": dict(probe_kwargs or {}),
        }
    candidate["trifield_base_contract"] = {
        "schema": "eventfieldnet_trifield_base_v1_contract_v1",
        "initialization": "scratch_random_initialization",
        "checkpoint_inheritance": False,
        "teacher": None,
        "ema": False,
        "projection": False,
        "veto": False,
        "old_loss_contracts_removed": True,
        "loss_names": ["rank", "evidence", "support", "transition", "endpoint"],
        "loss_weights": {
            "rank": 1.0,
            "evidence": 0.1,
            "support": 0.1,
            "transition": 0.1,
            "endpoint": 0.1,
        },
        "original_coordinates": True,
        "candidate_membership_unchanged": True,
        "epochs": int(epochs),
        "schedule_epochs": int(candidate["schedule_epochs"]),
        "warmup_epochs": int(candidate["warmup_epochs"]),
        "probe_epochs": list(DEFAULT_PROBE_EPOCHS),
        "fixed_panel_batches": 4,
        "fixed_panel_batch_size": 64,
        "checkpoint_retention": ["best_val.pt", "last.pt", "probes"],
        "official_evaluator": DEFAULT_EVALUATOR_TARGET,
        "config_source_sanitized": True,
    }
    rt.assert_scratch_config(candidate)
    if candidate["epochs"] < 1 or candidate["schedule_epochs"] < 1:
        raise rt.RuntimeConfigError("epochs and schedule_epochs must be positive")
    if candidate["schedule_epochs"] > candidate["epochs"]:
        raise rt.RuntimeConfigError("schedule_epochs cannot exceed epochs")
    if (
        candidate["warmup_epochs"] < 0
        or candidate["warmup_epochs"] > candidate["schedule_epochs"]
    ):
        raise rt.RuntimeConfigError("warmup_epochs must be within schedule_epochs")
    output = Path(output_path).resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    tmp.write_text(
        json.dumps(candidate, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, output)
    return output


def _json_arg(value: str | None) -> dict[str, Any]:
    if value in (None, ""):
        return {}
    result = json.loads(value)
    if not isinstance(result, dict):
        raise ValueError("expected JSON object")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="field_core child runtime")
    commands = parser.add_subparsers(dest="command", required=True)
    materialize = commands.add_parser("materialize")
    materialize.add_argument("--template", type=Path, required=True)
    materialize.add_argument("--output", type=Path, required=True)
    materialize.add_argument("--model-target", default=DEFAULT_MODEL_TARGET)
    materialize.add_argument("--model-kwargs")
    materialize.add_argument(
        "--probe-target", default="field_core.probes:build_probe_suite"
    )
    materialize.add_argument("--probe-kwargs")
    materialize.add_argument("--output-dir")
    materialize.add_argument("--root")
    materialize.add_argument("--project-root")
    materialize.add_argument("--wrapper")
    materialize.add_argument("--epochs", type=int, default=50)
    materialize.add_argument("--batch-size", type=int)
    materialize.add_argument("--device")
    materialize.add_argument("--learning-rate", type=float, default=1e-4)
    materialize.add_argument("--warmup-epochs", type=int, default=9)
    materialize.add_argument("--schedule-epochs", type=int, default=180)
    materialize.add_argument("--fixed-panel-path")
    for name in ("smoke", "train", "child"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--remote-root")
        command.add_argument("--output-dir")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "materialize":
        output = materialize_config(
            args.template,
            args.output,
            model_target=args.model_target,
            model_kwargs=_json_arg(args.model_kwargs),
            probe_target=args.probe_target,
            probe_kwargs=_json_arg(args.probe_kwargs),
            output_dir=args.output_dir,
            root=args.root,
            project_root=args.project_root,
            wrapper=args.wrapper,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device,
            learning_rate=args.learning_rate,
            warmup_epochs=args.warmup_epochs,
            schedule_epochs=args.schedule_epochs,
            fixed_panel_path=args.fixed_panel_path,
        )
        print(json.dumps({"status": "pass", "config": str(output)}))
        return 0
    if args.command == "smoke":
        result = run_smoke(
            args.config, remote_root=args.remote_root, output_dir=args.output_dir
        )
    else:
        result = run_training(
            args.config, remote_root=args.remote_root, output_dir=args.output_dir
        )
    print(json.dumps(rt._json_value(result), sort_keys=True))
    return 0


__all__ = [
    "DEFAULT_EVALUATOR_TARGET",
    "DEFAULT_MODEL_TARGET",
    "DEFAULT_PROBE_EPOCHS",
    "ProbeRequest",
    "install_import_roots",
    "materialize_config",
    "run_probe_suite",
    "run_smoke",
    "run_training",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
