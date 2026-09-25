"""Fixed-budget FP32 training with separate retrieval and highlight selectors."""

from __future__ import annotations

import copy
import importlib
import json
import math
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from training.config import RunnerConfig
from training.contracts import OfficialEvalRequest
from .optim import build_optimizer


def instantiate(spec, config):
    module, name = spec.target.split(":", 1)
    return getattr(importlib.import_module(module), name)(config=config, **spec.kwargs)


def normalize_task(raw):
    """Resolve the task once and reject a conflicting model-head setting."""
    raw = copy.deepcopy(raw)
    task = raw.get("task", "mr_hd")
    if task not in ("mr_hd", "mr"):
        raise ValueError("task must be 'mr_hd' (retrieval + highlights) or 'mr' (retrieval only)")
    expected = task == "mr_hd"
    kwargs = raw["model_factory"].setdefault("kwargs", {})
    if "enable_highlight" in kwargs and (
        not isinstance(kwargs["enable_highlight"], bool) or kwargs["enable_highlight"] != expected
    ):
        raise ValueError("task conflicts with model_factory.kwargs.enable_highlight")
    raw["task"] = task
    kwargs["enable_highlight"] = expected
    return raw


def load_config(path, *, seed=None, output=None, device=None):
    """Expand public training options or read a previously materialized config."""
    root = Path(__file__).resolve().parents[1]
    supplied = json.loads(Path(path).read_text(encoding="utf-8"))
    if "model_factory" in supplied:
        # Saved run/checkpoint configs already contain their complete fixed recipe.
        raw = supplied
    else:
        training_keys = {
            "seed", "output_dir", "device", "precision", "epochs", "batch_size",
            "num_workers", "learning_rate", "weight_decay", "warmup_epochs",
            "lr_schedule_milestones", "lr_schedule_gamma", "grad_clip", "task",
        }
        extra, missing = set(supplied) - training_keys, (training_keys - {"task"}) - set(supplied)
        if extra or missing:
            raise ValueError(f"Training configuration keys: unknown={sorted(extra)}, missing={sorted(missing)}")
        raw = json.loads((Path(__file__).parent / "recipe.json").read_text(encoding="utf-8"))
        raw.update(supplied)
    raw = json.loads(json.dumps(raw).replace("@ROOT@", root.as_posix()))
    for name, value in (("seed", seed), ("output_dir", output), ("device", device)):
        if value is not None:
            raw[name] = str(Path(value).resolve()) if name == "output_dir" else value
    return normalize_task(raw)


def strict_fp32():
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def loader_generators(bundle):
    found = {}
    seen = set()
    for name in ("train_loader", "val_loader"):
        loader = getattr(bundle, name)
        for key, owner in ((name, loader), (name + ".sampler", loader.sampler),
                           (name + ".batch_sampler", loader.batch_sampler)):
            generator = getattr(owner, "generator", None)
            if isinstance(generator, torch.Generator) and id(generator) not in seen:
                found[key] = generator
                seen.add(id(generator))
    return found


def capture_rng(bundle):
    state = {"python": random.getstate(), "numpy": np.random.get_state(),
             "torch_cpu": torch.get_rng_state(),
             "loaders": {name: g.get_state() for name, g in loader_generators(bundle).items()}}
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state, bundle):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([x.cpu() for x in state["torch_cuda"]])
    generators = loader_generators(bundle)
    if set(state["loaders"]) != set(generators):
        raise ValueError("Checkpoint loader RNG streams do not match the configured loaders")
    for name, value in state["loaders"].items():
        generators[name].set_state(value.cpu())


def create_scheduler(optimizer, updates_per_epoch, warmup_epochs=3,
                     milestones=(10, 20, 40), gamma=0.3):
    """The value for update 1 is 1/warmup_steps; decay starts after each milestone."""
    warmup_steps = warmup_epochs * updates_per_epoch
    boundaries = [epoch * updates_per_epoch for epoch in milestones]

    def factor(step):
        completed = step + 1
        if warmup_steps and completed <= warmup_steps:
            return completed / float(warmup_steps)
        return gamma ** sum(completed > boundary for boundary in boundaries)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def gradients_finite(parameters):
    """Check all gradients with one device-to-host synchronization."""
    checks = [torch.isfinite(p.grad).all() for p in parameters if p.grad is not None]
    return not checks or bool(torch.stack(checks).all())


def metric_values(metrics):
    """Copy scalar metrics together, keeping each source device and dtype."""
    converted, groups = {}, {}
    for name, value in metrics.items():
        if torch.is_tensor(value):
            if value.numel() != 1:
                raise ValueError(f"Metric {name} must contain one scalar")
            groups.setdefault((value.device, value.dtype), []).append((name, value.detach().reshape(())))
        else:
            converted[name] = float(value)
    for items in groups.values():
        values = torch.stack([value for _, value in items]).cpu().tolist()
        for (name, _), value in zip(items, values):
            converted[name] = float(value)
    return {name: converted[name] for name in metrics}


def release_batch_references(model):
    """Discard same-batch field caches after loss, update and diagnostics finish."""
    # These are read only during the current forward/loss/diagnostics. The next
    # forward/wrong-query pass creates them again; none is a parameter or buffer.
    for name in ("_last_inputs", "_last_field_output", "_last_wrong_query_field"):
        if hasattr(model, name):
            setattr(model, name, None)


def run_epoch(model, bundle, loader, device, epoch, *, optimizer=None,
              scheduler=None, grad_clip=1.0, registered_trainable_ids=None):
    training = optimizer is not None
    model.train(training)
    before_ids = registered_trainable_ids if registered_trainable_ids is not None else {id(p) for p in model.parameters() if p.requires_grad}
    model.set_epoch(epoch, training)
    if before_ids != {id(p) for p in model.parameters() if p.requires_grad}:
        raise RuntimeError("set_epoch changed optimizer membership")
    parameters = [p for p in model.parameters() if p.requires_grad]
    if training:
        optimizer.zero_grad(set_to_none=True)
    totals, examples, steps = {}, 0, 0
    for batch_index, raw in enumerate(loader):
        batch = bundle.prepare_batch(raw, device)
        if set(batch.inputs) & set(batch.targets):
            raise ValueError("Targets must not be present in model forward inputs")
        with torch.set_grad_enabled(training):
            outputs = model(batch.inputs)
            result = model.compute_loss(outputs, batch, None, epoch)
            if result.loss.ndim or not torch.isfinite(result.loss):
                raise FloatingPointError(f"Invalid loss at epoch {epoch}, batch {batch_index}")
        if training:
            result.loss.backward()
            if not gradients_finite(parameters):
                raise FloatingPointError(f"Nonfinite gradient at epoch {epoch}, batch {batch_index}")
            torch.nn.utils.clip_grad_norm_(parameters, grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            steps += 1
        metrics = {"loss": result.loss, **dict(result.metrics)}
        metrics.update({"diag/" + key: value for key, value in model.diagnostics(outputs, batch).items()})
        examples += batch.batch_size
        for name, value in metric_values(metrics).items():
            if not math.isfinite(value):
                raise FloatingPointError(f"Nonfinite metric {name}")
            totals[name] = totals.get(name, 0.0) + value * batch.batch_size
        # The next forward must not overlap with the previous dense span tensors.
        # Backward/optimizer and diagnostics have finished; no required graph is detached.
        release_batch_references(model)
        del outputs, result, metrics, batch, raw
    if not examples:
        raise RuntimeError("Empty data loader")
    return {**{k: v / examples for k, v in totals.items()}, "optimizer_steps": steps}


def metric(metrics, *names):
    for name in names:
        if name in metrics and math.isfinite(float(metrics[name])):
            return float(metrics[name])
    raise KeyError(f"Required official metric missing: {names}")


def selection_keys(metrics, epoch, task="mr_hd"):
    if task not in ("mr_hd", "mr"):
        raise ValueError("task must be 'mr_hd' or 'mr'")
    mr = metric(metrics, "MR-mAP", "MR-mAP-Full_Avg")
    retrieval = [mr, metric(metrics, "mAP@0.75", "MR-mAP-Full@0.75"),
                 metric(metrics, "R1@0.7", "MR-full-R1@0.7"), -epoch]
    keys = {"mr": retrieval}
    if task == "mr_hd":
        keys["hd"] = [metric(metrics, "official_HD/HL-min-VeryGood/HL-mAP"),
                      metric(metrics, "official_HD/HL-min-VeryGood/HL-Hit1"), mr, -epoch]
    return keys


def atomic_save(path, payload):
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp)
    temp.replace(path)


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def evaluate(evaluator, checkpoint, output, config_path, config):
    request = OfficialEvalRequest(
        project_root=Path(__file__).resolve().parents[1], checkpoint=checkpoint,
        output_dir=output, config_path=config_path, device=config.device,
        batch_size=config.batch_size, num_workers=config.num_workers, precision="fp32")
    result = evaluator.evaluate(request)
    return dict(result.get("metrics", result))


def configure_training(model, raw):
    """Apply the options used by the historical runner at run_phase entry.

    The clean constructor returns a runtime ready for formal training. Direct
    comparison with the historical factory must apply its two corresponding
    _apply_* helpers too, since that factory returns an unconfigured runtime.
    """
    for name in ("negative_selection_mode", "rank_length_reweight_gain",
                 "rank_topk_restrict", "support_shift_relaxed_fallback"):
        if name in raw:
            if not hasattr(model, name):
                raise AttributeError(f"Model is missing configured option {name}")
            setattr(model, name, raw[name])


def build_runtime(raw, device=None):
    """Construct data, model and optimizer in the historical initialization order."""
    raw = normalize_task(raw)
    if device is not None:
        raw["device"] = str(device)
    if raw.get("precision") != "fp32" or raw.get("grad_accumulation", 1) != 1:
        raise ValueError("The released recipe uses FP32 and one optimizer update per batch")
    if raw.get("teacher", {}).get("factory"):
        raise ValueError("The released recipe has no teacher model")
    if raw.get("optimizer_lr_scale", 1.0) != 1.0:
        raise ValueError("The released recipe uses the model's explicit group learning rates")
    strict_fp32()
    config = RunnerConfig.from_dict({**raw, "resume": False})
    seed_everything(config.seed)
    device = torch.device(config.device)
    cuda_rng = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
    bundle = instantiate(config.data_factory, config)
    model = instantiate(config.model_factory, config).to(device)
    configure_training(model, raw)
    optimizer, optimizer_info = build_optimizer(model, lr=raw.get("learning_rate", 1e-4),
                                                weight_decay=config.weight_decay)
    scheduler = create_scheduler(optimizer, len(bundle.train_loader), config.warmup_epochs,
                                 raw.get("lr_schedule_milestones", [10, 20, 40]),
                                 raw.get("lr_schedule_gamma", 0.3))
    evaluator = instantiate(config.evaluator_factory, config)
    if cuda_rng is not None:
        torch.cuda.set_rng_state_all(cuda_rng)
    return SimpleNamespace(config=config, bundle=bundle, model=model, optimizer=optimizer,
                           scheduler=scheduler, evaluator=evaluator, device=device,
                           optimizer_info=optimizer_info,
                           registered_trainable_ids={id(p) for p in model.parameters() if p.requires_grad})


def train(raw, *, resume=None):
    """Resume only at a committed epoch boundary, including all RNG streams."""
    raw = normalize_task(raw)
    output = Path(raw["output_dir"])
    if not resume:
        output.mkdir(parents=True, exist_ok=False)
    elif not output.is_dir():
        raise FileNotFoundError(output)
    config_path = output / "config.json"
    if resume and Path(resume).resolve() != (output / "last.pt").resolve():
        raise ValueError("Resume must use the existing output directory's last.pt")
    runtime = build_runtime(raw)
    config, bundle, model = runtime.config, runtime.bundle, runtime.model
    optimizer, scheduler, evaluator = runtime.optimizer, runtime.scheduler, runtime.evaluator
    device = runtime.device
    completed, history, best = 0, [], {}
    if resume:
        saved = torch.load(resume, map_location="cpu", weights_only=False)
        if saved.get("format") != "eventfieldnet_training_v1":
            raise ValueError("Historical checkpoints support inference; full training resume requires this engine's last.pt")
        saved["config"] = normalize_task(saved["config"])
        ignored = {"output_dir", "device", "source_roots"}
        if {k: v for k, v in saved["config"].items() if k not in ignored} != {k: v for k, v in raw.items() if k not in ignored}:
            raise ValueError("Resume recipe differs from the saved training configuration")
        if saved["status"] != "complete_epoch":
            raise ValueError("Resume from a committed last.pt, not an unevaluated checkpoint")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        restore_rng(saved["rng"], bundle)
        completed, history, best = saved["epoch"], saved["history"], saved["best"]
        expected_selectors = {"mr", "hd"} if raw["task"] == "mr_hd" else {"mr"}
        if set(best) != expected_selectors:
            raise ValueError("Checkpoint selectors do not match its task")
        if scheduler.last_epoch != completed * len(bundle.train_loader):
            raise ValueError("Scheduler update count does not match the resumed data loader")
    write_json(config_path, raw)
    write_json(output / "optimizer.json", runtime.optimizer_info)
    for epoch in range(completed + 1, config.epochs + 1):
        train_metrics = run_epoch(model, bundle, bundle.train_loader, device, epoch,
                                  optimizer=optimizer, scheduler=scheduler, grad_clip=config.grad_clip,
                                  registered_trainable_ids=runtime.registered_trainable_ids)
        val_metrics = run_epoch(model, bundle, bundle.val_loader, device, epoch,
                                registered_trainable_ids=runtime.registered_trainable_ids)
        payload = dict(format="eventfieldnet_training_v1", status="pending_evaluation", epoch=epoch,
                       config=raw, model={k: v.detach().cpu() for k, v in model.state_dict().items()},
                       optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                       rng=capture_rng(bundle), history=copy.deepcopy(history), best=copy.deepcopy(best))
        pending = output / "pending.pt"
        atomic_save(pending, payload)
        try:
            evaluation_dir = output / "validation" / f"epoch_{epoch:03d}"
            # Preserve incomplete evaluation output if an interrupted epoch is replayed.
            retry = 0
            while evaluation_dir.exists():
                retry += 1
                evaluation_dir = output / "validation" / f"epoch_{epoch:03d}_retry{retry}"
            if device.type == "cuda":
                # The isolated evaluator builds its own model on this GPU. Release
                # unused cached blocks; the live training model/AdamW remain intact.
                torch.cuda.empty_cache()
            official = evaluate(evaluator, pending, evaluation_dir, config_path, config)
        finally:
            restore_rng(payload["rng"], bundle)
        keys = selection_keys(official, epoch, raw["task"])
        improved = [name for name, key in keys.items() if name not in best or key > best[name]["key"]]
        for name in improved:
            best[name] = dict(epoch=epoch, key=keys[name], metrics=official)
        history.append(dict(epoch=epoch, train=train_metrics, val=val_metrics, official=official))
        payload.update(status="complete_epoch", official=official,
                       history=copy.deepcopy(history), best=copy.deepcopy(best))
        for name in improved:
            atomic_save(output / f"best_{name}.pt", payload)
        atomic_save(output / "last.pt", payload)
        pending.unlink()
        write_json(output / "history.json", history)
        write_json(output / "best.json", best)
        print(json.dumps(dict(epoch=epoch, official=official)), flush=True)
    return best
