"""R68 isolated optimizer schedule, sharing the unmodified official runner.

The learning rate used for update ``step+1`` is the inherited LambdaLR value,
times0.5 for actual parent-only optimizer groups from step10*updates_per_epoch.
No loss scaling, set_epoch hook, checkpoint inheritance, or source-file edit.
"""

from __future__ import annotations
import copy, hashlib, json, random, warnings
import numpy as np
import torch

LATE_MODES = frozenset(("late_parent", "neutral_late_parent"))
SCHEMA = "r68_actual_parent_lr_v1"


def _tensor_hash(rows):
    digest = hashlib.sha256()
    for name, tensor in rows:
        x = tensor.detach().cpu().contiguous()
        digest.update(str(name).encode())
        digest.update(str(x.dtype).encode())
        digest.update(str(tuple(x.shape)).encode())
        digest.update(x.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _baseline_factor(runtime, step):
    cfg = runtime.schedule_probe_config
    assert cfg["mode"] == "milestone"
    count = int(cfg["updates_per_epoch"])
    completed = int(step) + 1
    warmup = int(cfg["warmup_epochs"]) * count
    if warmup and completed <= warmup:
        return completed / float(warmup)
    return float(cfg["gamma"]) ** sum(
        completed > int(epoch) * count for epoch in cfg["milestones"]
    )


def _expected_rows(runtime, step=None):
    install = runtime.r68_runtime_installation
    step = runtime.scheduler.last_epoch if step is None else int(step)
    base_factor = _baseline_factor(runtime, step)
    epoch = step // install["updates_per_epoch"] + 1
    mult = 0.5 if install["late_parent_enabled"] and epoch >= 11 else 1.0
    assert float(runtime.model.r68_parent_lr_multiplier(epoch)) == mult
    rows = []
    for index, (group, base, meta) in enumerate(
        zip(
            runtime.optimizer.param_groups,
            runtime.scheduler.base_lrs,
            install["groups"],
        )
    ):
        factor = mult if meta["parent_only"] else 1.0
        expected = float(base) * base_factor * factor
        actual = float(group["lr"])
        rows.append(
            dict(
                index=index,
                name=group.get("name", str(index)),
                base_lr=float(base),
                parent_only=meta["parent_only"],
                parent_multiplier=factor,
                baseline_factor=base_factor,
                actual_lr=actual,
                expected_lr=expected,
                absolute_error=abs(actual - expected),
                parameter_count=len(group["params"]),
                parameter_names_sha256=meta["parameter_names_sha256"],
                cumulative_actual_lr=group.get("round31_cumulative_lr"),
                exposure_updates=group.get("round31_exposure_updates"),
            )
        )
    assert len(rows) == len(runtime.optimizer.param_groups)
    assert all(x["absolute_error"] < 1e-12 for x in rows), rows
    return rows


def install_runtime(runtime, raw):
    """Install once immediately after _build_phase_runtime; idempotent.

    Return a serializable installation proof. CPU/smoke/formal share this path.
    The exact inherited R67 reference is an explicit no-op.
    """
    model = runtime.model
    if not hasattr(model, "r68_mode"):
        return dict(status="reference_noop", reason="Non-R68 exact inherited reference")
    mode = model.r68_mode
    assert raw["model_factory"]["kwargs"]["r68_spec"] == {"mode": mode}
    cfg = raw.get("train", {})
    assert (
        cfg.get("schema") == SCHEMA and cfg.get("entrypoint") == "eventfieldnet.train"
    )
    assert cfg.get("late_parent_start_epoch") == 11
    late = mode in LATE_MODES
    assert cfg.get("late_parent_multiplier") == (0.5 if late else 1.0)
    assert (
        bool(model.r68_late_parent_enabled) == late
        and model.r68_late_parent_start_epoch == 11
    )
    assert raw["precision"] == "fp32" and int(raw.get("grad_accumulation", 1)) == 1
    assert raw["lr_schedule_mode"] == "milestone" and raw["lr_schedule_milestones"] == [
        10,
        20,
        40,
    ]
    assert raw["warmup_epochs"] == 3 and raw["lr_schedule_gamma"] == 0.3
    assert not raw.get("resume") and raw.get("base_checkpoint") is None
    assert len(raw["phases"]) == 1 and raw["phases"][0]["start_epoch"] == 0
    scheduler = runtime.scheduler
    assert isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR)
    if hasattr(runtime, "r68_runtime_installation"):
        assert runtime.r68_runtime_installation["mode"] == mode
        _expected_rows(runtime)
        return runtime.r68_runtime_installation
    updates = len(runtime.bundle.train_loader)
    assert updates > 0 and updates == runtime.schedule_probe_config["updates_per_epoch"]
    assert scheduler.last_epoch == 0, "Install before the first optimizer update"
    names = {id(p): n for n, p in model.named_parameters() if p.requires_grad}
    parent = {id(p) for p in model.parent_model.parameters() if p.requires_grad}
    assert parent and parent <= set(names)
    groups = []
    used = []
    original = list(scheduler.lr_lambdas)
    assert len(original) == len(runtime.optimizer.param_groups)
    for group in runtime.optimizer.param_groups:
        identities = [id(p) for p in group["params"]]
        used.extend(identities)
        parent_count = sum(p in parent for p in identities)
        assert parent_count in (0, len(identities)), (
            "Parent and extension parameters share an optimizer group"
        )
        listed = [names[p] for p in identities]
        groups.append(
            dict(
                name=group.get("name"),
                parent_only=parent_count > 0,
                parameter_count=len(identities),
                parameter_ids=identities,
                parameter_names=listed,
                parameter_names_sha256=hashlib.sha256(
                    json.dumps(listed).encode()
                ).hexdigest(),
            )
        )
    assert len(used) == len(set(used)) and set(used) == set(names), (
        "Optimizer membership is not exact"
    )
    assert any(g["parent_only"] for g in groups) and any(
        not g["parent_only"] for g in groups
    )
    before = [float(g["lr"]) for g in runtime.optimizer.param_groups]
    if late:
        wrapped = []
        for function, meta in zip(original, groups):

            def factor(
                step, base_function=function, parent_only=meta["parent_only"], n=updates
            ):
                base = base_function(step)
                return base * (0.5 if parent_only and int(step) >= 10 * n else 1.0)

            wrapped.append(factor)
        scheduler.lr_lambdas = wrapped
    proof = dict(
        status="installed",
        schema=SCHEMA,
        mode=mode,
        late_parent_enabled=late,
        updates_per_epoch=updates,
        first_halved_scheduler_step=10 * updates,
        first_halved_absolute_epoch=11,
        parent_multiplier_after_boundary=0.5 if late else 1.0,
        inherited_schedule="warmup3; milestones10,20,40; gamma0.3",
        groups=groups,
        weights_inherited=False,
        source_files_modified=False,
    )
    runtime.r68_runtime_installation = proof
    runtime.r68_original_lr_lambdas = original
    assert before == [float(g["lr"]) for g in runtime.optimizer.param_groups]
    _expected_rows(runtime)
    return proof


def _plain(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return True
    if isinstance(value, (tuple, list)):
        return all(_plain(x) for x in value)
    if isinstance(value, dict):
        return all(
            isinstance(k, (str, int, float, bool)) and _plain(v)
            for k, v in value.items()
        )
    return False


def _phase_attributes(model):
    # set_epoch delegates through inherited modules and mutates ordinary epoch,
    # routing and phase fields. Do not deepcopy nn.Module's registries/hooks.
    return [
        (
            module,
            {
                k: copy.deepcopy(v)
                for k, v in vars(module).items()
                if _phase_attribute(k, v)
            },
        )
        for module in model.modules()
    ]


def _phase_attribute(key, value):
    return (
        key not in ("_parameters", "_buffers", "_modules")
        and "hook" not in key
        and _plain(value)
    )


def scheduler_boundary_contract(runtime, raw):
    """Inspect real optimizer LR at e10/e11 without optimizer/weight updates.

    Uses the real full training-loader length. Restores scheduler/LRs/phase
    attributes/RNG and proves parameter+buffer identity before returning.
    """
    proof = install_runtime(runtime, raw)
    assert proof["status"] == "installed"
    from training.runner import set_model_epoch

    scheduler = runtime.scheduler
    optimizer = runtime.optimizer
    model = runtime.model
    original_scheduler = copy.deepcopy(scheduler.state_dict())
    original_lrs = [float(g["lr"]) for g in optimizer.param_groups]
    original_model = _tensor_hash(model.state_dict().items())
    original_grad = _tensor_hash(
        (n, p.grad) for n, p in model.named_parameters() if p.grad is not None
    )
    original_optimizer = copy.deepcopy(optimizer.state_dict())
    original_phase = _phase_attributes(model)
    original_torch = torch.get_rng_state().clone()
    original_numpy = np.random.get_state()
    original_python = random.getstate()
    original_cuda = (
        [s.clone() for s in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_initialized()
        else None
    )
    n = proof["updates_per_epoch"]
    observations = []
    try:
        for step in (9 * n, 10 * n - 1, 10 * n, 11 * n, 20 * n):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                scheduler.step(epoch=step)
            epoch = step // n + 1
            rows = _expected_rows(runtime)
            before = [g["lr"] for g in optimizer.param_groups]
            phases = []
            for training in (True, False, True, False):
                phases.append(
                    set_model_epoch(
                        model, epoch, training, set(runtime.registered_trainable_ids)
                    )
                )
                assert before == [g["lr"] for g in optimizer.param_groups], (
                    "set_epoch compounded/changed LR"
                )
                assert scheduler.last_epoch == step, "set_epoch advanced scheduler"
            observations.append(
                dict(
                    scheduler_step=step,
                    absolute_epoch=epoch,
                    groups=rows,
                    repeated_train_val_lr_unchanged=True,
                    phase_call_count=len(phases),
                )
            )
        assert _tensor_hash(model.state_dict().items()) == original_model, (
            "Boundary audit changed model tensors"
        )
        assert (
            _tensor_hash(
                (n, p.grad) for n, p in model.named_parameters() if p.grad is not None
            )
            == original_grad
        )

        # No optimizer.step is executed. Check all existing Adam moment tensors
        # and non-tensor state recursively, then leave their objects untouched.
        def equal(a, b):
            if isinstance(a, torch.Tensor):
                return isinstance(b, torch.Tensor) and torch.equal(a, b)
            if isinstance(a, dict):
                return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
            if isinstance(a, (tuple, list)):
                return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
            return a == b

        current = optimizer.state_dict()
        for group, before in zip(
            current["param_groups"], original_optimizer["param_groups"]
        ):
            group["lr"] = before["lr"]
        assert equal(current, original_optimizer), (
            "Boundary audit changed optimizer state"
        )
    finally:
        scheduler.load_state_dict(original_scheduler)
        for group, lr in zip(optimizer.param_groups, original_lrs):
            group["lr"] = lr
        for module, saved in original_phase:
            current = [k for k, v in vars(module).items() if _phase_attribute(k, v)]
            for key in current:
                if key not in saved:
                    delattr(module, key)
            for key, value in saved.items():
                setattr(module, key, value)
        torch.set_rng_state(original_torch)
        np.random.set_state(original_numpy)
        random.setstate(original_python)
        if original_cuda is not None:
            torch.cuda.set_rng_state_all(original_cuda)
    assert _tensor_hash(model.state_dict().items()) == original_model
    assert [float(g["lr"]) for g in optimizer.param_groups] == original_lrs
    assert scheduler.state_dict() == original_scheduler
    assert torch.equal(torch.get_rng_state(), original_torch)
    current_numpy = np.random.get_state()
    assert (
        current_numpy[0] == original_numpy[0]
        and np.array_equal(current_numpy[1], original_numpy[1])
        and current_numpy[2:] == original_numpy[2:]
    )
    assert random.getstate() == original_python
    if original_cuda is not None:
        assert all(
            torch.equal(a, b)
            for a, b in zip(torch.cuda.get_rng_state_all(), original_cuda)
        )
    for (module, saved), (_, now) in zip(original_phase, _phase_attributes(model)):
        assert saved == now
    _expected_rows(runtime)
    return dict(
        status="passed",
        schema=SCHEMA,
        mode=model.r68_mode,
        updates_per_epoch=n,
        observations=observations,
        actual_optimizer_group_lrs_verified=True,
        actual_parent_parameter_ids_verified=True,
        repeated_train_val_idempotent=True,
        optimizer_steps=0,
        parameters_unchanged=True,
        model_phase_attributes_restored=True,
        scheduler_restored=True,
        optimizer_state_unchanged=True,
        rng_restored=True,
        initial_final_model_sha256=original_model,
    )


def probe_actual_schedule(runtime, before, expected_updates):
    rows = _expected_rows(runtime)
    actual = runtime.scheduler.last_epoch - before
    assert actual == expected_updates, "Scheduler/update count diverged"
    return dict(
        runtime.schedule_probe_config,
        scheduler_last_epoch=runtime.scheduler.last_epoch,
        optimizer_updates=actual,
        expected_updates=expected_updates,
        groups=rows,
        formula_match=True,
        r68_schema=SCHEMA,
        r68_mode=runtime.model.r68_mode,
        next_update_absolute_epoch=runtime.scheduler.last_epoch
        // runtime.r68_runtime_installation["updates_per_epoch"]
        + 1,
    )


def install_runner_hooks():
    """Process-local facade only; inherited source files remain unchanged."""
    from .runtime import runner

    if getattr(runner, "r68_runtime_hooks_installed", False):
        return runner
    original_build = runner._build_phase_runtime
    original_probe = runner.probe_actual_schedule

    def build(*args, **kwargs):
        runtime, config, roots, audit = original_build(*args, **kwargs)
        raw = kwargs.get("raw", args[1] if len(args) > 1 else None)
        proof = install_runtime(runtime, raw)
        audit = dict(audit, train=proof)
        return runtime, config, roots, audit

    def probe(runtime, before, expected_updates):
        if hasattr(runtime, "r68_runtime_installation"):
            return probe_actual_schedule(runtime, before, expected_updates)
        return original_probe(runtime, before, expected_updates)

    runner._build_phase_runtime = build
    runner.probe_actual_schedule = probe
    runner.r68_runtime_hooks_installed = True
    return runner


def main():
    runner = install_runner_hooks()
    return runner.cli_main()


if __name__ == "__main__":
    raise SystemExit(main())
