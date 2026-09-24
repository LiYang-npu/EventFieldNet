"""Strict optimizer, parameter-drift, and source-integrity audits."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import torch
from torch import Tensor, nn

from .contracts import ParameterGroup


def build_optimizer(
    model: nn.Module, groups: Sequence[ParameterGroup], default_weight_decay: float
) -> tuple[torch.optim.Optimizer, Dict[str, Any]]:
    if not groups:
        raise ValueError("model returned no optimizer groups")
    named = dict(model.named_parameters())
    trainable_ids = {id(p) for p in named.values() if p.requires_grad}
    owner = {id(p): name for name, p in named.items()}
    seen: list[int] = []
    payload = []
    audit_groups: Dict[str, Any] = {}
    group_names: set[str] = set()
    for group in groups:
        if group.name in group_names:
            raise ValueError(f"duplicate optimizer group name: {group.name}")
        group_names.add(group.name)
        params = list(group.params)
        if not params:
            raise ValueError(f"empty optimizer group: {group.name}")
        bad = [owner.get(id(p), "<unknown>") for p in params if not p.requires_grad]
        if bad:
            raise ValueError(
                f"frozen parameters placed in optimizer group {group.name}: {bad[:5]}"
            )
        ids = [id(p) for p in params]
        seen.extend(ids)
        weight_decay = (
            default_weight_decay
            if group.weight_decay is None
            else float(group.weight_decay)
        )
        payload.append(
            {
                "name": group.name,
                "params": params,
                "lr": float(group.lr),
                "weight_decay": weight_decay,
            }
        )
        audit_groups[group.name] = {
            "base_lr": float(group.lr),
            "weight_decay": weight_decay,
            "tensor_count": len(params),
            "numel": sum(int(p.numel()) for p in params),
            "names": [owner.get(id(p), "<unknown>") for p in params],
        }
    if len(seen) != len(set(seen)):
        raise ValueError("optimizer parameter IDs contain duplicates")
    seen_set = set(seen)
    if seen_set != trainable_ids:
        missing = [
            name
            for name, p in named.items()
            if p.requires_grad and id(p) not in seen_set
        ]
        extra = [owner.get(pid, "<unknown>") for pid in seen_set - trainable_ids]
        raise ValueError(
            f"optimizer coverage mismatch; missing={missing[:10]} extra={extra[:10]}"
        )
    optimizer = torch.optim.AdamW(payload)
    audit = {
        "groups": audit_groups,
        "optimizer_parameter_id_count": len(seen),
        "optimizer_unique_id_count": len(set(seen)),
        "optimizer_id_exact_cover": seen_set == trainable_ids,
        "total_trainable_numel": sum(
            int(p.numel()) for p in named.values() if p.requires_grad
        ),
        "total_parameter_numel": sum(int(p.numel()) for p in named.values()),
    }
    return optimizer, audit


def current_learning_rates(optimizer: torch.optim.Optimizer) -> Dict[str, float]:
    return {
        str(group.get("name", index)): float(group["lr"])
        for index, group in enumerate(optimizer.param_groups)
    }


def snapshot_parameters(model: nn.Module) -> Dict[str, Tensor]:
    return {
        name: value.detach().cpu().clone() for name, value in model.named_parameters()
    }


def parameter_drift(model: nn.Module, before: Mapping[str, Tensor]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for name, value in model.named_parameters():
        if name not in before:
            raise KeyError(f"parameter missing from initial snapshot: {name}")
        delta = (value.detach().cpu() - before[name]).abs()
        result[name] = {
            "max_abs": float(delta.max()) if delta.numel() else 0.0,
            "l2": float(torch.linalg.vector_norm(delta.float()))
            if delta.numel()
            else 0.0,
            "trainable": bool(value.requires_grad),
        }
    return result


def assert_frozen_zero_drift(drift: Mapping[str, Mapping[str, Any]]) -> None:
    offenders = [
        name
        for name, item in drift.items()
        if not item["trainable"] and item["max_abs"] != 0.0
    ]
    if offenders:
        raise AssertionError(f"frozen parameters changed: {offenders[:20]}")


def assert_equal_parameter_contract(models: Sequence[nn.Module]) -> Dict[str, Any]:
    if len(models) < 2:
        raise ValueError("at least two models are required")
    schemas = []
    for model in models:
        schemas.append(
            [
                (name, tuple(p.shape), bool(p.requires_grad))
                for name, p in model.named_parameters()
            ]
        )
    first = schemas[0]
    for index, schema in enumerate(schemas[1:], start=1):
        if schema != first:
            raise AssertionError(f"parameter schema differs for model index {index}")
    return {
        "models": len(models),
        "tensor_count": len(first),
        "total_numel": sum(int(p.numel()) for p in models[0].parameters()),
    }


def assert_equal_optimizer_contract(audits: Sequence[Mapping[str, Any]]) -> None:
    if len(audits) < 2:
        raise ValueError("at least two optimizer audits are required")

    def signature(audit: Mapping[str, Any]) -> Any:
        return [
            (name, item["tensor_count"], item["numel"], item["names"])
            for name, item in audit["groups"].items()
        ]

    first = signature(audits[0])
    for index, audit in enumerate(audits[1:], start=1):
        if signature(audit) != first:
            raise AssertionError(
                f"optimizer parameter contract differs for model index {index}"
            )


def hash_sources(
    roots: Iterable[str | Path], suffixes: tuple[str, ...] = (".py", ".json", ".sh")
) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for raw_root in roots:
        root = Path(raw_root).resolve()
        paths = (
            [root]
            if root.is_file()
            else sorted(
                p for p in root.rglob("*") if p.is_file() and p.suffix in suffixes
            )
        )
        for path in paths:
            result[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result
