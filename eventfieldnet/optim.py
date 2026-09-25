"""AdamW parameter groups and exact trainable-parameter coverage."""

from __future__ import annotations
from typing import Any, Mapping, Optional

import torch
from torch import nn


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
