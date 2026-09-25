"""Unchanged SG-derived metric API used by the checkpoint evaluator."""
from typing import Any
import numpy as np
import torch
from sg_components.metrics.metrics_collection import get_metrics

def tensor_to_python(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu())
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: tensor_to_python(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [tensor_to_python(item) for item in value]
    return value
