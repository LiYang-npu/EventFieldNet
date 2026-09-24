"""Explicit adapters for the repository's official EventField evaluation."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence

from .contracts import OfficialEvalRequest


STANDARD_METRICS = (
    "MR-mAP",
    "R1@0.5",
    "R1@0.7",
    "mAP@0.75",
    "Long-mAP",
    "Middle-mAP",
    "Short-mAP",
    "HD-mAP",
    "Hit@1",
    "Oracle-mIoU",
    "Top1-mIoU",
    "Pearson",
    "Spearman",
)


METRIC_ALIASES = {
    "MR-mAP": ("MR-mAP", "mr_map", "MR full mAP"),
    "R1@0.5": ("R1@0.5", "r1_05", "MR R1@0.5"),
    "R1@0.7": ("R1@0.7", "r1_07", "MR R1@0.7"),
    "mAP@0.75": ("mAP@0.75", "ap75", "MR mAP@0.75"),
    "Long-mAP": ("Long-mAP", "long_map", "MR long mAP"),
    "Middle-mAP": ("Middle-mAP", "middle_map", "MR middle mAP"),
    "Short-mAP": ("Short-mAP", "short_map", "MR short mAP"),
    "HD-mAP": ("HD-mAP", "hd_map", "HL mAP"),
    "Hit@1": ("Hit@1", "hit1", "HL Hit1"),
    "Oracle-mIoU": ("Oracle-mIoU", "oracle_miou", "oracle"),
    "Top1-mIoU": ("Top1-mIoU", "top1_miou", "top1"),
    "Pearson": ("Pearson", "pearson"),
    "Spearman": ("Spearman", "spearman"),
}


def _flatten(raw: Mapping[str, Any], prefix: str = "") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in raw.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        flat[name] = value
        flat[str(key)] = value
        if isinstance(value, Mapping):
            flat.update(_flatten(value, name))
    return flat


def normalize_official_metrics(
    raw: Mapping[str, Any], require_all: bool = True
) -> Dict[str, float]:
    flat = _flatten(raw)
    result: Dict[str, float] = {}
    missing = []
    for canonical, aliases in METRIC_ALIASES.items():
        match = next(
            (
                flat[key]
                for key in aliases
                if key in flat and isinstance(flat[key], (int, float))
            ),
            None,
        )
        if match is None:
            missing.append(canonical)
        else:
            result[canonical] = float(match)
    if require_all and missing:
        raise KeyError(f"official evaluator did not emit required metrics: {missing}")
    return result


@dataclass
class RepositoryCliEvalAdapter:
    """Runs a model-specific wrapper around scripts.event_field_official_eval.

    The wrapper is required because the repository evaluator needs the plugin's
    model builder. This adapter never fabricates missing metrics.
    """

    wrapper: str
    project_root: str
    root: str
    extra_args: Sequence[str] = ()
    metrics_filename: str = "metrics.json"

    def evaluate(self, request: OfficialEvalRequest) -> Mapping[str, float]:
        request.output_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            self.wrapper,
            "--project-root",
            str(self.project_root or request.project_root),
            "--root",
            str(self.root),
            "--checkpoint",
            str(request.checkpoint),
            "--output",
            str(request.output_dir),
            "--device",
            request.device,
            "--batch-size",
            str(request.batch_size),
            "--num-workers",
            str(request.num_workers),
            "--amp",
            request.precision,
            *list(self.extra_args),
        ]
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            raise RuntimeError(
                f"official evaluation failed with exit code {completed.returncode}"
            )
        path = request.output_dir / self.metrics_filename
        if not path.exists():
            raise FileNotFoundError(f"official evaluator did not create {path}")
        return normalize_official_metrics(
            json.loads(path.read_text(encoding="utf-8")), require_all=True
        )
