"""Official-evaluation subprocess adapter and strict metric normalization."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Dict, Mapping

from training.contracts import OfficialEvalRequest


def _flatten(raw: Mapping[str, Any], prefix: str = "") -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in raw.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        result[name] = value
        result[str(key)] = value
        if isinstance(value, Mapping):
            result.update(_flatten(value, name))
    return result


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


ALIASES = {
    "MR-mAP": ("mrmapfullavg", "mrmap", "mrfullmap", "mrfullmeanap"),
    "R1@0.5": ("mrr1full05", "r105", "mrr105", "mrr1iou05", "mrfullr105"),
    "R1@0.7": ("mrr1full07", "r107", "mrr107", "mrr1iou07", "mrfullr107"),
    "mAP@0.75": ("mrmapfull075", "map075", "mrfullmap075", "mrmap075"),
    "Long-mAP": ("mrmaplongavg", "longmap", "mrlongmap"),
    "Middle-mAP": ("mrmapmiddleavg", "middlemap", "mrmiddlemap"),
    "Short-mAP": ("mrmapshortavg", "shortmap", "mrshortmap"),
    "HD-mAP": ("hlmapverygood", "hdmap", "hlmap", "hlminverygoodmap"),
    "Hit@1": ("hlhit1verygood", "hit1", "hlhit1", "hlminverygoodhit1"),
    "Oracle-mIoU": ("oracle", "oraclemiou", "oraclemioutop30"),
    "Top1-mIoU": ("top1", "top1miou"),
    "Pearson": ("pearson", "scoreioupearson"),
    "Spearman": ("spearman", "scoreiouspearman"),
}


def normalize_repository_metrics(raw: Mapping[str, Any]) -> Dict[str, float]:
    flat = _flatten(raw)
    numeric = {
        key: float(value)
        for key, value in flat.items()
        if isinstance(value, (int, float))
    }
    by_slug: Dict[str, float] = {}
    for key, value in numeric.items():
        # Preserve decimal points inside repository metric names (for example
        # MR-R1-Full_0.7); direct leaf keys are also emitted by _flatten.
        by_slug.setdefault(_slug(key), value)
    normalized: Dict[str, float] = {}
    missing = []
    for canonical, aliases in ALIASES.items():
        value = next((by_slug[name] for name in aliases if name in by_slug), None)
        if value is None:
            missing.append(canonical)
        else:
            if (
                canonical in {"HD-mAP", "Oracle-mIoU", "Top1-mIoU"}
                and abs(value) <= 1.5
            ):
                value *= 100.0
            normalized[canonical] = value
    if missing:
        raise KeyError(
            f"official metrics missing required values: {missing}; available={sorted(by_slug)}"
        )
    return normalized


@dataclass
class SegmentOfficialEvalAdapter:
    wrapper: str
    project_root: str
    root: str

    def evaluate(self, request: OfficialEvalRequest) -> Mapping[str, float]:
        request.output_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            self.wrapper,
            "--project-root",
            self.project_root,
            "--root",
            self.root,
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
        ]
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            raise RuntimeError(
                f"segment official evaluator failed with exit code {completed.returncode}"
            )
        path = request.output_dir / "metrics.json"
        if not path.exists():
            raise FileNotFoundError(path)
        return normalize_repository_metrics(
            json.loads(path.read_text(encoding="utf-8"))
        )


__all__ = ["SegmentOfficialEvalAdapter", "normalize_repository_metrics"]
