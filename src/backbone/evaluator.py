"""Official evaluator subprocess adapter for Stage55."""

from __future__ import annotations
import json, re, subprocess, sys
from dataclasses import dataclass
from typing import Any, Dict, Mapping
from training.contracts import OfficialEvalRequest


def _flat(x: Mapping[str, Any], prefix: str = "") -> Dict[str, Any]:
    out = {}
    for k, v in x.items():
        name = f"{prefix}.{k}" if prefix else str(k)
        out[name] = v
        out[str(k)] = v
        if isinstance(v, Mapping):
            out.update(_flat(v, name))
    return out


ALIASES = {
    "MR-mAP": ("mrmapfullavg", "mrmap", "mrfullmap"),
    "R1@0.5": ("mrr1full05", "r105", "mrr105"),
    "R1@0.7": ("mrr1full07", "r107", "mrr107"),
    "mAP@0.75": ("mrmapfull075", "map075", "mrmap075"),
    "Long-mAP": ("mrmaplongavg", "longmap"),
    "Middle-mAP": ("mrmapmiddleavg", "middlemap"),
    "Short-mAP": ("mrmapshortavg", "shortmap"),
    "HD-mAP": ("hlmapverygood", "hdmap"),
    "Hit@1": ("hlhit1verygood", "hit1"),
    "Oracle-mIoU": ("oracle", "oraclemiou", "oraclemioutop30"),
    "Top1-mIoU": ("top1", "top1miou"),
    "Pearson": ("pearson", "scoreioupearson"),
    "Spearman": ("spearman", "scoreiouspearman"),
}


def _slug(x: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", x.lower())


def normalize_repository_metrics(raw: Mapping[str, Any]) -> Dict[str, float]:
    flat = _flat(raw)
    numeric = {k: float(v) for k, v in flat.items() if isinstance(v, (int, float))}
    slugs = {_slug(k): v for k, v in numeric.items()}
    result = {}
    for canonical, aliases in ALIASES.items():
        value = next((slugs[a] for a in aliases if a in slugs), None)
        if value is None:
            raise KeyError(f"official metrics missing {canonical}")
        if canonical in {"HD-mAP", "Oracle-mIoU", "Top1-mIoU"} and abs(value) <= 1.5:
            value *= 100.0
        result[canonical] = value
    return result


@dataclass
class SegmentOfficialEvalAdapter:
    wrapper: str
    project_root: str
    root: str

    def evaluate(self, request: OfficialEvalRequest) -> Mapping[str, float]:
        request.output_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
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
        result = subprocess.run(cmd, check=False)
        if result.returncode:
            raise RuntimeError(f"Stage55 official evaluator exited {result.returncode}")
        path = request.output_dir / "metrics.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        return normalize_repository_metrics(
            json.loads(path.read_text(encoding="utf-8"))
        )
