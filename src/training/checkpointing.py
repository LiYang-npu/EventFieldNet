"""Reusable per-epoch checkpoint evaluation and retention lifecycle."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

import torch

from .config import CheckpointingConfig
from .contracts import OfficialEvalRequest
from .evaluator import normalize_official_metrics


SCHEMA_VERSION = "eventfield_checkpoint_lifecycle_v1"
SELECTION_METRICS = ("MR-mAP", "mAP@0.75", "R1@0.7")


def checkpoint_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selection_key(
    metrics: Mapping[str, float], epoch: int
) -> tuple[float, float, float, int]:
    """MR -> AP75 -> R1@0.7 -> earlier epoch, all deterministic."""
    return (
        float(metrics[SELECTION_METRICS[0]]),
        float(metrics[SELECTION_METRICS[1]]),
        float(metrics[SELECTION_METRICS[2]]),
        -int(epoch),
    )


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise TypeError("checkpoint lifecycle metadata tensors must be scalar")
        return value.detach().cpu().item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    raise TypeError(
        f"checkpoint lifecycle metadata is not JSON-safe: {type(value).__name__}"
    )


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(_json_safe(value), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(_json_safe(value), sort_keys=True, ensure_ascii=False) + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def _load_checkpoint(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _verify_loadable(path: Path) -> None:
    try:
        _load_checkpoint(path)
    except Exception as exc:
        raise RuntimeError(f"checkpoint is not reloadable: {path}") from exc


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(dict(payload), temporary)
    _verify_loadable(temporary)
    os.replace(temporary, path)


def _atomic_verified_copy(source: Path, target: Path, expected_sha256: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    shutil.copy2(source, temporary)
    if checkpoint_sha256(temporary) != expected_sha256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"checkpoint copy hash mismatch: {source} -> {target}")
    _verify_loadable(temporary)
    os.replace(temporary, target)


@dataclass(frozen=True)
class LifecyclePaths:
    output_dir: Path
    checkpoints_dir: Path
    official_by_epoch_dir: Path
    results_jsonl: Path
    results_json: Path
    best_checkpoint: Path
    last_checkpoint: Path
    best_manifest: Path
    cleanup_manifest: Path
    state: Path

    @classmethod
    def create(cls, output_dir: str | Path) -> "LifecyclePaths":
        root = Path(output_dir).resolve()
        return cls(
            output_dir=root,
            checkpoints_dir=root / "checkpoints",
            official_by_epoch_dir=root / "official_by_epoch",
            results_jsonl=root / "all_checkpoint_official_results.jsonl",
            results_json=root / "all_checkpoint_official_results.json",
            best_checkpoint=root / "best_val.pt",
            last_checkpoint=root / "last.pt",
            best_manifest=root / "best_val_manifest.json",
            cleanup_manifest=root / "checkpoint_cleanup_manifest.json",
            state=root / "checkpoint_lifecycle_state.json",
        )


class CheckpointLifecycle:
    """Evaluate every epoch online and retain only best/last by default."""

    def __init__(
        self,
        *,
        output_dir: str | Path,
        policy: CheckpointingConfig,
        evaluator: Any,
        request_template: OfficialEvalRequest,
        checkpoint_loader: Callable[[Path], Any] = _load_checkpoint,
    ) -> None:
        if not policy.enabled:
            raise ValueError("CheckpointLifecycle requires checkpointing.enabled=true")
        if evaluator is None or not callable(getattr(evaluator, "evaluate", None)):
            raise TypeError("official evaluator must provide evaluate(request)")
        self.paths = LifecyclePaths.create(output_dir)
        self.policy = policy
        self.evaluator = evaluator
        self.request_template = request_template
        self.checkpoint_loader = checkpoint_loader
        self.records: list[Dict[str, Any]] = []
        self.best_record: Optional[Dict[str, Any]] = None
        self.deleted: list[Dict[str, Any]] = []
        self.failed_checkpoints: list[str] = []
        self._initialized = False

    def initialize(self) -> None:
        """Create a fresh ledger; never silently append to an old experiment."""
        self.paths.output_dir.mkdir(parents=True, exist_ok=True)
        self.paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.paths.official_by_epoch_dir.mkdir(parents=True, exist_ok=True)
        protected = (
            self.paths.results_jsonl,
            self.paths.results_json,
            self.paths.best_manifest,
            self.paths.cleanup_manifest,
            self.paths.state,
            self.paths.best_checkpoint,
            self.paths.last_checkpoint,
        )
        existing = [str(path) for path in protected if path.exists()]
        if existing:
            raise FileExistsError(
                f"refusing to overwrite checkpoint lifecycle artifacts: {existing}"
            )
        self.paths.results_jsonl.write_text("", encoding="utf-8")
        self._write_results()
        self._write_state(status="initialized")
        self._write_cleanup(status="in_progress")
        self._initialized = True

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("CheckpointLifecycle.initialize() must be called first")

    def _write_results(self) -> None:
        _atomic_json(
            self.paths.results_json,
            {
                "schema": SCHEMA_VERSION,
                "selection_metrics": list(SELECTION_METRICS),
                "selection_policy": "MR_then_AP75_then_R07_then_earlier_epoch",
                "evaluator_version": self.policy.evaluator_version,
                "records": self.records,
            },
        )

    def _write_state(self, status: str) -> None:
        _atomic_json(
            self.paths.state,
            {
                "schema": SCHEMA_VERSION,
                "status": status,
                "completed_epochs": [row["epoch"] for row in self.records],
                "successful_epochs": [
                    row["epoch"] for row in self.records if row["status"] == "success"
                ],
                "failed_epochs": [
                    row["epoch"] for row in self.records if row["status"] != "success"
                ],
                "best_epoch": None
                if self.best_record is None
                else self.best_record["epoch"],
                "best_checkpoint": str(self.paths.best_checkpoint),
                "last_checkpoint": str(self.paths.last_checkpoint),
                "updated_unix": time.time(),
            },
        )

    def _write_best_manifest(self) -> None:
        if self.best_record is None:
            return
        _atomic_json(
            self.paths.best_manifest,
            {
                "schema": SCHEMA_VERSION,
                "selection_policy": "MR_then_AP75_then_R07_then_earlier_epoch",
                "selection_metrics": list(SELECTION_METRICS),
                "best_epoch": self.best_record["epoch"],
                "metrics": self.best_record["metrics"],
                "selection_key": self.best_record["selection_key"],
                "source_checkpoint_sha256": self.best_record["checkpoint_sha256"],
                "best_val": {
                    "path": str(self.paths.best_checkpoint),
                    "sha256": checkpoint_sha256(self.paths.best_checkpoint),
                },
                "selected_from": self.best_record["checkpoint_original_path"],
                "evaluator_version": self.policy.evaluator_version,
            },
        )

    def _write_cleanup(self, status: str) -> None:
        retained = []
        for role, path in (
            ("best_val", self.paths.best_checkpoint),
            ("last", self.paths.last_checkpoint),
        ):
            if path.is_file():
                retained.append(
                    {
                        "role": role,
                        "path": str(path),
                        "sha256": checkpoint_sha256(path),
                        "bytes": path.stat().st_size,
                    }
                )
        _atomic_json(
            self.paths.cleanup_manifest,
            {
                "schema": SCHEMA_VERSION,
                "status": status,
                "retained": retained,
                "deleted_epoch_checkpoints": self.deleted,
                "failed_checkpoints_retained": self.failed_checkpoints,
                "official_results": str(self.paths.results_json),
                "best_manifest": str(self.paths.best_manifest),
            },
        )

    def _record(self, row: Dict[str, Any]) -> None:
        self.records.append(row)
        _append_jsonl(self.paths.results_jsonl, row)
        self._write_results()

    def process_epoch(
        self,
        *,
        epoch: int,
        checkpoint_payload: Mapping[str, Any],
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Save, reload, officially evaluate, record, select, and clean one epoch."""
        self._ensure_initialized()
        epoch = int(epoch)
        if epoch < 1:
            raise ValueError("epoch must be positive")
        if self.records and epoch <= int(self.records[-1]["epoch"]):
            raise ValueError(
                "epochs must be processed once in strictly increasing order"
            )

        checkpoint = self.paths.checkpoints_dir / f"epoch_{epoch:03d}.pt"
        if checkpoint.exists():
            raise FileExistsError(checkpoint)
        _atomic_torch_save(checkpoint_payload, checkpoint)
        self.checkpoint_loader(checkpoint)
        digest = checkpoint_sha256(checkpoint)
        _atomic_verified_copy(checkpoint, self.paths.last_checkpoint, digest)

        official_dir = self.paths.official_by_epoch_dir / f"epoch_{epoch:03d}"
        request = replace(
            self.request_template,
            checkpoint=checkpoint,
            output_dir=official_dir,
        )
        started = time.time()
        try:
            raw_metrics = dict(self.evaluator.evaluate(request))
            metrics = normalize_official_metrics(raw_metrics, require_all=True)
        except Exception as exc:
            row = {
                "schema": SCHEMA_VERSION,
                "status": "evaluation_failed",
                "epoch": epoch,
                "checkpoint_original_path": str(checkpoint),
                "checkpoint_sha256": digest,
                "official_output_dir": str(official_dir),
                "evaluation_seconds": time.time() - started,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "metadata": _json_safe(metadata or {}),
            }
            self.failed_checkpoints.append(str(checkpoint))
            self._record(row)
            self._write_state(status="evaluation_failed")
            self._write_cleanup(status="evaluation_failed")
            if self.policy.stop_on_evaluation_failure:
                raise RuntimeError(
                    f"official evaluation failed at epoch {epoch}"
                ) from exc
            return row

        key = selection_key(metrics, epoch)
        improved = self.best_record is None or key > tuple(
            self.best_record["selection_key"]
        )
        row = {
            "schema": SCHEMA_VERSION,
            "status": "success",
            "epoch": epoch,
            "checkpoint_original_path": str(checkpoint),
            "checkpoint_sha256": digest,
            "checkpoint_bytes": checkpoint.stat().st_size,
            "official_output_dir": str(official_dir),
            "evaluation_seconds": time.time() - started,
            "metrics": metrics,
            "selection_key": list(key),
            "is_best_so_far": bool(improved),
            "metadata": _json_safe(metadata or {}),
        }
        if improved:
            _atomic_verified_copy(checkpoint, self.paths.best_checkpoint, digest)
            self.best_record = row
        row["best_epoch_after"] = self.best_record["epoch"]
        self._record(row)
        self._write_best_manifest()
        self._write_state(status="in_progress")

        if not self.policy.keep_epoch_checkpoints:
            size = checkpoint.stat().st_size
            checkpoint.unlink()
            self.deleted.append(
                {
                    "path": str(checkpoint),
                    "sha256": digest,
                    "bytes": size,
                    "epoch": epoch,
                }
            )
        self._write_cleanup(status="in_progress")
        return row

    def finalize(self, final_epoch: int) -> Dict[str, Any]:
        self._ensure_initialized()
        expected = list(range(1, int(final_epoch) + 1))
        actual = [int(row["epoch"]) for row in self.records]
        if actual != expected:
            raise RuntimeError(
                f"checkpoint ledger is not epoch-complete: {actual} != {expected}"
            )
        if self.best_record is None:
            raise RuntimeError(
                "no successful official evaluation; best_val.pt is unavailable"
            )
        _verify_loadable(self.paths.best_checkpoint)
        _verify_loadable(self.paths.last_checkpoint)
        failures = [row for row in self.records if row["status"] != "success"]
        status = "complete" if not failures else "complete_with_failures"
        self._write_state(status=status)
        self._write_cleanup(status=status)
        return {
            "status": status,
            "epochs": len(self.records),
            "best_epoch": self.best_record["epoch"],
            "best_metrics": self.best_record["metrics"],
            "best_checkpoint": str(self.paths.best_checkpoint),
            "last_checkpoint": str(self.paths.last_checkpoint),
            "last_official": self.records[-1].get("metrics"),
            "official_results": str(self.paths.results_json),
        }


__all__ = [
    "CheckpointLifecycle",
    "LifecyclePaths",
    "SCHEMA_VERSION",
    "SELECTION_METRICS",
    "checkpoint_sha256",
    "selection_key",
]
