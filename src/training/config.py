"""JSON-serializable configuration for the shared Stage47 runner."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


@dataclass(frozen=True)
class FactoryConfig:
    target: str
    kwargs: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_value(cls, value: Mapping[str, Any] | str) -> "FactoryConfig":
        if isinstance(value, str):
            return cls(target=value)
        return cls(target=str(value["target"]), kwargs=dict(value.get("kwargs", {})))


@dataclass(frozen=True)
class TeacherConfig:
    factory: Optional[FactoryConfig] = None
    end_epoch: int = 5

    def weight(self, epoch: int) -> float:
        """Linear 1 -> 0 schedule. The teacher is never called at zero weight."""
        if self.factory is None or epoch < 1 or epoch > self.end_epoch:
            return 0.0
        if self.end_epoch <= 1:
            return 1.0 if epoch == 1 else 0.0
        return max(0.0, float(self.end_epoch - epoch) / float(self.end_epoch - 1))


@dataclass(frozen=True)
class CheckpointingConfig:
    """Online official-evaluation and checkpoint-retention policy.

    When enabled, every epoch is saved to a transient checkpoint, evaluated
    with the official validation adapter, recorded, and folded into best/last.
    """

    enabled: bool = False
    keep_epoch_checkpoints: bool = False
    stop_on_evaluation_failure: bool = True
    official_device: Optional[str] = None
    official_batch_size: Optional[int] = None
    evaluator_version: str = "repository_official_validation"

    def __post_init__(self) -> None:
        if self.official_batch_size is not None and self.official_batch_size < 1:
            raise ValueError("checkpointing.official_batch_size must be positive")


@dataclass(frozen=True)
class RunnerConfig:
    model_factory: FactoryConfig
    data_factory: FactoryConfig
    output_dir: str
    seed: int = 2026
    epochs: int = 40
    batch_size: int = 64
    grad_accumulation: int = 1
    num_workers: int = 2
    device: str = "cuda:0"
    precision: str = "bf16"
    warmup_epochs: int = 3
    grad_clip: float = 1.0
    weight_decay: float = 1.0e-4
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    evaluator_factory: Optional[FactoryConfig] = None
    target_keys: List[str] = field(default_factory=list)
    source_roots: List[str] = field(default_factory=list)
    fixed_final: bool = True
    resume: bool = False
    fail_on_frozen_drift: bool = True
    checkpointing: CheckpointingConfig = field(default_factory=CheckpointingConfig)

    def __post_init__(self) -> None:
        if not self.fixed_final:
            raise ValueError("Stage47 only permits fixed-final model selection")
        if self.resume:
            raise ValueError("resume is disabled for causal fixed-budget comparisons")
        if self.epochs < 1 or self.batch_size < 1 or self.grad_accumulation < 1:
            raise ValueError(
                "epochs, batch_size, and grad_accumulation must be positive"
            )
        if self.precision not in {"bf16", "fp32"}:
            raise ValueError("precision must be 'bf16' or 'fp32'")
        if self.warmup_epochs < 0 or self.warmup_epochs > self.epochs:
            raise ValueError("warmup_epochs must be in [0, epochs]")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RunnerConfig":
        teacher_raw = raw.get("teacher", {})
        teacher_factory = teacher_raw.get("factory")
        teacher = TeacherConfig(
            factory=FactoryConfig.from_value(teacher_factory)
            if teacher_factory
            else None,
            end_epoch=int(teacher_raw.get("end_epoch", 5)),
        )
        evaluator = raw.get("evaluator_factory")
        checkpointing_raw = raw.get("checkpointing", {})
        checkpointing = CheckpointingConfig(
            enabled=bool(checkpointing_raw.get("enabled", False)),
            keep_epoch_checkpoints=bool(
                checkpointing_raw.get("keep_epoch_checkpoints", False)
            ),
            stop_on_evaluation_failure=bool(
                checkpointing_raw.get("stop_on_evaluation_failure", True)
            ),
            official_device=(
                None
                if checkpointing_raw.get("official_device") is None
                else str(checkpointing_raw["official_device"])
            ),
            official_batch_size=(
                None
                if checkpointing_raw.get("official_batch_size") is None
                else int(checkpointing_raw["official_batch_size"])
            ),
            evaluator_version=str(
                checkpointing_raw.get(
                    "evaluator_version", "repository_official_validation"
                )
            ),
        )
        return cls(
            model_factory=FactoryConfig.from_value(raw["model_factory"]),
            data_factory=FactoryConfig.from_value(raw["data_factory"]),
            output_dir=str(raw["output_dir"]),
            seed=int(raw.get("seed", 2026)),
            epochs=int(raw.get("epochs", 40)),
            batch_size=int(raw.get("batch_size", 64)),
            grad_accumulation=int(raw.get("grad_accumulation", 1)),
            num_workers=int(raw.get("num_workers", 2)),
            device=str(raw.get("device", "cuda:0")),
            precision=str(raw.get("precision", "bf16")),
            warmup_epochs=int(raw.get("warmup_epochs", 3)),
            grad_clip=float(raw.get("grad_clip", 1.0)),
            weight_decay=float(raw.get("weight_decay", 1.0e-4)),
            teacher=teacher,
            evaluator_factory=FactoryConfig.from_value(evaluator)
            if evaluator
            else None,
            target_keys=list(raw.get("target_keys", [])),
            source_roots=list(raw.get("source_roots", [])),
            fixed_final=bool(raw.get("fixed_final", True)),
            resume=bool(raw.get("resume", False)),
            fail_on_frozen_drift=bool(raw.get("fail_on_frozen_drift", True)),
            checkpointing=checkpointing,
        )

    @classmethod
    def load(cls, path: str | Path) -> "RunnerConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
