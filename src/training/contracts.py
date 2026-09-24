"""Runtime contracts used by both Stage47 model plugins."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Iterable,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)

import torch
from torch import Tensor, nn


@dataclass
class PreparedBatch:
    """Inputs and supervision are separated before model.forward is called."""

    inputs: Mapping[str, Any]
    targets: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    batch_size: int = 1


BatchAdapter = Callable[[Any, torch.device], PreparedBatch]


@dataclass
class DataBundle:
    train_loader: Iterable[Any]
    val_loader: Iterable[Any]
    prepare_batch: BatchAdapter
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class LossResult:
    loss: Tensor
    metrics: Mapping[str, Tensor | float] = field(default_factory=dict)


@dataclass
class ParameterGroup:
    name: str
    params: Sequence[nn.Parameter]
    lr: float
    weight_decay: Optional[float] = None


@runtime_checkable
class Stage47Model(Protocol):
    """A plugin model. Ground truth never appears in forward or decode."""

    def train(self, mode: bool = True) -> Any: ...

    def eval(self) -> Any: ...

    def to(self, device: torch.device | str) -> Any: ...

    def parameters(self, recurse: bool = True) -> Iterable[nn.Parameter]: ...

    def named_parameters(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterable[tuple[str, nn.Parameter]]: ...

    def state_dict(self, *args: Any, **kwargs: Any) -> Mapping[str, Tensor]: ...

    def __call__(self, inputs: Mapping[str, Any]) -> Any: ...

    def set_epoch(self, epoch: int, training: bool) -> Mapping[str, Any] | str | None:
        """Set the GT-free execution phase before any forward in an epoch.

        Implementations must not change requires_grad or optimizer membership.
        Phase-specific freezing uses gradient/path gates while all
        pre-registered parameters remain in the optimizer contract.
        """
        ...

    def compute_loss(
        self,
        outputs: Any,
        batch: PreparedBatch,
        teacher_outputs: Optional[Any],
        epoch: int,
    ) -> LossResult: ...

    def decode(self, outputs: Any, inputs: Mapping[str, Any]) -> Any: ...

    def diagnostics(
        self, outputs: Any, batch: PreparedBatch
    ) -> Mapping[str, Tensor | float]: ...

    def parameter_groups(self) -> Sequence[ParameterGroup]: ...


@dataclass(frozen=True)
class OfficialEvalRequest:
    project_root: Path
    checkpoint: Path
    output_dir: Path
    config_path: Path
    device: str
    batch_size: int
    num_workers: int
    precision: str


@runtime_checkable
class OfficialEvalAdapter(Protocol):
    def evaluate(self, request: OfficialEvalRequest) -> Mapping[str, float]: ...


ModelFactory = Callable[..., Stage47Model]
DataFactory = Callable[..., DataBundle]
