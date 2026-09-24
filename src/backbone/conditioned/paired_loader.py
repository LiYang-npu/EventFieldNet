"""Data factory with RNG streams isolated from model construction.

The repository's original training loader uses ``RandomSampler`` without an
explicit generator.  Its permutation therefore depends on how much global
PyTorch RNG a model constructor consumed.  That makes architecture/loss arms
see different examples in a different order even when ``config.seed`` is the
same.  This wrapper keeps the dataset and loader settings unchanged while
assigning dedicated, reproducible generators to sampling and worker seeding.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import RandomSampler

from .trifield_spanset_c3_plugin import build_repository_data_with_saliency


SCHEMA = "eventfieldnet_paired_loader_rng_v1"
_MAX_TORCH_SEED = (1 << 63) - 1
_STREAMS = {
    "train_sampler": 100_003,
    "train_workers": 200_003,
    "validation_workers": 300_007,
    "test_workers": 400_009,
}


def _derived_seed(base_seed: int, stream_offset: int) -> int:
    seed = (int(base_seed) * 1_000_003 + int(stream_offset)) % _MAX_TORCH_SEED
    return seed if seed > 0 else int(stream_offset)


def _generator(seed: int) -> torch.Generator:
    value = torch.Generator(device="cpu")
    value.manual_seed(int(seed))
    return value


def _install_train_generators(loader: Any, seed: int) -> None:
    sampler = getattr(loader, "sampler", None)
    if not isinstance(sampler, RandomSampler):
        raise TypeError(
            "paired RNG contract requires a torch RandomSampler for training; "
            f"got {type(sampler).__name__}"
        )
    loader.generator = _generator(_derived_seed(seed, _STREAMS["train_workers"]))
    sampler.generator = _generator(_derived_seed(seed, _STREAMS["train_sampler"]))


def _install_worker_generator(loader: Any, seed: int, stream: str) -> None:
    loader.generator = _generator(_derived_seed(seed, _STREAMS[stream]))


def build_repository_data_with_paired_rng(config: Any, root: str) -> Any:
    """Build the unchanged repository data bundle with arm-independent RNG."""

    bundle = build_repository_data_with_saliency(config, root=root)
    seed = int(config.seed)
    _install_train_generators(bundle.train_loader, seed)
    _install_worker_generator(bundle.val_loader, seed, "validation_workers")
    test_loader = getattr(bundle, "test_loader", None)
    if test_loader is not None:
        _install_worker_generator(test_loader, seed, "test_workers")
    return bundle


def paired_rng_contract(seed: int) -> dict[str, Any]:
    """Return the exact deterministic stream assignment for manifests/tests."""

    return {
        "schema": SCHEMA,
        "base_seed": int(seed),
        "train_sampler_seed": _derived_seed(seed, _STREAMS["train_sampler"]),
        "train_worker_seed": _derived_seed(seed, _STREAMS["train_workers"]),
        "validation_worker_seed": _derived_seed(seed, _STREAMS["validation_workers"]),
        "test_worker_seed": _derived_seed(seed, _STREAMS["test_workers"]),
        "model_global_rng_isolated_from_loader_order": True,
    }
