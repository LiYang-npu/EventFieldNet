"""Small CPU-only RNG helpers for deterministic local branches.

The parent runtime owns the process-level CUDA RNG.  Round31 model branches
must therefore seed only a CPU ``torch.Generator`` and, when a module
constructor needs the default CPU stream, install that state inside
``fork_rng(devices=[])``.  The caller's CPU state is restored by the context;
CUDA state is never seeded or rewritten by these helpers.
"""

from __future__ import annotations

import torch


def cpu_generator(seed: int) -> torch.Generator:
    """Return a deterministically seeded CPU generator without touching CUDA."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return generator


def install_cpu_generator_state(seed: int) -> torch.Generator:
    """Install a CPU generator state for code using the default CPU stream.

    This is intended to run inside ``torch.random.fork_rng(devices=[])``.
    ``torch.set_rng_state`` addresses the CPU stream only; unlike
    ``torch.manual_seed`` it does not seed the default CUDA generators.
    """

    generator = cpu_generator(seed)
    torch.set_rng_state(generator.get_state())
    return generator


__all__ = ["cpu_generator", "install_cpu_generator_state"]
