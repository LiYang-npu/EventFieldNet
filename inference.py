"""Evaluate one frozen checkpoint; MR and HD are reported from the same weights."""

import argparse
import json
import os
from pathlib import Path

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(name, "1")
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"

import torch

from eventfieldnet.engine import evaluate, instantiate, load_config, strict_fp32, write_json
from training.config import RunnerConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "configs/qvhighlights.json")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    strict_fp32()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_seed = int(checkpoint["config"]["seed"])
    if args.seed is not None and args.seed != checkpoint_seed:
        parser.error("--seed must match the checkpoint's original training seed")
    del checkpoint
    raw = load_config(args.config, seed=checkpoint_seed, output=args.output, device=args.device)
    args.output.mkdir(parents=True, exist_ok=False)
    config_path = args.output / "config.json"
    write_json(config_path, raw)
    config = RunnerConfig.from_dict(raw)
    evaluator = instantiate(config.evaluator_factory, config)
    metrics = evaluate(evaluator, args.checkpoint.resolve(), args.output / "evaluation", config_path, config)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
