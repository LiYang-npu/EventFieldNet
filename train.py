"""Train the final EventFieldNet recipe from scratch or resume an epoch boundary."""

import argparse
import os
from pathlib import Path

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(name, "1")
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"

from eventfieldnet.engine import load_config, train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "configs/qvhighlights.json")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", type=Path, help="A committed last.pt from the same run")
    args = parser.parse_args()
    raw = load_config(args.config, seed=args.seed, output=args.output, device=args.device)
    train(raw, resume=args.resume)


if __name__ == "__main__":
    main()
