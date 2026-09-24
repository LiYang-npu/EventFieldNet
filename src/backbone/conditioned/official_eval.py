"""Official repository evaluator with a config-resolved C3 extension factory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from backbone.conditioned.plugin import checkpoint_payload, resolve_factory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--amp", choices=("bf16", "fp32"), default="bf16")
    args = parser.parse_args()
    for path in (
        args.project_root,
        args.root,
        args.root / "src",
        args.root / "stage51",
        args.root / "src",
    ):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    payload = checkpoint_payload(args.checkpoint)
    config = payload.get("config", {})
    model_factory = config.get("model_factory", {})
    if not isinstance(model_factory, dict):
        raise TypeError("checkpoint config has no model_factory")
    target = str(model_factory["target"])
    kwargs = dict(model_factory.get("kwargs", {}))
    factory = resolve_factory(target)

    from scripts import event_field_official_eval as repository_eval

    repository_eval.build_model = lambda _kind: factory(None, **kwargs)
    sys.argv = [
        "event_field_official_eval.py",
        "--root",
        str(args.root),
        "--checkpoint",
        str(args.checkpoint),
        "--output",
        str(args.output),
        "--model-kind",
        "shared_full",
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--top-k",
        str(args.top_k),
        "--device",
        args.device,
        "--amp",
        args.amp,
    ]
    repository_eval.main()


if __name__ == "__main__":
    main()
