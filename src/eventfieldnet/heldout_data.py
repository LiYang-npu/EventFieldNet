"""Evaluation-only loader; original feature preprocessing, heldout annotations."""

import copy
import dataclasses
import hashlib
import json
from pathlib import Path


def build(config, root, annotation, annotation_sha256):
    from backbone.conditioned.paired_loader import build_repository_data_with_paired_rng
    from feature_bridge.plugin import f0

    p = Path(annotation)
    assert hashlib.sha256(p.read_bytes()).hexdigest() == annotation_sha256
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    assert len({str(x["qid"]) for x in rows}) == len(rows)
    bundle = build_repository_data_with_paired_rng(config, root=root)
    dataset = copy.copy(bundle.val_loader.dataset)
    dataset.data_path = str(p)
    dataset.data = rows
    loader = f0.make_loader(dataset, config.batch_size, config.num_workers, False)
    loader.generator = bundle.val_loader.generator
    # No training loader is exposed by this evaluation-only adapter.
    return dataclasses.replace(bundle, train_loader=[], val_loader=loader)
