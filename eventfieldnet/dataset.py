"""QVHighlights data preprocessing used by the fixed EventFieldNet recipe.

Feature normalization, 75-clip padding, training GT handling and loader settings
are preserved from the recorded runs. Independent sampler RNG is installed by
backbone.conditioned.paired_loader.
"""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Tuple
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from training.config import RunnerConfig
from training.contracts import DataBundle, PreparedBatch
from sg_components.dataset.collate import custom_collate, move_inputs_to_device
from sg_components.dataset.qvhighlights import QVHighlights
from sg_components.utils.span_utils import span_cxw_to_xx

def make_dataset(root: Path, split: str) -> QVHighlights:
    data_root = root / "data" / "qvhighlights"
    annotation = {
        "train": "highlight_train_release.jsonl",
        "val": "highlight_val_release.jsonl",
    }[split]
    return QVHighlights(
        data_path=str(data_root / "annotation" / annotation),
        video_feat_dir=str(data_root / "custom_features" / "video"),
        query_feat_dir=str(data_root / "custom_features" / "custom_text"),
        max_query_length=40,
        max_video_length=75,
        normalize_video=True,
        normalize_query=True,
        use_tef=True,
        clip_len=2,
        max_windows=10,
    )

def make_loader(
    dataset: QVHighlights,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=custom_collate,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

def pad_spans(
    span_labels: List[Dict[str, Tensor]],
    device: torch.device,
) -> Tuple[Tensor, Tensor]:
    sample_spans = [span_cxw_to_xx(item["spans"]) for item in span_labels]
    max_spans = max(len(spans) for spans in sample_spans)
    spans = torch.zeros(len(sample_spans), max_spans, 2, device=device)
    mask = torch.zeros(
        len(sample_spans),
        max_spans,
        dtype=torch.bool,
        device=device,
    )
    for index, values in enumerate(sample_spans):
        spans[index, : len(values)] = values
        mask[index, : len(values)] = True
    return spans, mask

def _prepare(raw: Any, device: torch.device):
    metas, raw_batch = raw
    inputs, labels = move_inputs_to_device(raw_batch, device)
    if labels is None:
        raise RuntimeError("labels missing")
    spans, mask = pad_spans(labels["span_labels"], device)
    return PreparedBatch(
        inputs={
            "src_vid": inputs["src_vid"],
            "src_txt": inputs["src_txt"],
            "video_padding_mask": ~inputs["src_vid_mask"].bool(),
            "query_padding_mask": ~inputs["src_txt_mask"].bool(),
        },
        targets={"gt_spans": spans, "gt_span_mask": mask},
        metadata={"metas": metas},
        batch_size=int(inputs["src_vid"].shape[0]),
    )

def build_repository_data(config: RunnerConfig, root: str):
    root = Path(root)
    return DataBundle(
        train_loader=make_loader(
            make_dataset(root, "train"), config.batch_size, config.num_workers, True
        ),
        val_loader=make_loader(
            make_dataset(root, "val"), config.batch_size, config.num_workers, False
        ),
        prepare_batch=_prepare,
        metadata={"root": str(root), "protocol": "qvhighlights-fixed-final"},
    )

def build_repository_data_with_saliency(config: RunnerConfig, root: str) -> DataBundle:
    base = build_repository_data(config, root=root)

    def prepare(raw: Any, device: torch.device):
        batch = base.prepare_batch(raw, device)
        _, raw_batch = raw
        _, labels = move_inputs_to_device(raw_batch, device)
        if labels is None:
            raise RuntimeError("repository labels missing")
        targets = dict(batch.targets)
        for key in (
            "saliency_pos_labels",
            "saliency_neg_labels",
            "saliency_all_labels",
            "relevant_clips",
        ):
            if key in labels:
                targets[key] = labels[key]
        return replace(batch, targets=targets)

    metadata = dict(base.metadata)
    metadata["semantic_supervision_preserved"] = True
    return DataBundle(
        train_loader=base.train_loader,
        val_loader=base.val_loader,
        prepare_batch=prepare,
        metadata=metadata,
    )
