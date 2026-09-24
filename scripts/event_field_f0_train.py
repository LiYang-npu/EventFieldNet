"""Train and evaluate the independent Event Field F0 model."""

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from sg_components.dataset.collate import custom_collate, move_inputs_to_device
from sg_components.dataset.qvhighlights import QVHighlights
from sg_components.model.event_field import (
    EventFieldF0,
    EventFieldLoss,
    build_event_field_targets,
)
from sg_components.model.event_field.unified import EventFieldF1
from sg_components.utils.span_utils import span_cxw_to_xx


@dataclass
class EpochMetrics:
    loss: float
    evidence_loss: float
    support_loss: float
    transition_loss: float
    evidence_corr: float
    support_soft_iou: float
    boundary_mae: float
    decoded_miou: float
    decoded_r1_05: float
    decoded_r1_07: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model", choices=("f0", "f1"), default="f0")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def safe_corr(prediction: Tensor, target: Tensor) -> Tensor:
    prediction = prediction - prediction.mean()
    target = target - target.mean()
    denominator = prediction.square().sum().sqrt() * target.square().sum().sqrt()
    if denominator <= 1e-8:
        return prediction.new_zeros(())
    return (prediction * target).sum() / denominator


def best_iou(predicted: Tensor, targets: Tensor) -> Tensor:
    left = torch.maximum(predicted[0], targets[:, 0])
    right = torch.minimum(predicted[1], targets[:, 1])
    intersection = (right - left).clamp_min(0)
    union = (
        (predicted[1] - predicted[0]).clamp_min(0)
        + (targets[:, 1] - targets[:, 0]).clamp_min(0)
        - intersection
    )
    return (intersection / union.clamp_min(1e-8)).max()


def field_metrics(
    outputs,
    targets,
    spans: Tensor,
    span_mask: Tensor,
    padding_mask: Tensor,
) -> Dict[str, float]:
    valid = ~padding_mask
    evidence = torch.sigmoid(outputs.evidence_logits)
    support = torch.sigmoid(outputs.support_logits)
    evidence_corr = safe_corr(
        evidence[valid],
        targets.evidence[valid],
    )
    support_intersection = (support[valid] * targets.support[valid]).sum()
    support_union = (
        support[valid]
        + targets.support[valid]
        - support[valid] * targets.support[valid]
    ).sum()
    support_soft_iou = support_intersection / support_union.clamp_min(1e-8)

    start_logits = outputs.start_transition_logits.masked_fill(padding_mask, -1e4)
    end_logits = outputs.end_transition_logits.masked_fill(padding_mask, -1e4)
    start_index = start_logits.argmax(dim=1)
    end_index = end_logits.argmax(dim=1)
    target_start = targets.start_transition.masked_fill(padding_mask, -1).argmax(dim=1)
    target_end = targets.end_transition.masked_fill(padding_mask, -1).argmax(dim=1)
    lengths = valid.sum(dim=1).clamp_min(1)
    boundary_mae = (
        (start_index - target_start).abs() + (end_index - target_end).abs()
    ).float() / (2 * lengths)

    ious = []
    for index in range(len(spans)):
        start = start_index[index].float() / lengths[index]
        end = (end_index[index].float() + 1) / lengths[index]
        if end < start:
            start, end = end, start
        predicted = torch.stack((start.clamp(0, 1), end.clamp(0, 1)))
        ious.append(best_iou(predicted, spans[index, span_mask[index]]))
    decoded_iou = torch.stack(ious)
    return {
        "evidence_corr": float(evidence_corr),
        "support_soft_iou": float(support_soft_iou),
        "boundary_mae": float(boundary_mae.mean()),
        "decoded_miou": float(decoded_iou.mean()),
        "decoded_r1_05": float((decoded_iou >= 0.5).float().mean()),
        "decoded_r1_07": float((decoded_iou >= 0.7).float().mean()),
    }


def run_epoch(
    model: EventFieldF0,
    criterion: EventFieldLoss,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> EpochMetrics:
    training = optimizer is not None
    model.train(training)
    totals: Dict[str, float] = {}
    examples = 0
    for _, raw_batch in loader:
        inputs, labels = move_inputs_to_device(raw_batch, device)
        assert labels is not None
        video_padding_mask = ~inputs["src_vid_mask"].bool()
        query_padding_mask = ~inputs["src_txt_mask"].bool()
        spans, span_mask = pad_spans(labels["span_labels"], device)
        with torch.set_grad_enabled(training):
            outputs = model(
                inputs["src_vid"],
                inputs["src_txt"],
                video_padding_mask,
                query_padding_mask,
            )
            targets = build_event_field_targets(
                spans,
                outputs.support_logits.shape[1],
                evidence_targets=(labels["saliency_all_labels"] / 12.0).clamp(0, 1),
                padding_mask=video_padding_mask,
                gt_span_mask=span_mask,
            )
            losses = criterion(outputs, targets, video_padding_mask)
            if training:
                optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        metrics = field_metrics(
            outputs,
            targets,
            spans,
            span_mask,
            video_padding_mask,
        )
        batch_size = len(spans)
        examples += batch_size
        values = {
            "loss": float(losses["loss"]),
            "evidence_loss": float(losses["loss_evidence"]),
            "support_loss": float(losses["loss_support"]),
            "transition_loss": float(losses["loss_transition"]),
            **metrics,
        }
        for name, value in values.items():
            totals[name] = totals.get(name, 0.0) + value * batch_size
    return EpochMetrics(**{name: value / examples for name, value in totals.items()})


def save_checkpoint(
    path: Path,
    model: EventFieldF0,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: EpochMetrics,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metrics": asdict(metrics),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    train_loader = make_loader(
        make_dataset(args.root, "train"),
        args.batch_size,
        args.num_workers,
        True,
    )
    val_loader = make_loader(
        make_dataset(args.root, "val"),
        args.batch_size,
        args.num_workers,
        False,
    )
    model_class = EventFieldF0 if args.model == "f0" else EventFieldF1
    model = model_class(video_dim=514, query_dim=512).to(device)
    criterion = EventFieldLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )
    best_loss = math.inf
    history_path = args.output / "history.jsonl"
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, criterion, train_loader, device, optimizer)
        val_metrics = run_epoch(model, criterion, val_loader, device, None)
        scheduler.step()
        record = {
            "epoch": epoch,
            "learning_rate": scheduler.get_last_lr()[0],
            "train": asdict(train_metrics),
            "val": asdict(val_metrics),
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        save_checkpoint(args.output / "last.pt", model, optimizer, epoch, val_metrics)
        if val_metrics.loss < best_loss:
            best_loss = val_metrics.loss
            save_checkpoint(
                args.output / "best.pt", model, optimizer, epoch, val_metrics
            )
    summary = {
        "seed": args.seed,
        "epochs": args.epochs,
        "selection": "fixed final epoch",
        "model": args.model,
        "final": asdict(val_metrics),
        "best_val_loss": best_loss,
    }
    (args.output / "final_metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
