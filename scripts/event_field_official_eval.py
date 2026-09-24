"""Evaluate EventField checkpoints with the unchanged SG-DETR metrics."""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts import event_field_f0_train as f0
from sg_components.dataset.collate import move_inputs_to_device
from sg_components.metrics.metrics_collection import get_metrics
from sg_components.model.event_field.framework_proven import EventFieldNetProven
from sg_components.model.event_field.framework_taskaware_deep import (
    EventFieldNetTaskAwareDeep,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model-kind",
        choices=("shared", "shared_full", "deep_temporal"),
        required=True,
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp", choices=("bf16", "fp32"), default="bf16")
    return parser.parse_args()


def build_model(kind: str):
    training = dict(
        video_dim=514,
        query_dim=512,
        hidden_dim=384,
        num_heads=8,
        feedforward_dim=1536,
        pair_dim=96,
        dropout=0.2,
        min_span_clips=2,
    )
    if kind == "deep_temporal":
        return EventFieldNetTaskAwareDeep(
            **training,
            architecture="deep_temporal",
        )
    return EventFieldNetProven(**training, architecture=kind)


def tensor_to_python(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu())
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: tensor_to_python(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [tensor_to_python(item) for item in value]
    return value


def temporal_iou(windows: np.ndarray, targets: np.ndarray) -> np.ndarray:
    left = np.maximum(windows[:, None, 0], targets[None, :, 0])
    right = np.minimum(windows[:, None, 1], targets[None, :, 1])
    intersection = np.maximum(right - left, 0.0)
    union = (
        np.maximum(windows[:, None, 1] - windows[:, None, 0], 0.0)
        + np.maximum(targets[None, :, 1] - targets[None, :, 0], 0.0)
        - intersection
    )
    return intersection / np.maximum(union, 1e-8)


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def safe_corr(left: list[float], right: list[float], rank: bool = False) -> float:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if rank:
        x, y = rankdata(x), rankdata(y)
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model = build_model(args.model_kind).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    incompatible = model.load_state_dict(checkpoint["model"], strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"checkpoint mismatch: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    model.eval()
    loader = f0.make_loader(
        f0.make_dataset(args.root, "val"),
        args.batch_size,
        args.num_workers,
        False,
    )
    official_metrics = get_metrics().cpu()
    submissions: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []
    candidate_scores: list[float] = []
    candidate_ious: list[float] = []
    recalls = {
        f"R@{count}_IoU{threshold}": []
        for count in (1, 5, 10, 30)
        for threshold in (0.5, 0.7)
    }
    oracle_ious: list[float] = []
    top1_ious: list[float] = []
    top1_oracle_gaps: list[float] = []
    too_short: list[float] = []
    too_long: list[float] = []
    forward_seconds = 0.0
    example_count = 0
    amp_context = lambda: (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if args.amp == "bf16"
        else torch.autocast(device_type="cuda", enabled=False)
    )

    with torch.no_grad():
        for metas, raw_batch in loader:
            inputs, _ = move_inputs_to_device(raw_batch, device)
            video_padding_mask = ~inputs["src_vid_mask"].bool()
            query_padding_mask = ~inputs["src_txt_mask"].bool()
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            with amp_context():
                outputs = model(
                    inputs["src_vid"],
                    inputs["src_txt"],
                    video_padding_mask,
                    query_padding_mask,
                )
            torch.cuda.synchronize(device)
            forward_seconds += time.perf_counter() - started
            batch_size, length, _ = outputs.span_logits.shape
            example_count += batch_size
            flat_logits = outputs.span_logits.flatten(1).masked_fill(
                ~outputs.span_valid_mask.flatten(1), -torch.inf
            )
            flat_probabilities = torch.softmax(flat_logits.float(), dim=1)
            if hasattr(outputs, "hypothesis_span_logits") and os.environ.get(
                "FS_HYPOTHESIS_SET_UNION", "0"
            ).lower() not in {"0", "false", "no", "off"}:
                valid_flat = outputs.span_valid_mask.flatten(1)
                primary_quota = min(
                    int(os.environ.get("FS_HYPOTHESIS_PRIMARY_QUOTA", "15")),
                    flat_logits.shape[1],
                )
                mode_quota = min(
                    int(os.environ.get("FS_HYPOTHESIS_MODE_QUOTA", "5")),
                    flat_logits.shape[1],
                )
                merged_scores = 0.1 * flat_probabilities
                primary_scores, primary_indices = flat_probabilities.topk(
                    primary_quota, dim=1
                )
                merged_scores.scatter_(1, primary_indices, 3.0 + primary_scores)
                mode_logits = outputs.hypothesis_span_logits.flatten(2).masked_fill(
                    ~valid_flat[:, None], -torch.inf
                )
                mode_probabilities = torch.softmax(mode_logits.float(), dim=2)
                calibrated_union = hasattr(
                    outputs, "hypothesis_quality_logits"
                ) and os.environ.get(
                    "FS_HYPOTHESIS_CALIBRATED_UNION", "0"
                ).lower() not in {"0", "false", "no", "off"}
                if calibrated_union:
                    pool_count = min(args.top_k, flat_logits.shape[1])
                    primary_pool = flat_probabilities.topk(pool_count, dim=1).indices
                    primary_quality = outputs.quality_logits.flatten(1).float()
                    merged_scores = torch.full_like(primary_quality, -torch.inf)
                    merged_scores.scatter_(
                        1,
                        primary_pool,
                        primary_quality.gather(1, primary_pool),
                    )
                    mode_quality = outputs.hypothesis_quality_logits.flatten(2).float()
                    for mode in range(mode_probabilities.shape[1]):
                        mode_indices = (
                            mode_probabilities[:, mode].topk(mode_quota, dim=1).indices
                        )
                        proposed = mode_quality[:, mode].gather(1, mode_indices)
                        existing = merged_scores.gather(1, mode_indices)
                        merged_scores.scatter_(
                            1, mode_indices, torch.maximum(existing, proposed)
                        )
                    primary_top = flat_probabilities.argmax(dim=1, keepdim=True)
                    forced_top = merged_scores.amax(dim=1, keepdim=True) + 1e-3
                    merged_scores.scatter_(1, primary_top, forced_top)
                else:
                    for mode in range(mode_probabilities.shape[1]):
                        mode_scores, mode_indices = mode_probabilities[:, mode].topk(
                            mode_quota, dim=1
                        )
                        proposed = 1.0 + mode_scores
                        existing = merged_scores.gather(1, mode_indices)
                        merged_scores.scatter_(
                            1, mode_indices, torch.maximum(existing, proposed)
                        )
                flat_probabilities = torch.softmax(
                    merged_scores.masked_fill(~valid_flat, -torch.inf), dim=1
                )
            evidence = torch.sigmoid(outputs.evidence_logits.float())
            valid_lengths = (~video_padding_mask).sum(dim=1)

            for index, meta in enumerate(metas):
                valid_count = int(outputs.span_valid_mask[index].sum())
                count = min(args.top_k, valid_count)
                scores, flat_indices = torch.topk(flat_probabilities[index], count)
                start_indices = torch.div(flat_indices, length, rounding_mode="floor")
                end_indices = flat_indices.remainder(length)
                valid_length = int(valid_lengths[index])
                duration = float(meta["duration"])
                scale = duration / max(valid_length, 1)
                windows = torch.stack(
                    (
                        start_indices.float() * scale,
                        (end_indices.float() + 1.0) * scale,
                        scores,
                    ),
                    dim=1,
                ).cpu()
                saliency = evidence[index, :valid_length].cpu()
                submission = {
                    "qid": meta["qid"],
                    "query": meta.get("query", ""),
                    "vid": meta["vid"],
                    "pred_relevant_windows": windows,
                    "pred_saliency_scores": saliency,
                }
                submissions.append(submission)
                targets.append(meta)

                gt = np.asarray(meta["relevant_windows"], dtype=np.float64)
                predicted = windows[:, :2].numpy()
                ious = temporal_iou(predicted, gt).max(axis=1)
                oracle = float(ious.max())
                top1 = float(ious[0])
                oracle_ious.append(oracle)
                top1_ious.append(top1)
                top1_oracle_gaps.append(oracle - top1)
                candidate_scores.extend(windows[:, 2].numpy().tolist())
                candidate_ious.extend(ious.tolist())
                best_gt = gt[int(temporal_iou(predicted[:1], gt)[0].argmax())]
                predicted_length = max(predicted[0, 1] - predicted[0, 0], 1e-8)
                gt_length = max(best_gt[1] - best_gt[0], 1e-8)
                ratio = predicted_length / gt_length
                too_short.append(float(ratio < 0.8))
                too_long.append(float(ratio > 1.25))
                for recall_count in (1, 5, 10, 30):
                    available = ious[: min(recall_count, len(ious))]
                    for threshold in (0.5, 0.7):
                        recalls[f"R@{recall_count}_IoU{threshold}"].append(
                            float((available >= threshold).any())
                        )

    official_metrics.update(submissions=submissions, targets=targets)
    official = tensor_to_python(official_metrics.compute())
    diagnostics = {
        "oracle_mIoU_top30": float(np.mean(oracle_ious)),
        "top1_mIoU": float(np.mean(top1_ious)),
        "top1_oracle_gap": float(np.mean(top1_oracle_gaps)),
        "score_iou_pearson": safe_corr(candidate_scores, candidate_ious),
        "score_iou_spearman": safe_corr(candidate_scores, candidate_ious, rank=True),
        "top1_too_short_rate": float(np.mean(too_short)),
        "top1_too_long_rate": float(np.mean(too_long)),
        **{name: float(np.mean(values)) for name, values in recalls.items()},
    }
    summary = {
        "model_kind": args.model_kind,
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "examples": example_count,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "forward_ms_per_example": 1000.0 * forward_seconds / example_count,
        "official_metrics": official,
        "candidate_diagnostics": diagnostics,
    }
    (args.output / "metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with (args.output / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for submission in submissions:
            serializable = {
                **submission,
                "pred_relevant_windows": submission["pred_relevant_windows"].tolist(),
                "pred_saliency_scores": submission["pred_saliency_scores"].tolist(),
            }
            handle.write(json.dumps(serializable, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
