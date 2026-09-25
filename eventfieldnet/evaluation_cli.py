"""Independent official evaluation command for trifield round31.

The training worker delegates evaluation here so repository imports and
CUDA state stay in a separate process.  Candidate selection is deterministic:
native torch top-30 is kept as the raw policy, then the complete valid span
grid is traversed with the native prefix followed by stable score order and
hard temporal NMS at IoU 0.5.  Ground truth is read only for diagnostics
after both candidate sets have been selected.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
from pathlib import Path
import secrets
import runpy
import sys
import time
from contextlib import nullcontext
from typing import Any, Mapping, Optional, Sequence

try:
    from .evaluator import (
        MAX_CANDIDATES,
        NMS_IOU_THRESHOLD,
        SCHEMA,
        SELECTION_POLICY,
        Round31EvaluatorError,
        _atomic_json,
        _canonical_hash,
        _json_safe,
        _sha256_file,
        candidate_diagnostics,
        official_metrics,
        select_full_grid_candidates,
    )
except ImportError:
    from evaluator import (  # type: ignore
        MAX_CANDIDATES,
        NMS_IOU_THRESHOLD,
        SCHEMA,
        SELECTION_POLICY,
        Round31EvaluatorError,
        _atomic_json,
        _canonical_hash,
        _json_safe,
        _sha256_file,
        candidate_diagnostics,
        official_metrics,
        select_full_grid_candidates,
    )


def _load_json(path: Path) -> Mapping[str, Any]:
    if not path.exists() or not path.is_file():
        raise Round31EvaluatorError(f"config file does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise Round31EvaluatorError(f"invalid JSON config: {path}") from exc
    if not isinstance(value, Mapping):
        raise Round31EvaluatorError(f"config must be a JSON object: {path}")
    return value


def _torch_load(path: Path, torch: Any) -> Mapping[str, Any]:
    if not path.exists() or not path.is_file():
        raise Round31EvaluatorError(f"checkpoint does not exist: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise Round31EvaluatorError("checkpoint payload must be a mapping")
    return payload


def _checkpoint_config(
    payload: Mapping[str, Any],
    config_path: Optional[Path],
) -> dict[str, Any]:
    value = payload.get("config")
    if isinstance(value, Mapping):
        raw = dict(value)
    elif config_path is not None:
        raw = dict(_load_json(config_path))
    else:
        raise Round31EvaluatorError(
            "checkpoint has no materialized config and --config was not supplied"
        )
    for name in ("model_factory", "data_factory"):
        spec = raw.get(name)
        if not isinstance(spec, Mapping) or not spec.get("target"):
            raise Round31EvaluatorError(f"checkpoint config is missing {name}.target")
    return raw


def _path_list(root: Path, project_root: Path) -> list[Path]:
    return [
        root / "src",
        project_root,
        project_root / "scripts",
        root / "src",
        root / "stage51",
        root / "src",
        root,
        Path(__file__).resolve().parent.parent,
    ]


def _install_import_roots(raw, *, root, project_root):
    paths = (Path(root).resolve(), Path(project_root).resolve())
    for path in reversed(paths):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return paths


def _repository_metric_api(project_root: Path) -> tuple[Any, Any, Path]:
    script = project_root / "scripts" / "event_field_official_eval.py"
    if not script.exists():
        raise Round31EvaluatorError(
            f"official evaluator script does not exist: {script}"
        )
    try:
        namespace = runpy.run_path(
            str(script),
            run_name="eventfieldnet.repository_official_eval",
        )
    except Exception as exc:
        raise Round31EvaluatorError(
            f"failed to load official evaluator script: {script}"
        ) from exc
    get_metrics = namespace.get("get_metrics")
    tensor_to_python = namespace.get("tensor_to_python")
    if not callable(get_metrics) or not callable(tensor_to_python):
        raise Round31EvaluatorError(
            "official evaluator must expose get_metrics and tensor_to_python"
        )
    return get_metrics, tensor_to_python, script


def _config_value(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _meta_value(meta: Any, name: str, default: Any = None) -> Any:
    if isinstance(meta, Mapping):
        return meta.get(name, default)
    return getattr(meta, name, default)


def _device_from_arg(torch: Any, value: str) -> Any:
    device = torch.device(str(value))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise Round31EvaluatorError(f"CUDA unavailable: {device}")
    if device.type not in {"cuda", "cpu"}:
        raise Round31EvaluatorError(f"unsupported evaluation device: {device}")
    return device


def _autocast(torch: Any, device: Any, precision: str) -> Any:
    if precision == "fp32":
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


def _prepare_model_and_bundle(
    raw: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    torch: Any,
    device: Any,
) -> tuple[Any, Any, Any, int]:
    try:
        common_config = importlib.import_module("training.config")
        shared = importlib.import_module("eventfieldnet.engine")
        runner_config = getattr(common_config, "RunnerConfig")
        instantiate = getattr(shared, "instantiate")
        seed_everything = getattr(shared, "seed_everything")
    except (ImportError, ModuleNotFoundError, AttributeError) as exc:
        raise Round31EvaluatorError(
            "repository training.config/training.runner contract is unavailable"
        ) from exc
    try:
        config = runner_config.from_dict(dict(raw))
    except Exception as exc:
        raise Round31EvaluatorError(
            "checkpoint config is not a valid RunnerConfig"
        ) from exc
    seed = int(_config_value(config, "seed", raw.get("seed", 0)))
    seed_everything(seed)
    try:
        bundle = instantiate(config.data_factory, config)
        model = instantiate(config.model_factory, config)
    except Exception as exc:
        raise Round31EvaluatorError(
            "data_factory/model_factory construction failed"
        ) from exc
    if not isinstance(model, torch.nn.Module):
        raise Round31EvaluatorError("model_factory did not return torch.nn.Module")
    state = payload.get("model")
    if not isinstance(state, Mapping):
        raise Round31EvaluatorError("checkpoint has no model state mapping")
    try:
        model.load_state_dict(state, strict=True)
        model.to(device)
        model.eval()
    except Exception as exc:
        raise Round31EvaluatorError("checkpoint model reconstruction failed") from exc
    return config, bundle, model, seed


def _numpy(value: Any) -> Any:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "cpu") and hasattr(value, "numpy"):
        return value.cpu().numpy()
    return value


def _span_candidates(
    output: Any,
    prepared: Any,
    row: int,
    *,
    torch: Any,
) -> tuple[Any, Any, Any, Any]:
    try:
        logits = output.span_logits
        valid_mask = output.span_valid_mask.bool()
        padding = prepared.inputs["video_padding_mask"][row].bool()
    except (AttributeError, KeyError, TypeError) as exc:
        raise Round31EvaluatorError(
            "prepared/model output misses span_logits, span_valid_mask, or video_padding_mask"
        ) from exc
    flat_logits = logits.flatten(1)
    flat_valid = valid_mask.flatten(1)
    if row >= int(flat_logits.shape[0]):
        raise Round31EvaluatorError(f"model output row out of range: {row}")
    valid = _numpy(flat_valid[row]).astype(bool, copy=False).reshape(-1)
    valid_count = int(valid.sum())
    if valid_count <= 0:
        raise Round31EvaluatorError(f"query row {row} has no valid span candidates")
    masked = flat_logits[row].masked_fill(~flat_valid[row], -torch.inf)
    probabilities = torch.softmax(masked.float(), dim=0)
    scores = _numpy(probabilities).astype("float64", copy=False).reshape(-1)
    if len(scores) != len(valid):
        raise Round31EvaluatorError("span score/mask lengths differ")
    native = probabilities.topk(min(MAX_CANDIDATES, valid_count)).indices
    native_order = _numpy(native).astype("int64", copy=False).reshape(-1)
    span_length = int(logits.shape[-1])
    if span_length <= 0:
        raise Round31EvaluatorError("model produced an empty span grid")
    valid_video_clips = int((~padding).sum().item())
    if valid_video_clips <= 0:
        raise Round31EvaluatorError("query has no valid video clips")
    return logits, valid, scores, (native_order, span_length, valid_video_clips)


def _geometry(
    scores: Any,
    *,
    span_length: int,
    valid_video_clips: int,
    duration: Any,
) -> Any:
    duration_value = float(duration)
    if not math.isfinite(duration_value) or duration_value <= 0:
        raise Round31EvaluatorError(f"invalid video duration: {duration!r}")
    import numpy as np

    indexes = np.arange(len(scores))
    scale = duration_value / float(valid_video_clips)
    return np.stack(
        [
            (indexes // int(span_length)).astype(np.float32) * scale,
            ((indexes % int(span_length)) + 1).astype(np.float32) * scale,
        ],
        axis=1,
    )


def _evidence_scores(output: Any, row: int, valid_video_clips: int, torch: Any) -> Any:
    try:
        return (
            torch.sigmoid(output.evidence_logits[row, :valid_video_clips].float())
            .detach()
            .cpu()
        )
    except (AttributeError, IndexError) as exc:
        raise Round31EvaluatorError("model output misses evidence_logits") from exc


def _prediction(
    meta: Any,
    windows: Any,
    scores: Any,
    ids: Any,
    saliency: Any,
    torch: Any,
) -> dict[str, Any]:
    import numpy as np

    selected = np.asarray(ids, dtype=np.int64)
    predicted = np.asarray(windows, dtype=np.float32)[selected]
    score_array = np.asarray(scores, dtype=np.float32)[selected]
    return {
        "qid": _meta_value(meta, "qid"),
        "query": _meta_value(meta, "query", ""),
        "vid": _meta_value(meta, "vid"),
        "pred_relevant_windows": torch.tensor(
            np.column_stack([predicted, score_array]),
            dtype=torch.float32,
        ),
        "pred_saliency_scores": saliency,
    }


def _prediction_json(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    for key in ("pred_relevant_windows", "pred_saliency_scores"):
        tensor = value.get(key)
        if hasattr(tensor, "detach"):
            result[key] = tensor.detach().cpu().tolist()
        else:
            result[key] = tensor
    return _json_safe(result)


def _compute_official_metrics(
    get_metrics: Any,
    tensor_to_python: Any,
    submissions: Sequence[Any],
    targets: Sequence[Any],
) -> dict[str, float]:
    try:
        metric = get_metrics()
    except Exception as exc:
        raise Round31EvaluatorError("official get_metrics construction failed") from exc
    cpu = getattr(metric, "cpu", None)
    if callable(cpu):
        moved = cpu()
        if moved is not None:
            metric = moved
    try:
        metric.update(submissions=submissions, targets=targets)
        computed = metric.compute()
        converted = tensor_to_python(computed)
    except Exception as exc:
        raise Round31EvaluatorError("official metric update/compute failed") from exc
    return official_metrics(converted)


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    import numpy as np

    if not rows:
        return {"query_count": 0}
    result: dict[str, Any] = {"query_count": int(len(rows))}
    numeric_keys = sorted(
        key
        for key in rows[0]
        if key != "qid"
        and all(
            isinstance(row.get(key), (int, float))
            and not isinstance(row.get(key), bool)
            and math.isfinite(float(row[key]))
            for row in rows
        )
    )
    for key in numeric_keys:
        values = [float(row[key]) for row in rows]
        result[f"{key}_mean"] = float(np.mean(values))
        if key.endswith("_count"):
            result[f"{key}_total"] = int(round(sum(values)))
    gt_total = float(sum(float(row.get("gt_count", 0)) for row in rows))
    for key in sorted(rows[0]):
        if key.startswith("gt_coverage_") and key.endswith("_count"):
            total = float(sum(float(row.get(key, 0)) for row in rows))
            result[f"{key}_rate"] = total / gt_total if gt_total else 0.0
    return result


def _save_predictions(
    path: Path, predictions: Sequence[Mapping[str, Any]], torch: Any
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(list(predictions), path)


def evaluate_checkpoint(
    *,
    checkpoint: str | Path,
    output_dir: str | Path,
    root: str | Path,
    project_root: str | Path,
    config: str | Path | None = None,
    wrapper: str | Path | None = None,
    device: str = "cuda:0",
    batch_size: int = 64,
    num_workers: int = 2,
    precision: str = "bf16",
    max_candidates: int = MAX_CANDIDATES,
    nms_iou: float = NMS_IOU_THRESHOLD,
    request_id: str | None = None,
    checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    del wrapper, batch_size, num_workers
    if int(max_candidates) != MAX_CANDIDATES:
        raise Round31EvaluatorError("round31 fixes max candidates at 30")
    if abs(float(nms_iou) - NMS_IOU_THRESHOLD) > 1.0e-12:
        raise Round31EvaluatorError("round31 fixes temporal NMS IoU at 0.5")
    if precision not in {"bf16", "fp32"}:
        raise Round31EvaluatorError(f"unsupported precision: {precision}")
    checkpoint_path = Path(checkpoint).resolve()
    request_id = str(request_id or secrets.token_hex(16))
    expected_checkpoint_sha256 = (
        None if checkpoint_sha256 in (None, "") else str(checkpoint_sha256)
    )
    checkpoint_sha256 = expected_checkpoint_sha256
    output_path = Path(output_dir).resolve()
    root_path = Path(root).resolve()
    project_path = Path(project_root).resolve()
    config_path = None if config in (None, "") else Path(config).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    if output_path.is_symlink():
        raise Round31EvaluatorError(f"refusing symlink output directory: {output_path}")
    started = time.monotonic()
    try:
        checkpoint_sha256 = _sha256_file(checkpoint_path)
        if (
            expected_checkpoint_sha256 is not None
            and checkpoint_sha256 != expected_checkpoint_sha256
        ):
            raise Round31EvaluatorError(
                "checkpoint hash supplied by facade does not match checkpoint"
            )
        import torch

        payload = _torch_load(checkpoint_path, torch)
        raw = _checkpoint_config(payload, config_path)
        import_roots = _install_import_roots(
            raw, root=root_path, project_root=project_path
        )
        get_metrics, tensor_to_python, official_script = _repository_metric_api(
            project_path
        )
        device_obj = _device_from_arg(torch, device)
        _runner_config, bundle, model, seed = _prepare_model_and_bundle(
            raw, payload, torch=torch, device=device_obj
        )
        val_loader = getattr(bundle, "val_loader", None)
        prepare_batch = getattr(bundle, "prepare_batch", None)
        if val_loader is None or not callable(prepare_batch):
            raise Round31EvaluatorError(
                "data bundle must expose val_loader and prepare_batch"
            )
        submissions: dict[str, list[dict[str, Any]]] = {"raw": [], "nms": []}
        targets: list[Any] = []
        rows: list[dict[str, Any]] = []
        with torch.no_grad():
            for batch_index, batch in enumerate(val_loader):
                try:
                    prepared = prepare_batch(batch, device_obj)
                    with _autocast(torch, device_obj, precision):
                        output = model(prepared.inputs)
                except Exception as exc:
                    raise Round31EvaluatorError(
                        f"validation forward failed at batch {batch_index}"
                    ) from exc
                metas = _meta_value(getattr(prepared, "metadata", None), "metas")
                if metas is None:
                    raise Round31EvaluatorError("prepared batch metadata lacks metas")
                for row_index, meta in enumerate(metas):
                    _logits, valid, scores, native_info = _span_candidates(
                        output, prepared, row_index, torch=torch
                    )
                    native_order, span_length, valid_video_clips = native_info

                    windows = _geometry(
                        scores,
                        span_length=span_length,
                        valid_video_clips=valid_video_clips,
                        duration=_meta_value(meta, "duration"),
                    )
                    native, _full_order, nms = select_full_grid_candidates(
                        windows,
                        scores,
                        valid,
                        native_order=native_order,
                        iou_threshold=NMS_IOU_THRESHOLD,
                        max_candidates=MAX_CANDIDATES,
                    )
                    saliency = _evidence_scores(
                        output, row_index, valid_video_clips, torch
                    )
                    submissions["raw"].append(
                        _prediction(meta, windows, scores, native, saliency, torch)
                    )
                    submissions["nms"].append(
                        _prediction(meta, windows, scores, nms, saliency, torch)
                    )
                    targets.append(meta)
                    gt_windows = _meta_value(meta, "relevant_windows")
                    if hasattr(gt_windows, "detach"):
                        gt_windows = gt_windows.detach().cpu().numpy()
                    row = candidate_diagnostics(
                        windows,
                        scores,
                        valid,
                        native,
                        nms,
                        gt_windows=gt_windows,
                    )
                    row.update(
                        {
                            "qid": _json_safe(_meta_value(meta, "qid")),
                            "vid": _json_safe(_meta_value(meta, "vid")),
                            "raw_ids": [int(value) for value in native.tolist()],
                            "nms_ids": [int(value) for value in nms.tolist()],
                            "native_top30_count": int(len(native)),
                            "gt_used_for_selection": False,
                        }
                    )
                    rows.append(row)
        raw_metrics = _compute_official_metrics(
            get_metrics, tensor_to_python, submissions["raw"], targets
        )
        nms_metrics = _compute_official_metrics(
            get_metrics, tensor_to_python, submissions["nms"], targets
        )
        top1_consistent = all(bool(row["top1_same"]) for row in rows)
        if not top1_consistent:
            raise Round31EvaluatorError("raw and NMS candidate top-1 diverged")
        _save_predictions(output_path / "predictions_raw.pt", submissions["raw"], torch)
        _save_predictions(output_path / "predictions_nms.pt", submissions["nms"], torch)
        _atomic_json(
            output_path / "predictions.json",
            {
                "raw": [_prediction_json(value) for value in submissions["raw"]],
                "nms": [_prediction_json(value) for value in submissions["nms"]],
            },
        )
        diagnostics = {
            "query_count": len(rows),
            "selection_gt_used": False,
            "top1_consistent": top1_consistent,
            "raw": _aggregate(rows),
            "nms": _aggregate(rows),
            "per_query": rows,
        }
        _atomic_json(output_path / "candidate_diagnostics.json", diagnostics)
        result = {
            "schema": SCHEMA,
            "status": "complete",
            "seed": seed,
            "epoch": payload.get("epoch"),
            "request_id": request_id,
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint": str(checkpoint_path),
            "checkpoint_config_hash": _canonical_hash(raw),
            "official_script": str(official_script),
            "root": str(root_path),
            "project_root": str(project_path),
            "import_roots": [str(path) for path in import_roots],
            "selection": {
                "policy": SELECTION_POLICY,
                "iou_threshold": NMS_IOU_THRESHOLD,
                "max_candidates": MAX_CANDIDATES,
                "native_top30_prefix": True,
                "native_tie_order": "torch.topk prefix, then stable full-grid score order",
                "score_source": "softmax(span_logits masked by span_valid_mask)",
                "geometry": "grid index with duration / valid video clips",
                "gt_used_for_selection": False,
                "top1_consistent": top1_consistent,
            },
            "metrics": nms_metrics,
            "raw_metrics": raw_metrics,
            "diagnostics": {
                "query_count": len(rows),
                "raw": diagnostics["raw"],
                "nms": diagnostics["nms"],
                "path": str(output_path / "candidate_diagnostics.json"),
            },
            "predictions": {
                "raw": str(output_path / "predictions_raw.pt"),
                "nms": str(output_path / "predictions_nms.pt"),
                "json": str(output_path / "predictions.json"),
            },
            "config": raw,
            "elapsed_seconds": time.monotonic() - started,
        }
        _atomic_json(output_path / "raw_metrics.json", raw_metrics)
        _atomic_json(output_path / "nms_metrics.json", nms_metrics)
        _atomic_json(output_path / "metrics.json", result)
        _atomic_json(
            output_path / "evaluator_status.json",
            {
                "schema": SCHEMA,
                "status": "complete",
                "request_id": request_id,
                "checkpoint_sha256": checkpoint_sha256,
                "metrics": nms_metrics,
                "raw_metrics": raw_metrics,
                "elapsed_seconds": result["elapsed_seconds"],
            },
        )
        return result
    except Exception as exc:
        _atomic_json(
            output_path / "evaluator_status.json",
            {
                "schema": SCHEMA,
                "status": "failed",
                "request_id": request_id,
                "checkpoint_sha256": checkpoint_sha256,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "checkpoint": str(checkpoint_path),
                "elapsed_seconds": time.monotonic() - started,
            },
        )
        if isinstance(exc, Round31EvaluatorError):
            raise
        raise Round31EvaluatorError("round31 evaluation failed") from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run official round31 raw and fixed temporal-NMS evaluation"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--wrapper", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--max-candidates", type=int, default=MAX_CANDIDATES)
    parser.add_argument("--nms-iou", type=float, default=NMS_IOU_THRESHOLD)
    parser.add_argument("--request-id")
    parser.add_argument("--checkpoint-sha256")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = evaluate_checkpoint(
            checkpoint=args.checkpoint,
            config=args.config,
            output_dir=args.output_dir,
            root=args.root,
            project_root=args.project_root,
            wrapper=args.wrapper,
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            precision=args.precision,
            max_candidates=args.max_candidates,
            nms_iou=args.nms_iou,
            request_id=args.request_id,
            checkpoint_sha256=args.checkpoint_sha256,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "schema": SCHEMA,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(
            {
                "schema": result["schema"],
                "status": result["status"],
                "metrics": result["metrics"],
                "raw_metrics": result["raw_metrics"],
                "query_count": result["diagnostics"]["query_count"],
                "elapsed_seconds": result["elapsed_seconds"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
