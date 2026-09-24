"""Round31 official evaluator and deterministic full-grid temporal NMS.

The training worker only sees the Round31OfficialEvaluator facade. Each
evaluation is delegated to an independent python -m trifield_round31.evaluation_cli
process, so repository model/data imports and CUDA allocations do not leak into
the training process.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import signal
import secrets
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np


SCHEMA = "eventfieldnet_trifield_round31_evaluator_v1"
SELECTION_POLICY = "full_grid_hard_temporal_nms"
NMS_IOU_THRESHOLD = 0.5
MAX_CANDIDATES = 30


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Round31EvaluatorError(RuntimeError):
    """Configuration, subprocess, or official metric contract failure."""


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    try:
        import torch

        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return _json_safe(value.detach().cpu().item())
            return {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "device": str(value.device),
            }
    except Exception:
        pass
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    temp.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temp, path)


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def temporal_iou(a: Any, b: Any) -> np.ndarray:
    """Pairwise IoU for [start, end] windows; no labels or GT are consulted."""
    left = np.asarray(a, dtype=np.float64).reshape(-1, 2)
    right = np.asarray(b, dtype=np.float64).reshape(-1, 2)
    if not len(left) or not len(right):
        return np.zeros((len(left), len(right)), dtype=np.float64)
    intersection = np.maximum(
        0.0,
        np.minimum(left[:, None, 1], right[None, :, 1])
        - np.maximum(left[:, None, 0], right[None, :, 0]),
    )
    left_width = np.maximum(0.0, left[:, 1] - left[:, 0])
    right_width = np.maximum(0.0, right[:, 1] - right[:, 0])
    union = np.maximum(
        1.0e-12, left_width[:, None] + right_width[None, :] - intersection
    )
    return intersection / union


def stable_score_order(scores: Any, valid: Any) -> np.ndarray:
    scores_array = np.asarray(scores, dtype=np.float64).reshape(-1)
    valid_array = np.asarray(valid, dtype=bool).reshape(-1)
    indices = np.flatnonzero(valid_array)
    return indices[np.argsort(-scores_array[indices], kind="stable")]


def hard_temporal_nms(
    windows: Any,
    order: Sequence[int],
    *,
    iou_threshold: float = NMS_IOU_THRESHOLD,
    max_candidates: int = MAX_CANDIDATES,
) -> np.ndarray:
    """Greedy hard NMS in the caller-provided score/tie order."""
    if not 0.0 <= float(iou_threshold) <= 1.0:
        raise Round31EvaluatorError("iou_threshold must be in [0, 1]")
    windows_array = np.asarray(windows, dtype=np.float64).reshape(-1, 2)
    remaining = np.asarray(list(order), dtype=np.int64)
    keep: list[int] = []
    while len(remaining) and len(keep) < int(max_candidates):
        chosen = int(remaining[0])
        keep.append(chosen)
        remaining = remaining[1:]
        if len(remaining):
            overlaps = temporal_iou(
                windows_array[chosen : chosen + 1], windows_array[remaining]
            )[0]
            remaining = remaining[overlaps <= float(iou_threshold)]
    return np.asarray(keep, dtype=np.int64)


def select_full_grid_candidates(
    windows: Any,
    scores: Any,
    valid: Any,
    *,
    native_order: Optional[Sequence[int]] = None,
    iou_threshold: float = NMS_IOU_THRESHOLD,
    max_candidates: int = MAX_CANDIDATES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return raw native top-k, full-grid NMS order, and NMS-selected IDs.

    native_order is supplied by torch.topk in the CLI. Keeping it as the
    prefix makes native top-k tie behavior explicit and guarantees that NMS
    and raw share the same top-1 candidate.
    """
    scores_array = np.asarray(scores, dtype=np.float64).reshape(-1)
    valid_array = np.asarray(valid, dtype=bool).reshape(-1)
    windows_array = np.asarray(windows, dtype=np.float64).reshape(-1, 2)
    if len(scores_array) != len(valid_array) or len(scores_array) != len(windows_array):
        raise Round31EvaluatorError(
            "scores, valid mask, and windows must have equal length"
        )
    all_valid = stable_score_order(scores_array, valid_array)
    if native_order is None:
        native = all_valid[: int(max_candidates)]
    else:
        native = np.asarray(list(native_order), dtype=np.int64)
        if len(native) > int(max_candidates):
            native = native[: int(max_candidates)]
        if len(set(native.tolist())) != len(native):
            raise Round31EvaluatorError("native top-k contains duplicate candidate IDs")
        if any(
            int(index) < 0
            or int(index) >= len(scores_array)
            or not valid_array[int(index)]
            for index in native
        ):
            raise Round31EvaluatorError("native top-k contains an invalid candidate")
    native_set = set(int(index) for index in native.tolist())
    suffix = np.asarray(
        [int(index) for index in all_valid.tolist() if int(index) not in native_set],
        dtype=np.int64,
    )
    full_order = np.concatenate([native, suffix]).astype(np.int64, copy=False)
    nms = hard_temporal_nms(
        windows_array,
        full_order,
        iou_threshold=iou_threshold,
        max_candidates=max_candidates,
    )
    if len(native) and (not len(nms) or int(nms[0]) != int(native[0])):
        raise Round31EvaluatorError("NMS top-1 diverged from native top-k top-1")
    return native, full_order, nms


def _window_tuples(windows: np.ndarray) -> list[tuple[float, float]]:
    return [
        (float(row[0]), float(row[1])) for row in np.asarray(windows).reshape(-1, 2)
    ]


def _matching_coverage(matrix: np.ndarray, threshold: float = 0.7) -> int:
    assigned: dict[int, int] = {}

    def match(candidate: int, seen: set[int]) -> bool:
        for gt in np.flatnonzero(matrix[candidate] >= threshold):
            gt_index = int(gt)
            if gt_index in seen:
                continue
            seen.add(gt_index)
            if gt_index not in assigned or match(assigned[gt_index], seen):
                assigned[gt_index] = candidate
                return True
        return False

    for candidate in range(matrix.shape[0]):
        match(candidate, set())
    return len(assigned)


def candidate_diagnostics(
    windows: Any,
    scores: Any,
    valid: Any,
    raw_ids: Sequence[int],
    nms_ids: Sequence[int],
    *,
    gt_windows: Optional[Any] = None,
) -> dict[str, Any]:
    """Record candidate repetition, suppression, and post-selection coverage."""
    windows_array = np.asarray(windows, dtype=np.float64).reshape(-1, 2)
    np.asarray(scores, dtype=np.float64).reshape(-1)
    valid_array = np.asarray(valid, dtype=bool).reshape(-1)
    raw = np.asarray(raw_ids, dtype=np.int64)
    nms = np.asarray(nms_ids, dtype=np.int64)
    raw_windows = windows_array[raw] if len(raw) else np.zeros((0, 2))
    nms_windows = windows_array[nms] if len(nms) else np.zeros((0, 2))

    def pair_stats(selected: np.ndarray) -> tuple[float, float]:
        if len(selected) < 2:
            return 0.0, 0.0
        pair = temporal_iou(windows_array[selected], windows_array[selected])
        triangle = pair[np.triu_indices(len(selected), 1)]
        return float(triangle.mean()), float((triangle > 0.7).mean())

    raw_pair_mean, raw_pair_high = pair_stats(raw)
    nms_pair_mean, nms_pair_high = pair_stats(nms)
    result: dict[str, Any] = {
        "valid_candidate_count": int(valid_array.sum()),
        "raw_count": int(len(raw)),
        "nms_count": int(len(nms)),
        "nms_suppressed_count": int(
            len(raw) - len(set(raw.tolist()) & set(nms.tolist()))
        ),
        "nms_fill_in_count": int(len(nms) - len(set(raw.tolist()) & set(nms.tolist()))),
        "raw_unique_window_count": int(len(set(_window_tuples(raw_windows)))),
        "nms_unique_window_count": int(len(set(_window_tuples(nms_windows)))),
        "raw_duplicate_count": int(len(raw) - len(set(_window_tuples(raw_windows)))),
        "nms_duplicate_count": int(len(nms) - len(set(_window_tuples(nms_windows)))),
        "raw_pair_iou_mean": raw_pair_mean,
        "raw_pair_iou_gt07_rate": raw_pair_high,
        "nms_pair_iou_mean": nms_pair_mean,
        "nms_pair_iou_gt07_rate": nms_pair_high,
        "native_top1": int(raw[0]) if len(raw) else None,
        "nms_top1": int(nms[0]) if len(nms) else None,
        "top1_same": bool(
            (not len(raw) and not len(nms))
            or (len(raw) and len(nms) and int(raw[0]) == int(nms[0]))
        ),
        "native_prefix_preserved_count": int(
            len(set(raw.tolist()) & set(nms.tolist()))
        ),
        "native_prefix_removed_count": int(len(set(raw.tolist()) - set(nms.tolist()))),
    }
    if gt_windows is None:
        return result
    gt = np.asarray(gt_windows, dtype=np.float64).reshape(-1, 2)
    result["gt_count"] = int(len(gt))
    if not len(gt):
        result["matching_coverage07_count"] = 0
        return result
    matrix = temporal_iou(nms_windows, gt)
    for k in (10, 30):
        prefix = matrix[: min(k, len(matrix))]
        best = prefix.max(0) if len(prefix) else np.zeros(len(gt))
        for threshold in (0.5, 0.7):
            result[f"gt_coverage_top{k}_{threshold:g}_count"] = int(
                (best >= threshold - 1.0e-6).sum()
            )
    result["matching_coverage07_count"] = _matching_coverage(matrix)
    return result


def metric_aliases(metrics: Mapping[str, Any]) -> dict[str, float]:
    """Keep official full metric names and add runner-compatible aliases."""
    result = {
        str(key): float(value) for key, value in metrics.items() if _finite(value)
    }
    aliases = {
        "MR-mAP": ("MR-mAP-Full_Avg", "MR-mAP-Full-Avg", "MR_mAP_Full_Avg"),
        "mAP@0.75": ("MR-mAP-Full_0.75", "MR-mAP-Full-0.75", "mAP75"),
        "R1@0.7": ("MR-R1-Full_0.7", "MR-R1-Full-0.7", "R1@70"),
    }
    for canonical, candidates in aliases.items():
        if canonical not in result:
            for key in candidates:
                if key in result:
                    result[canonical] = result[key]
                    break
    return result


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def official_metrics(value: Any) -> dict[str, float]:
    if isinstance(value, Mapping) and isinstance(value.get("metrics"), Mapping):
        value = value["metrics"]
    if not isinstance(value, Mapping):
        raise Round31EvaluatorError("official metric result must be a mapping")
    metrics = metric_aliases(value)
    required = ("MR-mAP", "mAP@0.75", "R1@0.7")
    missing = [key for key in required if key not in metrics]
    if missing:
        raise Round31EvaluatorError(
            f"official metrics missing required aliases: {missing}"
        )
    return metrics


def _config_get(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _spec_kwargs(config: Any, name: str) -> Mapping[str, Any]:
    spec = _config_get(config, name, {})
    if isinstance(spec, Mapping) and isinstance(spec.get("kwargs"), Mapping):
        return spec["kwargs"]
    return {}


def _resolve_path(value: Any, *, base: Optional[Path] = None) -> Optional[Path]:
    if value in (None, ""):
        return None
    path = Path(str(value))
    if not path.is_absolute() and base is not None:
        path = base / path
    return path.resolve()


def _dedupe_paths(paths: Sequence[Path]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for path in paths:
        value = str(path)
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _subprocess_pythonpath(root: Path, project_root: Path) -> str:
    local_package = Path(__file__).resolve().parent.parent
    paths = [
        local_package,
        root / "src",
        project_root,
        root,
        root / "src",
        root / "stage51",
        root / "src",
        project_root / "scripts",
    ]
    existing = os.environ.get("PYTHONPATH")
    if existing:
        paths.extend(Path(value) for value in existing.split(os.pathsep) if value)
    return os.pathsep.join(_dedupe_paths(paths))


def _request_value(request: Any, name: str, default: Any = None) -> Any:
    if isinstance(request, Mapping):
        return request.get(name, default)
    return getattr(request, name, default)


def _process_group_kwargs() -> dict[str, Any]:
    if os.name == "nt":
        return {}
    # The evaluator intentionally runs in its own session so the worker can
    # reap it precisely. On Linux, also arm a kernel parent-death signal so
    # an external watchdog killing the worker cannot leave this child alive.
    kwargs: dict[str, Any] = {
        "start_new_session": True,
        "preexec_fn": _set_parent_death_signal,
    }
    return kwargs


def _set_parent_death_signal() -> None:
    if not sys.platform.startswith("linux"):
        return
    import ctypes

    parent_pid = os.getppid()
    libc = ctypes.CDLL(None, use_errno=True)
    prctl = libc.prctl
    prctl.argtypes = [
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
    ]
    prctl.restype = ctypes.c_int
    # PR_SET_PDEATHSIG is 1 on Linux. SIGKILL makes the cleanup fail-closed.
    if prctl(1, int(signal.SIGKILL), 0, 0, 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, "prctl(PR_SET_PDEATHSIG) failed")
    # Close the tiny fork/exec race: if the parent already died, terminate
    # rather than inheriting a long-lived orphan.
    if os.getppid() != parent_pid:
        os.kill(os.getpid(), signal.SIGKILL)


def _kill_process_group(process: Any) -> None:
    try:
        if os.name != "nt":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except (ProcessLookupError, OSError):
        try:
            process.kill()
        except (ProcessLookupError, OSError):
            pass


def _reap_process(process: Any) -> tuple[str, str]:
    try:
        return process.communicate(timeout=5.0)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except (ProcessLookupError, OSError):
            pass
        return process.communicate()


def _run_process(
    command: Sequence[str],
    *,
    env: Mapping[str, str],
    cwd: Optional[Path],
    timeout_seconds: float,
) -> tuple[int, str, str]:
    process = subprocess.Popen(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=dict(env),
        cwd=str(cwd) if cwd is not None and cwd.exists() else None,
        **_process_group_kwargs(),
    )
    try:
        stdout, stderr = process.communicate(timeout=float(timeout_seconds))
    except subprocess.TimeoutExpired as exc:
        _kill_process_group(process)
        stdout, stderr = _reap_process(process)
        raise Round31EvaluatorError(
            f"official evaluation timed out after {timeout_seconds:g}s"
        ) from exc
    except BaseException:
        _kill_process_group(process)
        _reap_process(process)
        raise
    return process.returncode, stdout, stderr


@dataclass
class Round31OfficialEvaluator:
    """Official evaluator facade used by the round1-compatible runtime."""

    root: Path
    project_root: Path
    wrapper: Optional[Path] = None
    python_executable: str = sys.executable
    timeout_seconds: float = 240.0
    batch_size: int = 64
    num_workers: int = 2
    precision: str = "bf16"
    nms_iou_threshold: float = NMS_IOU_THRESHOLD
    max_candidates: int = MAX_CANDIDATES

    def __post_init__(self) -> None:
        self.root = Path(self.root).resolve()
        self.project_root = Path(self.project_root).resolve()
        self.wrapper = None if self.wrapper is None else Path(self.wrapper).resolve()
        if int(self.max_candidates) != MAX_CANDIDATES:
            raise Round31EvaluatorError("round31 selection fixes max_candidates=30")
        if abs(float(self.nms_iou_threshold) - NMS_IOU_THRESHOLD) > 1.0e-12:
            raise Round31EvaluatorError("round31 selection fixes temporal NMS IoU=0.5")
        if float(self.timeout_seconds) <= 0:
            raise Round31EvaluatorError("evaluation timeout must be positive")
        if self.precision not in {"bf16", "fp32"}:
            raise Round31EvaluatorError(
                f"unsupported evaluator precision: {self.precision}"
            )

    def _command(
        self,
        request: Any,
        output_dir: Path,
        *,
        request_id: str,
        checkpoint_sha256: str,
    ) -> list[str]:
        checkpoint = _resolve_path(_request_value(request, "checkpoint"))
        if checkpoint is None:
            raise Round31EvaluatorError("official request has no checkpoint")
        config_path = _resolve_path(_request_value(request, "config_path"))
        device = str(_request_value(request, "device", "cuda:0"))
        batch_size = int(_request_value(request, "batch_size", self.batch_size))
        num_workers = int(_request_value(request, "num_workers", self.num_workers))
        precision = str(_request_value(request, "precision", self.precision))
        command = [
            self.python_executable,
            "-u",
            "-m",
            "eventfieldnet.evaluation_cli",
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(output_dir),
            "--root",
            str(self.root),
            "--project-root",
            str(self.project_root),
            "--device",
            device,
            "--batch-size",
            str(batch_size),
            "--num-workers",
            str(num_workers),
            "--precision",
            precision,
            "--max-candidates",
            str(MAX_CANDIDATES),
            "--nms-iou",
            str(NMS_IOU_THRESHOLD),
            "--request-id",
            request_id,
            "--checkpoint-sha256",
            checkpoint_sha256,
        ]
        if config_path is not None:
            command.extend(["--config", str(config_path)])
        if self.wrapper is not None:
            command.extend(["--wrapper", str(self.wrapper)])
        return command

    def evaluate(self, request: Any) -> dict[str, Any]:
        output_dir = _resolve_path(_request_value(request, "output_dir"))
        if output_dir is None:
            raise Round31EvaluatorError("official request has no output_dir")
        if output_dir.is_symlink():
            raise Round31EvaluatorError(
                f"refusing symlink evaluator output: {output_dir}"
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = _resolve_path(_request_value(request, "checkpoint"))
        if checkpoint is None or not checkpoint.exists():
            raise Round31EvaluatorError("official request checkpoint does not exist")
        checkpoint_sha256 = _sha256_file(checkpoint)
        request_id = secrets.token_hex(16)
        env = dict(os.environ)
        env["PYTHONPATH"] = _subprocess_pythonpath(self.root, self.project_root)
        command = self._command(
            request,
            output_dir,
            request_id=request_id,
            checkpoint_sha256=checkpoint_sha256,
        )
        started = time.monotonic()
        try:
            returncode, stdout, stderr = _run_process(
                command,
                env=env,
                cwd=self.project_root,
                timeout_seconds=self.timeout_seconds,
            )
        except BaseException as exc:
            _atomic_json(
                output_dir / "evaluator_status.json",
                {
                    "schema": SCHEMA,
                    "status": "failed",
                    "request_id": request_id,
                    "checkpoint_sha256": checkpoint_sha256,
                    "command": command,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "elapsed_seconds": time.monotonic() - started,
                },
            )
            raise
        (output_dir / "subprocess.stdout.txt").write_text(stdout, encoding="utf-8")
        (output_dir / "subprocess.stderr.txt").write_text(stderr, encoding="utf-8")
        if returncode != 0:
            _atomic_json(
                output_dir / "evaluator_status.json",
                {
                    "schema": SCHEMA,
                    "status": "failed",
                    "request_id": request_id,
                    "checkpoint_sha256": checkpoint_sha256,
                    "returncode": returncode,
                    "command": command,
                    "elapsed_seconds": time.monotonic() - started,
                    "stderr_tail": stderr[-4000:],
                },
            )
            raise Round31EvaluatorError(
                f"official evaluation subprocess exited {returncode}: {stderr[-1000:]}"
            )
        metrics_path = output_dir / "metrics.json"
        if not metrics_path.exists():
            raise Round31EvaluatorError(
                f"evaluation subprocess produced no metrics.json: {output_dir}"
            )
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping) or payload.get("status") != "complete":
            raise Round31EvaluatorError(
                f"invalid round31 evaluation result: {metrics_path}"
            )
        if payload.get("request_id") != request_id:
            raise Round31EvaluatorError(
                "evaluation result request_id does not match this invocation"
            )
        if payload.get("checkpoint_sha256") != checkpoint_sha256:
            raise Round31EvaluatorError(
                "evaluation result checkpoint hash does not match this invocation"
            )
        metrics = official_metrics(payload.get("metrics", {}))
        result = dict(payload)
        result["metrics"] = metrics
        result["raw_metrics"] = official_metrics(payload.get("raw_metrics", {}))
        result["evaluator_elapsed_seconds"] = time.monotonic() - started
        _atomic_json(
            output_dir / "evaluator_status.json",
            {
                "schema": SCHEMA,
                "status": "complete",
                "request_id": request_id,
                "checkpoint_sha256": checkpoint_sha256,
                "command": command,
                "elapsed_seconds": result["evaluator_elapsed_seconds"],
                "metrics": metrics,
                "raw_metrics": result["raw_metrics"],
            },
        )
        return result

    __call__ = evaluate


def build_evaluator(
    config: Any = None,
    *,
    root: str | Path | None = None,
    project_root: str | Path | None = None,
    wrapper: str | Path | None = None,
    python_executable: str | None = None,
    timeout_seconds: float = 240.0,
    batch_size: int = 64,
    num_workers: int = 2,
    precision: str = "bf16",
    nms_iou_threshold: float = NMS_IOU_THRESHOLD,
    max_candidates: int = MAX_CANDIDATES,
    **kwargs: Any,
) -> Round31OfficialEvaluator:
    """Config-first factory compatible with training.instantiate/evaluator_from_config."""
    eval_kwargs = _spec_kwargs(config, "evaluator_factory")
    if root is None:
        root = eval_kwargs.get("root")
    if project_root is None:
        project_root = eval_kwargs.get("project_root")
    if wrapper is None:
        wrapper = eval_kwargs.get("wrapper")
    if root is None:
        raise Round31EvaluatorError("build_evaluator requires root")
    root_path = Path(str(root)).resolve()
    project_path = (
        Path(str(project_root)).resolve()
        if project_root not in (None, "")
        else (root_path / "code").resolve()
    )
    wrapper_path = _resolve_path(wrapper, base=root_path)
    if wrapper_path is None:
        candidate = root_path / "src" / "backbone" / "conditioned" / "official_eval.py"
        wrapper_path = candidate if candidate.exists() else None
    return Round31OfficialEvaluator(
        root=root_path,
        project_root=project_path,
        wrapper=wrapper_path,
        python_executable=str(python_executable or sys.executable),
        timeout_seconds=float(timeout_seconds),
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        precision=str(precision),
        nms_iou_threshold=float(nms_iou_threshold),
        max_candidates=int(max_candidates),
    )


__all__ = [
    "SCHEMA",
    "SELECTION_POLICY",
    "NMS_IOU_THRESHOLD",
    "MAX_CANDIDATES",
    "Round31EvaluatorError",
    "Round31OfficialEvaluator",
    "build_evaluator",
    "temporal_iou",
    "hard_temporal_nms",
    "select_full_grid_candidates",
    "candidate_diagnostics",
    "metric_aliases",
    "official_metrics",
]
