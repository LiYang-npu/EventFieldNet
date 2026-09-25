"""Official evaluator with observed strict-FP32 execution, including subprocesses."""

import json
from contextlib import contextmanager
from pathlib import Path
from . import evaluation_cli as original
from .precision import configure_precision, snapshot_precision

_evaluate_checkpoint = original.evaluate_checkpoint


def evaluate_checkpoint(*args, **kwargs):
    precision = kwargs.get("precision", "fp32")
    if precision != "fp32":
        raise ValueError("Every R66 official evaluation must use strict FP32")
    kwargs["precision"] = "fp32"
    configure_precision("fp32")
    previous = original._autocast
    observed = []

    @contextmanager
    def checked_context(torch, device, requested):
        assert requested == "fp32"
        with previous(torch, device, requested):
            state = snapshot_precision()
            state.update(
                actual_cuda_autocast_enabled=bool(torch.is_autocast_enabled()),
                actual_cpu_autocast_enabled=bool(torch.is_autocast_cpu_enabled()),
            )
            assert (
                not state["actual_cuda_autocast_enabled"]
                and not state["actual_cpu_autocast_enabled"]
            )
            assert not state["matmul_allow_tf32"] and not state["cudnn_allow_tf32"]
            assert state["float32_matmul_precision"] == "highest"
            observed.append(state)
            yield

    original._autocast = checked_context
    try:
        result = _evaluate_checkpoint(*args, **kwargs)
    finally:
        original._autocast = previous
    assert observed, "No actual official model-forward precision context observed"
    receipt = dict(
        schema="r66_actual_official_precision_v1",
        status="passed",
        actual_evaluation_precision="fp32",
        forward_context_count=len(observed),
        checkpoint_sha256=result["checkpoint_sha256"],
        request_id=result["request_id"],
        query_count=result["diagnostics"]["query_count"],
        saved_training_precision=result.get("config", {}).get("precision"),
        precision_policy=observed[0],
        every_forward_context_strict_fp32=True,
    )
    Path(kwargs["output_dir"]).joinpath("precision_receipt.json").write_text(
        json.dumps(receipt, indent=2)
    )
    return result


def main(argv=None):
    previous = original.evaluate_checkpoint
    original.evaluate_checkpoint = evaluate_checkpoint
    try:
        return original.main(argv)
    finally:
        original.evaluate_checkpoint = previous


if __name__ == "__main__":
    raise SystemExit(main())
