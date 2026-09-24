"""Preserve official requests while overriding their evaluation precision only."""

import dataclasses
import json
from pathlib import Path
from . import evaluator as original


def fp32_request(request):
    if not dataclasses.is_dataclass(request):
        raise TypeError("R66 requires the existing OfficialEvalRequest dataclass")
    revised = dataclasses.replace(request, precision="fp32")
    for field in dataclasses.fields(request):
        if field.name != "precision":
            assert getattr(revised, field.name) is getattr(request, field.name)
    return revised


class Evaluator(original.Round31OfficialEvaluator):
    def _command(self, request, output_dir, **kwargs):
        command = super()._command(fp32_request(request), output_dir, **kwargs)
        index = command.index("-m") + 1
        assert command[index] == "eventfieldnet.evaluation_cli"
        command[index] = "eventfieldnet.precision_evaluation"
        assert command[command.index("--precision") + 1] == "fp32"
        return command

    def evaluate(self, request):
        revised = fp32_request(request)
        result = super().evaluate(revised)
        dest = Path(revised.output_dir)
        receipt = json.loads((dest / "precision_receipt.json").read_text())
        assert (
            receipt["status"] == "passed"
            and receipt["every_forward_context_strict_fp32"]
        )
        assert receipt["actual_evaluation_precision"] == "fp32"
        return result

    __call__ = evaluate


def build_evaluator(config=None, **kwargs):
    kwargs["precision"] = "fp32"
    instance = original.build_evaluator(config, **kwargs)
    instance.__class__ = Evaluator
    return instance
