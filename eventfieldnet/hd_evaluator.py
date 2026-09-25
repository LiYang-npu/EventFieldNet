"""Run strict FP32 retrieval and task-selected highlight evaluation in a subprocess."""
from .validation import Evaluator as MomentEvaluator, build_evaluator as build_moment_evaluator


class Evaluator(MomentEvaluator):
    def _command(self, *args, **kwargs):
        command = super()._command(*args, **kwargs)
        index = command.index('-m') + 1
        command[index] = 'eventfieldnet.evaluation'
        return command


def build_evaluator(config=None, **kwargs):
    evaluator = build_moment_evaluator(config, **kwargs)
    evaluator.__class__ = Evaluator
    return evaluator
