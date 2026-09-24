"""R68 fixed-checkpoint field deletion with unchanged anchor selection."""

import types
import torch
from . import model_factory


def build_model(config, *, audit_omit=None, **kwargs):
    model = model_factory.build_model(config, **kwargs)
    if audit_omit is not None:
        names = {
            "E": ("evidence",),
            "S": ("support",),
            "T": ("transition_start", "transition_end"),
        }[audit_omit]
        original = model.selector.compose_score

        def compose(self, field, overrides=None, **other):
            changed = dict(overrides or {})
            for name in names:
                changed[name] = torch.zeros_like(getattr(field, name))
            return original(field, changed, **other)

        model.selector.compose_score = types.MethodType(compose, model.selector)
    return model
