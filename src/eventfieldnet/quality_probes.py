"""Actual weighted loss derivatives and applied routing, never proxy slopes."""

import torch

LOSS_NAMES = ("rank", "evidence", "support", "transition", "endpoint")


def norm(gs):
    return (
        sum(float(g.detach().float().square().sum()) for g in gs if g is not None)
        ** 0.5
    )


def scalar_tree(value):
    if isinstance(value, torch.Tensor):
        return (
            float(value.detach())
            if value.numel() == 1
            else {"shape": list(value.shape)}
        )
    if isinstance(value, dict):
        return {str(k): scalar_tree(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [scalar_tree(v) for v in value]
    return value


def cosine(a, b):
    numerator = sum(
        float((x.detach().float() * y.detach().float()).sum()) for x, y in zip(a, b)
    )
    denominator = norm(a) * norm(b)
    return numerator / denominator if denominator > 1e-20 else 0.0


def actual_probe(model, field, terms, result, strict=True):
    """No extra forwards or random draws; retain graph for real training.

    `carrier_raw` is the actual original raw tensor retained by the R65
    selector. dL/draw is measured by autograd, including all centering and
    scaling derivatives; a local tanh slope is not called this quantity.
    """
    terms.geometry.valid.bool()
    weighted = {
        name: model.loss_weights[name] * getattr(terms, name) for name in LOSS_NAMES
    }
    plain_total = sum(weighted.values())
    numeric_error = float((result.loss.detach() - plain_total.detach()).abs())
    record = {
        "mode": model.r65_mode,
        "actual_loss": float(result.loss.detach()),
        "recomposition_numeric_error": numeric_error,
        "routing_zero_value_only_not_zero_gradient": model.r65_mode
        == "protect_s_gradient",
        "carrier_stats": scalar_tree(getattr(field, "r65_carrier_stats", {})),
        "training_details": scalar_tree(getattr(model, "r65_last_training", {})),
        "support_details": scalar_tree(getattr(field, "r65_s_stats", {})),
        "derivative_scope": "autograd of each actual weighted loss; actual_total includes zero-valued routing correction",
        "loss_to_field_gradients": {},
    }
    if strict:
        assert numeric_error <= max(2e-6, abs(float(result.loss.detach())) * 2e-6), (
            "R65 actual total is not weighted loss recomposition"
        )

    field_inputs = [
        (name, getattr(field, name))
        for name in (
            "carrier",
            "evidence",
            "support",
            "transition_start",
            "transition_end",
        )
    ]
    carrier_raw = getattr(field, "r65_carrier_raw", None)
    if carrier_raw is not None:
        field_inputs.append(("carrier_raw", carrier_raw))
    params = [
        (name, p)
        for name, p in model.selector.s_projection.named_parameters()
        if p.requires_grad
    ]
    requested = [tensor for _, tensor in field_inputs] + [p for _, p in params]
    active_ids = [i for i, tensor in enumerate(requested) if tensor.requires_grad]
    active = [requested[i] for i in active_ids]
    gradients = {}
    for key, value in dict(weighted, actual_total=result.loss).items():
        raw_grads = (
            torch.autograd.grad(value, active, retain_graph=True, allow_unused=True)
            if value.requires_grad
            else [None] * len(active)
        )
        all_grads = [None] * len(requested)
        for i, grad in zip(active_ids, raw_grads):
            all_grads[i] = grad
        part = {}
        for (name, tensor), grad in zip(field_inputs, all_grads[: len(field_inputs)]):
            entry = {"connected": grad is not None, "norm": norm([grad])}
            if grad is not None:
                entry["abs_mean"] = float(grad.detach().float().abs().mean())
                entry["finite"] = bool(torch.isfinite(grad).all())
                if strict:
                    assert entry["finite"], (
                        "nonfinite actual derivative: " + key + " -> " + name
                    )
            part[name] = entry
        record["loss_to_field_gradients"][key] = part
        gradients[key] = [
            torch.zeros_like(p) if grad is None else grad.detach()
            for (_, p), grad in zip(params, all_grads[len(field_inputs) :])
        ]
    rank, support = gradients["rank"], gradients["support"]
    total = gradients["actual_total"]
    other = [
        sum(gradients[name][i] for name in LOSS_NAMES if name != "rank")
        for i in range(len(params))
    ]
    effective_rank = [a - b for a, b in zip(total, other)]
    plain = [sum(gradients[name][i] for name in LOSS_NAMES) for i in range(len(params))]
    delta = [a - b for a, b in zip(total, plain)]
    routing = {
        "scope": "only selector.s_projection; other groups require their own probes",
        "raw_rank_norm": norm(rank),
        "raw_support_aux_norm": norm(support),
        "raw_s_aux_norm": norm(support),
        "raw_rank_support_cosine": cosine(rank, support),
        "plain_sum_norm": norm(plain),
        "actual_total_norm": norm(total),
        "effective_rank_norm": norm(effective_rank),
        "effective_rank_support_cosine": cosine(effective_rank, support),
        "actual_total_minus_plain_norm": norm(delta),
    }
    # For the protected route, compare the actual total derivative with the
    # declared projection, not just an algebraically precomputed correction.
    if model.r65_mode == "protect_s_gradient":
        dot = sum((a.float() * b.float()).sum() for a, b in zip(rank, support))
        denom = sum(b.float().square().sum() for b in support)
        coefficient = torch.where(
            denom > 0.0,
            torch.minimum(dot, dot.new_zeros(())) / denom.clamp_min(1e-30),
            dot.new_zeros(()),
        )
        expected = [a - coefficient * b for a, b in zip(rank, support)]
        error = norm([a - b for a, b in zip(effective_rank, expected)])
        routing.update(
            expected_projected_rank_norm=norm(expected),
            applied_projection_error=error,
            projection_coefficient=float(coefficient),
        )
        if strict:
            assert error <= max(1e-5, norm(expected) * 2e-4), (
                "R65 actual applied S projection differs from declared route"
            )
    record["s_encoder_routing"] = routing
    record["routing"] = routing
    record["support_to_deployed_s_gradient"] = record["loss_to_field_gradients"][
        "support"
    ]["support"]["norm"]
    return record
