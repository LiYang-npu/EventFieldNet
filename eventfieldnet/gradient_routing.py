"""Parameters whose ranking-loss gradients use the fixed support multiplier."""


def local_parameters(model):
    return [
        ("selector.r59_local_support.weight", model.selector.r59_local_support.weight),
        *[
            ("selector.r66_centered_s_readout." + name, parameter)
            for name, parameter in model.selector.r66_centered_s_readout.named_parameters()
        ],
    ]
