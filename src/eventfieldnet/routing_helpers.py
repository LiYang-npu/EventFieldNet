"""R67 coverage-only local S readout, retaining R66 bounded composition.

Only the deterministic local readout module is replaced. The inherited
edge-S, token gate, anchors, eligibility and score composition stay in R66.
No training labels, random draws or new auxiliary objectives are used here.
"""

import torch
from torch import nn
from torch.nn import functional as F


class CoverageOnlyReadout(nn.Module):
    """One scalar times coverage excess over its exact uniform-gate null.

    At uniform gates, coverage == 2 * uniform and the residual is zero for
    any scalar weight. For the same anchor/reference, expanding outside
    that reference leaves this local score unchanged when coverage and
    inside count do not change, even if whole-window density changes.

    Initial weight 1 matches only the partial derivative with respect to
    coverage of the original J07 z=2*(coverage*purity-uniform) at purity=.5.
    The complete token-gate Jacobian differs: this arm removes purity's
    derivative. This is a structural contrast, not an identical-gradient
    reparameterization or a guarantee that the entire S is expansion-neutral.
    """

    def __init__(self):
        super().__init__()
        # torch.tensor consumes no random initialization draws. Retain the
        # existing registered module slot so the removed 3-vector is not a
        # dead trainable parameter and R66's bounded composition is reused.
        self.weight = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))

    def forward(self, details):
        with torch.autocast(device_type=details["coverage"].device.type, enabled=False):
            coverage = details["coverage"].float()
            uniform = details["uniform"].float()
            features = (coverage - 2.0 * uniform)[..., None]
            raw = F.linear(features, self.weight.float()[None]).squeeze(-1)
            local = torch.tanh(raw).masked_fill(~details["eligible"], 0.0)
            return local, features


def prepare_coverage_residual(selector):
    """Replace J07's existing readout slot; leave all other modules intact."""
    if not hasattr(selector, "r59_local_support"):
        raise ValueError("R67 coverage readout requires the D07 token gate")
    if not hasattr(selector, "r66_centered_s_readout"):
        raise ValueError("R67 coverage readout replaces an existing J07 readout")
    if isinstance(selector.r66_centered_s_readout, CoverageOnlyReadout):
        raise ValueError("R67 coverage readout is already installed")
    selector.r66_centered_s_readout = CoverageOnlyReadout()
