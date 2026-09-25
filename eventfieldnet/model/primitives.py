"""CPU-reviewable candidate primitives; not imported by any training release.

Edges index i joins tokens i and i+1. Spans use inclusive [s,e].
No annotation or candidate quality enters these inference primitives.
"""

import torch
from torch import nn


def edge_span_mean(edge, token_valid):
    """Mean internal edge score; single-token spans are neutral, not connected."""
    b, length = token_valid.shape
    if edge.shape != (b, max(0, length - 1)):
        raise ValueError("edge shape must be B x (L-1)")
    if length == 0:
        raise ValueError("empty temporal axis")
    alive = token_valid.bool()
    edge_valid = alive[:, :-1] & alive[:, 1:]
    prefix = torch.nn.functional.pad(
        edge.float().masked_fill(~edge_valid, 0).cumsum(-1), (1, 0)
    )
    ids = torch.arange(length, device=edge.device)
    s, e = ids[:, None], ids[None, :]
    widths = e - s + 1
    token_count = torch.nn.functional.pad(alive.long().cumsum(-1), (1, 0))
    valid = (widths > 0)[None] & (
        token_count[:, e + 1] - token_count[:, s] == widths[None]
    )
    score = (prefix[:, e] - prefix[:, s]) / (e - s).clamp_min(1)
    return score.masked_fill(~valid | (widths == 1)[None], 0), valid


class OneStepLatentComposition(nn.Module):
    """3d parameters; two unlabeled modes, not natural-stage probabilities.

    All messages use the original tokens, preventing recursive propagation.
    S, when enabled, supplies edge connection in [-1,1]; otherwise gate is 1.
    """

    def __init__(self, dimension):
        super().__init__()
        # Deterministic initialization consumes no global random stream.
        self.gate_weight = nn.Parameter(torch.zeros(dimension))
        self.mode_a = nn.Parameter(torch.full((dimension,), -0.01))
        self.mode_b = nn.Parameter(torch.full((dimension,), 0.01))

    def forward(self, tokens, token_valid, support=None):
        if tokens.ndim != 3 or token_valid.shape != tokens.shape[:2]:
            raise ValueError("tokens B x L x D and validity B x L required")
        h = tokens.float().masked_fill(~token_valid.bool()[..., None], 0)
        edges = token_valid[:, :-1].bool() & token_valid[:, 1:].bool()
        if support is not None and support.shape != edges.shape:
            raise ValueError("support must describe adjacent edges")
        p = torch.sigmoid(((h[:, 1:] - h[:, :-1]) * self.gate_weight).sum(-1))
        connection = (
            torch.ones_like(p) if support is None else (support.float() + 1) / 2
        )
        mix = (1 - p[..., None]) * self.mode_a + p[..., None] * self.mode_b
        message = (connection[..., None] * mix * h[:, :-1]).masked_fill(
            ~edges[..., None], 0
        )
        update = torch.nn.functional.pad(message, (0, 0, 1, 0))
        return h + update, {"latent_gate": p, "message": message, "valid_edges": edges}
