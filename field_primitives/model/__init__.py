"""Independent round1 model/loss/probe package."""

from .adapter import (
    DEFAULT_PARENT_FACTORY,
    ROUND1_VARIANTS,
    Round1BaseModel,
    Round1Options,
    TriFieldRound1Model,
    build_model,
    build_trifield_round1_model,
)
from .losses import (
    IndependentEndpointTargets,
    compute_round1_loss_terms,
    independent_endpoint_targets,
    official_ordinal_margin_loss,
)

from .selector import Round1FieldScoreHeads

__all__ = [
    "DEFAULT_PARENT_FACTORY",
    "ROUND1_VARIANTS",
    "Round1BaseModel",
    "Round1Options",
    "TriFieldRound1Model",
    "Round1FieldScoreHeads",
    "IndependentEndpointTargets",
    "compute_round1_loss_terms",
    "independent_endpoint_targets",
    "official_ordinal_margin_loss",
    "build_model",
    "build_trifield_round1_model",
]
