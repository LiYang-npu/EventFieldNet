"""Three-field model construction, span scoring, and supervised objectives."""

from .adapter import (
    DEFAULT_PARENT_FACTORY,
    ROUND31_VARIANTS,
    FieldModel,
    ThreeFieldOptions,
    ThreeFieldModel,
    build_model,
    build_trifield_round31_model,
)
from .losses import (
    IndependentEndpointTargets,
    compute_loss_terms,
    gt_balanced_kl_rank_loss,
    independent_endpoint_targets,
    official_ordinal_margin_loss,
)

from .selector import ThreeFieldScoreHeads

__all__ = [
    "DEFAULT_PARENT_FACTORY",
    "ROUND31_VARIANTS",
    'FieldModel',
    'ThreeFieldOptions',
    'ThreeFieldModel',
    'ThreeFieldScoreHeads',
    "IndependentEndpointTargets",
    "compute_loss_terms",
    "gt_balanced_kl_rank_loss",
    "independent_endpoint_targets",
    "official_ordinal_margin_loss",
    "build_model",
    "build_trifield_round31_model",
]
