"""Independent round31 trifield model, losses, and probes."""

from .adapter import (
    DEFAULT_PARENT_FACTORY,
    ROUND31_VARIANTS,
    Round31BaseModel,
    Round31Options,
    TriFieldRound31Model,
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
from .probes import (
    Round31ProbeSuite,
    build_probe_suite,
    probe_round31_all_losses,
    probe_round31_candidate_pairs,
    probe_round31_score_bounds,
    probe_round31_support_centering,
    probe_round31_support_pair,
    run_round31_probes,
)
from .selector import Round31FieldScoreHeads

__all__ = [
    "DEFAULT_PARENT_FACTORY",
    "ROUND31_VARIANTS",
    "Round31BaseModel",
    "Round31Options",
    "TriFieldRound31Model",
    "Round31FieldScoreHeads",
    "IndependentEndpointTargets",
    "compute_loss_terms",
    "gt_balanced_kl_rank_loss",
    "independent_endpoint_targets",
    "official_ordinal_margin_loss",
    "build_model",
    "build_trifield_round31_model",
    "Round31ProbeSuite",
    "build_probe_suite",
    "probe_round31_all_losses",
    "probe_round31_candidate_pairs",
    "probe_round31_score_bounds",
    "probe_round31_support_centering",
    "probe_round31_support_pair",
    "run_round31_probes",
]
