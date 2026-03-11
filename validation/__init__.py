"""
validation
==========

Statistical quality and privacy-risk validation for anonymised / synthetic datasets.

Public API
----------
Quality checks:
    QualityReport          : Dataclass holding all quality-check results.
    run_quality_checks     : Orchestrate all statistical quality checks.
    compare_marginals      : Per-column chi-square / KS tests.
    compare_correlations   : Correlation-matrix fidelity.
    cross_tab_divergence   : Jensen-Shannon divergence on cross-tabulations.

Privacy metrics:
    membership_inference_auc   : AUC of a classifier that tries to tell real from synthetic.
    duplicate_class_rate       : Fraction of synthetic rows that exactly match a real row.
    new_row_rate               : Fraction of synthetic rows that are entirely novel.
    nearest_neighbor_distance  : Distribution of nearest-neighbour distances.

Adversarial attacks:
    AttackResult     : Dataclass summarising a single attack outcome.
    BaseAttack       : Abstract base class for all attack implementations.
    RedTeamConfig    : Configuration dataclass loaded from YAML.
    RedTeamRunner    : Orchestrates every registered attack and emits a summary.
"""

from .quality import (
    QualityReport,
    compare_marginals,
    compare_correlations,
    cross_tab_divergence,
    run_quality_checks,
)
from .privacy_metrics import (
    membership_inference_auc,
    duplicate_class_rate,
    new_row_rate,
    nearest_neighbor_distance,
)
from .red_team import (
    AttackResult,
    BaseAttack,
    RedTeamConfig,
    RedTeamRunner,
)

__all__ = [
    # quality
    "QualityReport",
    "compare_marginals",
    "compare_correlations",
    "cross_tab_divergence",
    "run_quality_checks",
    # privacy metrics
    "membership_inference_auc",
    "duplicate_class_rate",
    "new_row_rate",
    "nearest_neighbor_distance",
    # red team
    "AttackResult",
    "BaseAttack",
    "RedTeamConfig",
    "RedTeamRunner",
]
