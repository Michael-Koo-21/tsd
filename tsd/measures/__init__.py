"""
Evaluation measures for synthetic data quality.

Available measures:
- propensity_auc: Fidelity measure (distinguishability)
- dcr_privacy: Distance to closest record privacy measure
- membership_inference_privacy: Membership inference attack privacy
- tstr_utility: Train on Synthetic, Test on Real utility
- fairness_gap: Maximum subgroup utility gap for fairness
"""

from tsd.measures.dcr_privacy import (
    compare_dcr_distributions,
    dcr_privacy,
    dcr_privacy_per_record,
)
from tsd.measures.fairness_gap import (
    compare_fairness_across_methods,
    fairness_gap,
)
from tsd.measures.membership_inference import (
    membership_inference_privacy,
)
from tsd.measures.propensity_auc import (
    propensity_auc,
    propensity_auc_holdout,
)
from tsd.measures.tstr_utility import (
    tstr_utility,
    tstr_utility_per_subgroup,
)

__all__ = [
    "propensity_auc",
    "propensity_auc_holdout",
    "dcr_privacy",
    "dcr_privacy_per_record",
    "compare_dcr_distributions",
    "membership_inference_privacy",
    "tstr_utility",
    "tstr_utility_per_subgroup",
    "fairness_gap",
    "compare_fairness_across_methods",
]
