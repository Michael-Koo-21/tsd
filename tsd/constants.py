"""
Shared constants for the TSD framework.

Single source of truth for method labels, metric mappings, stakeholder archetypes,
colors, and runtime estimates used across analysis, visualization, and VOI modules.
"""

from dataclasses import dataclass


@dataclass
class UserProfile:
    """Predefined user profile with attribute weights."""

    name: str
    description: str
    weights: dict[str, float]


# ── Method metadata ──────────────────────────────────────────────────────────

METHOD_LABELS = {
    "independent_marginals": "Indep. Marginals",
    "ctgan": "CTGAN",
    "dpbn": "DP-BN (PrivBayes)",
    "synthpop": "Synthpop",
    "great": "GReaT",
}

COLORS = {
    "independent_marginals": "#4C72B0",
    "ctgan": "#55A868",
    "dpbn": "#C44E52",
    "synthpop": "#8172B3",
    "great": "#CCB974",
}

# Default runtime data (in minutes) based on empirical observations.
# GReaT: ~90 min on Google Colab T4 GPU; others: <1 min on CPU.
DEFAULT_RUNTIME_MINUTES = {
    "independent_marginals": 0.5,
    "synthpop": 0.5,
    "ctgan": 1.0,
    "dpbn": 0.5,
    "great": 90.0,
}

# Normalized efficiency scores [0,1], higher = faster.
# Derived from approximate observed runtimes (DEFAULT_RUNTIME_MINUTES) with
# qualitative judgment. These are estimates, not precisely instrumented measurements.
# Single source of truth: imported by voi_analysis.py and verify.py.
NORMALIZED_EFFICIENCY = {
    "independent_marginals": 1.0,  # <1 min on CPU
    "synthpop": 1.0,  # <1 min on CPU
    "ctgan": 0.90,  # ~1 min on CPU
    "dpbn": 1.0,  # <1 min on CPU
    "great": 0.24,  # ~90 min on Google Colab T4 GPU
}

# ── Metric definitions ───────────────────────────────────────────────────────

# Map from CSV column name → abstract objective name
METRIC_MAP = {
    "fidelity_auc": "fidelity",
    "privacy_dcr": "privacy",
    "utility_tstr": "utility",
    "fairness_gap": "fairness",
    "efficiency_time": "efficiency",
}

# Direction: True = higher is better, False = lower is better.
# After inversion in normalize_scores(), higher normalized value = better.
METRIC_DIRECTION = {
    "fidelity_auc": False,  # Lower AUC = better fidelity (closer to 0.5)
    "privacy_dcr": True,  # Higher DCR = better privacy
    "utility_tstr": True,  # Higher TSTR ratio = better utility
    "fairness_gap": False,  # Lower gap = better fairness
    "efficiency_time": False,  # Lower time = better efficiency
}

# Human-readable labels for plots
METRIC_LABELS = {
    "fidelity_auc": "Fidelity\n(Propensity AUC)",
    "privacy_dcr": "Privacy\n(DCR)",
    "utility_tstr": "Utility\n(TSTR)",
    "fairness_gap": "Fairness\n(Gap \u2193)",
}

# The four core metrics (excluding efficiency, which is derived from runtime)
CORE_METRICS = ["fidelity_auc", "privacy_dcr", "utility_tstr", "fairness_gap"]

# ── Stakeholder archetypes (Paper Table 3) ────────────────────────────────────

ARCHETYPES = {
    "privacy_first": {
        "name": "Privacy-First Practitioner",
        "description": "Prioritizes data privacy and protection against re-identification",
        "weights": {
            "fidelity": 0.25,
            "privacy": 0.45,
            "utility": 0.15,
            "fairness": 0.10,
            "efficiency": 0.05,
        },
    },
    "utility_first": {
        "name": "Utility-First Practitioner",
        "description": "Prioritizes downstream ML model performance",
        "weights": {
            "fidelity": 0.30,
            "privacy": 0.10,
            "utility": 0.40,
            "fairness": 0.10,
            "efficiency": 0.10,
        },
    },
    "balanced": {
        "name": "Balanced Practitioner",
        "description": "Balanced importance across all objectives",
        "weights": {
            "fidelity": 0.25,
            "privacy": 0.25,
            "utility": 0.25,
            "fairness": 0.15,
            "efficiency": 0.10,
        },
    },
}

# UserProfile versions of archetypes (used by mada_framework)
PROFILES = {
    key: UserProfile(
        name=val["name"],
        description=val["description"],
        weights=val["weights"],
    )
    for key, val in ARCHETYPES.items()
}
