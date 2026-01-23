"""
Synthetic data generators.

Available generators:
- IndependentMarginalsGenerator: Baseline sampling from marginal distributions
- CTGANGenerator: Deep learning with Conditional Tabular GAN
- DPBayesianNetworkGenerator: Differentially private Bayesian network
- generate_synthpop: R synthpop wrapper (requires R installation)
- GReaTGenerator: Language model-based generation (requires GPU)
"""

from src.generators.independent_marginals import (
    IndependentMarginalsGenerator,
    generate_independent_marginals,
)
from src.generators.ctgan_generator import (
    CTGANGenerator,
    generate_ctgan,
)
from src.generators.dp_bayesian_network import (
    DPBayesianNetworkGenerator,
    generate_dp_bayesian_network,
)
from src.generators.synthpop_wrapper import generate_synthpop
from src.generators.great_generator import (
    GReaTGenerator,
    generate_great,
)

__all__ = [
    'IndependentMarginalsGenerator',
    'generate_independent_marginals',
    'CTGANGenerator',
    'generate_ctgan',
    'DPBayesianNetworkGenerator',
    'generate_dp_bayesian_network',
    'generate_synthpop',
    'GReaTGenerator',
    'generate_great',
]
