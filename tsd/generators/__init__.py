"""
Synthetic data generators.

Available generators:
- IndependentMarginalsGenerator: Baseline sampling from marginal distributions
- CTGANGenerator: Deep learning with Conditional Tabular GAN (requires sdv)
- DPBayesianNetworkGenerator: Differentially private Bayesian network (requires DataSynthesizer)
- generate_synthpop: R synthpop wrapper (requires R installation)
- GReaTGenerator: Language model-based generation (requires be-great + GPU)

Heavy-dependency generators are imported lazily to avoid ImportError
when optional packages are not installed.
"""

from tsd.generators.independent_marginals import (
    IndependentMarginalsGenerator,
    generate_independent_marginals,
)


def __getattr__(name):
    """Lazy imports for generators that require optional dependencies."""
    _ctgan = {"CTGANGenerator", "generate_ctgan"}
    _dpbn = {"DPBayesianNetworkGenerator", "generate_dp_bayesian_network"}
    _synthpop = {"generate_synthpop"}
    _great = {"GReaTGenerator", "generate_great"}

    if name in _ctgan:
        from tsd.generators.ctgan_generator import CTGANGenerator, generate_ctgan

        return CTGANGenerator if name == "CTGANGenerator" else generate_ctgan

    if name in _dpbn:
        from tsd.generators.dp_bayesian_network import (
            DPBayesianNetworkGenerator,
            generate_dp_bayesian_network,
        )

        return (
            DPBayesianNetworkGenerator
            if name == "DPBayesianNetworkGenerator"
            else generate_dp_bayesian_network
        )

    if name in _synthpop:
        from tsd.generators.synthpop_wrapper import generate_synthpop

        return generate_synthpop

    if name in _great:
        from tsd.generators.great_generator import GReaTGenerator, generate_great

        return GReaTGenerator if name == "GReaTGenerator" else generate_great

    raise AttributeError(f"module 'tsd.generators' has no attribute {name!r}")


__all__ = [
    "IndependentMarginalsGenerator",
    "generate_independent_marginals",
    "CTGANGenerator",
    "generate_ctgan",
    "DPBayesianNetworkGenerator",
    "generate_dp_bayesian_network",
    "generate_synthpop",
    "GReaTGenerator",
    "generate_great",
]
