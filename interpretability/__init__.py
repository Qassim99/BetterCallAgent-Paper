"""Offline interpretability utilities for BetterCallAgent.

The package consumes saved research artifacts. It never calls retrieval,
language-model, or verifier services.
"""

from .error_attribution import attribute_citation_errors
from .gate_surrogate import fit_logistic_additive_surrogate
from .perturbation_faithfulness import evaluate_perturbation_faithfulness
from .ragas_style import evaluate_ragas_style

__all__ = [
    "attribute_citation_errors",
    "evaluate_perturbation_faithfulness",
    "evaluate_ragas_style",
    "fit_logistic_additive_surrogate",
]

__version__ = "0.1.0"
