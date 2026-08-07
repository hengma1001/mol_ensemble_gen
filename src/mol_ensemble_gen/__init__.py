#   -------------------------------------------------------------
#   Licensed under the MIT License. See LICENSE in project root for information.
#   -------------------------------------------------------------
"""mol_ensemble_gen — biomolecular configuration ensembles with ESMFold2.

Public entry point: :class:`ESMFold2Ensemble` (see :mod:`mol_ensemble_gen.ensemble`).
"""
from __future__ import annotations

from .analysis import EnsembleAnalysis, analyze_run
from .ensemble import (
    EnsembleMember,
    EnsembleSpec,
    ESMFold2Ensemble,
    SamplingParams,
    derive_seed,
)

__version__ = "0.0.2"

__all__ = [
    "ESMFold2Ensemble",
    "EnsembleSpec",
    "SamplingParams",
    "EnsembleMember",
    "derive_seed",
    "analyze_run",
    "EnsembleAnalysis",
]
