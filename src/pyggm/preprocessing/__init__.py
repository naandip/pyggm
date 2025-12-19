"""Preprocessing utilities for pyggm."""

from .nonparanormal import NonparanormalTransformer
from .correlation import rank_correlation

# Alias for convenience
Nonparanormal = NonparanormalTransformer

__all__ = ["NonparanormalTransformer", "Nonparanormal", "rank_correlation"]
