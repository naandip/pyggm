"""
PyGGM: Gaussian Graphical Models in Python
===========================================

A Python package for Gaussian Graphical Models with multiple regularization
selection methods (StARS, EBIC, CV) and nonparanormal transformations.

Main classes
------------
GaussianGraphicalModel : Main estimator for precision matrix estimation
NonparanormalTransformer : Copula transformation for non-Gaussian data

Utility functions
-----------------
rank_correlation : Compute rank-based correlations with sine transformation
project_to_psd : Project matrix to nearest PSD correlation matrix
is_psd : Check if matrix is positive semi-definite

Examples
--------
>>> import numpy as np
>>> from pyggm import GaussianGraphicalModel
>>> X = np.random.randn(200, 50)
>>> model = GaussianGraphicalModel(method='stars', n_jobs=-1)
>>> model.fit(X)
>>> print(f"Selected alpha: {model.alpha_:.4f}")
>>> print(f"Number of edges: {model.n_edges_}")
"""

from .estimators.ggm import GaussianGraphicalModel
from .preprocessing.nonparanormal import NonparanormalTransformer
from .preprocessing.correlation import rank_correlation
from .utils.psd import project_to_psd, is_psd

# Alias for convenience
Nonparanormal = NonparanormalTransformer

__version__ = "0.1.0"

__all__ = [
    "GaussianGraphicalModel",
    "NonparanormalTransformer",
    "Nonparanormal",  # Alias
    "rank_correlation",
    "project_to_psd",
    "is_psd",
]
