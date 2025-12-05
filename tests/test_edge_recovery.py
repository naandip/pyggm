"""Tests for edge recovery."""

import numpy as np
from pyggm import GaussianGraphicalModel


def test_recovers_strong_edges():
    """Check that strong true edges are more likely to be recovered."""
    np.random.seed(42)
    p = 20
    n = 300

    # Create sparse precision matrix with known strong edges
    true_precision = np.eye(p)
    # Add a few strong edges
    strong_edges = [(0, 1, 0.6), (2, 3, 0.5), (4, 5, 0.4)]
    for i, j, val in strong_edges:
        true_precision[i, j] = val
        true_precision[j, i] = val

    # Ensure PSD by adding to diagonal if needed
    min_eig = np.linalg.eigvalsh(true_precision).min()
    if min_eig < 0.1:
        true_precision += (0.1 - min_eig) * np.eye(p)

    # Generate data
    true_cov = np.linalg.inv(true_precision)
    X = np.random.multivariate_normal(np.zeros(p), true_cov, n)

    # Fit model
    model = GaussianGraphicalModel(method='ebic', gamma=0.5)
    model.fit(X)

    # Check that at least 2 of the 3 strong edges are recovered
    recovered = 0
    for i, j, _ in strong_edges:
        if model.adjacency_[i, j] != 0:
            recovered += 1

    assert recovered >= 2, f"Only recovered {recovered}/3 strong edges"


def test_no_spurious_edges_independence():
    """Independent features should have few edges."""
    np.random.seed(42)
    n, p = 200, 15

    # Generate independent features
    X = np.random.randn(n, p)

    # Fit model with relatively high regularization
    model = GaussianGraphicalModel(method='ebic', gamma=0.5)
    model.fit(X)

    # Should have few edges (allowing some false positives)
    assert model.n_edges_ < p, f"Too many edges: {model.n_edges_} (expected < {p})"
