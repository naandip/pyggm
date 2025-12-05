"""Tests for StARS algorithm."""

import numpy as np
import pytest
from pyggm.estimators.stars import stars_select, _compute_instability


def test_compute_instability_range():
    """Instability D should be in [0, 0.5]."""
    # All edges have θ=0.5 (max instability)
    p = 10
    edge_probs = np.full((p, p), 0.5)
    np.fill_diagonal(edge_probs, 0)
    D = _compute_instability(edge_probs)
    assert 0 <= D <= 0.5
    assert np.isclose(D, 0.5, atol=0.01)  # Should be near max

    # All edges have θ=0 or θ=1 (no instability)
    edge_probs_stable = np.zeros((p, p))
    D_stable = _compute_instability(edge_probs_stable)
    assert np.isclose(D_stable, 0.0)


def test_stars_select_returns_valid_alpha():
    """stars_select should return an alpha from the grid."""
    np.random.seed(42)
    X = np.random.randn(100, 10)
    alphas = np.logspace(-2, 0, 10)

    best_alpha, instabilities, edge_probs, prec, cov = stars_select(
        X, alphas, n_subsamples=10, beta=0.05
    )

    assert best_alpha in alphas
    assert len(instabilities) == len(alphas)
    assert edge_probs.shape == (10, 10)
    assert prec.shape == (10, 10)
    assert cov.shape == (10, 10)


def test_stars_instabilities_decreasing():
    """Instabilities should generally decrease with increasing alpha."""
    np.random.seed(42)
    X = np.random.randn(200, 20)
    alphas = np.logspace(-2, 0, 15)

    _, instabilities, _, _, _ = stars_select(X, alphas, n_subsamples=20, beta=0.05)

    # Allow some noise, but overall trend should be decreasing
    # Check that last instability is less than first
    assert instabilities[-1] <= instabilities[0] + 0.1


def test_stars_reproducible():
    """StARS should give same results with same random_state."""
    np.random.seed(42)
    X = np.random.randn(100, 10)
    alphas = np.logspace(-2, 0, 8)

    result1 = stars_select(X, alphas, n_subsamples=10, random_state=42)
    result2 = stars_select(X, alphas, n_subsamples=10, random_state=42)

    assert result1[0] == result2[0]  # Same alpha
    assert np.allclose(result1[1], result2[1])  # Same instabilities


def test_stars_edge_probs_range():
    """Edge probabilities should be in [0, 1]."""
    np.random.seed(42)
    X = np.random.randn(80, 8)
    alphas = np.logspace(-2, 0, 5)

    _, _, edge_probs, _, _ = stars_select(X, alphas, n_subsamples=10, beta=0.05)

    assert np.all(edge_probs >= 0)
    assert np.all(edge_probs <= 1)


def test_stars_symmetric_edge_probs():
    """Edge probabilities matrix should be symmetric."""
    np.random.seed(42)
    X = np.random.randn(100, 10)
    alphas = np.logspace(-2, 0, 5)

    _, _, edge_probs, _, _ = stars_select(X, alphas, n_subsamples=10, beta=0.05)

    assert np.allclose(edge_probs, edge_probs.T)
