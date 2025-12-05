"""Tests for rank correlation methods."""

import numpy as np
import pytest
from pyggm.preprocessing.correlation import rank_correlation


def test_spearman_perfect_monotone():
    """Perfect monotone relationship should give correlation ~1."""
    X = np.column_stack([np.arange(100), np.arange(100)])
    corr = rank_correlation(X, method='spearman')
    assert np.isclose(corr[0, 1], 1.0, atol=1e-10)
    assert np.allclose(np.diag(corr), 1.0)


def test_kendall_perfect_monotone():
    """Perfect monotone relationship should give correlation ~1."""
    X = np.column_stack([np.arange(50), np.arange(50)])
    corr = rank_correlation(X, method='kendall')
    assert np.isclose(corr[0, 1], 1.0, atol=1e-10)


def test_sine_transform_range():
    """Transformed correlations should be in [-1, 1]."""
    np.random.seed(42)
    X = np.random.randn(100, 10)
    for method in ['spearman', 'kendall']:
        corr = rank_correlation(X, method=method)
        assert np.all(corr >= -1.0)
        assert np.all(corr <= 1.0)
        assert np.allclose(np.diag(corr), 1.0)


def test_spearman_independence():
    """Independent variables should have correlation near 0."""
    np.random.seed(42)
    X = np.random.randn(200, 2)
    corr = rank_correlation(X, method='spearman')
    # Should be close to 0, but allow some random variation
    assert np.abs(corr[0, 1]) < 0.2


def test_correlation_symmetric():
    """Correlation matrix should be symmetric."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    for method in ['spearman', 'kendall']:
        corr = rank_correlation(X, method=method)
        assert np.allclose(corr, corr.T)


def test_invalid_method():
    """Invalid method should raise ValueError."""
    X = np.random.randn(50, 3)
    with pytest.raises(ValueError, match="method must be"):
        rank_correlation(X, method='invalid')
