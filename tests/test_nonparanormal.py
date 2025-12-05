"""Tests for Nonparanormal transformer."""

import numpy as np
import pytest
from scipy import stats
from pyggm import NonparanormalTransformer


def test_reduces_skewness():
    """Transformed marginals should have skewness closer to 0."""
    np.random.seed(42)
    # Create skewed data (e.g., exponential)
    X = np.random.exponential(scale=2.0, size=(200, 5))

    transformer = NonparanormalTransformer()
    X_trans = transformer.fit_transform(X)

    for j in range(X.shape[1]):
        skew_before = stats.skew(X[:, j])
        skew_after = stats.skew(X_trans[:, j])
        # Transformed should be closer to 0 (normal has skew=0)
        assert abs(skew_after) < abs(skew_before)


def test_output_shape_and_dtype():
    """Output should match input shape and be float64."""
    np.random.seed(42)
    X = np.random.randn(100, 10)

    transformer = NonparanormalTransformer()
    X_trans = transformer.fit_transform(X)

    assert X_trans.shape == X.shape
    assert X_trans.dtype == np.float64


def test_transform_raises_on_different_features():
    """Transform should raise if n_features differs from fit."""
    np.random.seed(42)
    X_train = np.random.randn(100, 5)
    X_test = np.random.randn(50, 3)

    transformer = NonparanormalTransformer()
    transformer.fit(X_train)

    with pytest.raises(ValueError, match="features"):
        transformer.transform(X_test)


def test_gaussian_data_unchanged():
    """Gaussian data should remain approximately Gaussian."""
    np.random.seed(42)
    X = np.random.randn(200, 3)

    transformer = NonparanormalTransformer()
    X_trans = transformer.fit_transform(X)

    # Should have similar means and stds
    for j in range(X.shape[1]):
        assert np.abs(np.mean(X_trans[:, j])) < 0.5
        assert np.abs(np.std(X_trans[:, j]) - 1.0) < 0.5


def test_no_infinities():
    """Transformation should not produce infinities."""
    np.random.seed(42)
    X = np.random.exponential(size=(100, 5))

    transformer = NonparanormalTransformer(truncate=True)
    X_trans = transformer.fit_transform(X)

    assert np.all(np.isfinite(X_trans))


def test_fit_returns_self():
    """fit() should return self for chaining."""
    X = np.random.randn(50, 3)
    transformer = NonparanormalTransformer()
    result = transformer.fit(X)
    assert result is transformer
