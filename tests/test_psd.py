"""Tests for PSD projection utilities."""

import numpy as np
import pytest
from pyggm.utils.psd import project_to_psd, is_psd


def test_project_to_psd_already_psd():
    """PSD matrix should be unchanged (approximately)."""
    S = np.array([[1.0, 0.5], [0.5, 1.0]])
    assert is_psd(S)
    S_proj = project_to_psd(S)
    assert np.allclose(S, S_proj, atol=1e-6)


def test_project_to_psd_non_psd():
    """Non-PSD matrix should become PSD."""
    S = np.array([[1.0, 1.5], [1.5, 1.0]])  # Not PSD (eigenvalues: 2.5, -0.5)
    assert not is_psd(S)
    S_proj = project_to_psd(S)
    assert is_psd(S_proj)
    assert np.allclose(np.diag(S_proj), 1.0)  # Diagonal should be 1


def test_project_to_psd_preserves_symmetry():
    """Projected matrix should remain symmetric."""
    np.random.seed(42)
    S = np.random.randn(10, 10)
    S = (S + S.T) / 2
    S_proj = project_to_psd(S)
    assert np.allclose(S_proj, S_proj.T)


def test_is_psd_identity():
    """Identity matrix is PSD."""
    I = np.eye(5)
    assert is_psd(I)


def test_is_psd_negative_eigenvalues():
    """Matrix with negative eigenvalues is not PSD."""
    S = np.array([[1, 2], [2, 1]])  # Eigenvalues: 3, -1
    assert not is_psd(S)


def test_project_to_psd_unit_diagonal():
    """Projected correlation matrix should have unit diagonal."""
    np.random.seed(123)
    S = np.random.randn(5, 5)
    S = (S + S.T) / 2
    S_proj = project_to_psd(S)
    assert np.allclose(np.diag(S_proj), 1.0, atol=1e-10)
