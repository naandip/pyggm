"""Utilities for positive semi-definite matrix projection."""

import numpy as np


def project_to_psd(S, eps=1e-7):
    """
    Project a symmetric matrix to a PSD correlation matrix.

    We first apply eigenvalue clipping, which yields the nearest PSD
    matrix in Frobenius norm. We then renormalize to unit diagonal
    to obtain a valid correlation matrix.

    Parameters
    ----------
    S : ndarray of shape (p, p)
        Symmetric matrix (e.g., correlation matrix)
    eps : float, default=1e-7
        Minimum eigenvalue threshold

    Returns
    -------
    S_psd : ndarray of shape (p, p)
        PSD correlation matrix with unit diagonal

    Examples
    --------
    >>> import numpy as np
    >>> from pyggm.utils.psd import project_to_psd
    >>> S = np.array([[1.0, 1.5], [1.5, 1.0]])  # Not PSD
    >>> S_psd = project_to_psd(S)
    >>> np.linalg.eigvalsh(S_psd).min() > 0
    True
    """
    # Ensure symmetry
    S = (S + S.T) / 2

    # Eigenvalue decomposition
    w, V = np.linalg.eigh(S)

    # Clip eigenvalues to be at least eps
    w_clipped = np.clip(w, eps, None)

    # Reconstruct matrix
    S_psd = (V * w_clipped) @ V.T

    # Renormalize to correlation matrix (unit diagonal)
    d = np.sqrt(np.diag(S_psd))
    d[d == 0] = 1.0  # Avoid division by zero
    S_corr = S_psd / np.outer(d, d)
    np.fill_diagonal(S_corr, 1.0)

    return S_corr


def is_psd(S, tol=1e-8):
    """
    Check if matrix is positive semi-definite.

    Parameters
    ----------
    S : ndarray of shape (p, p)
        Matrix to check
    tol : float, default=1e-8
        Tolerance for negative eigenvalues

    Returns
    -------
    is_psd : bool
        True if all eigenvalues are >= -tol

    Examples
    --------
    >>> import numpy as np
    >>> from pyggm.utils.psd import is_psd
    >>> S = np.eye(3)
    >>> is_psd(S)
    True
    >>> S = np.array([[1.0, 2.0], [2.0, 1.0]])  # Negative eigenvalue
    >>> is_psd(S)
    False
    """
    eigvals = np.linalg.eigvalsh(S)
    return np.all(eigvals >= -tol)
