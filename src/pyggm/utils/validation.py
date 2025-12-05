"""Input validation utilities."""

import numpy as np


def check_array_2d(X, name="X", dtype=None, ensure_finite=True):
    """
    Validate input array is 2D with appropriate properties.

    Parameters
    ----------
    X : array-like
        Input array to validate
    name : str, default="X"
        Name of the array for error messages
    dtype : numpy dtype or None, default=None
        Required dtype. If None, converts to float64.
    ensure_finite : bool, default=True
        Check that array contains only finite values

    Returns
    -------
    X : ndarray
        Validated 2D array

    Raises
    ------
    ValueError
        If input is not 2D or contains invalid values
    """
    X = np.asarray(X, dtype=dtype if dtype is not None else np.float64)

    if X.ndim != 2:
        raise ValueError(
            f"{name} must be a 2D array. Got {X.ndim}D array instead."
        )

    if ensure_finite and not np.all(np.isfinite(X)):
        raise ValueError(
            f"{name} contains NaN or infinite values."
        )

    return X


def check_symmetric(S, name="S", tol=1e-10):
    """
    Check if a matrix is symmetric.

    Parameters
    ----------
    S : ndarray of shape (n, n)
        Matrix to check
    name : str, default="S"
        Name for error messages
    tol : float, default=1e-10
        Tolerance for symmetry check

    Raises
    ------
    ValueError
        If matrix is not square or not symmetric
    """
    if S.shape[0] != S.shape[1]:
        raise ValueError(
            f"{name} must be square. Got shape {S.shape}."
        )

    if not np.allclose(S, S.T, atol=tol):
        raise ValueError(
            f"{name} must be symmetric."
        )
