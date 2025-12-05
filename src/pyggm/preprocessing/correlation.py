"""Rank-based correlation methods for nonparanormal estimation."""

import numpy as np
from scipy import stats


def rank_correlation(X, method='spearman'):
    """
    Compute rank-based correlation with sine transformation.

    For Spearman: Sigma_jk = 2 * sin(pi/6 * rho_jk)
    For Kendall:  Sigma_jk = sin(pi/2 * tau_jk)

    These transformations approximate the Pearson correlation of the
    underlying Gaussian copula (Liu et al., 2012).

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Data matrix
    method : str, default='spearman'
        Correlation method: 'spearman' or 'kendall'

    Returns
    -------
    corr : ndarray of shape (n_features, n_features)
        Transformed correlation matrix

    References
    ----------
    Liu, H., Lafferty, J., & Wasserman, L. (2009). The nonparanormal:
    Semiparametric estimation of high dimensional undirected graphs.
    Journal of Machine Learning Research, 10, 2295-2328.

    Examples
    --------
    >>> import numpy as np
    >>> from pyggm.preprocessing.correlation import rank_correlation
    >>> X = np.random.randn(100, 5)
    >>> corr = rank_correlation(X, method='spearman')
    >>> corr.shape
    (5, 5)
    """
    X = np.asarray(X, dtype=np.float64)
    n, p = X.shape

    if method == 'spearman':
        # Compute Spearman correlation for all pairs
        rho, _ = stats.spearmanr(X, axis=0)

        # Handle scalar output when p=2
        if p == 2:
            rho = np.array([[1.0, rho], [rho, 1.0]])

        # Apply sine transformation: 2 * sin(pi/6 * rho)
        corr = 2 * np.sin(np.pi / 6 * rho)

    elif method == 'kendall':
        # Kendall's tau requires pairwise computation
        corr = np.eye(p)

        for i in range(p):
            for j in range(i + 1, p):
                tau, _ = stats.kendalltau(X[:, i], X[:, j])
                # Handle NaN (can occur with constant columns)
                if np.isnan(tau):
                    tau = 0.0
                # Apply sine transformation: sin(pi/2 * tau)
                corr[i, j] = np.sin(np.pi / 2 * tau)
                corr[j, i] = corr[i, j]
    else:
        raise ValueError(
            f"method must be 'spearman' or 'kendall', got '{method}'"
        )

    # Ensure diagonal is exactly 1.0
    np.fill_diagonal(corr, 1.0)

    return corr
