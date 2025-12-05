"""Extended BIC (EBIC) for graphical lasso regularization selection."""

import numpy as np
from sklearn.covariance import GraphicalLasso


def ebic_select(X, alphas, gamma=0.5, verbose=0):
    """
    Extended BIC selection for graphical lasso.

    EBIC_gamma = -2 * log_likelihood + |E| * log(n) + 4 * |E| * gamma * log(p)

    where |E| is the number of edges (non-zero off-diagonal elements in
    precision matrix divided by 2).

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Data matrix (already transformed if nonparanormal)
    alphas : array-like
        Regularization values to evaluate
    gamma : float, default=0.5
        Sparsity penalty (0 = standard BIC, 0.5 recommended for high-dim)
    verbose : int, default=0
        Verbosity level

    Returns
    -------
    best_alpha : float
        Alpha with minimum EBIC score
    ebic_scores : ndarray
        EBIC scores for each alpha
    best_precision : ndarray
        Precision matrix at best alpha
    best_covariance : ndarray
        Covariance matrix at best alpha

    References
    ----------
    Foygel, R., & Drton, M. (2010). Extended Bayesian information criteria
    for Gaussian graphical models. Advances in Neural Information Processing
    Systems, 23, 604-612.
    """
    X = np.asarray(X, dtype=np.float64)
    n, p = X.shape
    alphas = np.asarray(alphas)

    # Compute sample covariance
    S = np.cov(X, rowvar=False, bias=False)

    ebic_scores = np.zeros(len(alphas))
    precisions = []
    covariances = []

    for i, alpha in enumerate(alphas):
        if verbose > 0:
            print(f"EBIC: Fitting alpha {i+1}/{len(alphas)}: {alpha:.6f}")

        # Fit GraphicalLasso
        model = GraphicalLasso(alpha=alpha, max_iter=100, tol=1e-4, mode='cd')
        model.fit(X)

        precision = model.precision_
        covariance = model.covariance_

        # Compute EBIC
        ebic = _compute_ebic(precision, S, n, gamma)
        ebic_scores[i] = ebic

        precisions.append(precision)
        covariances.append(covariance)

    # Select alpha with minimum EBIC
    best_idx = np.argmin(ebic_scores)
    best_alpha = alphas[best_idx]
    best_precision = precisions[best_idx]
    best_covariance = covariances[best_idx]

    if verbose > 0:
        print(f"EBIC: Selected alpha = {best_alpha:.6f} (EBIC = {ebic_scores[best_idx]:.2f})")

    return best_alpha, ebic_scores, best_precision, best_covariance


def _compute_ebic(precision, S, n, gamma):
    """
    Compute EBIC for a single precision matrix estimate.

    Parameters
    ----------
    precision : ndarray of shape (p, p)
        Estimated precision matrix
    S : ndarray of shape (p, p)
        Sample covariance matrix
    n : int
        Number of samples
    gamma : float
        Sparsity penalty parameter

    Returns
    -------
    ebic : float
        Extended BIC score (lower is better)
    """
    p = precision.shape[0]

    # Compute log-likelihood
    # log L = (n/2) * (log(det(Theta)) - trace(S @ Theta))
    sign, logdet = np.linalg.slogdet(precision)
    if sign <= 0:
        # Precision matrix should be PD; if not, return large penalty
        return np.inf

    log_likelihood = (n / 2) * (logdet - np.trace(S @ precision))

    # Count edges (non-zero off-diagonal elements / 2)
    # Use threshold to handle numerical zeros
    threshold = 1e-10
    off_diag = np.abs(precision) > threshold
    np.fill_diagonal(off_diag, False)
    n_edges = np.sum(off_diag) / 2

    # EBIC formula
    ebic = -2 * log_likelihood + n_edges * np.log(n) + 4 * n_edges * gamma * np.log(p)

    return ebic
