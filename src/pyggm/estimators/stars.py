"""StARS: Stability Approach to Regularization Selection."""

import numpy as np
from sklearn.covariance import GraphicalLasso
from joblib import Parallel, delayed


def stars_select(X, alphas, n_subsamples=50, subsample_size=None,
                 beta=0.05, n_jobs=1, random_state=None, verbose=0):
    """
    Stability Approach to Regularization Selection.

    Selects regularization parameter by evaluating edge stability across
    multiple subsamples. Returns the smallest alpha where instability <= beta.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Data matrix (already transformed if nonparanormal)
    alphas : array-like
        Regularization values to evaluate. Should be in ascending order
        (from low regularization / dense graphs to high regularization /
        sparse graphs). Early stopping at first alpha where D(alpha) <= beta.
    n_subsamples : int, default=50
        Number of subsamples (B). Uses subsampling without replacement.
    subsample_size : int or None, default=None
        If None, use floor(10 * sqrt(n))
    beta : float, default=0.05
        Instability threshold (Liu et al. 2010 recommend 0.05)
    n_jobs : int, default=1
        Number of parallel jobs (-1 for all cores)
    random_state : int or None, default=None
        Random seed for reproducible subsampling
    verbose : int, default=0
        Verbosity level

    Returns
    -------
    best_alpha : float
        Selected alpha (smallest with instability <= beta)
    instabilities : ndarray
        D(alpha) for each alpha, in [0, 0.5]
    edge_probs : ndarray of shape (n_features, n_features)
        Edge inclusion probabilities theta_ij at best_alpha
    best_precision : ndarray
        Precision matrix at best_alpha on full data
    best_covariance : ndarray
        Covariance matrix at best_alpha on full data

    References
    ----------
    Liu, H., Roeder, K., & Wasserman, L. (2010). Stability approach to
    regularization selection (stars) for high dimensional graphical models.
    Advances in Neural Information Processing Systems, 23, 1432-1440.
    """
    X = np.asarray(X, dtype=np.float64)
    n, p = X.shape
    alphas = np.asarray(alphas)

    # Determine subsample size
    if subsample_size is None:
        subsample_size = int(np.floor(10 * np.sqrt(n)))
        # Ensure subsample size is reasonable
        subsample_size = min(subsample_size, n - 1)
        subsample_size = max(subsample_size, int(n / 2))

    if verbose > 0:
        print(f"StARS: n={n}, p={p}, subsample_size={subsample_size}, "
              f"n_subsamples={n_subsamples}, beta={beta}")

    # Set random state
    rng = np.random.RandomState(random_state)

    # Generate subsample indices
    subsample_indices = []
    for _ in range(n_subsamples):
        idx = rng.choice(n, size=subsample_size, replace=False)
        subsample_indices.append(idx)

    # Compute edge probabilities for each alpha
    instabilities = np.zeros(len(alphas))
    all_edge_probs = []

    for i, alpha in enumerate(alphas):
        if verbose > 0:
            print(f"StARS: Evaluating alpha {i+1}/{len(alphas)}: {alpha:.6f}")

        # Fit graphical lasso on each subsample in parallel
        adjacencies = Parallel(n_jobs=n_jobs)(
            delayed(_fit_subsample)(X[idx], alpha)
            for idx in subsample_indices
        )

        # Compute edge probabilities: theta_ij = proportion of subsamples with edge (i,j)
        edge_probs = np.mean(adjacencies, axis=0)

        # Compute instability
        instability = _compute_instability(edge_probs)
        instabilities[i] = instability

        all_edge_probs.append(edge_probs)

        if verbose > 0:
            n_edges = np.sum(edge_probs > 0.5) / 2
            print(f"  Instability: {instability:.4f}, Avg edges: {n_edges:.1f}")

    # Apply monotonicity constraint (required for correct lambda selection)
    # Instability should be non-increasing with INCREASING alpha
    # For ascending alphas [small...large], cummin enforces this directly
    instabilities_monotone = np.minimum.accumulate(instabilities)

    # Select optimal alpha: smallest alpha (densest graph) with instability <= beta
    # This ensures maximum power while maintaining stability (matches R's huge)
    valid_alphas = np.where(instabilities_monotone <= beta)[0]

    if len(valid_alphas) == 0:
        # No alpha satisfies criterion; use most regularized (largest alpha)
        if verbose > 0:
            print(f"Warning: No alpha with instability <= {beta}. Using most regularized.")
        best_idx = len(instabilities) - 1
    else:
        # Select FIRST valid index = smallest alpha = densest graph (matches R's huge)
        best_idx = valid_alphas[0]

    best_alpha = alphas[best_idx]
    edge_probs = all_edge_probs[best_idx]

    if verbose > 0:
        print(f"StARS: Selected alpha = {best_alpha:.6f} "
              f"(instability = {instabilities[best_idx]:.4f})")

    # Refit on full data with selected alpha
    model = GraphicalLasso(alpha=best_alpha, max_iter=100, tol=1e-4, mode='cd')
    model.fit(X)
    best_precision = model.precision_
    best_covariance = model.covariance_

    return best_alpha, instabilities, edge_probs, best_precision, best_covariance


def _fit_subsample(X_sub, alpha):
    """
    Fit GraphicalLasso on a subsample and return binary adjacency matrix.

    Parameters
    ----------
    X_sub : ndarray of shape (n_sub, p)
        Subsample data
    alpha : float
        Regularization parameter

    Returns
    -------
    adjacency : ndarray of shape (p, p)
        Binary adjacency matrix (1 if edge present, 0 otherwise)
    """
    try:
        model = GraphicalLasso(alpha=alpha, max_iter=100, tol=1e-4, mode='cd')
        model.fit(X_sub)
        precision = model.precision_

        # Threshold to binary adjacency
        threshold = 1e-10
        adjacency = (np.abs(precision) > threshold).astype(float)
        np.fill_diagonal(adjacency, 0)

        return adjacency
    except Exception as e:
        # If fitting fails (e.g., numerical issues), return all zeros
        p = X_sub.shape[1]
        return np.zeros((p, p))


def _compute_instability(edge_probs):
    """
    Compute total instability D from edge probabilities.

    D = (2 / (p*(p-1))) * sum_{i<j} 2 * theta_ij * (1 - theta_ij)

    This measures the average variability in edge selection. The factor of 2
    accounts for the fact that maximum instability occurs when theta_ij = 0.5,
    giving D in [0, 0.5].

    Parameters
    ----------
    edge_probs : ndarray of shape (p, p)
        Edge inclusion probabilities (0 to 1)

    Returns
    -------
    instability : float
        Total instability in [0, 0.5]
    """
    p = edge_probs.shape[0]

    # Extract upper triangle (exclude diagonal)
    upper_tri_idx = np.triu_indices(p, k=1)
    theta = edge_probs[upper_tri_idx]

    # Compute variability: 2 * theta * (1 - theta)
    variability = 2 * theta * (1 - theta)

    # Average over all pairs
    instability = np.mean(variability)

    return instability
