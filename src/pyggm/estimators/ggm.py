"""Gaussian Graphical Model estimator with multiple regularization selection methods."""

import numpy as np
import networkx as nx
from sklearn.base import BaseEstimator
from sklearn.covariance import GraphicalLasso, GraphicalLassoCV

from ..utils.psd import project_to_psd, is_psd
from ..utils.validation import check_array_2d
from ..preprocessing.correlation import rank_correlation
from .stars import stars_select
from .ebic import ebic_select


class GaussianGraphicalModel(BaseEstimator):
    """
    Scikit-learn compatible Gaussian Graphical Model estimator.

    Estimates the precision matrix (inverse covariance) of a multivariate
    Gaussian distribution using graphical lasso with regularization selection
    via StARS, EBIC, or cross-validation.

    Parameters
    ----------
    method : str, default='stars'
        Regularization selection method: 'stars', 'ebic', or 'cv'
    correlation : str, default='pearson'
        Correlation type: 'pearson', 'spearman', or 'kendall'
    alpha : float or None, default=None
        Fixed regularization parameter. If None, selected via `method`.
    alphas : array-like or None, default=None
        Custom array of alpha values to evaluate. If provided, overrides
        n_alphas and alpha_min_ratio. Useful for benchmarking.
    n_alphas : int, default=20
        Number of alpha values to evaluate (ignored if alphas is provided)
    alpha_min_ratio : float, default=0.01
        Ratio of min to max alpha (ignored if alphas is provided)
    beta : float, default=0.05
        StARS instability threshold (only used if method='stars')
    n_subsamples : int, default=50
        Number of subsamples for StARS
    subsample_size : int or None, default=None
        Size of each subsample. If None, uses floor(10 * sqrt(n))
    gamma : float, default=0.5
        EBIC gamma parameter (only used if method='ebic')
    cv : int or CV splitter, default=5
        Cross-validation strategy (only used if method='cv').
        If int, number of folds. If CV splitter object, uses custom folds.
    ensure_psd : bool, default=True
        Project correlation matrix to nearest PSD if needed
    n_jobs : int, default=1
        Number of parallel jobs (-1 for all cores)
    random_state : int or None, default=None
        Random seed for reproducibility
    verbose : int, default=0
        Verbosity level

    Attributes
    ----------
    precision_ : ndarray of shape (n_features, n_features)
        Estimated precision matrix
    covariance_ : ndarray of shape (n_features, n_features)
        Estimated covariance matrix
    adjacency_ : ndarray of shape (n_features, n_features)
        Binary adjacency matrix (edges where precision != 0)
    alpha_ : float
        Selected regularization parameter
    n_edges_ : int
        Number of edges in estimated graph
    alphas_ : ndarray of shape (n_alphas,)
        Grid of regularization values evaluated
    instabilities_ : ndarray (if method='stars')
        Instability values for each alpha
    ebic_scores_ : ndarray (if method='ebic')
        EBIC scores for each alpha
    cv_scores_ : ndarray (if method='cv')
        CV log-likelihood scores for each alpha

    Examples
    --------
    >>> import numpy as np
    >>> from pyggm import GaussianGraphicalModel
    >>> X = np.random.randn(200, 50)
    >>> model = GaussianGraphicalModel(method='stars', n_jobs=-1)
    >>> model.fit(X)
    GaussianGraphicalModel(...)
    >>> print(f"Selected alpha: {model.alpha_:.4f}")
    >>> print(f"Number of edges: {model.n_edges_}")
    """

    def __init__(self, method='stars', correlation='pearson', alpha=None,
                 alphas=None, n_alphas=20, alpha_min_ratio=0.01, beta=0.05,
                 n_subsamples=50, subsample_size=None, gamma=0.5, cv=5,
                 ensure_psd=True, n_jobs=1, random_state=None, verbose=0):
        self.method = method
        self.correlation = correlation
        self.alpha = alpha
        self.alphas = alphas
        self.n_alphas = n_alphas
        self.alpha_min_ratio = alpha_min_ratio
        self.beta = beta
        self.n_subsamples = n_subsamples
        self.subsample_size = subsample_size
        self.gamma = gamma
        self.cv = cv
        self.ensure_psd = ensure_psd
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y=None):
        """
        Fit the model to data X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : Ignored
            Not used, present for API consistency

        Returns
        -------
        self : object
            Fitted estimator
        """
        # Validate input
        X = check_array_2d(X, name="X")
        n, p = X.shape

        if self.verbose > 0:
            print(f"Fitting GaussianGraphicalModel: n={n}, p={p}, "
                  f"method={self.method}, correlation={self.correlation}")

        # Step 1: Compute correlation/covariance matrix
        if self.correlation == 'pearson':
            # Use sample covariance
            S = np.cov(X, rowvar=False, bias=False)
        elif self.correlation in ['spearman', 'kendall']:
            # Use rank-based correlation with sine transformation
            S = rank_correlation(X, method=self.correlation)
        else:
            raise ValueError(
                f"correlation must be 'pearson', 'spearman', or 'kendall', "
                f"got '{self.correlation}'"
            )

        # Step 2: Ensure PSD if requested
        if self.ensure_psd and not is_psd(S):
            if self.verbose > 0:
                print("Projecting correlation matrix to PSD")
            S = project_to_psd(S)

        # Step 3: Build alpha grid or use fixed/custom alphas
        if self.alpha is not None:
            # Use fixed alpha
            self.alphas_ = np.array([self.alpha])
            self.alpha_ = self.alpha

            # Fit directly
            model = GraphicalLasso(alpha=self.alpha_, max_iter=100, tol=1e-4)
            model.fit(X)
            self.precision_ = model.precision_
            self.covariance_ = model.covariance_

        else:
            # Use custom or automatically generated alpha grid
            if self.alphas is not None:
                # Use custom alpha array (for benchmarking)
                self.alphas_ = np.asarray(self.alphas)
                if self.verbose > 0:
                    print(f"Using custom alpha grid: {len(self.alphas_)} values")
            else:
                # Build alpha grid automatically
                alpha_max = np.max(np.abs(S - np.diag(np.diag(S))))
                self.alphas_ = np.logspace(
                    np.log10(alpha_max * self.alpha_min_ratio),
                    np.log10(alpha_max),
                    num=self.n_alphas
                )

            # Step 4: Select regularization parameter
            if self.method == 'stars':
                result = stars_select(
                    X, self.alphas_,
                    n_subsamples=self.n_subsamples,
                    subsample_size=self.subsample_size,
                    beta=self.beta,
                    n_jobs=self.n_jobs,
                    random_state=self.random_state,
                    verbose=self.verbose
                )
                self.alpha_, self.instabilities_, self.edge_probs_, \
                    self.precision_, self.covariance_ = result

            elif self.method == 'ebic':
                result = ebic_select(
                    X, self.alphas_,
                    gamma=self.gamma,
                    verbose=self.verbose
                )
                self.alpha_, self.ebic_scores_, self.precision_, self.covariance_ = result

            elif self.method == 'cv':
                # Use sklearn's GraphicalLassoCV
                model = GraphicalLassoCV(
                    alphas=self.alphas_,
                    cv=self.cv,
                    n_jobs=self.n_jobs,
                    verbose=self.verbose > 0
                )
                model.fit(X)

                self.alpha_ = model.alpha_
                self.precision_ = model.precision_
                self.covariance_ = model.covariance_
                # Note: GraphicalLassoCV doesn't expose cv_results_ in a consistent way
                # We just store the alphas that were evaluated
                if hasattr(model, 'cv_alphas'):
                    self.alphas_ = model.cv_alphas
                # Store empty cv_scores for consistency (actual scores not exposed by sklearn)
                self.cv_scores_ = np.array([])

            else:
                raise ValueError(
                    f"method must be 'stars', 'ebic', or 'cv', got '{self.method}'"
                )

        # Step 5: Populate derived attributes
        threshold = 1e-10
        self.adjacency_ = (np.abs(self.precision_) > threshold).astype(int)
        np.fill_diagonal(self.adjacency_, 0)
        self.n_edges_ = np.sum(self.adjacency_) // 2

        if self.verbose > 0:
            print(f"Fit complete: alpha={self.alpha_:.6f}, n_edges={self.n_edges_}")

        return self

    def score(self, X, y=None):
        """
        Return average log-likelihood of X under the model.

        Computed as: (1/n) * sum_i log p(x_i | precision_)

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to score
        y : Ignored
            Not used, present for API consistency

        Returns
        -------
        score : float
            Average log-likelihood (higher is better)

        Notes
        -----
        Returns log-likelihood up to an additive constant
        (excludes the -(p/2)*log(2π) term).
        """
        X = check_array_2d(X, name="X")
        n, p = X.shape

        # Compute log-likelihood
        # log p(X | Theta) = (n/2) * (log|Theta| - trace(S @ Theta))
        S = np.cov(X, rowvar=False, bias=False)
        sign, logdet = np.linalg.slogdet(self.precision_)

        if sign <= 0:
            return -np.inf

        log_likelihood = (n / 2) * (logdet - np.trace(S @ self.precision_))

        # Return average per sample
        return log_likelihood / n

    def to_networkx(self, threshold=0.0, weighted=True, labels=None):
        """
        Convert precision matrix to NetworkX graph.

        Parameters
        ----------
        threshold : float, default=0.0
            Minimum absolute value |precision_ij| to include an edge.
            Edges with |precision_ij| <= threshold are excluded.
        weighted : bool, default=True
            If True, edge weights are set to precision_ij values.
        labels : list of str or None
            Node labels. If None, uses integer indices.

        Returns
        -------
        G : networkx.Graph
            Undirected graph with edges where |precision_ij| > threshold.

        Examples
        --------
        >>> model.fit(X)
        >>> G = model.to_networkx(threshold=0.01)
        >>> print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        """
        p = self.precision_.shape[0]

        # Create graph
        G = nx.Graph()

        # Add nodes
        if labels is None:
            labels = list(range(p))
        G.add_nodes_from(labels)

        # Add edges
        for i in range(p):
            for j in range(i + 1, p):
                weight = self.precision_[i, j]
                if np.abs(weight) > threshold:
                    if weighted:
                        G.add_edge(labels[i], labels[j], weight=weight)
                    else:
                        G.add_edge(labels[i], labels[j])

        return G
