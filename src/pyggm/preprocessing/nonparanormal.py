"""Nonparanormal (Gaussian copula) transformation."""

import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin


class NonparanormalTransformer(BaseEstimator, TransformerMixin):
    """
    Transform data via nonparanormal (Gaussian copula) transformation.

    X_transformed_j = Phi^{-1}(F_hat_j(X_j))

    where F_hat is the empirical CDF and Phi^{-1} is the inverse normal CDF.

    This transformation converts arbitrary continuous distributions to
    approximately Gaussian marginals while preserving dependence structure.

    Parameters
    ----------
    truncate : bool, default=True
        If True, truncate ranks to avoid infinite values in inverse CDF

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit
    empirical_cdfs_ : list of arrays
        Sorted values for each feature used to compute empirical CDF

    References
    ----------
    Liu, H., Lafferty, J., & Wasserman, L. (2009). The nonparanormal:
    Semiparametric estimation of high dimensional undirected graphs.
    Journal of Machine Learning Research, 10, 2295-2328.

    Examples
    --------
    >>> import numpy as np
    >>> from pyggm import NonparanormalTransformer
    >>> X = np.random.exponential(size=(100, 5))
    >>> transformer = NonparanormalTransformer()
    >>> X_trans = transformer.fit_transform(X)
    >>> X_trans.shape
    (100, 5)
    """

    def __init__(self, truncate=True):
        self.truncate = truncate

    def fit(self, X, y=None):
        """
        Store empirical CDF information for each feature.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : Ignored
            Not used, present for API consistency

        Returns
        -------
        self : object
            Fitted transformer
        """
        X = np.asarray(X, dtype=np.float64)
        n, p = X.shape

        self.n_features_in_ = p
        self.empirical_cdfs_ = []

        # Store sorted values for each feature
        for j in range(p):
            sorted_vals = np.sort(X[:, j])
            self.empirical_cdfs_.append(sorted_vals)

        return self

    def transform(self, X):
        """
        Apply nonparanormal transformation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform

        Returns
        -------
        X_trans : ndarray of shape (n_samples, n_features), dtype float64
            Transformed data with approximately Gaussian marginals
        """
        X = np.asarray(X, dtype=np.float64)
        n, p = X.shape

        if p != self.n_features_in_:
            raise ValueError(
                f"X has {p} features, but transformer was fitted with "
                f"{self.n_features_in_} features."
            )

        X_trans = np.zeros_like(X)

        for j in range(p):
            # Compute empirical CDF values using ranks
            # Use searchsorted to find ranks efficiently
            sorted_vals = self.empirical_cdfs_[j]
            n_train = len(sorted_vals)

            # Get ranks (position in sorted training data)
            ranks = np.searchsorted(sorted_vals, X[:, j], side='right')

            # Convert ranks to uniform [0,1] values
            # Use midrank approach for ties
            uniform_vals = (ranks + 0.5) / (n_train + 1)

            if self.truncate:
                # Truncate to avoid infinities in inverse normal CDF
                eps = 1e-10
                uniform_vals = np.clip(uniform_vals, eps, 1 - eps)

            # Apply inverse normal CDF
            X_trans[:, j] = stats.norm.ppf(uniform_vals)

        return X_trans
