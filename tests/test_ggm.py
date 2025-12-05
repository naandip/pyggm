"""Tests for GaussianGraphicalModel estimator."""

import numpy as np
import pytest
from pyggm import GaussianGraphicalModel


class TestGaussianGraphicalModel:

    @pytest.fixture
    def sample_data(self):
        """Generate sample data with known structure."""
        np.random.seed(42)
        n, p = 200, 20
        # Create sparse precision matrix
        precision = np.eye(p)
        precision[0, 1] = precision[1, 0] = 0.5
        precision[2, 3] = precision[3, 2] = 0.3
        covariance = np.linalg.inv(precision)
        X = np.random.multivariate_normal(np.zeros(p), covariance, n)
        return X, precision

    def test_fit_stars(self, sample_data):
        X, true_prec = sample_data
        model = GaussianGraphicalModel(method='stars', n_jobs=1, n_subsamples=10)
        model.fit(X)
        assert model.precision_.shape == (X.shape[1], X.shape[1])
        assert model.alpha_ > 0
        assert hasattr(model, 'instabilities_')

    def test_fit_ebic(self, sample_data):
        X, _ = sample_data
        model = GaussianGraphicalModel(method='ebic', gamma=0.5)
        model.fit(X)
        assert hasattr(model, 'ebic_scores_')
        assert model.precision_ is not None

    def test_fit_cv(self, sample_data):
        X, _ = sample_data
        model = GaussianGraphicalModel(method='cv', cv=3)
        model.fit(X)
        assert model.precision_ is not None
        assert model.alpha_ > 0

    def test_spearman_correlation(self, sample_data):
        X, _ = sample_data
        model = GaussianGraphicalModel(correlation='spearman', method='ebic')
        model.fit(X)
        assert model.precision_ is not None

    def test_kendall_correlation(self, sample_data):
        X, _ = sample_data
        # Use smaller data for Kendall (slower)
        model = GaussianGraphicalModel(correlation='kendall', method='ebic')
        model.fit(X[:50, :10])
        assert model.precision_ is not None

    def test_to_networkx(self, sample_data):
        X, _ = sample_data
        model = GaussianGraphicalModel(method='ebic')
        model.fit(X)
        G = model.to_networkx(threshold=0.01)
        assert G.number_of_nodes() == X.shape[1]

    def test_score_returns_float(self, sample_data):
        """Test that score() returns a float."""
        X, _ = sample_data
        model = GaussianGraphicalModel(method='ebic')
        model.fit(X)
        score = model.score(X)
        assert isinstance(score, (float, np.floating))

    def test_stars_instabilities_reasonable(self, sample_data):
        """Check StARS instabilities are in valid range."""
        X, _ = sample_data
        model = GaussianGraphicalModel(method='stars', n_subsamples=10)
        model.fit(X)
        # Instabilities should be between 0 and 0.5
        assert np.all(model.instabilities_ >= 0)
        assert np.all(model.instabilities_ <= 0.5)

    def test_fixed_alpha(self, sample_data):
        """Test fitting with fixed alpha."""
        X, _ = sample_data
        model = GaussianGraphicalModel(alpha=0.1)
        model.fit(X)
        assert model.alpha_ == 0.1
        assert len(model.alphas_) == 1

    def test_adjacency_symmetric(self, sample_data):
        """Adjacency matrix should be symmetric."""
        X, _ = sample_data
        model = GaussianGraphicalModel(method='ebic')
        model.fit(X)
        assert np.allclose(model.adjacency_, model.adjacency_.T)

    def test_n_edges_count(self, sample_data):
        """n_edges should match adjacency matrix."""
        X, _ = sample_data
        model = GaussianGraphicalModel(method='ebic')
        model.fit(X)
        expected_edges = np.sum(model.adjacency_) // 2
        assert model.n_edges_ == expected_edges

    def test_to_networkx_weighted(self, sample_data):
        """to_networkx with weighted=True should have edge weights."""
        X, _ = sample_data
        model = GaussianGraphicalModel(method='ebic')
        model.fit(X)
        G = model.to_networkx(threshold=0.01, weighted=True)
        if G.number_of_edges() > 0:
            # Check that at least one edge has a weight attribute
            edge = list(G.edges())[0]
            assert 'weight' in G[edge[0]][edge[1]]

    def test_to_networkx_labels(self, sample_data):
        """to_networkx with custom labels."""
        X, _ = sample_data
        X = X[:, :5]
        model = GaussianGraphicalModel(method='ebic')
        model.fit(X)
        labels = ['A', 'B', 'C', 'D', 'E']
        G = model.to_networkx(labels=labels)
        assert set(G.nodes()) == set(labels)

    def test_invalid_method(self, sample_data):
        """Invalid method should raise ValueError."""
        X, _ = sample_data
        model = GaussianGraphicalModel(method='invalid')
        with pytest.raises(ValueError, match="method must be"):
            model.fit(X)

    def test_invalid_correlation(self, sample_data):
        """Invalid correlation should raise ValueError."""
        X, _ = sample_data
        model = GaussianGraphicalModel(correlation='invalid')
        with pytest.raises(ValueError, match="correlation must be"):
            model.fit(X)
