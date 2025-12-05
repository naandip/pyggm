"""Tests for visualization functions."""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import pytest

from pyggm import GaussianGraphicalModel
from pyggm.visualization import (
    plot_network, plot_precision_matrix, plot_stars_path,
    plot_edge_stability, plot_model_selection
)


def test_plot_network_smoke():
    """plot_network runs without error and returns fig, ax, G."""
    precision = np.eye(5)
    precision[0, 1] = precision[1, 0] = 0.3
    fig, ax, G = plot_network(precision)
    assert fig is not None
    assert ax is not None
    assert G is not None
    plt.close(fig)


def test_plot_precision_matrix_smoke():
    """plot_precision_matrix runs without error and returns fig, ax."""
    precision = np.eye(5)
    precision[0, 1] = precision[1, 0] = 0.3
    fig, ax = plot_precision_matrix(precision)
    assert fig is not None
    assert ax is not None
    plt.close(fig)


def test_plot_network_with_labels():
    """plot_network with custom labels."""
    precision = np.eye(3)
    precision[0, 1] = precision[1, 0] = 0.5
    labels = ['A', 'B', 'C']
    fig, ax, G = plot_network(precision, labels=labels)
    assert set(G.nodes()) == set(labels)
    plt.close(fig)


def test_plot_network_layouts():
    """plot_network with different layouts."""
    precision = np.eye(5)
    precision[0, 1] = precision[1, 0] = 0.3

    for layout in ['spring', 'circular', 'spectral']:
        fig, ax, G = plot_network(precision, layout=layout)
        assert fig is not None
        plt.close(fig)


def test_plot_stars_path():
    """plot_stars_path with fitted StARS model."""
    np.random.seed(42)
    X = np.random.randn(100, 10)
    model = GaussianGraphicalModel(method='stars', n_subsamples=10)
    model.fit(X)

    fig, ax = plot_stars_path(model)
    assert fig is not None
    assert ax is not None
    plt.close(fig)


def test_plot_stars_path_raises_without_stars():
    """plot_stars_path should raise if model not fitted with StARS."""
    np.random.seed(42)
    X = np.random.randn(100, 10)
    model = GaussianGraphicalModel(method='ebic')
    model.fit(X)

    with pytest.raises(ValueError, match="method='stars'"):
        plot_stars_path(model)


def test_plot_edge_stability():
    """plot_edge_stability with fitted StARS model."""
    np.random.seed(42)
    X = np.random.randn(80, 8)
    model = GaussianGraphicalModel(method='stars', n_subsamples=10)
    model.fit(X)

    fig, ax = plot_edge_stability(model)
    assert fig is not None
    assert ax is not None
    plt.close(fig)


def test_plot_edge_stability_raises_without_stars():
    """plot_edge_stability should raise if model not fitted with StARS."""
    np.random.seed(42)
    X = np.random.randn(100, 10)
    model = GaussianGraphicalModel(method='ebic')
    model.fit(X)

    with pytest.raises(ValueError, match="method='stars'"):
        plot_edge_stability(model)


def test_plot_model_selection_ebic():
    """plot_model_selection with EBIC model."""
    np.random.seed(42)
    X = np.random.randn(100, 10)
    model = GaussianGraphicalModel(method='ebic')
    model.fit(X)

    fig, ax = plot_model_selection(model)
    assert fig is not None
    assert ax is not None
    plt.close(fig)


def test_plot_model_selection_stars():
    """plot_model_selection with StARS model."""
    np.random.seed(42)
    X = np.random.randn(80, 8)
    model = GaussianGraphicalModel(method='stars', n_subsamples=10)
    model.fit(X)

    fig, ax = plot_model_selection(model)
    assert fig is not None
    assert ax is not None
    plt.close(fig)


def test_plot_precision_matrix_with_labels():
    """plot_precision_matrix with custom labels."""
    precision = np.eye(3)
    precision[0, 1] = precision[1, 0] = 0.5
    labels = ['X1', 'X2', 'X3']
    fig, ax = plot_precision_matrix(precision, labels=labels)
    assert fig is not None
    plt.close(fig)
