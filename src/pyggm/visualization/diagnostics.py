"""Diagnostic plots for regularization selection."""

import numpy as np
import matplotlib.pyplot as plt


def plot_stars_path(model, ax=None):
    """
    Plot StARS instability curve D(alpha) vs alpha.

    Shows threshold line at beta and marks selected alpha.

    Parameters
    ----------
    model : GaussianGraphicalModel
        Fitted model with method='stars'
    ax : matplotlib Axes or None, default=None
        Axes to plot on

    Returns
    -------
    fig : matplotlib Figure
    ax : matplotlib Axes

    Raises
    ------
    ValueError
        If model was not fitted with method='stars'

    Examples
    --------
    >>> from pyggm import GaussianGraphicalModel
    >>> from pyggm.visualization import plot_stars_path
    >>> model = GaussianGraphicalModel(method='stars')
    >>> model.fit(X)
    >>> fig, ax = plot_stars_path(model)
    """
    if not hasattr(model, 'instabilities_'):
        raise ValueError("Model must be fitted with method='stars' to plot StARS path")

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    alphas = model.alphas_[:len(model.instabilities_)]
    instabilities = model.instabilities_

    # Plot instability curve
    ax.plot(alphas, instabilities, 'o-', linewidth=2, markersize=6, label='Instability')

    # Plot threshold line
    ax.axhline(y=model.beta, color='red', linestyle='--', linewidth=2,
               label=f'Threshold (β = {model.beta})')

    # Mark selected alpha
    ax.axvline(x=model.alpha_, color='green', linestyle=':', linewidth=2,
               label=f'Selected α = {model.alpha_:.4f}')

    ax.set_xlabel('Regularization parameter (α)', fontsize=12)
    ax.set_ylabel('Instability D(α)', fontsize=12)
    ax.set_title('StARS Regularization Path', fontsize=14)
    ax.set_xscale('log')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    return fig, ax


def plot_edge_stability(model, labels=None, ax=None):
    """
    Plot heatmap of edge inclusion probabilities theta_ij.

    Parameters
    ----------
    model : GaussianGraphicalModel
        Fitted model with method='stars'
    labels : list of str or None, default=None
        Node labels
    ax : matplotlib Axes or None, default=None
        Axes to plot on

    Returns
    -------
    fig : matplotlib Figure
    ax : matplotlib Axes

    Raises
    ------
    ValueError
        If model was not fitted with method='stars'
    """
    if not hasattr(model, 'edge_probs_'):
        raise ValueError("Model must be fitted with method='stars' to plot edge stability")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))
    else:
        fig = ax.figure

    edge_probs = model.edge_probs_
    p = edge_probs.shape[0]

    # Plot heatmap
    im = ax.imshow(edge_probs, cmap='viridis', vmin=0, vmax=1, aspect='auto')

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Edge Probability', rotation=270, labelpad=20)

    # Set ticks
    if labels is not None:
        ax.set_xticks(range(p))
        ax.set_yticks(range(p))
        ax.set_xticklabels(labels, rotation=90)
        ax.set_yticklabels(labels)
    else:
        if p <= 50:
            ax.set_xticks(range(p))
            ax.set_yticks(range(p))

    ax.set_title('Edge Inclusion Probabilities (θ)', fontsize=14)
    plt.tight_layout()

    return fig, ax


def plot_regularization_path(model, ax=None):
    """
    Plot number of edges vs alpha for all methods.

    Parameters
    ----------
    model : GaussianGraphicalModel
        Fitted model
    ax : matplotlib Axes or None, default=None
        Axes to plot on

    Returns
    -------
    fig : matplotlib Figure
    ax : matplotlib Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    # Compute number of edges for each alpha
    # This requires refitting, which is expensive, so we approximate from instabilities
    # or show a message
    ax.text(0.5, 0.5, 'Regularization path visualization\nrequires fitting multiple models',
            ha='center', va='center', transform=ax.transAxes, fontsize=12)

    ax.set_xlabel('Regularization parameter (α)', fontsize=12)
    ax.set_ylabel('Number of edges', fontsize=12)
    ax.set_title('Regularization Path', fontsize=14)

    plt.tight_layout()

    return fig, ax


def plot_model_selection(model, ax=None):
    """
    Plot model selection criterion vs alpha.

    Automatically detects which criterion was used (EBIC, StARS, or CV).

    Parameters
    ----------
    model : GaussianGraphicalModel
        Fitted model
    ax : matplotlib Axes or None, default=None
        Axes to plot on

    Returns
    -------
    fig : matplotlib Figure
    ax : matplotlib Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    alphas = model.alphas_

    if hasattr(model, 'ebic_scores_'):
        # EBIC plot
        scores = model.ebic_scores_
        ylabel = 'EBIC'
        title = 'EBIC Model Selection'
        marker_label = 'EBIC'

    elif hasattr(model, 'instabilities_'):
        # StARS plot (redirect to plot_stars_path)
        return plot_stars_path(model, ax=ax)

    elif hasattr(model, 'cv_scores_'):
        # CV plot
        scores = model.cv_scores_
        ylabel = 'CV Log-Likelihood'
        title = 'Cross-Validation Model Selection'
        marker_label = 'CV Score'

    else:
        ax.text(0.5, 0.5, 'No model selection scores available',
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        return fig, ax

    # Plot scores
    alphas_plot = alphas[:len(scores)]
    ax.plot(alphas_plot, scores, 'o-', linewidth=2, markersize=6, label=marker_label)

    # Mark selected alpha
    ax.axvline(x=model.alpha_, color='green', linestyle=':', linewidth=2,
               label=f'Selected α = {model.alpha_:.4f}')

    ax.set_xlabel('Regularization parameter (α)', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xscale('log')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    return fig, ax
