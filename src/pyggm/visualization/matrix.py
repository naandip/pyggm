"""Matrix visualization for precision and covariance matrices."""

import numpy as np
import matplotlib.pyplot as plt


def plot_precision_matrix(precision, labels=None, cmap='RdBu_r', ax=None, **kwargs):
    """
    Plot precision matrix as a heatmap.

    Parameters
    ----------
    precision : ndarray of shape (p, p)
        Precision matrix to visualize
    labels : list of str or None, default=None
        Tick labels for rows/columns
    cmap : str, default='RdBu_r'
        Colormap name
    ax : matplotlib Axes or None, default=None
        Axes to plot on
    **kwargs : dict
        Additional arguments passed to plt.imshow

    Returns
    -------
    fig : matplotlib Figure
    ax : matplotlib Axes

    Examples
    --------
    >>> import numpy as np
    >>> from pyggm.visualization import plot_precision_matrix
    >>> precision = np.eye(20)
    >>> precision[0, 1] = precision[1, 0] = 0.3
    >>> fig, ax = plot_precision_matrix(precision)
    """
    p = precision.shape[0]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))
    else:
        fig = ax.figure

    # Compute symmetric range for colorbar
    vmax = np.max(np.abs(precision[~np.eye(p, dtype=bool)]))
    if vmax == 0:
        vmax = 1.0

    # Plot heatmap
    im = ax.imshow(precision, cmap=cmap, vmin=-vmax, vmax=vmax,
                   aspect='auto', **kwargs)

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Precision', rotation=270, labelpad=20)

    # Set ticks
    if labels is not None:
        ax.set_xticks(range(p))
        ax.set_yticks(range(p))
        ax.set_xticklabels(labels, rotation=90)
        ax.set_yticklabels(labels)
    else:
        # Show tick labels for small matrices only
        if p <= 50:
            ax.set_xticks(range(p))
            ax.set_yticks(range(p))

    ax.set_title('Precision Matrix', fontsize=14)
    plt.tight_layout()

    return fig, ax


def plot_covariance_matrix(covariance, labels=None, cmap='viridis', ax=None, **kwargs):
    """
    Plot covariance matrix as a heatmap.

    Parameters
    ----------
    covariance : ndarray of shape (p, p)
        Covariance matrix to visualize
    labels : list of str or None, default=None
        Tick labels for rows/columns
    cmap : str, default='viridis'
        Colormap name
    ax : matplotlib Axes or None, default=None
        Axes to plot on
    **kwargs : dict
        Additional arguments passed to plt.imshow

    Returns
    -------
    fig : matplotlib Figure
    ax : matplotlib Axes
    """
    p = covariance.shape[0]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))
    else:
        fig = ax.figure

    # Plot heatmap
    im = ax.imshow(covariance, cmap=cmap, aspect='auto', **kwargs)

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Covariance', rotation=270, labelpad=20)

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

    ax.set_title('Covariance Matrix', fontsize=14)
    plt.tight_layout()

    return fig, ax
