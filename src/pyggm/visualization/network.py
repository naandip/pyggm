"""Network visualization for precision matrices."""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def plot_network(precision, labels=None, threshold=0.01, layout='spring',
                 node_size='degree', edge_width='weight', edge_color='sign',
                 ax=None, **kwargs):
    """
    Plot precision matrix as a network graph.

    Parameters
    ----------
    precision : ndarray of shape (p, p)
        Precision matrix
    labels : list of str or None, default=None
        Node labels
    threshold : float, default=0.01
        Minimum |precision_ij| to draw edge
    layout : str, default='spring'
        Layout algorithm: 'spring', 'circular', 'spectral',
        'kamada_kawai', or 'shell'
    node_size : str or array, default='degree'
        'degree' (proportional to node degree) or array of sizes
    edge_width : str or array, default='weight'
        'weight' (proportional to |precision_ij|) or array
    edge_color : str, default='sign'
        'sign' (red=negative, blue=positive) or single color
    ax : matplotlib Axes or None, default=None
        Axes to plot on
    **kwargs : dict
        Additional arguments passed to nx.draw_networkx

    Returns
    -------
    fig : matplotlib Figure
    ax : matplotlib Axes
    G : networkx.Graph
        The constructed graph

    Examples
    --------
    >>> import numpy as np
    >>> from pyggm.visualization import plot_network
    >>> precision = np.eye(10)
    >>> precision[0, 1] = precision[1, 0] = 0.3
    >>> fig, ax, G = plot_network(precision)
    """
    p = precision.shape[0]

    # Create NetworkX graph
    G = nx.Graph()

    # Add nodes
    if labels is None:
        labels = [str(i) for i in range(p)]
    G.add_nodes_from(labels)

    # Add edges
    edges_positive = []
    edges_negative = []
    edge_weights = []

    for i in range(p):
        for j in range(i + 1, p):
            weight = precision[i, j]
            if np.abs(weight) > threshold:
                if weight > 0:
                    edges_positive.append((labels[i], labels[j]))
                else:
                    edges_negative.append((labels[i], labels[j]))
                G.add_edge(labels[i], labels[j], weight=weight)
                edge_weights.append(np.abs(weight))

    # Compute layout
    if layout == 'spring':
        pos = nx.spring_layout(G, seed=42)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    elif layout == 'spectral':
        pos = nx.spectral_layout(G)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    elif layout == 'shell':
        pos = nx.shell_layout(G)
    else:
        raise ValueError(f"Unknown layout: {layout}")

    # Compute node sizes
    if node_size == 'degree':
        degrees = dict(G.degree())
        node_sizes = [300 + 100 * degrees[node] for node in G.nodes()]
    elif isinstance(node_size, (list, np.ndarray)):
        node_sizes = node_size
    else:
        node_sizes = 300

    # Compute edge widths
    if edge_width == 'weight':
        if len(edge_weights) > 0:
            max_weight = np.max(edge_weights)
            edge_widths_pos = [5 * np.abs(G[u][v]['weight']) / max_weight
                               for u, v in edges_positive]
            edge_widths_neg = [5 * np.abs(G[u][v]['weight']) / max_weight
                               for u, v in edges_negative]
        else:
            edge_widths_pos = []
            edge_widths_neg = []
    elif isinstance(edge_width, (list, np.ndarray)):
        edge_widths_pos = edge_width
        edge_widths_neg = edge_width
    else:
        edge_widths_pos = 2.0
        edge_widths_neg = 2.0

    # Create plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        fig = ax.figure

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightblue',
                           ax=ax, **kwargs)

    # Draw edges
    if edge_color == 'sign':
        if len(edges_positive) > 0:
            nx.draw_networkx_edges(G, pos, edgelist=edges_positive,
                                   width=edge_widths_pos, edge_color='blue',
                                   alpha=0.6, ax=ax)
        if len(edges_negative) > 0:
            nx.draw_networkx_edges(G, pos, edgelist=edges_negative,
                                   width=edge_widths_neg, edge_color='red',
                                   alpha=0.6, ax=ax)
    else:
        nx.draw_networkx_edges(G, pos, width=edge_widths_pos,
                               edge_color=edge_color, alpha=0.6, ax=ax)

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)

    ax.axis('off')
    ax.set_title(f'Network Graph ({G.number_of_edges()} edges)', fontsize=14)

    plt.tight_layout()

    return fig, ax, G
