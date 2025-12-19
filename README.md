# PyGGM: Gaussian Graphical Models in Python

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python package for Gaussian Graphical Models with multiple regularization selection methods and nonparanormal transformations.

## Features

- **Multiple Regularization Selection Methods**:
  - **StARS** (Stability Approach to Regularization Selection)
  - **EBIC** (Extended Bayesian Information Criterion)
  - **Cross-Validation**

- **Nonparanormal Methods**:
  - Gaussian copula transformation
  - Rank-based correlations (Spearman/Kendall) with sine transformation

- **Robust Utilities**:
  - PSD projection for numerical stability
  - Comprehensive input validation

- **Visualization Tools**:
  - Network graphs with customizable layouts
  - Precision matrix heatmaps
  - Diagnostic plots for model selection

- **Performance**:
  - Parallelization support via joblib
  - Scikit-learn compatible API

## Installation

### From source

```bash
git clone https://github.com/naandip/pyggm.git
cd pyggm
pip install -e .
```

### Development installation

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
import numpy as np
from pyggm import GaussianGraphicalModel
from pyggm.visualization import plot_network, plot_stars_path

# Generate or load data
np.random.seed(42)
X = np.random.randn(200, 50)

# Fit model with StARS selection
model = GaussianGraphicalModel(
    method='stars',
    correlation='pearson',
    n_subsamples=50,
    n_jobs=-1  # Use all cores
)
model.fit(X)

# Results
print(f"Selected alpha: {model.alpha_:.4f}")
print(f"Number of edges: {model.n_edges_}")

# Visualize
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
plot_network(model.precision_, ax=axes[0])
plot_stars_path(model, ax=axes[1])
plt.show()
```

## Usage Examples

### 1. Basic Usage with Different Methods

#### StARS (Recommended for sparse graphs)

```python
from pyggm import GaussianGraphicalModel

model = GaussianGraphicalModel(
    method='stars',
    beta=0.05,           # Instability threshold
    n_subsamples=50,     # Number of subsamples
    n_jobs=-1            # Parallel processing
)
model.fit(X)
```

#### EBIC (Good for model selection)

```python
model = GaussianGraphicalModel(
    method='ebic',
    gamma=0.5            # Sparsity penalty (0 = BIC, 0.5 = recommended)
)
model.fit(X)
```

#### Cross-Validation

```python
model = GaussianGraphicalModel(
    method='cv',
    cv=5,                # Number of folds
    n_jobs=-1
)
model.fit(X)
```

### 2. Nonparanormal Transformation

For non-Gaussian data, use the nonparanormal transformer:

```python
from pyggm import NonparanormalTransformer, GaussianGraphicalModel

# Transform data to approximately Gaussian
transformer = NonparanormalTransformer()
X_transformed = transformer.fit_transform(X)

# Fit graphical model
model = GaussianGraphicalModel(method='stars')
model.fit(X_transformed)
```

### 3. Rank-Based Correlations

Use rank correlations for robustness to outliers:

```python
# Spearman correlation
model = GaussianGraphicalModel(
    method='stars',
    correlation='spearman'  # or 'kendall'
)
model.fit(X)
```

### 4. Fixed Regularization Parameter

If you already know the regularization parameter:

```python
model = GaussianGraphicalModel(alpha=0.1)
model.fit(X)
```

### 5. Model Evaluation

```python
# Compute log-likelihood on test data
score = model.score(X_test)

# Access model attributes
print(f"Precision matrix: {model.precision_.shape}")
print(f"Covariance matrix: {model.covariance_.shape}")
print(f"Adjacency matrix: {model.adjacency_.shape}")
print(f"Number of edges: {model.n_edges_}")
```

### 6. Network Analysis

Convert the estimated graph to NetworkX for further analysis:

```python
import networkx as nx

# Convert to NetworkX graph
G = model.to_networkx(threshold=0.01, weighted=True)

# Network statistics
print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")
print(f"Average degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")

# Centrality measures
degree_centrality = nx.degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G)

# Find connected components
components = list(nx.connected_components(G))
print(f"Number of connected components: {len(components)}")
```

### 7. Visualization

```python
from pyggm.visualization import (
    plot_network,
    plot_precision_matrix,
    plot_stars_path,
    plot_edge_stability,
    plot_model_selection
)
import matplotlib.pyplot as plt

# Create comprehensive visualization
fig = plt.figure(figsize=(16, 10))

# Network plot
ax1 = plt.subplot(2, 3, 1)
plot_network(model.precision_, threshold=0.01, layout='spring', ax=ax1)

# Precision matrix heatmap
ax2 = plt.subplot(2, 3, 2)
plot_precision_matrix(model.precision_, ax=ax2)

# StARS path (if using StARS)
if hasattr(model, 'instabilities_'):
    ax3 = plt.subplot(2, 3, 3)
    plot_stars_path(model, ax=ax3)

    ax4 = plt.subplot(2, 3, 4)
    plot_edge_stability(model, ax=ax4)

# Model selection plot
ax5 = plt.subplot(2, 3, 5)
plot_model_selection(model, ax=ax5)

plt.tight_layout()
plt.show()
```

### 8. Complete Pipeline Example

```python
import numpy as np
from pyggm import GaussianGraphicalModel, NonparanormalTransformer
from pyggm.visualization import plot_network, plot_stars_path
import matplotlib.pyplot as plt

# 1. Load/generate data
np.random.seed(42)
n, p = 200, 30

# Create data with known structure
true_precision = np.eye(p)
true_precision[0, 1] = true_precision[1, 0] = 0.5
true_precision[2, 3] = true_precision[3, 2] = 0.4
true_cov = np.linalg.inv(true_precision)
X = np.random.multivariate_normal(np.zeros(p), true_cov, n)

# 2. Optional: Apply nonparanormal transformation
# (Skip if data is already Gaussian)
transformer = NonparanormalTransformer()
X_transformed = transformer.fit_transform(X)

# 3. Fit model
model = GaussianGraphicalModel(
    method='stars',
    beta=0.05,
    n_subsamples=50,
    n_jobs=-1,
    random_state=42,
    verbose=1
)
model.fit(X_transformed)

# 4. Evaluate
print(f"\nModel Summary:")
print(f"  Selected alpha: {model.alpha_:.4f}")
print(f"  Number of edges: {model.n_edges_}")
print(f"  Sparsity: {1 - model.n_edges_ / (p * (p-1) / 2):.2%}")

# 5. Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
plot_network(model.precision_, threshold=0.01, ax=axes[0])
plot_stars_path(model, ax=axes[1])
plt.savefig('ggm_results.png', dpi=300, bbox_inches='tight')
plt.show()

# 6. Export to NetworkX for further analysis
G = model.to_networkx(threshold=0.01)
print(f"\nNetwork statistics:")
print(f"  Nodes: {G.number_of_nodes()}")
print(f"  Edges: {G.number_of_edges()}")
print(f"  Density: {2 * G.number_of_edges() / (p * (p-1)):.3f}")
```

## API Reference

### `GaussianGraphicalModel`

Main estimator class for Gaussian Graphical Models.

**Parameters:**
- `method` (str): Regularization selection method ('stars', 'ebic', or 'cv')
- `correlation` (str): Correlation type ('pearson', 'spearman', or 'kendall')
- `alpha` (float or None): Fixed regularization parameter
- `n_alphas` (int): Number of alpha values to evaluate
- `alpha_min_ratio` (float): Ratio of min to max alpha
- `beta` (float): StARS instability threshold
- `n_subsamples` (int): Number of subsamples for StARS
- `gamma` (float): EBIC gamma parameter
- `cv` (int): Number of CV folds
- `ensure_psd` (bool): Project correlation matrix to PSD
- `n_jobs` (int): Number of parallel jobs
- `random_state` (int or None): Random seed
- `verbose` (int): Verbosity level

**Attributes:**
- `precision_`: Estimated precision matrix
- `covariance_`: Estimated covariance matrix
- `adjacency_`: Binary adjacency matrix
- `alpha_`: Selected regularization parameter
- `n_edges_`: Number of edges
- `alphas_`: Grid of alphas evaluated
- `instabilities_`: StARS instabilities (if method='stars')
- `ebic_scores_`: EBIC scores (if method='ebic')
- `cv_scores_`: CV scores (if method='cv')

**Methods:**
- `fit(X)`: Fit the model
- `score(X)`: Compute average log-likelihood
- `to_networkx(threshold, weighted, labels)`: Convert to NetworkX graph

### `NonparanormalTransformer`

Gaussian copula transformation for non-Gaussian data.

**Parameters:**
- `truncate` (bool): Whether to truncate extreme values

**Methods:**
- `fit(X)`: Learn empirical CDFs
- `transform(X)`: Apply transformation
- `fit_transform(X)`: Fit and transform in one step

## Algorithm Details

### StARS (Stability Approach to Regularization Selection)

StARS selects the regularization parameter by evaluating the stability of edge selection across multiple subsamples. It chooses the smallest alpha where the instability is below a threshold β (default 0.05).

**Key parameters:**
- `n_subsamples`: More subsamples = more stable selection (default 50)
- `subsample_size`: Defaults to floor(10 * sqrt(n))
- `beta`: Lower values = sparser graphs (default 0.05)

**Reference:** Liu, H., Roeder, K., & Wasserman, L. (2010). Stability approach to regularization selection (stars) for high dimensional graphical models. NIPS.

### EBIC (Extended BIC)

EBIC extends the standard BIC with an additional penalty for graph sparsity:

```
EBIC = -2 * log_likelihood + |E| * log(n) + 4 * |E| * gamma * log(p)
```

**Key parameters:**
- `gamma`: Controls sparsity penalty (0 = standard BIC, 0.5 recommended for high-dimensional settings)

**Reference:** Foygel, R., & Drton, M. (2010). Extended Bayesian information criteria for Gaussian graphical models. NIPS.

### Nonparanormal

The nonparanormal assumes a Gaussian copula: after transforming each variable's marginal distribution to Gaussian, the joint distribution is multivariate Gaussian.

**Transformations:**
- Copula: X_j → Φ^(-1)(F_hat_j(X_j))
- Spearman: ρ → 2 * sin(π/6 * ρ)
- Kendall: τ → sin(π/2 * τ)

**Reference:** Liu, H., Lafferty, J., & Wasserman, L. (2009). The nonparanormal: Semiparametric estimation of high dimensional undirected graphs. JMLR.

## Performance Tips

1. **Use parallelization**: Set `n_jobs=-1` to use all CPU cores
2. **For large p**: Consider using `method='ebic'` (faster than StARS)
3. **For robust estimation**: Use `correlation='spearman'` or apply `NonparanormalTransformer`
4. **For interpretability**: Use `method='stars'` with default `beta=0.05`

## Testing

Run the test suite:

```bash
# Install test dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run with coverage
pytest --cov=pyggm --cov-report=html

# Run specific test file
pytest tests/test_ggm.py -v
```

## Development

### Code formatting

```bash
# Format code
black src/ tests/

# Check with ruff
ruff check src/ tests/ --fix
```

### Type checking

```bash
mypy src/pyggm/
```

## Citation

If you use PyGGM in your research, please cite the relevant papers:

```bibtex
@article{liu2010stars,
  title={Stability approach to regularization selection (stars) for high dimensional graphical models},
  author={Liu, Han and Roeder, Kathryn and Wasserman, Larry},
  journal={Advances in neural information processing systems},
  volume={23},
  year={2010}
}

@article{foygel2010ebic,
  title={Extended Bayesian information criteria for Gaussian graphical models},
  author={Foygel, Rina and Drton, Mathias},
  journal={Advances in neural information processing systems},
  volume={23},
  pages={604--612},
  year={2010}
}

@article{liu2009nonparanormal,
  title={The nonparanormal: Semiparametric estimation of high dimensional undirected graphs},
  author={Liu, Han and Lafferty, John and Wasserman, Larry},
  journal={Journal of Machine Learning Research},
  volume={10},
  number={10},
  year={2009}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

This package wraps scikit-learn's `GraphicalLasso` implementation and implements the StARS and EBIC selection procedures based on the cited papers.

## Support

- Issues: [GitHub Issues](https://github.com/naandip/pyggm/issues)
- Documentation: [docs/](docs/)

---

**PyGGM** - A Python package for Gaussian Graphical Models
