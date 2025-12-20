# PyGGM Tutorials

This folder contains comprehensive tutorials and examples for the **PyGGM** package.


## Notebooks

### 1. Complete Tutorial with Real Data
**File:** `Complete_Tutorial_Real_Data.ipynb`

A comprehensive end-to-end tutorial using real flow cytometry data (11 proteins, 7,466 samples).
   - Learn the complete workflow
   - Understand normality testing and transformation
   - Compare different methods


**Topics covered:**
- **Normality testing** - Critical first step for GGMs
- **Nonparanormal (NPN) transformation** - Handling non-Gaussian data
- **Method comparison:**
  - StARS (Stability Selection)
  - EBIC (Extended BIC)
  - Cross-Validation
- **Advanced visualizations:**
  - Network plots (spring, circular layouts)
  - Precision matrices and adjacency matrices
  - StARS instability paths
  - Edge stability heatmaps
- **Bootstrap stability analysis**
- **Fixed alpha analysis**

---

### 2. Validation on Simulated Networks
**File:** `Validation_Simulated_Network.ipynb`

Demonstrates PyGGM's performance on synthetic networks with known ground truth.
   - See validation on known structures
   - Understand regularization paths
   - Learn about different network types

**Network types tested:**
1. **Hub network** - Central node connected to many others (gene regulation, protein interactions)
2. **Chain network** - Sequential connections (time series, pathways)
3. **Random network** - Erdős-Rényi structure (testing robustness)
4. **Block diagonal network** - Community structure (modular systems)
5. **Scale-free network** - Power-law degree distribution (biological networks, social networks)

**Topics covered:**
- Ground truth comparison
- Performance metrics (precision, recall, F1-score)
- Regularization path visualization
- Network recovery across different structures
- Custom network layouts for visualization

---

## Data

### `flow_cytometry_data.csv`
### Sachs et al. 2005 Flow Cytometry Data
Real biological data measuring protein expression levels in single cells.
- **Size:** 7,466 samples × 11 features
- **Features:** Signaling proteins (praf, pmek, plcg, PIP2, PIP3, P44, pakts, PKA, PKC, P38, pjnk)
- **Source:** Flow cytometry measurements
- **Used in:** `Complete_Tutorial_Real_Data.ipynb`

---

## Getting Started

### Installation

Install PyGGM and required dependencies:

```bash
pip install pyggm jupyter matplotlib seaborn networkx pandas numpy scipy scikit-learn
```

### Running the Notebooks

1. Clone the repository or download the docs folder
2. Navigate to the docs directory:
   ```bash
   cd docs
   ```
3. Launch Jupyter:
   ```bash
   jupyter notebook
   ```
4. Open either notebook and run cells sequentially

### Quick Start Example

```python
from pyggm import GaussianGraphicalModel
from pyggm.visualization import plot_network
import numpy as np

# Generate sample data
X = np.random.randn(500, 10)

# Fit model using StARS
model = GaussianGraphicalModel(method='stars', beta=0.05)
model.fit(X)

# Visualize network
plot_network(model.precision_, threshold=0.01, layout='spring')

print(f"Estimated {model.n_edges_} edges")
```

---

## Methods Comparison

PyGGM supports three regularization selection methods:

| Method | Description | Best For | Speed |
|--------|-------------|----------|-------|
| **StARS** | Stability Approach to Regularization Selection | Reproducibility, minimizing false positives | Slow |
| **EBIC** | Extended Bayesian Information Criterion | Balance between fit and sparsity, exploration | Fast |
| **CV** | Cross-Validation | Predictive performance, standard ML approach | Medium |

---

## Key Features Demonstrated

### Handling Non-Gaussian Data
```python
from pyggm import Nonparanormal

# Transform non-Gaussian data
npn = Nonparanormal()
X_transformed = npn.fit_transform(X_raw)

# Fit model on transformed data
model.fit(X_transformed)
```

### Comprehensive Visualizations
```python
from pyggm.visualization import (
    plot_network,
    plot_precision_matrix,
    plot_stars_path,
    plot_edge_stability,
    plot_regularization_path,
    plot_model_selection
)
```

### Network Analysis
```python
# Export to NetworkX for further analysis
G = model.to_networkx()

# Access model outputs
precision_matrix = model.precision_
adjacency_matrix = model.adjacency_
n_edges = model.n_edges_
selected_lambda = model.alpha_
```

---

## Learn More

- **PyGGM GitHub:** [https://github.com/naandip/pyggm](https://github.com/naandip/pyggm)
- **PyPI Package:** [https://pypi.org/project/pyggm/](https://pypi.org/project/pyggm/)
- **Documentation:** See notebooks in this folder

---

## Citation

If you use PyGGM in your research, please cite:

```bibtex
@software{pyggm,
  title = {PyGGM: Gaussian Graphical Models in Python},
  author = {Prasad, Nandini},
  year = {2025},
  url = {https://github.com/naandip/pyggm}
}
```



