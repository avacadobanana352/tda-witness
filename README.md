# tda-witness

Minimal, from-scratch implementation of Topological Data Analysis in Python. Point cloud in, Betti numbers out. Vibe coded from the original [MATLAB learning code](https://www.mathworks.com/matlabcentral/fileexchange/47009-topological-data-analysis-learning-code).

This is mainly intended as a learning tool for people who may be studying the topic. I still find this an interesting topic in math, so thought it'd be a good one to bring up to date with Claude.

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/avacadobanana352/tda-witness/blob/main/examples/tda_witness_demo.ipynb)

<p align="center">
  <img src="examples/mnist8_filtration.gif" alt="Filtration on MNIST digit 8 — detecting two loops" width="420">
</p>

## Install

```bash
pip install -e ".[all]"
```

## Quick Start

```python
import numpy as np
from tda import analyze

# sample a circle
t = np.linspace(0, 2 * np.pi, 50, endpoint=False)
data = np.column_stack([np.cos(t), np.sin(t)])

result = analyze(data, threshold=0.3)
print(result["betti"])  # [1, 1, ...] — one component, one loop
```

Visualize:

```python
from tda.visualization import plot_complex, save_html

fig = plot_complex(result["data"], result["landmarks"], result["graph"], result["complex"])
fig.show()
```

Sweep thresholds with an interactive slider:

```python
from tda.visualization import plot_filtration

fig = plot_filtration(data, thresholds=np.linspace(0.1, 0.5, 20), seed=42)
fig.show()
```

## MNIST: Classifying Digits by Topology

A "0" has one hole. A "1" has none. An "8" has two. With just $\beta_1$ (loop count), we can classify handwritten digits — no training needed.

```python
# convert a 28x28 digit image to a point cloud
ys, xs = np.where(image > 128)
point_cloud = np.column_stack([xs, 27 - ys]).astype(float)

result = analyze(point_cloud, threshold=1.5, n_landmarks=40, normalize=False)
print(result["betti"])  # [1, 2] for an "8" — one component, two loops
```

Gets **76% accuracy** over 200 test samples with zero training. See the full example: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/avacadobanana352/tda-witness/blob/main/examples/mnist_topology.ipynb)

## CLI

```bash
tda compute data.csv -r 0.3 --plot          # betti numbers + visualization
tda persist data.csv --dim 2 --plot          # persistent homology
tda generate circle -n 100 -o circle.csv     # synthetic data
```

## How It Works

```
Point Cloud → Normalize → Select Landmarks (farthest-point sampling)
                                ↓
                        Distance Matrix
                                ↓
                  Witness Complex (1-skeleton)
                                ↓
                   Vietoris-Rips (k-skeleton)
                                ↓
                ┌───────────────┴───────────────┐
                ↓                               ↓
    Boundary Matrices                  Filtration (by birth time)
                ↓                               ↓
      SNF over GF(2)                   Column Reduction
                ↓                               ↓
        Betti Numbers                  Persistence Pairs
```

**Betti numbers** count topological features at a fixed threshold:

| $\beta_0$ | $\beta_1$ | $\beta_2$ |
|:---------:|:---------:|:---------:|
| components | loops | voids |

**Persistent homology** sweeps all thresholds, tracking when features appear (birth) and disappear (death). No threshold parameter needed:

```python
from tda import persistent_homology

result = persistent_homology(data, simplex_dim=2)
# result["pairs"] → [{"dim": 0, "birth": 0.0, "death": inf}, ...]
```

## API

| Function | What it does |
|----------|-------------|
| `analyze(data, threshold=0.4, ...)` | Point cloud → Betti numbers |
| `persistent_homology(data, ...)` | Point cloud → persistence pairs |
| `plot_complex(data, landmarks, graph, complex_)` | Draw the simplicial complex |
| `plot_filtration(data, thresholds, ...)` | Interactive threshold slider |
| `plot_betti_summary(thresholds, betti_seqs)` | Betti numbers vs. R |
| `plot_persistence_diagram(pairs)` | Birth vs. death scatter |
| `plot_barcode(pairs)` | Lifetime bars by dimension |
| `make_circle / make_torus / make_sphere / ...` | Synthetic datasets |

## References

- De Silva & Carlsson (2004). *Topological estimation using witness complexes.*
- Zomorodian (2010). *Fast construction of the Vietoris-Rips complex.* Computers & Graphics.
- Edelsbrunner, Letscher & Zomorodian (2002). *Topological persistence and simplification.* DCG.
- Edelsbrunner & Harer (2010). *Computational Topology: An Introduction.* AMS.

## License

MIT

