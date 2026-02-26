"""Filtered simplicial complex construction with exact birth times."""

from __future__ import annotations

import numpy as np
from itertools import combinations

from tda.complex.vietoris_rips import compute_vr_complex


def build_filtration(
    distances: np.ndarray,
    witness_param: int,
    max_dim: int,
) -> tuple[list[tuple[int, ...]], np.ndarray, np.ndarray]:
    """Build a filtered witness/VR complex with exact birth times.

    For each simplex, the *birth time* is the smallest threshold R at which
    it first appears in the witness complex:

    - Vertices are born at time 0.
    - Edge (a, b) is born at ``min_i(max(D[a,i], D[b,i]) - m[i])`` where
      *m[i]* is the witness threshold for witness *i*.
    - A higher simplex is born at the maximum birth time of its edges
      (Vietoris-Rips property).

    Parameters
    ----------
    distances : np.ndarray, shape (n_landmarks, n_witnesses)
        Distance matrix from landmarks to all data points.
    witness_param : int
        Witness parameter (*v*).  ``v=0`` for weak witnesses.
    max_dim : int
        Maximum simplex dimension.

    Returns
    -------
    simplices : list of tuple[int, ...]
        All simplices in filtration order (ascending birth time,
        ties broken by dimension then lexicographic order).
    birth_times : np.ndarray, shape (n_simplices,)
        Birth time for each simplex.
    dimensions : np.ndarray, shape (n_simplices,)
        Dimension of each simplex (0 = vertex, 1 = edge, ...).
    """
    n_landmarks, n_witnesses = distances.shape

    # Witness thresholds: m[i] = v-th nearest landmark distance for witness i
    if witness_param > 0:
        sorted_dists = np.sort(distances, axis=0)
        witness_threshold = sorted_dists[witness_param - 1, :]
    else:
        witness_threshold = np.zeros(n_witnesses)

    # --- Compute edge birth times (vectorized) ---
    # For edge (a,b), birth = max(0, min_i(max(D[a,i], D[b,i]) - m[i]))
    # Compute all at once: max_dists[a, b, i] = max(D[a,i], D[b,i])
    max_dists = np.maximum(
        distances[:, None, :],  # (L, 1, W)
        distances[None, :, :],  # (1, L, W)
    )  # (L, L, W)
    effective = max_dists - witness_threshold[None, None, :]  # (L, L, W)
    edge_birth_matrix = np.maximum(np.min(effective, axis=2), 0.0)  # (L, L)

    # --- Build full VR complex on the complete landmark graph ---
    full_graph = np.ones((n_landmarks, n_landmarks), dtype=int)
    np.fill_diagonal(full_graph, 0)
    complex_, achieved_dim = compute_vr_complex(full_graph, max_dim)

    # --- Assign birth times to all simplices ---
    all_simplices: list[tuple[int, ...]] = []
    all_births: list[float] = []
    all_dims: list[int] = []

    # Vertices: born at 0
    for v in range(n_landmarks):
        all_simplices.append((v,))
        all_births.append(0.0)
        all_dims.append(0)

    # Edges: read from the birth matrix
    if len(complex_) > 1:
        for edge in complex_[1]:
            a, b = int(min(edge)), int(max(edge))
            all_simplices.append((a, b))
            all_births.append(float(edge_birth_matrix[a, b]))
            all_dims.append(1)

    # Higher simplices: birth = max edge birth among all edges in the simplex
    for d in range(2, len(complex_)):
        for simplex in complex_[d]:
            simplex_sorted = tuple(int(x) for x in sorted(simplex))
            max_edge_birth = 0.0
            for a, b in combinations(simplex_sorted, 2):
                max_edge_birth = max(max_edge_birth, float(edge_birth_matrix[a, b]))
            all_simplices.append(simplex_sorted)
            all_births.append(max_edge_birth)
            all_dims.append(d)

    # --- Sort by (birth_time, dimension, lexicographic) ---
    order = sorted(
        range(len(all_simplices)),
        key=lambda i: (all_births[i], all_dims[i], all_simplices[i]),
    )

    simplices = [all_simplices[i] for i in order]
    birth_times = np.array([all_births[i] for i in order])
    dimensions = np.array([all_dims[i] for i in order])

    return simplices, birth_times, dimensions
