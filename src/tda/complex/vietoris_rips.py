"""Vietoris-Rips complex construction (Zomorodian's inductive algorithm)."""

from __future__ import annotations

import numpy as np


def _lower_neighbours(G: np.ndarray, u: int) -> np.ndarray:
    """Return indices of vertices *j < u* connected to *u* in graph *G*."""
    return np.where(G[:u, u] == 1)[0]


def compute_vr_complex(
    graph: np.ndarray,
    max_dim: int,
) -> tuple[list[np.ndarray], int]:
    """Compute the k-skeleton of the Vietoris-Rips complex.

    Uses Zomorodian's inductive algorithm (Zomorodian, 2010).

    Parameters
    ----------
    graph : np.ndarray, shape (n, n)
        Symmetric adjacency matrix of the 1-skeleton (0-indexed vertices).
    max_dim : int
        Maximum simplex dimension to compute.

    Returns
    -------
    k_skeleton : list of np.ndarray
        ``k_skeleton[d]`` has shape ``(n_d_simplices, d+1)`` and contains the
        *d*-simplices as rows of sorted vertex indices.
    achieved_dim : int
        Actual maximum simplex dimension achieved (may be < *max_dim*).

    Raises
    ------
    ValueError
        If *graph* is not square or *max_dim* < 1.
    """
    if graph.ndim != 2 or graph.shape[0] != graph.shape[1]:
        raise ValueError("graph must be a square 2-D array")
    if max_dim < 1:
        raise ValueError("max_dim must be >= 1")

    n_points = graph.shape[0]
    achieved_dim = max_dim

    # 0-simplices (vertices)
    k_skeleton: list[np.ndarray] = [np.arange(n_points).reshape(-1, 1)]

    for dim in range(max_dim):
        next_level: list[np.ndarray] = []
        prev_simplices = k_skeleton[dim]

        for simplex in prev_simplices:
            # Find vertices w < min(simplex) connected to every vertex in simplex
            reachable = _lower_neighbours(graph, simplex[0])
            for vertex in simplex[1:]:
                reachable = reachable[np.isin(reachable, _lower_neighbours(graph, vertex))]

            for new_vertex in reachable:
                next_level.append(np.sort(np.append(simplex, new_vertex)))

        if not next_level:
            achieved_dim = dim
            return k_skeleton, achieved_dim

        next_array = np.unique(np.array(next_level), axis=0)
        sort_idx = np.argsort(next_array[:, 0], kind="stable")
        k_skeleton.append(next_array[sort_idx])

    return k_skeleton, achieved_dim
