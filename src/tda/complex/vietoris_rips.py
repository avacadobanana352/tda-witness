"""Vietoris-Rips complex construction (Zomorodian's inductive algorithm)."""

from __future__ import annotations

import numpy as np


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

    # Pre-compute adjacency sets and lower-neighbor sets for fast intersection
    adj_sets: list[set[int]] = [set() for _ in range(n_points)]
    lower_adj: list[set[int]] = [set() for _ in range(n_points)]
    for u in range(n_points):
        neighbors = set(np.flatnonzero(graph[u]).tolist())
        neighbors.discard(u)
        adj_sets[u] = neighbors
        lower_adj[u] = {v for v in neighbors if v < u}

    # 0-simplices (vertices)
    k_skeleton: list[np.ndarray] = [np.arange(n_points).reshape(-1, 1)]

    for dim in range(max_dim):
        next_level: list[tuple[int, ...]] = []
        prev_simplices = k_skeleton[dim]

        for row in prev_simplices:
            simplex = row.tolist()
            # Candidates: vertices < simplex[0] connected to every vertex in simplex
            candidates = lower_adj[simplex[0]].copy()
            for v in simplex[1:]:
                candidates &= adj_sets[v]
                if not candidates:
                    break

            for new_vertex in candidates:
                next_level.append(tuple(sorted([new_vertex] + simplex)))

        if not next_level:
            achieved_dim = dim
            return k_skeleton, achieved_dim

        # Deduplicate (each simplex is generated exactly once by construction,
        # but use a set for safety) and sort
        unique = sorted(set(next_level))
        k_skeleton.append(np.array(unique, dtype=int))

    return k_skeleton, achieved_dim
