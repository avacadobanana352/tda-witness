"""Witness complex construction."""

from __future__ import annotations

import numpy as np


def build_witness_graph(
    distances: np.ndarray,
    threshold: float,
    witness_param: int,
) -> np.ndarray:
    """Build the 1-skeleton adjacency matrix using the witness complex.

    A landmark pair (a, b) is connected if at least one data point *i*
    (witness) satisfies ``max(D[a,i], D[b,i]) <= threshold + m[i]``, where
    *m[i]* is the ``witness_param``-th smallest landmark distance for
    witness *i*.

    Parameters
    ----------
    distances : np.ndarray, shape (n_landmarks, n_witnesses)
        Distance matrix from landmarks (rows) to all data points (columns).
    threshold : float
        Connectivity threshold (*R*).
    witness_param : int
        Witness parameter (*v*).  ``v=0`` uses weak witnesses (``m[i]=0``).

    Returns
    -------
    np.ndarray, shape (n_landmarks, n_landmarks)
        Symmetric integer adjacency matrix with zeros on the diagonal.

    Raises
    ------
    ValueError
        If *threshold* is negative or *witness_param* is negative.
    """
    if threshold < 0:
        raise ValueError("threshold must be >= 0")
    if witness_param < 0:
        raise ValueError("witness_param must be >= 0")

    n_landmarks, n_witnesses = distances.shape
    graph = np.zeros((n_landmarks, n_landmarks), dtype=int)

    # witness_threshold[i] = v-th nearest landmark distance for witness i
    if witness_param > 0:
        sorted_dists = np.sort(distances, axis=0)
        witness_threshold = sorted_dists[witness_param - 1, :]
    else:
        witness_threshold = np.zeros(n_witnesses)

    # Vectorized: for each witness, compute pairwise max distances and check
    # against the threshold.  This replaces the original O(L^2 * W) Python
    # loop with O(W) vectorized passes, each O(L^2) in NumPy C code.
    for i in range(n_witnesses):
        d_col = distances[:, i]  # shape (n_landmarks,)
        max_dists = np.maximum(d_col[:, None], d_col[None, :])
        connected = max_dists <= threshold + witness_threshold[i]
        graph |= connected.astype(int)

    np.fill_diagonal(graph, 0)
    # Ensure symmetry
    graph = np.maximum(graph, graph.T)
    return graph
