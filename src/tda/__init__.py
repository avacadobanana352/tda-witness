"""Topological Data Analysis: Betti numbers via witness complexes.

Quick start::

    import numpy as np
    from tda import analyze

    # Points sampled from a circle
    t = np.linspace(0, 2 * np.pi, 50, endpoint=False)
    data = np.column_stack([np.cos(t), np.sin(t)])

    result = analyze(data, threshold=0.3, simplex_dim=2)
    print(result["betti"])  # [1, 1] -- one component, one loop
"""

from __future__ import annotations

import numpy as np

from tda.preprocessing import normalize_data, get_landmarks, pairwise_distances
from tda.complex.witness import build_witness_graph
from tda.complex.vietoris_rips import compute_vr_complex
from tda.complex.filtration import build_filtration
from tda.homology.boundary import compute_boundary_matrices
from tda.homology.betti import compute_betti_numbers
from tda.homology.persistence import compute_persistence, pairs_to_barcodes

__version__ = "0.3.0"

__all__ = [
    "analyze",
    "persistent_homology",
    "normalize_data",
    "get_landmarks",
    "build_witness_graph",
    "compute_vr_complex",
    "build_filtration",
    "compute_boundary_matrices",
    "compute_betti_numbers",
    "compute_persistence",
    "pairs_to_barcodes",
]


def analyze(
    data: np.ndarray,
    *,
    n_landmarks: int | None = None,
    simplex_dim: int = 2,
    threshold: float = 0.4,
    witness_param: int = 1,
    normalize: bool = True,
    seed: int | None = None,
) -> dict:
    """End-to-end TDA pipeline: point cloud to Betti numbers.

    Parameters
    ----------
    data : np.ndarray, shape (n_points, n_dims)
        Input point cloud.
    n_landmarks : int or None
        Number of landmark points.  Defaults to all points.
    simplex_dim : int
        Maximum simplex dimension (*k*).
    threshold : float
        Witness complex connectivity threshold (*R*).
    witness_param : int
        Witness parameter (*v*).  ``v=0`` for weak witnesses.
    normalize : bool
        Whether to min-max normalize the data to [0, 1].
    seed : int or None
        Random seed for landmark selection (for reproducibility).

    Returns
    -------
    dict with keys:
        ``"betti"``
            list[int] -- Betti numbers.
        ``"graph"``
            np.ndarray -- Adjacency matrix of the 1-skeleton.
        ``"complex"``
            list[np.ndarray] -- k-skeleton of the VR complex.
        ``"landmarks"``
            np.ndarray -- Indices of selected landmark points.
        ``"data"``
            np.ndarray -- (Possibly normalized) input data.
        ``"boundary_matrices"``
            list[np.ndarray] -- Boundary matrices for each dimension.
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError(f"data must be 2-D, got {data.ndim}-D array")

    if normalize:
        data = normalize_data(data)

    n_points = data.shape[0]
    if n_landmarks is None:
        n_landmarks = n_points

    rng = np.random.default_rng(seed)
    landmarks = get_landmarks(data, n_landmarks, rng=rng)

    # Distance from landmarks to all points
    distances = pairwise_distances(data[landmarks], data)

    # Build simplicial complex
    graph = build_witness_graph(distances, threshold, witness_param)
    complex_, achieved_dim = compute_vr_complex(graph, simplex_dim)

    # Compute boundary matrices and Betti numbers
    n_levels = len(complex_)
    boundary_mats = compute_boundary_matrices(complex_, n_levels)
    betti = compute_betti_numbers(boundary_mats)

    # The highest Betti number (beta_d for simplex_dim=d) is unreliable
    # when the complex hit the requested ceiling: without (d+1)-simplices,
    # beta_d counts ALL d-cycles as surviving, inflating the result.
    # Only drop when achieved_dim == simplex_dim (the complex may have
    # more simplices we didn't compute).  When the complex stopped early
    # (achieved_dim < simplex_dim), all Betti numbers are reliable.
    if achieved_dim == simplex_dim and len(betti) > 1:
        betti = betti[:-1]

    return {
        "betti": betti,
        "graph": graph,
        "complex": complex_,
        "landmarks": landmarks,
        "data": data,
        "boundary_matrices": boundary_mats,
    }


def persistent_homology(
    data: np.ndarray,
    *,
    n_landmarks: int | None = None,
    simplex_dim: int = 2,
    witness_param: int = 1,
    normalize: bool = True,
    seed: int | None = None,
) -> dict:
    """Compute persistent homology of a point cloud.

    Unlike :func:`analyze`, this does not require a fixed threshold.
    It sweeps all thresholds automatically, tracking when topological
    features (components, loops, voids) appear and disappear.

    Parameters
    ----------
    data : np.ndarray, shape (n_points, n_dims)
        Input point cloud.
    n_landmarks : int or None
        Number of landmark points.  Defaults to all points.
    simplex_dim : int
        Maximum simplex dimension (*k*).
    witness_param : int
        Witness parameter (*v*).
    normalize : bool
        Whether to min-max normalize the data to [0, 1].
    seed : int or None
        Random seed for landmark selection.

    Returns
    -------
    dict with keys:
        ``"pairs"``
            list[dict] -- Persistence pairs, each with ``"dim"``,
            ``"birth"``, ``"death"`` (``inf`` for essential features).
        ``"barcodes"``
            dict[int, list[tuple]] -- Bars grouped by dimension.
        ``"filtration_simplices"``
            list[tuple] -- Simplices in filtration order.
        ``"birth_times"``
            np.ndarray -- Birth time of each simplex.
        ``"landmarks"``
            np.ndarray -- Landmark indices.
        ``"data"``
            np.ndarray -- (Possibly normalized) data.
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError(f"data must be 2-D, got {data.ndim}-D array")

    if normalize:
        data = normalize_data(data)

    n_points = data.shape[0]
    if n_landmarks is None:
        n_landmarks = n_points

    rng = np.random.default_rng(seed)
    landmarks = get_landmarks(data, n_landmarks, rng=rng)

    distances = pairwise_distances(data[landmarks], data)

    simplices, birth_times, dimensions = build_filtration(
        distances, witness_param, simplex_dim,
    )

    all_pairs = compute_persistence(simplices, birth_times, dimensions)

    # Drop the top-dimension homology: with simplex_dim=k, H_k is
    # artificially inflated because there are no (k+1)-simplices to
    # kill k-cycles.  H_0 through H_{k-1} are computed correctly.
    pairs = [p for p in all_pairs if p["dim"] < simplex_dim]
    barcodes = pairs_to_barcodes(pairs)

    return {
        "pairs": pairs,
        "barcodes": barcodes,
        "filtration_simplices": simplices,
        "birth_times": birth_times,
        "landmarks": landmarks,
        "data": data,
    }
