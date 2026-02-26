"""Data preprocessing: normalization and landmark selection."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


def normalize_data(data: np.ndarray) -> np.ndarray:
    """Min-max normalize each column of *data* to [0, 1].

    Parameters
    ----------
    data : np.ndarray, shape (n_points, n_dims)

    Returns
    -------
    np.ndarray, same shape as *data*, with each column in [0, 1].

    Raises
    ------
    ValueError
        If *data* is not 2-D or is empty.
    """
    if data.ndim != 2:
        raise ValueError(f"data must be 2-D, got {data.ndim}-D array")
    if data.shape[0] == 0:
        raise ValueError("data must have at least one point")

    col_min = data.min(axis=0)
    col_range = data.max(axis=0) - col_min
    col_range[col_range == 0] = 1.0
    return (data - col_min) / col_range


def get_landmarks(
    data: np.ndarray,
    n_landmarks: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Select landmarks using farthest-point (MinMax) sampling.

    Iteratively picks the data point farthest from the current landmark set,
    starting from a random seed.  This gives approximately uniform coverage.

    Parameters
    ----------
    data : np.ndarray, shape (n_points, n_dims)
    n_landmarks : int
        Number of landmarks to select (<= n_points).
    rng : np.random.Generator or None
        Random generator for the initial seed (for reproducibility).

    Returns
    -------
    np.ndarray, shape (n_landmarks,)
        0-indexed row indices of the selected landmark points.

    Raises
    ------
    ValueError
        If *n_landmarks* is out of range.
    """
    if n_landmarks < 1:
        raise ValueError("n_landmarks must be >= 1")
    if n_landmarks > data.shape[0]:
        raise ValueError(
            f"n_landmarks ({n_landmarks}) exceeds n_points ({data.shape[0]})"
        )

    if rng is None:
        rng = np.random.default_rng()

    n_points = data.shape[0]
    distances = cdist(data, data, metric="euclidean")

    landmarks = np.empty(n_landmarks, dtype=int)
    landmarks[0] = rng.integers(0, n_points)
    remaining = np.ones(n_points, dtype=bool)
    remaining[landmarks[0]] = False

    for i in range(1, n_landmarks):
        dists_to_landmarks = distances[landmarks[:i]][:, remaining]
        min_dists = dists_to_landmarks.min(axis=0)
        best = np.argmax(min_dists)
        chosen = np.where(remaining)[0][best]
        landmarks[i] = chosen
        remaining[chosen] = False

    return landmarks
