"""Persistent homology via column reduction over GF(2).

The standard persistence algorithm processes the boundary matrix column by
column in filtration order.  Each column either reduces to zero (creating a
new homological feature) or has a unique lowest nonzero entry (killing an
existing feature), producing a birth-death pair.

Reference: Edelsbrunner, Letscher & Zomorodian (2002),
           "Topological Persistence and Simplification".
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def compute_persistence(
    simplices: list[tuple[int, ...]],
    birth_times: np.ndarray,
    dimensions: np.ndarray,
) -> list[dict[str, Any]]:
    """Compute persistent homology from a filtered simplicial complex.

    Parameters
    ----------
    simplices : list of tuple[int, ...]
        Simplices in filtration order (ascending birth time).
    birth_times : np.ndarray, shape (n,)
        Birth time for each simplex.
    dimensions : np.ndarray, shape (n,)
        Dimension of each simplex.

    Returns
    -------
    list of dict
        Each dict has keys ``"dim"`` (int), ``"birth"`` (float),
        ``"death"`` (float — ``inf`` for essential features).
        Sorted by (dimension, birth, death).
    """
    n = len(simplices)

    # Map each simplex tuple to its filtration index
    simplex_to_idx: dict[tuple[int, ...], int] = {}
    for i, s in enumerate(simplices):
        simplex_to_idx[s] = i

    # Build boundary columns as sets of filtration indices (sparse GF(2))
    columns: list[set[int]] = []
    for j, simplex in enumerate(simplices):
        if len(simplex) <= 1:
            # 0-simplices have empty boundary
            columns.append(set())
        else:
            # Faces: remove each vertex in turn
            boundary: set[int] = set()
            for k in range(len(simplex)):
                face = simplex[:k] + simplex[k + 1:]
                face_idx = simplex_to_idx[face]
                boundary.add(face_idx)
            columns.append(boundary)

    # --- Column reduction ---
    # pivot_col[i] = column index j whose pivot (lowest entry) is row i
    pivot_col: dict[int, int] = {}

    for j in range(n):
        _reduce_column(columns, j, pivot_col)

    # --- Extract persistence pairs ---
    paired_as_death: set[int] = set()  # indices that appear as a death simplex
    paired_as_birth: set[int] = set()  # indices that appear as a birth simplex

    pairs: list[dict[str, Any]] = []

    for j in range(n):
        pivot = _get_pivot(columns[j])
        if pivot is not None:
            # Column j kills the feature born by simplex pivot
            paired_as_death.add(j)
            paired_as_birth.add(pivot)
            pairs.append({
                "dim": int(dimensions[pivot]),
                "birth": float(birth_times[pivot]),
                "death": float(birth_times[j]),
            })

    # Essential features: simplices that created a feature that never dies
    for j in range(n):
        if j not in paired_as_birth and j not in paired_as_death:
            # This simplex created a feature (its column reduced to zero)
            # and was never killed
            if len(columns[j]) == 0 or _get_pivot(columns[j]) is None:
                pairs.append({
                    "dim": int(dimensions[j]),
                    "birth": float(birth_times[j]),
                    "death": math.inf,
                })

    # Sort by (dimension, birth, death)
    pairs.sort(key=lambda p: (p["dim"], p["birth"], p["death"]))
    return pairs


def _get_pivot(col: set[int]) -> int | None:
    """Return the largest index in a column (the pivot), or None if empty."""
    return max(col) if col else None


def _reduce_column(
    columns: list[set[int]],
    j: int,
    pivot_col: dict[int, int],
) -> None:
    """Reduce column *j* by adding earlier columns with matching pivots."""
    while True:
        pivot = _get_pivot(columns[j])
        if pivot is None:
            break
        if pivot not in pivot_col:
            # This pivot is unique; record it
            pivot_col[pivot] = j
            break
        # Add (XOR) the earlier column that has the same pivot
        other = pivot_col[pivot]
        columns[j] = columns[j].symmetric_difference(columns[other])


def pairs_to_barcodes(
    pairs: list[dict[str, Any]],
) -> dict[int, list[tuple[float, float]]]:
    """Group persistence pairs into barcodes by dimension.

    Returns
    -------
    dict mapping dimension -> list of (birth, death) tuples.
    """
    barcodes: dict[int, list[tuple[float, float]]] = {}
    for p in pairs:
        dim = p["dim"]
        if dim not in barcodes:
            barcodes[dim] = []
        barcodes[dim].append((p["birth"], p["death"]))
    return barcodes
