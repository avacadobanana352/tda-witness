"""Smith Normal Form reduction and GF(2) rank computation.

Algorithm from *Computational Topology: An Introduction*
(Edelsbrunner & Harer, 2008).
"""

from __future__ import annotations

import numpy as np


def snf_gf2(matrix: np.ndarray) -> np.ndarray:
    """Compute the Smith Normal Form of *matrix* over GF(2).

    The input boundary matrix may have entries in {-1, 0, 1}.  It is first
    mapped to GF(2) via ``abs(.) % 2``, then reduced iteratively so that
    the result is diagonal with leading 1s followed by 0s.

    Parameters
    ----------
    matrix : np.ndarray
        Integer matrix (entries in {-1, 0, 1}).

    Returns
    -------
    np.ndarray
        Reduced matrix in Smith Normal Form over GF(2).
    """
    result = np.abs(matrix).astype(np.uint8) % 2

    pivot = 0
    while pivot < min(result.shape):
        # Find the first 1 in the sub-matrix from (pivot, pivot) onward
        rows, cols = np.where(result[pivot:, pivot:] == 1)
        if len(rows) == 0:
            break

        pivot_row = rows[0] + pivot
        pivot_col = cols[0] + pivot

        # Move the leading 1 to the (pivot, pivot) position
        if pivot_row != pivot:
            result[[pivot, pivot_row], :] = result[[pivot_row, pivot], :].copy()
        if pivot_col != pivot:
            result[:, [pivot, pivot_col]] = result[:, [pivot_col, pivot]].copy()

        # Eliminate all other 1s in the pivot column (vectorized)
        elim_rows = np.flatnonzero(result[:, pivot])
        elim_rows = elim_rows[elim_rows != pivot]
        if len(elim_rows) > 0:
            result[elim_rows] ^= result[pivot]

        # Eliminate all other 1s in the pivot row (vectorized)
        elim_cols = np.flatnonzero(result[pivot, :])
        elim_cols = elim_cols[elim_cols != pivot]
        if len(elim_cols) > 0:
            result[:, elim_cols] ^= result[:, pivot : pivot + 1]

        pivot += 1

    return result


def rank_gf2(matrix: np.ndarray) -> int:
    """Compute the rank of *matrix* over GF(2) via row echelon form.

    Much faster than full SNF when only the rank is needed: uses only
    forward row elimination with vectorized XOR, no column operations.

    Parameters
    ----------
    matrix : np.ndarray
        Integer matrix (entries in {-1, 0, 1}).

    Returns
    -------
    int
        Rank of the matrix over GF(2).
    """
    A = np.abs(matrix).astype(np.uint8) % 2
    rows, cols = A.shape
    pivot_row = 0

    for col in range(cols):
        if pivot_row >= rows:
            break

        # Find first nonzero entry in this column at or below pivot_row
        nonzero = np.flatnonzero(A[pivot_row:, col])
        if len(nonzero) == 0:
            continue

        # Swap to pivot position
        first_nz = nonzero[0] + pivot_row
        if first_nz != pivot_row:
            A[[pivot_row, first_nz]] = A[[first_nz, pivot_row]]

        # Vectorized XOR elimination of all rows below pivot
        eliminate = np.flatnonzero(A[pivot_row + 1 :, col]) + pivot_row + 1
        if len(eliminate) > 0:
            A[eliminate] ^= A[pivot_row]

        pivot_row += 1

    return pivot_row
