"""Smith Normal Form reduction over GF(2).

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
    result = np.abs(matrix).astype(int) % 2

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

        # Eliminate all other 1s in the pivot column
        for row in range(result.shape[0]):
            if row != pivot and result[row, pivot] == 1:
                result[row, :] = (result[pivot, :] + result[row, :]) % 2

        # Eliminate all other 1s in the pivot row
        for col in range(result.shape[1]):
            if col != pivot and result[pivot, col] == 1:
                result[:, col] = (result[:, pivot] + result[:, col]) % 2

        pivot += 1

    return result
