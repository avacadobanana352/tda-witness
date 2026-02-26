"""Betti number computation from boundary matrices."""

from __future__ import annotations

import numpy as np

from tda.homology.smith_normal_form import snf_gf2


def _rank_gf2(snf_matrix: np.ndarray) -> int:
    """Count nonzero diagonal entries of an SNF matrix (= rank over GF(2))."""
    diag_len = min(snf_matrix.shape)
    return int(np.sum(np.diag(snf_matrix)[:diag_len] != 0))


def compute_betti_numbers(boundary_matrices: list[np.ndarray]) -> list[int]:
    """Compute all Betti numbers from a sequence of boundary matrices.

    Uses the rank-nullity theorem over GF(2):

        beta_k = dim(ker del_k) - dim(im del_{k+1})

    Parameters
    ----------
    boundary_matrices : list of np.ndarray
        ``boundary_matrices[k]`` is the boundary matrix for dimension ``k+1``
        (i.e., index 0 = boundary of vertices, index 1 = boundary of edges, ...).

    Returns
    -------
    list of int
        Betti numbers ``[beta_0, beta_1, ..., beta_{n-1}]``.
    """
    betti: list[int] = []

    for k in range(len(boundary_matrices)):
        snf_k = snf_gf2(boundary_matrices[k])
        kernel_dim = snf_k.shape[1] - _rank_gf2(snf_k)

        if k + 1 < len(boundary_matrices):
            snf_above = snf_gf2(boundary_matrices[k + 1])
            image_dim = _rank_gf2(snf_above)
        else:
            image_dim = 0

        betti.append(kernel_dim - image_dim)

    return betti
