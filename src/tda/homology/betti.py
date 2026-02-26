"""Betti number computation from boundary matrices."""

from __future__ import annotations

import numpy as np

from tda.homology.smith_normal_form import rank_gf2


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
    # Compute all ranks once up front — avoids the old code's redundant
    # SNF computation where each intermediate boundary matrix was reduced
    # twice (once as del_k, once as del_{k+1} for the previous dimension).
    ranks = [rank_gf2(bm) for bm in boundary_matrices]

    betti: list[int] = []
    for k in range(len(boundary_matrices)):
        kernel_dim = boundary_matrices[k].shape[1] - ranks[k]
        image_dim = ranks[k + 1] if k + 1 < len(ranks) else 0
        betti.append(kernel_dim - image_dim)

    return betti
