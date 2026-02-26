"""Boundary matrix computation for simplicial complexes."""

from __future__ import annotations

import numpy as np


def compute_boundary_matrix(
    k_skeleton: list[np.ndarray],
    simplex_dimension: int,
) -> np.ndarray:
    """Compute the boundary matrix for a given simplex dimension.

    The boundary matrix has rows indexed by (k-1)-simplices and columns by
    k-simplices.  Entry ``(i, j)`` is ``(-1)^p`` when the *i*-th face is
    obtained from the *j*-th simplex by dropping the vertex at position *p*.

    Parameters
    ----------
    k_skeleton : list of np.ndarray
        ``k_skeleton[d]`` has rows of sorted vertex indices for *d*-simplices.
    simplex_dimension : int
        1-indexed dimension level (1 = vertices, 2 = edges, ...).

    Returns
    -------
    np.ndarray
        Boundary matrix with entries in {-1, 0, 1}.
    """
    simplices = k_skeleton[simplex_dimension - 1]

    # Boundary of 0-simplices is the zero map
    if simplex_dimension == 1:
        return np.zeros((1, simplices.shape[0]))

    faces = k_skeleton[simplex_dimension - 2]
    n_simplices = simplices.shape[0]
    n_faces = faces.shape[0]
    k = simplices.shape[1]  # vertices per simplex

    # Build face lookup: tuple(face) -> row index
    face_lookup = {tuple(row): idx for idx, row in enumerate(faces)}

    # Pre-compute signs: (-1)^drop_pos
    signs = np.array([(-1) ** p for p in range(k)], dtype=int)

    # Build COO-style sparse entries, then scatter into dense matrix
    row_idx = np.empty(n_simplices * k, dtype=int)
    col_idx = np.empty(n_simplices * k, dtype=int)
    vals = np.empty(n_simplices * k, dtype=int)
    count = 0

    for j in range(n_simplices):
        simplex = simplices[j]
        for drop_pos in range(k):
            # Build face key without np.delete (avoid array allocation)
            face_key = tuple(simplex[:drop_pos]) + tuple(simplex[drop_pos + 1:])
            face_idx = face_lookup.get(face_key)
            if face_idx is not None:
                row_idx[count] = face_idx
                col_idx[count] = j
                vals[count] = signs[drop_pos]
                count += 1

    boundary_matrix = np.zeros((n_faces, n_simplices), dtype=int)
    boundary_matrix[row_idx[:count], col_idx[:count]] = vals[:count]

    return boundary_matrix


def compute_boundary_matrices(
    k_skeleton: list[np.ndarray],
    simplex_dimension: int,
) -> list[np.ndarray]:
    """Compute boundary matrices for all dimensions up to *simplex_dimension*.

    Parameters
    ----------
    k_skeleton : list of np.ndarray
        The simplicial complex skeleton.
    simplex_dimension : int
        Maximum 1-indexed dimension level (inclusive).

    Returns
    -------
    list of np.ndarray
        ``boundary_matrices[k]`` is the boundary matrix for dimension ``k+1``.
    """
    return [
        compute_boundary_matrix(k_skeleton, k)
        for k in range(1, simplex_dimension + 1)
    ]
