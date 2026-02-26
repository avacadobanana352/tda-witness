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

    boundary_matrix = np.zeros((n_faces, n_simplices), dtype=int)

    # For each simplex, generate its k faces by dropping one vertex at a time.
    # This is O(n_simplices * k) instead of the original O(n_faces * n_simplices).
    for j in range(n_simplices):
        simplex = simplices[j]
        for drop_pos in range(k):
            face_key = tuple(np.delete(simplex, drop_pos))
            face_idx = face_lookup.get(face_key)
            if face_idx is not None:
                boundary_matrix[face_idx, j] = (-1) ** (drop_pos % 2)

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
