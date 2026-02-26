"""Tests for tda.homology.boundary."""

import numpy as np

from tda.homology.boundary import compute_boundary_matrix, compute_boundary_matrices
from tda.complex.vietoris_rips import compute_vr_complex


class TestComputeBoundaryMatrix:
    def _triangle_skeleton(self):
        """Full triangle: vertices {0,1,2}, edges {01,02,12}, triangle {012}."""
        g = np.ones((3, 3), dtype=int)
        np.fill_diagonal(g, 0)
        skeleton, _ = compute_vr_complex(g, max_dim=2)
        return skeleton

    def test_vertex_boundary_is_zero(self):
        skeleton = self._triangle_skeleton()
        bd = compute_boundary_matrix(skeleton, 1)
        assert bd.shape[0] == 1  # trivial row
        assert bd.shape[1] == 3  # 3 vertices
        np.testing.assert_array_equal(bd, np.zeros_like(bd))

    def test_edge_boundary_shape(self):
        skeleton = self._triangle_skeleton()
        bd = compute_boundary_matrix(skeleton, 2)
        # 3 vertices (rows) x 3 edges (cols)
        assert bd.shape == (3, 3)

    def test_triangle_boundary_shape(self):
        skeleton = self._triangle_skeleton()
        bd = compute_boundary_matrix(skeleton, 3)
        # 3 edges (rows) x 1 triangle (col)
        assert bd.shape == (3, 1)

    def test_boundary_entries_in_minus1_0_1(self):
        skeleton = self._triangle_skeleton()
        for dim in range(1, 4):
            bd = compute_boundary_matrix(skeleton, dim)
            assert set(np.unique(bd)).issubset({-1.0, 0.0, 1.0})

    def test_boundary_of_boundary_is_zero(self):
        """del_{k} @ del_{k+1} == 0 for all k."""
        g = np.ones((4, 4), dtype=int)
        np.fill_diagonal(g, 0)
        skeleton, _ = compute_vr_complex(g, max_dim=3)
        mats = compute_boundary_matrices(skeleton, len(skeleton))

        for k in range(len(mats) - 1):
            product = mats[k] @ mats[k + 1]
            np.testing.assert_array_equal(
                product, np.zeros_like(product),
                err_msg=f"del_{k} @ del_{k+1} != 0"
            )


class TestComputeBoundaryMatrices:
    def test_returns_correct_count(self):
        g = np.ones((3, 3), dtype=int)
        np.fill_diagonal(g, 0)
        skeleton, _ = compute_vr_complex(g, max_dim=2)
        mats = compute_boundary_matrices(skeleton, len(skeleton))
        assert len(mats) == len(skeleton)
