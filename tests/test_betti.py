"""Tests for tda.homology.betti."""

import numpy as np

from tda.homology.betti import compute_betti_numbers
from tda.homology.smith_normal_form import snf_gf2, rank_gf2
from tda.homology.boundary import compute_boundary_matrices
from tda.complex.vietoris_rips import compute_vr_complex


def _full_graph(n):
    g = np.ones((n, n), dtype=int)
    np.fill_diagonal(g, 0)
    return g


class TestRankGf2:
    def test_identity(self):
        assert rank_gf2(np.eye(3, dtype=int)) == 3

    def test_zero(self):
        assert rank_gf2(np.zeros((3, 4), dtype=int)) == 0


class TestComputeBettiNumbers:
    def test_filled_triangle(self):
        """Triangle with 2-simplex filled: betti = [1, 0]."""
        g = _full_graph(3)
        skeleton, _ = compute_vr_complex(g, max_dim=2)
        mats = compute_boundary_matrices(skeleton, len(skeleton))
        betti = compute_betti_numbers(mats)
        assert betti[0] == 1  # one component
        assert betti[1] == 0  # no holes (triangle filled)

    def test_hollow_triangle(self):
        """Triangle with only edges (k=1): betti = [1, 1]."""
        g = _full_graph(3)
        skeleton, _ = compute_vr_complex(g, max_dim=1)
        mats = compute_boundary_matrices(skeleton, len(skeleton))
        betti = compute_betti_numbers(mats)
        assert betti[0] == 1  # one component
        assert betti[1] == 1  # one hole

    def test_two_disconnected_edges(self):
        """Graph: 0-1 and 2-3 disconnected: betti0 = 2."""
        g = np.zeros((4, 4), dtype=int)
        g[0, 1] = g[1, 0] = 1
        g[2, 3] = g[3, 2] = 1
        skeleton, _ = compute_vr_complex(g, max_dim=1)
        mats = compute_boundary_matrices(skeleton, len(skeleton))
        betti = compute_betti_numbers(mats)
        assert betti[0] == 2

    def test_single_vertex(self):
        """One vertex: betti = [1]."""
        g = np.zeros((1, 1), dtype=int)
        skeleton, _ = compute_vr_complex(g, max_dim=1)
        mats = compute_boundary_matrices(skeleton, len(skeleton))
        betti = compute_betti_numbers(mats)
        assert betti[0] == 1

    def test_filled_tetrahedron(self):
        """Complete graph on 4, k=3: solid tetrahedron, betti = [1, 0, 0]."""
        g = _full_graph(4)
        skeleton, _ = compute_vr_complex(g, max_dim=3)
        mats = compute_boundary_matrices(skeleton, len(skeleton))
        betti = compute_betti_numbers(mats)
        assert betti[0] == 1
        assert betti[1] == 0
        assert betti[2] == 0

    def test_square_cycle(self):
        """Square (4 edges, no diagonals, k=1): one cycle -> betti1 = 1."""
        g = np.zeros((4, 4), dtype=int)
        for i in range(4):
            j = (i + 1) % 4
            g[i, j] = g[j, i] = 1
        skeleton, _ = compute_vr_complex(g, max_dim=1)
        mats = compute_boundary_matrices(skeleton, len(skeleton))
        betti = compute_betti_numbers(mats)
        assert betti[0] == 1
        assert betti[1] == 1
