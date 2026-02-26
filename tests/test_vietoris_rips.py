"""Tests for tda.complex.vietoris_rips."""

import numpy as np
import pytest
from math import comb

from tda.complex.vietoris_rips import compute_vr_complex


def _complete_graph(n: int) -> np.ndarray:
    g = np.ones((n, n), dtype=int)
    np.fill_diagonal(g, 0)
    return g


def _path_graph(n: int) -> np.ndarray:
    g = np.zeros((n, n), dtype=int)
    for i in range(n - 1):
        g[i, i + 1] = 1
        g[i + 1, i] = 1
    return g


class TestComputeVrComplex:
    def test_complete_4_k2(self):
        """Complete graph on 4 vertices, k=2: C(4,1)=4 verts, C(4,2)=6 edges, C(4,3)=4 tris."""
        g = _complete_graph(4)
        skeleton, dim = compute_vr_complex(g, max_dim=2)
        assert dim == 2
        assert len(skeleton) == 3
        assert skeleton[0].shape == (4, 1)  # 4 vertices
        assert skeleton[1].shape == (6, 2)  # 6 edges
        assert skeleton[2].shape == (4, 3)  # 4 triangles

    def test_complete_5_k3(self):
        """Complete graph on 5, k=3: should have C(5,4)=5 tetrahedra."""
        g = _complete_graph(5)
        skeleton, dim = compute_vr_complex(g, max_dim=3)
        assert dim == 3
        assert skeleton[3].shape == (comb(5, 4), 4)

    def test_path_graph_no_triangles(self):
        """Path 0-1-2: 3 verts, 2 edges, 0 triangles."""
        g = _path_graph(3)
        skeleton, dim = compute_vr_complex(g, max_dim=2)
        assert dim == 1  # No triangles formed
        assert skeleton[0].shape == (3, 1)
        assert skeleton[1].shape == (2, 2)

    def test_single_vertex(self):
        g = np.zeros((1, 1), dtype=int)
        skeleton, dim = compute_vr_complex(g, max_dim=2)
        assert len(skeleton) == 1
        assert skeleton[0].shape == (1, 1)

    def test_no_edges(self):
        """Discrete graph: only vertices."""
        g = np.zeros((4, 4), dtype=int)
        skeleton, dim = compute_vr_complex(g, max_dim=2)
        assert len(skeleton) == 1
        assert dim == 0

    def test_simplices_are_sorted(self):
        g = _complete_graph(5)
        skeleton, _ = compute_vr_complex(g, max_dim=2)
        for level in skeleton:
            for simplex in level:
                assert list(simplex) == sorted(simplex)

    def test_no_duplicate_simplices(self):
        g = _complete_graph(5)
        skeleton, _ = compute_vr_complex(g, max_dim=2)
        for level in skeleton:
            unique = np.unique(level, axis=0)
            assert unique.shape == level.shape

    def test_rejects_non_square(self):
        with pytest.raises(ValueError, match="square"):
            compute_vr_complex(np.zeros((3, 4), dtype=int), max_dim=2)

    def test_rejects_low_dim(self):
        with pytest.raises(ValueError, match="max_dim"):
            compute_vr_complex(np.zeros((3, 3), dtype=int), max_dim=0)
