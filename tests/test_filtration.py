"""Tests for tda.complex.filtration."""

import numpy as np

from tda.complex.filtration import build_filtration
from tda.preprocessing import pairwise_distances


class TestBuildFiltration:
    def _distances(self, data):
        """All points are landmarks."""
        return pairwise_distances(data, data)

    def test_vertices_born_at_zero(self):
        data = np.array([[0, 0], [1, 0], [0.5, 0.866]], dtype=float)
        simplices, births, dims = build_filtration(self._distances(data), 1, 2)
        vertex_births = births[dims == 0]
        np.testing.assert_array_equal(vertex_births, 0.0)

    def test_filtration_order(self):
        """Simplices should be sorted by (birth, dim, lex)."""
        data = np.array([[0, 0], [1, 0], [2, 0], [0.5, 1]], dtype=float)
        simplices, births, dims = build_filtration(self._distances(data), 1, 2)
        # Check birth times are non-decreasing
        assert all(births[i] <= births[i + 1] for i in range(len(births) - 1))
        # When births are equal, lower dimension comes first
        for i in range(len(births) - 1):
            if births[i] == births[i + 1]:
                assert dims[i] <= dims[i + 1]

    def test_edge_births_positive(self):
        """Edge birth times should be >= 0."""
        data = np.array([[0, 0], [1, 0], [0.5, 0.866]], dtype=float)
        simplices, births, dims = build_filtration(self._distances(data), 1, 2)
        edge_births = births[dims == 1]
        assert np.all(edge_births >= 0)

    def test_higher_simplex_birth_geq_edge_births(self):
        """A triangle's birth must be >= all its edge births."""
        data = np.array([[0, 0], [1, 0], [0.5, 0.866]], dtype=float)
        simplices, births, dims = build_filtration(self._distances(data), 1, 2)

        # Build edge birth lookup
        edge_births_map = {}
        for i, (s, d) in enumerate(zip(simplices, dims)):
            if d == 1:
                edge_births_map[s] = births[i]

        # Check triangles
        from itertools import combinations
        for i, (s, d) in enumerate(zip(simplices, dims)):
            if d == 2:
                for edge in combinations(s, 2):
                    assert births[i] >= edge_births_map[edge]

    def test_correct_simplex_count(self):
        """3 vertices + 3 edges + 1 triangle = 7 simplices for a triangle."""
        data = np.array([[0, 0], [1, 0], [0.5, 0.866]], dtype=float)
        simplices, births, dims = build_filtration(self._distances(data), 1, 2)
        assert sum(dims == 0) == 3
        assert sum(dims == 1) == 3
        assert sum(dims == 2) == 1

    def test_single_point(self):
        data = np.array([[0.0, 0.0]])
        simplices, births, dims = build_filtration(self._distances(data), 1, 1)
        assert len(simplices) == 1
        assert simplices[0] == (0,)
        assert births[0] == 0.0
