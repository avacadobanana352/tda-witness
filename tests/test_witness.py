"""Tests for tda.complex.witness."""

import numpy as np
import pytest

from tda.complex.witness import build_witness_graph


class TestBuildWitnessGraph:
    def test_fully_connected_when_threshold_large(self):
        """All landmarks connect when threshold is huge."""
        distances = np.array([[0.1, 0.2], [0.3, 0.4], [0.2, 0.1]])
        graph = build_witness_graph(distances, threshold=10.0, witness_param=1)
        expected = np.ones((3, 3), dtype=int)
        np.fill_diagonal(expected, 0)
        np.testing.assert_array_equal(graph, expected)

    def test_no_connections_when_threshold_zero(self):
        """No edges when threshold is zero and distances are positive."""
        distances = np.array([[0.5, 0.6], [0.7, 0.8]])
        graph = build_witness_graph(distances, threshold=0.0, witness_param=1)
        np.testing.assert_array_equal(graph, np.zeros((2, 2), dtype=int))

    def test_symmetry(self):
        rng = np.random.default_rng(42)
        distances = rng.random((5, 10))
        graph = build_witness_graph(distances, threshold=0.5, witness_param=1)
        np.testing.assert_array_equal(graph, graph.T)

    def test_diagonal_zero(self):
        distances = np.zeros((3, 3))
        graph = build_witness_graph(distances, threshold=1.0, witness_param=0)
        np.testing.assert_array_equal(np.diag(graph), [0, 0, 0])

    def test_weak_witnesses(self):
        """With v=0, m[i]=0, so edge condition is max(D[a,i], D[b,i]) <= R."""
        distances = np.array([[0.1, 0.5], [0.2, 0.5]])
        graph = build_witness_graph(distances, threshold=0.3, witness_param=0)
        # Witness 0: max(0.1, 0.2) = 0.2 <= 0.3 -> connected
        assert graph[0, 1] == 1
        assert graph[1, 0] == 1

    def test_rejects_negative_threshold(self):
        with pytest.raises(ValueError, match="threshold"):
            build_witness_graph(np.zeros((2, 2)), threshold=-1.0, witness_param=0)

    def test_rejects_negative_witness_param(self):
        with pytest.raises(ValueError, match="witness_param"):
            build_witness_graph(np.zeros((2, 2)), threshold=1.0, witness_param=-1)
