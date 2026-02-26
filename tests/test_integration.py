"""End-to-end integration tests for tda.analyze()."""

import numpy as np
import pytest

from tda import analyze


class TestAnalyzeEndToEnd:
    def test_zomorodian_regression(self, zomorodian_data):
        """Regression: must match the original tda_testing.py output."""
        result = analyze(
            zomorodian_data,
            threshold=0.4,
            simplex_dim=2,
            witness_param=1,
        )
        assert result["betti"] == [1, 1]

    def test_triangle_k2(self, triangle_2d):
        """Filled triangle: [1, 0]."""
        result = analyze(triangle_2d, threshold=1.0, simplex_dim=2, normalize=False)
        assert result["betti"][0] == 1
        assert result["betti"][1] == 0

    def test_triangle_k1(self, triangle_2d):
        """simplex_dim=1: only beta_0 is reliable (beta_1 dropped)."""
        result = analyze(triangle_2d, threshold=1.0, simplex_dim=1, normalize=False)
        assert result["betti"] == [1]

    def test_circle(self, circle_points):
        """50 pts on circle with moderate R: should detect the loop."""
        result = analyze(
            circle_points,
            threshold=0.15,
            simplex_dim=2,
            witness_param=1,
            normalize=False,
        )
        assert result["betti"][0] == 1
        assert result["betti"][1] == 1

    def test_single_point(self, single_point):
        result = analyze(single_point, threshold=1.0, simplex_dim=1, normalize=False)
        assert result["betti"][0] == 1

    def test_returns_expected_keys(self, triangle_2d):
        result = analyze(triangle_2d, threshold=1.0, simplex_dim=2, normalize=False)
        assert set(result.keys()) == {
            "betti", "graph", "complex", "landmarks", "data", "boundary_matrices"
        }

    def test_seed_reproducibility(self, circle_points):
        r1 = analyze(circle_points, threshold=0.15, seed=42)
        r2 = analyze(circle_points, threshold=0.15, seed=42)
        assert r1["betti"] == r2["betti"]
        np.testing.assert_array_equal(r1["landmarks"], r2["landmarks"])

    def test_subset_landmarks(self, zomorodian_data):
        """Using fewer landmarks still produces valid results."""
        result = analyze(
            zomorodian_data,
            n_landmarks=8,
            threshold=0.4,
            simplex_dim=2,
            seed=0,
        )
        assert len(result["betti"]) >= 1
        assert result["landmarks"].shape[0] == 8

    def test_rejects_1d_data(self):
        with pytest.raises(ValueError):
            analyze(np.array([1.0, 2.0, 3.0]))
