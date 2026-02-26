"""Tests for tda.preprocessing."""

import numpy as np
import pytest

from tda.preprocessing import normalize_data, get_landmarks


class TestNormalizeData:
    def test_basic_normalization(self):
        data = np.array([[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]])
        result = normalize_data(data)
        assert result.shape == data.shape
        np.testing.assert_allclose(result.min(axis=0), [0, 0])
        np.testing.assert_allclose(result.max(axis=0), [1, 1])

    def test_already_normalized(self):
        data = np.array([[0, 0], [1, 1]], dtype=float)
        result = normalize_data(data)
        np.testing.assert_array_equal(result, data)

    def test_constant_column_no_nan(self):
        """Constant column should become all 0s, not NaN."""
        data = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])
        result = normalize_data(data)
        assert not np.any(np.isnan(result))
        np.testing.assert_array_equal(result[:, 0], [0, 0, 0])

    def test_single_point(self, single_point):
        result = normalize_data(single_point)
        assert not np.any(np.isnan(result))

    def test_rejects_1d(self):
        with pytest.raises(ValueError, match="2-D"):
            normalize_data(np.array([1.0, 2.0, 3.0]))

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="at least one point"):
            normalize_data(np.empty((0, 2)))


class TestGetLandmarks:
    def test_returns_correct_count(self, triangle_2d, rng):
        lm = get_landmarks(triangle_2d, 2, rng=rng)
        assert len(lm) == 2

    def test_all_unique(self, circle_points, rng):
        lm = get_landmarks(circle_points, 10, rng=rng)
        assert len(set(lm)) == 10

    def test_indices_in_bounds(self, circle_points, rng):
        lm = get_landmarks(circle_points, 10, rng=rng)
        assert all(0 <= idx < 50 for idx in lm)

    def test_all_points_when_n_equals_total(self, triangle_2d, rng):
        lm = get_landmarks(triangle_2d, 3, rng=rng)
        assert set(lm) == {0, 1, 2}

    def test_single_landmark(self, triangle_2d, rng):
        lm = get_landmarks(triangle_2d, 1, rng=rng)
        assert len(lm) == 1

    def test_reproducibility(self, circle_points):
        lm1 = get_landmarks(circle_points, 5, rng=np.random.default_rng(99))
        lm2 = get_landmarks(circle_points, 5, rng=np.random.default_rng(99))
        np.testing.assert_array_equal(lm1, lm2)

    def test_rejects_too_many(self, triangle_2d):
        with pytest.raises(ValueError, match="exceeds"):
            get_landmarks(triangle_2d, 10)

    def test_rejects_zero(self, triangle_2d):
        with pytest.raises(ValueError, match=">= 1"):
            get_landmarks(triangle_2d, 0)
