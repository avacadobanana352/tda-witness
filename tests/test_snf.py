"""Tests for tda.homology.smith_normal_form."""

import numpy as np

from tda.homology.smith_normal_form import snf_gf2


class TestSnfGf2:
    def test_identity(self):
        result = snf_gf2(np.eye(3, dtype=int))
        np.testing.assert_array_equal(result, np.eye(3, dtype=int))

    def test_zero_matrix(self):
        result = snf_gf2(np.zeros((3, 4), dtype=int))
        np.testing.assert_array_equal(result, np.zeros((3, 4), dtype=int))

    def test_single_one(self):
        m = np.zeros((3, 3), dtype=int)
        m[1, 2] = 1
        result = snf_gf2(m)
        # Should have exactly one 1 on the diagonal
        assert np.sum(np.diag(result)) == 1

    def test_handles_negative_entries(self):
        """Boundary matrices have -1 entries; these map to 1 in GF(2)."""
        m = np.array([[-1, 1], [1, -1]], dtype=int)
        result = snf_gf2(m)
        # rank should be 1 (rows are linearly dependent mod 2)
        assert np.sum(np.diag(result) != 0) == 1

    def test_gf2_correctness(self):
        """Regression: plain addition (without %2) gives wrong rank."""
        # In GF(2): [1,1] + [1,1] = [0,0], so rank should be 1
        m = np.array([[1, 1], [1, 1]], dtype=int)
        result = snf_gf2(m)
        assert np.sum(np.diag(result) != 0) == 1

    def test_rectangular(self):
        m = np.array([[1, 0, 1], [0, 1, 1]], dtype=int)
        result = snf_gf2(m)
        # Rank 2 over GF(2)
        diag_len = min(result.shape)
        rank = np.sum(np.diag(result)[:diag_len] != 0)
        assert rank == 2

    def test_does_not_modify_input(self):
        m = np.array([[1, 1], [0, 1]], dtype=int)
        m_copy = m.copy()
        snf_gf2(m)
        np.testing.assert_array_equal(m, m_copy)

    def test_result_is_diagonal(self):
        """The off-diagonal of the result should be all zeros."""
        rng = np.random.default_rng(42)
        m = rng.integers(0, 2, size=(8, 6))
        result = snf_gf2(m)
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                if i != j:
                    assert result[i, j] == 0, f"Non-zero off-diagonal at ({i},{j})"
