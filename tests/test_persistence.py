"""Tests for tda.homology.persistence and tda.persistent_homology."""

import math
import numpy as np
import pytest

from tda.complex.filtration import build_filtration
from tda.homology.persistence import compute_persistence, pairs_to_barcodes
from tda.preprocessing import pairwise_distances
from tda import persistent_homology


def _run_persistence(data, simplex_dim=2, witness_param=1):
    """Helper: run persistence on raw data (all points as landmarks)."""
    distances = pairwise_distances(data, data)
    simplices, births, dims = build_filtration(distances, witness_param, simplex_dim)
    return compute_persistence(simplices, births, dims)


class TestComputePersistence:
    def test_two_points(self):
        """Two points: 2 H0 births, 1 dies when edge connects them."""
        data = np.array([[0, 0], [1, 0]], dtype=float)
        pairs = _run_persistence(data, simplex_dim=1)
        h0 = [p for p in pairs if p["dim"] == 0]

        assert len(h0) == 2
        essential = [p for p in h0 if math.isinf(p["death"])]
        finite = [p for p in h0 if not math.isinf(p["death"])]
        assert len(essential) == 1
        assert len(finite) == 1
        assert essential[0]["birth"] == 0.0
        assert finite[0]["birth"] == 0.0

    def test_triangle_h0(self):
        """Triangle: exactly 1 essential H0 component."""
        data = np.array([[0, 0], [1, 0], [0.5, 0.866]], dtype=float)
        pairs = _run_persistence(data, simplex_dim=2)
        h0_essential = [p for p in pairs
                        if p["dim"] == 0 and math.isinf(p["death"])]
        assert len(h0_essential) == 1

    def test_triangle_h0_deaths(self):
        """Triangle: 2 finite H0 bars (components merge into 1)."""
        data = np.array([[0, 0], [1, 0], [0.5, 0.866]], dtype=float)
        pairs = _run_persistence(data, simplex_dim=2)
        h0_finite = [p for p in pairs
                     if p["dim"] == 0 and not math.isinf(p["death"])]
        assert len(h0_finite) == 2

    def test_circle_has_significant_h1(self):
        """Circle should have at least one H1 bar with significant lifetime."""
        t = np.linspace(0, 2 * np.pi, 20, endpoint=False)
        data = np.column_stack([np.cos(t), np.sin(t)])
        pairs = _run_persistence(data, simplex_dim=2)

        h1 = [p for p in pairs if p["dim"] == 1]
        lifetimes = [p["death"] - p["birth"] for p in h1
                     if not math.isinf(p["death"])]

        assert len(lifetimes) > 0
        assert max(lifetimes) > 0.5  # the main loop should be prominent

    def test_births_before_deaths(self):
        """Every pair should have birth <= death."""
        data = np.array([[0, 0], [1, 0], [0.5, 0.866], [2, 1]], dtype=float)
        pairs = _run_persistence(data, simplex_dim=2)
        for p in pairs:
            assert p["birth"] <= p["death"]

    def test_all_births_nonnegative(self):
        data = np.array([[0, 0], [1, 0], [0.5, 0.866]], dtype=float)
        pairs = _run_persistence(data, simplex_dim=2)
        for p in pairs:
            assert p["birth"] >= 0.0

    def test_single_point(self):
        """One point: 1 essential H0 bar, nothing else."""
        data = np.array([[0.0, 0.0]])
        pairs = _run_persistence(data, simplex_dim=1)
        assert len(pairs) == 1
        assert pairs[0]["dim"] == 0
        assert pairs[0]["birth"] == 0.0
        assert math.isinf(pairs[0]["death"])


class TestPairsToBarcodes:
    def test_grouping(self):
        pairs = [
            {"dim": 0, "birth": 0.0, "death": 1.0},
            {"dim": 0, "birth": 0.0, "death": math.inf},
            {"dim": 1, "birth": 0.5, "death": 1.5},
        ]
        barcodes = pairs_to_barcodes(pairs)
        assert len(barcodes[0]) == 2
        assert len(barcodes[1]) == 1


class TestPersistentHomologyAPI:
    def test_returns_expected_keys(self):
        data = np.array([[0, 0], [1, 0], [0.5, 0.866]], dtype=float)
        result = persistent_homology(data, simplex_dim=2, normalize=False)
        assert set(result.keys()) == {
            "pairs", "barcodes", "filtration_simplices",
            "birth_times", "landmarks", "data",
        }

    def test_seed_reproducibility(self):
        t = np.linspace(0, 2 * np.pi, 20, endpoint=False)
        data = np.column_stack([np.cos(t), np.sin(t)])
        r1 = persistent_homology(data, seed=42)
        r2 = persistent_homology(data, seed=42)
        assert r1["pairs"] == r2["pairs"]

    def test_rejects_1d_data(self):
        with pytest.raises(ValueError):
            persistent_homology(np.array([1.0, 2.0, 3.0]))
