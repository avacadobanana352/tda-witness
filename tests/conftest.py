"""Shared test fixtures for TDA tests."""

import pytest
import numpy as np


@pytest.fixture
def triangle_2d():
    """3 points forming a triangle."""
    return np.array([[0, 0], [1, 0], [0.5, 0.866]], dtype=float)


@pytest.fixture
def square_2d():
    """4 points at unit-square corners."""
    return np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)


@pytest.fixture
def circle_points():
    """50 points on the unit circle."""
    t = np.linspace(0, 2 * np.pi, 50, endpoint=False)
    return np.column_stack([np.cos(t), np.sin(t)])


@pytest.fixture
def tetrahedron_3d():
    """4 points of a regular tetrahedron."""
    return np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float)


@pytest.fixture
def single_point():
    return np.array([[0.0, 0.0]])


@pytest.fixture
def collinear_points():
    """4 points on a line."""
    return np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=float)


@pytest.fixture
def zomorodian_data():
    """13-point sample dataset (estimated from Zomorodian, 2010)."""
    return np.array([
        [13, 56], [35, 86], [40, 46], [69, 114], [80, 30],
        [85, 119], [125, 136], [123, 108], [135, 70], [155, 59],
        [129, 34], [144, 35], [169, 8],
    ], dtype=float)


@pytest.fixture
def rng():
    return np.random.default_rng(42)
