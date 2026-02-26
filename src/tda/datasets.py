"""Synthetic point-cloud generators for examples and testing."""

from __future__ import annotations

import numpy as np


def make_circle(
    n_points: int = 100,
    noise: float = 0.05,
    seed: int | None = None,
) -> np.ndarray:
    """Sample *n_points* from the unit circle in R^2 with Gaussian noise.

    Expected topology (suitable R): Betti = [1, 1].
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    pts = np.column_stack([np.cos(t), np.sin(t)])
    pts += rng.normal(scale=noise, size=pts.shape)
    return pts


def make_figure_eight(
    n_points: int = 100,
    noise: float = 0.05,
    seed: int | None = None,
) -> np.ndarray:
    """Sample from a figure-eight (two tangent circles) in R^2.

    Expected topology (suitable R): Betti = [1, 2].
    """
    rng = np.random.default_rng(seed)
    half = n_points // 2
    t1 = np.linspace(0, 2 * np.pi, half, endpoint=False)
    t2 = np.linspace(0, 2 * np.pi, n_points - half, endpoint=False)
    c1 = np.column_stack([np.cos(t1) - 1, np.sin(t1)])
    c2 = np.column_stack([np.cos(t2) + 1, np.sin(t2)])
    pts = np.vstack([c1, c2])
    pts += rng.normal(scale=noise, size=pts.shape)
    return pts


def make_sphere(
    n_points: int = 200,
    noise: float = 0.05,
    seed: int | None = None,
) -> np.ndarray:
    """Sample from the unit 2-sphere in R^3 (Fibonacci lattice + noise).

    Expected topology (suitable R): Betti = [1, 0, 1].
    """
    rng = np.random.default_rng(seed)
    golden_ratio = (1 + np.sqrt(5)) / 2
    indices = np.arange(n_points)
    theta = np.arccos(1 - 2 * (indices + 0.5) / n_points)
    phi = 2 * np.pi * indices / golden_ratio
    pts = np.column_stack([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ])
    pts += rng.normal(scale=noise, size=pts.shape)
    return pts


def make_torus(
    n_points: int = 300,
    noise: float = 0.05,
    seed: int | None = None,
    R: float = 2.0,
    r: float = 1.0,
) -> np.ndarray:
    """Sample uniformly from a torus in R^3.

    Uses rejection sampling so point density is uniform over the surface.
    The area element is proportional to (R + r*cos(theta)), so naive
    uniform (theta, phi) over-samples the inner equator.

    Parameters
    ----------
    R : float  Major radius.
    r : float  Minor radius.

    Expected topology (suitable threshold): Betti = [1, 2, 1].
    """
    rng = np.random.default_rng(seed)

    # Rejection sampling for area-uniform theta
    theta = np.empty(n_points)
    count = 0
    while count < n_points:
        batch = max(n_points - count, 256)
        t_cand = rng.uniform(0, 2 * np.pi, batch)
        accept_prob = (R + r * np.cos(t_cand)) / (R + r)
        mask = rng.uniform(size=batch) < accept_prob
        accepted = t_cand[mask]
        take = min(len(accepted), n_points - count)
        theta[count : count + take] = accepted[:take]
        count += take

    phi = rng.uniform(0, 2 * np.pi, n_points)
    pts = np.column_stack([
        (R + r * np.cos(theta)) * np.cos(phi),
        (R + r * np.cos(theta)) * np.sin(phi),
        r * np.sin(theta),
    ])
    pts += rng.normal(scale=noise, size=pts.shape)
    return pts


def make_swiss_roll(
    n_points: int = 300,
    noise: float = 0.05,
    seed: int | None = None,
) -> np.ndarray:
    """Sample from a Swiss roll in R^3."""
    rng = np.random.default_rng(seed)
    t = 1.5 * np.pi * (1 + 2 * rng.uniform(size=n_points))
    height = 10 * rng.uniform(size=n_points)
    pts = np.column_stack([t * np.cos(t), height, t * np.sin(t)])
    pts += rng.normal(scale=noise, size=pts.shape)
    return pts
