"""Interactive Plotly visualizations for TDA results.

Every public function returns a ``plotly.graph_objects.Figure`` so users can
further customize before showing or saving.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Lazy import helper
# ---------------------------------------------------------------------------

def _import_plotly():
    try:
        import plotly.graph_objects as go
        return go
    except ImportError:
        raise ImportError(
            "Plotly is required for visualization.  Install it with:\n"
            "  pip install tda-witness[viz]"
        ) from None


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

_COLORS = {
    "points": "rgba(65, 105, 225, 0.7)",     # royal blue
    "landmarks": "rgba(220, 50, 47, 0.9)",   # red
    "edges": "rgba(130, 130, 130, 0.5)",      # grey
    "triangles": "rgba(135, 206, 250, 0.25)", # light blue
    "betti0": "#1f77b4",
    "betti1": "#d62728",
    "betti2": "#2ca02c",
    "betti3": "#9467bd",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_point_cloud(
    data: np.ndarray,
    landmarks: np.ndarray | None = None,
    title: str | None = None,
) -> "go.Figure":
    """Interactive scatter plot of a point cloud.

    Parameters
    ----------
    data : np.ndarray, shape (n, 2) or (n, 3)
    landmarks : np.ndarray or None
        Indices of landmark points to highlight.
    title : str or None

    Returns
    -------
    plotly.graph_objects.Figure
    """
    go = _import_plotly()
    is_3d = data.shape[1] >= 3
    fig = go.Figure()

    hover = [f"Point {i}<br>({', '.join(f'{v:.3f}' for v in row)})"
             for i, row in enumerate(data)]

    if is_3d:
        fig.add_trace(go.Scatter3d(
            x=data[:, 0], y=data[:, 1], z=data[:, 2],
            mode="markers",
            marker=dict(size=3, color=_COLORS["points"]),
            hovertext=hover, hoverinfo="text",
            name="Points",
        ))
    else:
        fig.add_trace(go.Scatter(
            x=data[:, 0], y=data[:, 1],
            mode="markers",
            marker=dict(size=6, color=_COLORS["points"]),
            hovertext=hover, hoverinfo="text",
            name="Points",
        ))

    if landmarks is not None:
        lm_data = data[landmarks]
        lm_hover = [f"Landmark {i} (pt {landmarks[i]})" for i in range(len(landmarks))]
        if is_3d:
            fig.add_trace(go.Scatter3d(
                x=lm_data[:, 0], y=lm_data[:, 1], z=lm_data[:, 2],
                mode="markers",
                marker=dict(size=5, color=_COLORS["landmarks"], symbol="diamond"),
                hovertext=lm_hover, hoverinfo="text",
                name="Landmarks",
            ))
        else:
            fig.add_trace(go.Scatter(
                x=lm_data[:, 0], y=lm_data[:, 1],
                mode="markers",
                marker=dict(size=10, color=_COLORS["landmarks"], symbol="diamond"),
                hovertext=lm_hover, hoverinfo="text",
                name="Landmarks",
            ))

    fig.update_layout(
        title=title or "Point Cloud",
        template="plotly_white",
        showlegend=True,
    )
    if not is_3d:
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def plot_complex(
    data: np.ndarray,
    landmarks: np.ndarray,
    graph: np.ndarray,
    complex_: list[np.ndarray],
    title: str | None = None,
    show_triangles: bool = True,
) -> "go.Figure":
    """Visualize the simplicial complex overlaid on the point cloud.

    Parameters
    ----------
    data : np.ndarray, shape (n, 2) or (n, 3)
    landmarks : np.ndarray
        Landmark indices.
    graph : np.ndarray
        Adjacency matrix.
    complex_ : list of np.ndarray
        k-skeleton from ``compute_vr_complex``.
    title : str or None
    show_triangles : bool
        Whether to draw filled 2-simplices.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    go = _import_plotly()
    lm_coords = data[landmarks]
    is_3d = data.shape[1] >= 3
    fig = go.Figure()

    # Layer 1: edges
    edge_rows, edge_cols = np.where(np.triu(graph) == 1)
    if len(edge_rows) > 0:
        if is_3d:
            ex, ey, ez = [], [], []
            for r, c in zip(edge_rows, edge_cols):
                ex.extend([lm_coords[r, 0], lm_coords[c, 0], None])
                ey.extend([lm_coords[r, 1], lm_coords[c, 1], None])
                ez.extend([lm_coords[r, 2], lm_coords[c, 2], None])
            fig.add_trace(go.Scatter3d(
                x=ex, y=ey, z=ez,
                mode="lines",
                line=dict(color=_COLORS["edges"], width=2),
                name="Edges",
                hoverinfo="skip",
            ))
        else:
            ex, ey = [], []
            for r, c in zip(edge_rows, edge_cols):
                ex.extend([lm_coords[r, 0], lm_coords[c, 0], None])
                ey.extend([lm_coords[r, 1], lm_coords[c, 1], None])
            fig.add_trace(go.Scatter(
                x=ex, y=ey,
                mode="lines",
                line=dict(color=_COLORS["edges"], width=1.5),
                name="Edges",
                hoverinfo="skip",
            ))

    # Layer 2: triangles (2-simplices)
    if show_triangles and len(complex_) > 2:
        triangles = complex_[2]
        if is_3d:
            fig.add_trace(go.Mesh3d(
                x=lm_coords[:, 0], y=lm_coords[:, 1], z=lm_coords[:, 2],
                i=triangles[:, 0], j=triangles[:, 1], k=triangles[:, 2],
                opacity=0.2,
                color="lightblue",
                name="Triangles",
                hoverinfo="skip",
            ))
        else:
            first = True
            for tri in triangles:
                verts = lm_coords[tri]
                fig.add_trace(go.Scatter(
                    x=[verts[0, 0], verts[1, 0], verts[2, 0], verts[0, 0]],
                    y=[verts[0, 1], verts[1, 1], verts[2, 1], verts[0, 1]],
                    fill="toself",
                    fillcolor=_COLORS["triangles"],
                    line=dict(width=0),
                    legendgroup="triangles",
                    name="Triangles" if first else "",
                    showlegend=first,
                    hoverinfo="skip",
                ))
                first = False

    # Layer 3: all points
    if is_3d:
        fig.add_trace(go.Scatter3d(
            x=data[:, 0], y=data[:, 1], z=data[:, 2],
            mode="markers",
            marker=dict(size=3, color=_COLORS["points"]),
            name="Points",
        ))
    else:
        fig.add_trace(go.Scatter(
            x=data[:, 0], y=data[:, 1],
            mode="markers",
            marker=dict(size=5, color=_COLORS["points"]),
            name="Points",
        ))

    # Layer 4: landmarks
    lm_hover = [f"Landmark {i}" for i in range(len(landmarks))]
    if is_3d:
        fig.add_trace(go.Scatter3d(
            x=lm_coords[:, 0], y=lm_coords[:, 1], z=lm_coords[:, 2],
            mode="markers",
            marker=dict(size=5, color=_COLORS["landmarks"], symbol="diamond"),
            hovertext=lm_hover, hoverinfo="text",
            name="Landmarks",
        ))
    else:
        fig.add_trace(go.Scatter(
            x=lm_coords[:, 0], y=lm_coords[:, 1],
            mode="markers",
            marker=dict(size=9, color=_COLORS["landmarks"], symbol="diamond"),
            hovertext=lm_hover, hoverinfo="text",
            name="Landmarks",
        ))

    fig.update_layout(
        title=title or "Simplicial Complex",
        template="plotly_white",
        showlegend=True,
    )
    if not is_3d:
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def plot_filtration(
    data: np.ndarray,
    thresholds: np.ndarray | list[float],
    *,
    n_landmarks: int | None = None,
    simplex_dim: int = 2,
    witness_param: int = 1,
    normalize: bool = True,
    seed: int | None = None,
    title: str | None = None,
) -> "go.Figure":
    """Animated filtration: a slider sweeps through threshold values.

    Parameters
    ----------
    data : np.ndarray, shape (n, 2) or (n, 3)
    thresholds : array-like of float
        R values to sweep.
    n_landmarks, simplex_dim, witness_param, normalize, seed
        Forwarded to ``tda.analyze``.
    title : str or None

    Returns
    -------
    plotly.graph_objects.Figure
    """
    go = _import_plotly()
    from tda.preprocessing import normalize_data, get_landmarks, pairwise_distances
    from tda.complex.witness import build_witness_graph
    from tda.complex.vietoris_rips import compute_vr_complex
    from tda.homology.boundary import compute_boundary_matrices
    from tda.homology.betti import compute_betti_numbers

    data = np.asarray(data, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)
    is_3d = data.shape[1] >= 3

    # Precompute threshold-independent work once
    if normalize:
        data = normalize_data(data)
    n_points = data.shape[0]
    if n_landmarks is None:
        n_landmarks = n_points
    rng = np.random.default_rng(seed)
    landmarks = get_landmarks(data, n_landmarks, rng=rng)
    distances = pairwise_distances(data[landmarks], data)
    lm_coords = data[landmarks]

    # Pre-compute for each threshold
    frames = []
    slider_steps = []

    for idx, R in enumerate(thresholds):
        graph = build_witness_graph(distances, float(R), witness_param)
        complex_, achieved_dim = compute_vr_complex(graph, simplex_dim)
        n_levels = len(complex_)
        boundary_mats = compute_boundary_matrices(complex_, n_levels)
        betti = compute_betti_numbers(boundary_mats)
        if achieved_dim == simplex_dim and len(betti) > 1:
            betti = betti[:-1]
        betti_str = ", ".join(f"B{i}={b}" for i, b in enumerate(betti))

        # Build edge traces
        edge_rows, edge_cols = np.where(np.triu(graph) == 1)
        traces = []

        if is_3d:
            ex, ey, ez = [], [], []
            for r, c in zip(edge_rows, edge_cols):
                ex.extend([lm_coords[r, 0], lm_coords[c, 0], None])
                ey.extend([lm_coords[r, 1], lm_coords[c, 1], None])
                ez.extend([lm_coords[r, 2], lm_coords[c, 2], None])
            traces.append(go.Scatter3d(
                x=ex, y=ey, z=ez,
                mode="lines", line=dict(color=_COLORS["edges"], width=2),
                hoverinfo="skip",
            ))
            traces.append(go.Scatter3d(
                x=data[:, 0], y=data[:, 1], z=data[:, 2],
                mode="markers", marker=dict(size=3, color=_COLORS["points"]),
            ))
            traces.append(go.Scatter3d(
                x=lm_coords[:, 0], y=lm_coords[:, 1], z=lm_coords[:, 2],
                mode="markers",
                marker=dict(size=5, color=_COLORS["landmarks"], symbol="diamond"),
            ))
        else:
            ex, ey = [], []
            for r, c in zip(edge_rows, edge_cols):
                ex.extend([lm_coords[r, 0], lm_coords[c, 0], None])
                ey.extend([lm_coords[r, 1], lm_coords[c, 1], None])
            traces.append(go.Scatter(
                x=ex, y=ey,
                mode="lines", line=dict(color=_COLORS["edges"], width=1.5),
                hoverinfo="skip",
            ))
            traces.append(go.Scatter(
                x=data[:, 0], y=data[:, 1],
                mode="markers", marker=dict(size=5, color=_COLORS["points"]),
            ))
            traces.append(go.Scatter(
                x=lm_coords[:, 0], y=lm_coords[:, 1],
                mode="markers",
                marker=dict(size=9, color=_COLORS["landmarks"], symbol="diamond"),
            ))

        frame = go.Frame(data=traces, name=str(idx))
        frames.append(frame)
        slider_steps.append(dict(
            method="animate",
            args=[[str(idx)], dict(mode="immediate",
                                   frame=dict(duration=0, redraw=True),
                                   transition=dict(duration=0))],
            label=f"R={R:.3f}  ({betti_str})",
        ))

    # Initial figure = first frame
    fig = go.Figure(data=frames[0].data, frames=frames)
    fig.update_layout(
        title=title or "Filtration",
        template="plotly_white",
        sliders=[dict(
            active=0,
            steps=slider_steps,
            currentvalue=dict(prefix="", visible=True),
            pad=dict(t=50),
        )],
        showlegend=False,
    )
    if not is_3d:
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def plot_betti_summary(
    thresholds: np.ndarray | list[float],
    betti_sequences: list[list[int]],
    title: str | None = None,
) -> "go.Figure":
    """Step plot of Betti numbers vs. threshold *R*.

    Parameters
    ----------
    thresholds : array-like of float
    betti_sequences : list of list of int
        ``betti_sequences[i]`` is the Betti vector at ``thresholds[i]``.
    title : str or None

    Returns
    -------
    plotly.graph_objects.Figure
    """
    go = _import_plotly()
    thresholds = np.asarray(thresholds)
    max_dim = max(len(b) for b in betti_sequences)

    color_cycle = [_COLORS["betti0"], _COLORS["betti1"],
                   _COLORS["betti2"], _COLORS["betti3"]]

    fig = go.Figure()
    for d in range(max_dim):
        values = [b[d] if d < len(b) else 0 for b in betti_sequences]
        fig.add_trace(go.Scatter(
            x=thresholds, y=values,
            mode="lines+markers",
            line=dict(shape="hv", color=color_cycle[d % len(color_cycle)], width=2),
            marker=dict(size=5),
            name=f"Betti-{d}",
        ))

    fig.update_layout(
        title=title or "Betti Numbers vs. Threshold",
        xaxis_title="Threshold (R)",
        yaxis_title="Betti Number",
        template="plotly_white",
        showlegend=True,
    )
    return fig


def plot_persistence_diagram(
    pairs: list[dict],
    title: str | None = None,
) -> "go.Figure":
    """Persistence diagram: birth vs. death scatter plot.

    Parameters
    ----------
    pairs : list of dict
        Each dict has keys ``"dim"``, ``"birth"``, ``"death"``
        (as returned by ``compute_persistence``).
    title : str or None

    Returns
    -------
    plotly.graph_objects.Figure
    """
    go = _import_plotly()
    import math

    color_cycle = [_COLORS["betti0"], _COLORS["betti1"],
                   _COLORS["betti2"], _COLORS["betti3"]]

    # Determine axis limits — cap infinity at a visible value
    finite_vals = [p["death"] for p in pairs if not math.isinf(p["death"])]
    all_births = [p["birth"] for p in pairs]
    max_val = max(finite_vals + all_births) if (finite_vals or all_births) else 1.0
    cap = max_val * 1.3  # extra room for infinity markers

    fig = go.Figure()

    # Diagonal reference line
    fig.add_trace(go.Scatter(
        x=[0, cap], y=[0, cap],
        mode="lines",
        line=dict(color="lightgrey", dash="dash", width=1),
        showlegend=False, hoverinfo="skip",
    ))

    # Group by dimension
    dims = sorted(set(p["dim"] for p in pairs))
    for d in dims:
        dim_pairs = [p for p in pairs if p["dim"] == d]
        births = [p["birth"] for p in dim_pairs]
        deaths = [min(p["death"], cap) for p in dim_pairs]
        is_inf = [math.isinf(p["death"]) for p in dim_pairs]
        lifetimes = [p["death"] - p["birth"] if not math.isinf(p["death"])
                     else "inf" for p in dim_pairs]

        hover = [
            f"H{d}: birth={b:.4f}, death={'inf' if inf else f'{de:.4f}'}, "
            f"lifetime={lt}"
            for b, de, inf, lt in zip(births, deaths, is_inf, lifetimes)
        ]

        symbols = ["diamond" if inf else "circle" for inf in is_inf]

        fig.add_trace(go.Scatter(
            x=births, y=deaths,
            mode="markers",
            marker=dict(
                size=8,
                color=color_cycle[d % len(color_cycle)],
                symbol=symbols,
                line=dict(width=1, color="white"),
            ),
            hovertext=hover, hoverinfo="text",
            name=f"H{d}",
        ))

    fig.update_layout(
        title=title or "Persistence Diagram",
        xaxis_title="Birth",
        yaxis_title="Death",
        template="plotly_white",
        showlegend=True,
        xaxis=dict(range=[-max_val * 0.05, cap]),
        yaxis=dict(range=[-max_val * 0.05, cap]),
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def plot_barcode(
    pairs: list[dict],
    title: str | None = None,
    min_lifetime: float = 0.0,
) -> "go.Figure":
    """Persistence barcode: horizontal bars showing feature lifetimes.

    Parameters
    ----------
    pairs : list of dict
        Each dict has keys ``"dim"``, ``"birth"``, ``"death"``.
    title : str or None
    min_lifetime : float
        Hide bars shorter than this (filters noise).

    Returns
    -------
    plotly.graph_objects.Figure
    """
    go = _import_plotly()
    import math

    color_cycle = [_COLORS["betti0"], _COLORS["betti1"],
                   _COLORS["betti2"], _COLORS["betti3"]]

    # Cap infinity bars
    finite_vals = [p["death"] for p in pairs if not math.isinf(p["death"])]
    all_births = [p["birth"] for p in pairs]
    max_val = max(finite_vals + all_births) if (finite_vals or all_births) else 1.0
    cap = max_val * 1.2

    # Sort by dimension, then by lifetime (longest first)
    sorted_pairs = sorted(
        pairs,
        key=lambda p: (p["dim"], -(p["death"] - p["birth"])
                       if not math.isinf(p["death"]) else -float("inf")),
    )

    # Filter short-lived bars
    sorted_pairs = [
        p for p in sorted_pairs
        if math.isinf(p["death"])
        or (p["death"] - p["birth"]) >= min_lifetime
    ]

    fig = go.Figure()
    dims = sorted(set(p["dim"] for p in sorted_pairs))

    # One trace per dimension (fast, clean legend)
    y_pos = 0
    dim_start: dict[int, int] = {}  # dim -> first y position

    for d in dims:
        dim_pairs = [p for p in sorted_pairs if p["dim"] == d]
        dim_start[d] = y_pos
        color = color_cycle[d % len(color_cycle)]

        # Build bar coordinates with None separators
        xs: list[float | None] = []
        ys: list[float | None] = []
        hovers: list[str | None] = []

        for p in dim_pairs:
            birth = p["birth"]
            death = min(p["death"], cap)
            is_inf = math.isinf(p["death"])
            lt = "inf" if is_inf else f"{p['death'] - birth:.4f}"
            hover = f"H{d}: [{birth:.4f}, {'inf' if is_inf else f'{death:.4f}'})  lifetime={lt}"

            xs.extend([birth, death, None])
            ys.extend([y_pos, y_pos, None])
            hovers.extend([hover, hover, None])
            y_pos += 1

        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="lines",
            line=dict(color=color, width=5),
            name=f"H{d}",
            hovertext=hovers, hoverinfo="text",
        ))

        # Arrow markers for infinite bars
        inf_xs = [min(p["death"], cap) for p in dim_pairs if math.isinf(p["death"])]
        inf_ys = [dim_start[d] + i for i, p in enumerate(dim_pairs) if math.isinf(p["death"])]
        if inf_xs:
            fig.add_trace(go.Scatter(
                x=inf_xs, y=inf_ys,
                mode="markers",
                marker=dict(size=8, symbol="triangle-right", color=color),
                showlegend=False, hoverinfo="skip",
            ))

        y_pos += 1  # gap between dimensions

    # Y-axis: one label per dimension group
    tick_vals = [dim_start[d] + (y_pos - dim_start[d]) // 2 - 1
                 for d in dims]
    tick_text = [f"H{d}" for d in dims]

    fig.update_layout(
        title=title or "Persistence Barcode",
        xaxis_title="Filtration Value",
        yaxis=dict(
            tickvals=tick_vals,
            ticktext=tick_text,
            title="",
            showgrid=False,
        ),
        template="plotly_white",
        showlegend=True,
        height=400,
        width=700,
    )
    return fig


def save_html(fig: "go.Figure", path: str) -> None:
    """Save a Plotly figure as a standalone HTML file.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
    path : str
        Output file path.
    """
    fig.write_html(path, include_plotlyjs="cdn")
