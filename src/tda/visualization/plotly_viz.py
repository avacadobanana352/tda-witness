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
# Dark theme palette
# ---------------------------------------------------------------------------

_BG       = "#0f0f1a"
_GRID     = "#1e1e32"
_TEXT     = "#c8c8e0"

_COLORS = {
    "points":    "rgba(80, 140, 255, 0.7)",   # blue
    "landmarks": "rgba(255, 70, 120, 0.95)",  # pink-red
    "edges":     "rgba(160, 160, 210, 0.45)", # grey-blue
    "triangles": "rgba(80, 200, 255, 0.12)",  # faint cyan
    "betti0":    "#00d4ff",   # electric cyan
    "betti1":    "#ff2d78",   # hot pink
    "betti2":    "#7eff5a",   # neon green
    "betti3":    "#ffb700",   # amber
}

_BETTI_COLORS = [
    _COLORS["betti0"], _COLORS["betti1"],
    _COLORS["betti2"], _COLORS["betti3"],
]

_DARK_LAYOUT = dict(
    paper_bgcolor=_BG,
    plot_bgcolor=_BG,
    font=dict(color=_TEXT, family="monospace"),
    xaxis=dict(gridcolor=_GRID, linecolor=_GRID, zerolinecolor=_GRID),
    yaxis=dict(gridcolor=_GRID, linecolor=_GRID, zerolinecolor=_GRID),
)


def _dark_layout(**extra):
    """Merge dark base layout with caller overrides."""
    layout = dict(_DARK_LAYOUT)
    layout.update(extra)
    return layout


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_point_cloud(
    data: np.ndarray,
    landmarks: np.ndarray | None = None,
    title: str | None = None,
) -> "go.Figure":
    """Interactive scatter plot of a point cloud."""
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

    fig.update_layout(**_dark_layout(
        title=title or "Point Cloud",
        showlegend=True,
    ))
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
    """Visualize the simplicial complex overlaid on the point cloud."""
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
                name="Edges", hoverinfo="skip",
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
                name="Edges", hoverinfo="skip",
            ))

    # Layer 2: triangles (2-simplices)
    if show_triangles and len(complex_) > 2:
        triangles = complex_[2]
        if is_3d:
            fig.add_trace(go.Mesh3d(
                x=lm_coords[:, 0], y=lm_coords[:, 1], z=lm_coords[:, 2],
                i=triangles[:, 0], j=triangles[:, 1], k=triangles[:, 2],
                opacity=0.18,
                color=_COLORS["betti2"],
                name="Triangles", hoverinfo="skip",
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

    fig.update_layout(**_dark_layout(
        title=title or "Simplicial Complex",
        showlegend=True,
    ))
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
    """Animated filtration: a slider sweeps through threshold values."""
    go = _import_plotly()
    from tda.preprocessing import normalize_data, get_landmarks, pairwise_distances
    from tda.complex.witness import build_witness_graph
    from tda.complex.vietoris_rips import compute_vr_complex
    from tda.homology.boundary import compute_boundary_matrices
    from tda.homology.betti import compute_betti_numbers

    data = np.asarray(data, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)
    is_3d = data.shape[1] >= 3

    if normalize:
        data = normalize_data(data)
    n_points = data.shape[0]
    if n_landmarks is None:
        n_landmarks = n_points
    rng = np.random.default_rng(seed)
    landmarks = get_landmarks(data, n_landmarks, rng=rng)
    distances = pairwise_distances(data[landmarks], data)
    lm_coords = data[landmarks]

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

        frames.append(go.Frame(data=traces, name=str(idx)))
        slider_steps.append(dict(
            method="animate",
            args=[[str(idx)], dict(mode="immediate",
                                   frame=dict(duration=0, redraw=True),
                                   transition=dict(duration=0))],
            label=f"R={R:.3f}  ({betti_str})",
        ))

    fig = go.Figure(data=frames[0].data, frames=frames)
    fig.update_layout(**_dark_layout(
        title=title or "Filtration",
        sliders=[dict(
            active=0,
            steps=slider_steps,
            currentvalue=dict(prefix="", visible=True, font=dict(color=_TEXT)),
            bgcolor=_GRID,
            bordercolor=_GRID,
            pad=dict(t=50),
        )],
        showlegend=False,
    ))
    if not is_3d:
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def plot_betti_summary(
    thresholds: np.ndarray | list[float],
    betti_sequences: list[list[int]],
    title: str | None = None,
) -> "go.Figure":
    """Step plot of Betti numbers vs. threshold R."""
    go = _import_plotly()
    thresholds = np.asarray(thresholds)
    max_dim = max(len(b) for b in betti_sequences)

    fig = go.Figure()
    for d in range(max_dim):
        values = [b[d] if d < len(b) else 0 for b in betti_sequences]
        fig.add_trace(go.Scatter(
            x=thresholds, y=values,
            mode="lines+markers",
            line=dict(shape="hv", color=_BETTI_COLORS[d % len(_BETTI_COLORS)], width=2),
            marker=dict(size=5, color=_BETTI_COLORS[d % len(_BETTI_COLORS)]),
            name=f"β{d}",
        ))

    fig.update_layout(**_dark_layout(
        title=title or "Betti Numbers vs. Threshold",
        xaxis_title="Threshold (R)",
        yaxis_title="Betti number",
        showlegend=True,
    ))
    return fig


def plot_persistence_diagram(
    pairs: list[dict],
    title: str | None = None,
) -> "go.Figure":
    """Persistence diagram: birth vs. death scatter plot."""
    go = _import_plotly()
    import math

    finite_vals = [p["death"] for p in pairs if not math.isinf(p["death"])]
    all_births = [p["birth"] for p in pairs]
    max_val = max(finite_vals + all_births) if (finite_vals or all_births) else 1.0
    cap = max_val * 1.3

    fig = go.Figure()

    # Diagonal reference
    fig.add_trace(go.Scatter(
        x=[0, cap], y=[0, cap],
        mode="lines",
        line=dict(color="#333355", dash="dash", width=1),
        showlegend=False, hoverinfo="skip",
    ))

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
        color = _BETTI_COLORS[d % len(_BETTI_COLORS)]

        fig.add_trace(go.Scatter(
            x=births, y=deaths,
            mode="markers",
            marker=dict(
                size=9,
                color=color,
                symbol=["diamond" if inf else "circle" for inf in is_inf],
                line=dict(width=1, color=_BG),
            ),
            hovertext=hover, hoverinfo="text",
            name=f"H{d}",
        ))

    fig.update_layout(**_dark_layout(
        title=title or "Persistence Diagram",
        xaxis_title="Birth",
        yaxis_title="Death",
        xaxis=dict(gridcolor=_GRID, linecolor=_GRID, zerolinecolor=_GRID,
                   range=[-max_val * 0.05, cap]),
        yaxis=dict(gridcolor=_GRID, linecolor=_GRID, zerolinecolor=_GRID,
                   range=[-max_val * 0.05, cap]),
        showlegend=True,
    ))
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def plot_barcode(
    pairs: list[dict],
    title: str | None = None,
    min_lifetime: float = 0.0,
):
    """Persistence barcode — one horizontal line per feature.

    Parameters
    ----------
    pairs : list of dict
        Each dict has keys ``"dim"``, ``"birth"``, ``"death"``.
    title : str or None
    min_lifetime : float
        Hide bars shorter than this (filters noise).
    """
    import math
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    neon = {0: _COLORS["betti0"], 1: _COLORS["betti1"],
            2: _COLORS["betti2"], 3: _COLORS["betti3"]}

    # Cap infinity
    finite_vals = [p["death"] for p in pairs if not math.isinf(p["death"])]
    all_births = [p["birth"] for p in pairs]
    max_val = max(finite_vals + all_births) if (finite_vals or all_births) else 1.0
    cap = max_val * 1.15

    # Filter noise, sort by dim then lifetime descending
    filtered = sorted(
        (p for p in pairs
         if math.isinf(p["death"]) or (p["death"] - p["birth"]) >= min_lifetime),
        key=lambda p: (p["dim"], -(p["death"] - p["birth"])
                       if not math.isinf(p["death"]) else -float("inf")),
    )

    dims = sorted(set(p["dim"] for p in filtered))

    fig, ax = plt.subplots(figsize=(7, max(2.5, len(filtered) * 0.18 + 1)))
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_BG)

    y = 0
    dim_label_pos = {}
    for d in dims:
        dim_pairs = [p for p in filtered if p["dim"] == d]
        color = neon.get(d, "#aaaaaa")
        y_start = y
        births = [p["birth"] for p in dim_pairs]
        deaths = [min(p["death"], cap) for p in dim_pairs]
        ys = list(range(y, y + len(dim_pairs)))
        ax.hlines(ys, births, deaths,
                  colors=color, linewidth=1.5, alpha=0.9)
        # Arrow for infinite bars
        for i, p in enumerate(dim_pairs):
            if math.isinf(p["death"]):
                ax.annotate("", xy=(cap + 0.01 * cap, y + i),
                            xytext=(cap, y + i),
                            arrowprops=dict(arrowstyle="->", color=color, lw=1.2))
        dim_label_pos[d] = (y_start + y + len(dim_pairs) - 1) / 2
        y += len(dim_pairs) + 2  # gap between dimension groups

    ax.set_yticks(list(dim_label_pos.values()))
    ax.set_yticklabels([f"$H_{d}$" for d in dim_label_pos],
                       color=_TEXT, fontsize=11)
    ax.set_xlabel("Filtration value", color=_TEXT, fontsize=10)
    ax.set_title(title or "Persistence Barcode", color=_TEXT, fontsize=12, pad=10)
    ax.tick_params(colors=_TEXT, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(_GRID)
    ax.set_xlim(-cap * 0.02, cap * 1.08)

    # Legend patches
    patches = [mpatches.Patch(color=neon.get(d, "#aaa"), label=f"$H_{d}$")
               for d in dims]
    ax.legend(handles=patches, loc="lower right", fontsize=9,
              facecolor=_GRID, edgecolor="none", labelcolor=_TEXT)

    plt.tight_layout()
    plt.show()
    return fig


def save_html(fig: "go.Figure", path: str) -> None:
    """Save a Plotly figure as a standalone HTML file."""
    fig.write_html(path, include_plotlyjs="cdn")
