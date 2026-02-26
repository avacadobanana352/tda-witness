"""Command-line interface for TDA."""

from __future__ import annotations

import click
import numpy as np


@click.group()
@click.version_option(package_name="tda-witness")
def main():
    """Topological Data Analysis: compute Betti numbers from point clouds."""


@main.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("-r", "--threshold", default=0.4, show_default=True,
              help="Connectivity threshold R.")
@click.option("-k", "--dim", default=2, show_default=True,
              help="Max simplex dimension.")
@click.option("-v", "--witness-param", default=1, show_default=True,
              help="Witness parameter.")
@click.option("-n", "--n-landmarks", default=None, type=int,
              help="Number of landmarks (default: all points).")
@click.option("--normalize/--no-normalize", default=True, show_default=True,
              help="Min-max normalize the data.")
@click.option("--seed", default=None, type=int, help="Random seed.")
@click.option("-p", "--plot", is_flag=True, help="Open interactive plot in browser.")
@click.option("--output-html", type=click.Path(),
              help="Save visualization to an HTML file.")
def compute(input_file, threshold, dim, witness_param, n_landmarks,
            normalize, seed, plot, output_html):
    """Compute Betti numbers from a CSV or .npy point-cloud file."""
    from tda import analyze

    if input_file.endswith(".npy"):
        data = np.load(input_file)
    else:
        data = np.loadtxt(input_file, delimiter=",")

    result = analyze(
        data,
        n_landmarks=n_landmarks,
        simplex_dim=dim,
        threshold=threshold,
        witness_param=witness_param,
        normalize=normalize,
        seed=seed,
    )

    click.echo("Betti numbers:")
    for i, b in enumerate(result["betti"]):
        click.echo(f"  B{i} = {b}")

    n_simplices = [c.shape[0] for c in result["complex"]]
    click.echo(f"Simplices per dimension: {n_simplices}")

    if plot or output_html:
        from tda.visualization import plot_complex, save_html

        fig = plot_complex(
            result["data"], result["landmarks"],
            result["graph"], result["complex"],
        )
        if output_html:
            save_html(fig, output_html)
            click.echo(f"Saved to {output_html}")
        if plot:
            fig.show()


@main.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("-k", "--dim", default=2, show_default=True,
              help="Max simplex dimension.")
@click.option("-v", "--witness-param", default=1, show_default=True,
              help="Witness parameter.")
@click.option("-n", "--n-landmarks", default=None, type=int,
              help="Number of landmarks (default: all points).")
@click.option("--normalize/--no-normalize", default=True, show_default=True,
              help="Min-max normalize the data.")
@click.option("--seed", default=None, type=int, help="Random seed.")
@click.option("-p", "--plot", is_flag=True,
              help="Open persistence diagram in browser.")
@click.option("--output-html", type=click.Path(),
              help="Save diagram to an HTML file.")
def persist(input_file, dim, witness_param, n_landmarks,
            normalize, seed, plot, output_html):
    """Compute persistent homology from a CSV or .npy point-cloud file."""
    import math
    from tda import persistent_homology

    if input_file.endswith(".npy"):
        data = np.load(input_file)
    else:
        data = np.loadtxt(input_file, delimiter=",")

    result = persistent_homology(
        data,
        n_landmarks=n_landmarks,
        simplex_dim=dim,
        witness_param=witness_param,
        normalize=normalize,
        seed=seed,
    )

    click.echo("Persistence pairs:")
    for p in result["pairs"]:
        death = "inf" if math.isinf(p["death"]) else f'{p["death"]:.4f}'
        click.echo(f'  H{p["dim"]}: [{p["birth"]:.4f}, {death})')

    barcodes = result["barcodes"]
    for d in sorted(barcodes):
        n_bars = len(barcodes[d])
        n_essential = sum(1 for b, de in barcodes[d] if math.isinf(de))
        click.echo(f"  H{d}: {n_bars} bars ({n_essential} essential)")

    if plot or output_html:
        from tda.visualization import plot_persistence_diagram, save_html

        fig = plot_persistence_diagram(result["pairs"])
        if output_html:
            save_html(fig, output_html)
            click.echo(f"Saved to {output_html}")
        if plot:
            fig.show()


@main.command()
@click.argument("shape", type=click.Choice(
    ["circle", "figure_eight", "sphere", "torus", "swiss_roll"]))
@click.option("-n", "--n-points", default=100, show_default=True)
@click.option("--noise", default=0.05, show_default=True)
@click.option("--seed", default=None, type=int)
@click.option("-o", "--output", type=click.Path(),
              help="Save data to CSV (prints to stdout if omitted).")
def generate(shape, n_points, noise, seed, output):
    """Generate a synthetic point-cloud dataset."""
    from tda import datasets

    generators = {
        "circle": datasets.make_circle,
        "figure_eight": datasets.make_figure_eight,
        "sphere": datasets.make_sphere,
        "torus": datasets.make_torus,
        "swiss_roll": datasets.make_swiss_roll,
    }
    data = generators[shape](n_points=n_points, noise=noise, seed=seed)

    if output:
        np.savetxt(output, data, delimiter=",", fmt="%.6f")
        click.echo(f"Saved {data.shape[0]} points ({data.shape[1]}D) to {output}")
    else:
        for row in data:
            click.echo(",".join(f"{v:.6f}" for v in row))
