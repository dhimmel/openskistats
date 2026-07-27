"""
Dartmouth Skiway manuscript figure:
run segments with one compass direction highlighted and an inset ski rose.
Ported from the retired R implementation in r/02.plot.R.
"""

import math

import polars as pl
from matplotlib.figure import Figure

from openskistats.plot import plot_orientation
from openskistats.utils import get_images_data_directory

HIGHLIGHT_COLOR = "#d33c44"
MUTED_COLOR = "#cccccc"

SKIWAY_NAME = "Dartmouth Skiway"
"""Ski area for the example figure, chosen for its two distinctly oriented ledges."""


def load_skiway_segments() -> pl.DataFrame:
    """
    Segments between consecutive coordinates of Dartmouth Skiway runs.
    In skiway_run_coordinates.parquet, segment attributes like bearing are
    stored on the segment's ending coordinate,
    hence the backwards shift to combine them with the starting coordinate.
    """
    path = get_images_data_directory().joinpath("skiway_run_coordinates.parquet")
    return (
        pl.read_parquet(path)
        .sort("run_id", "index")
        .select(
            "run_id",
            "longitude",
            "latitude",
            longitude_end=pl.col("longitude").shift(-1).over("run_id"),
            latitude_end=pl.col("latitude").shift(-1).over("run_id"),
            bearing=pl.col("bearing").shift(-1).over("run_id"),
        )
        .drop_nulls()
    )


def load_skiway_bearings(num_bins: int = 32) -> pl.DataFrame:
    from openskistats.analyze import load_bearing_distribution_pl

    return load_bearing_distribution_pl(
        ski_area_filters=[pl.col("ski_area_name") == SKIWAY_NAME]
    ).filter(pl.col("num_bins") == num_bins)


def bearing_to_bin_index(bearing: pl.Expr, num_bins: int) -> pl.Expr:
    """Compass bin of a bearing in degrees, 1-indexed with bin 1 centered at north."""
    return ((bearing / (360 / num_bins)).round(0) % num_bins + 1).cast(pl.Int64)


def plot_skiway_segments_with_rose(
    highlight_bin_label: str = "NNE",
    num_bins: int = 32,
    bearings: pl.DataFrame | None = None,
) -> Figure:
    """
    Plot Dartmouth Skiway run segments as arrows colored by whether their
    bearing falls in the highlighted compass bin,
    with an inset ski rose highlighting that bin's petal.
    """
    if bearings is None:
        bearings = load_skiway_bearings(num_bins=num_bins)
    (highlight_bin_index,) = bearings.filter(
        pl.col("bin_label") == highlight_bin_label
    )["bin_index"]
    segments = load_skiway_segments().with_columns(
        highlight=bearing_to_bin_index(pl.col("bearing"), num_bins=num_bins)
        == highlight_bin_index
    )
    # a figure unmanaged by pyplot to avoid spawning interactive backend windows
    fig = Figure(figsize=(8, 6))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()
    margin = 0.03
    bounds = segments.select(
        x_min=pl.min_horizontal("longitude", "longitude_end").min(),
        x_max=pl.max_horizontal("longitude", "longitude_end").max(),
        y_min=pl.min_horizontal("latitude", "latitude_end").min(),
        y_max=pl.max_horizontal("latitude", "latitude_end").max(),
    ).row(0, named=True)
    x_pad = margin * (bounds["x_max"] - bounds["x_min"])
    y_pad = margin * (bounds["y_max"] - bounds["y_min"])
    ax.set_xlim(bounds["x_min"] - x_pad, bounds["x_max"] + x_pad)
    ax.set_ylim(bounds["y_min"] - y_pad, bounds["y_max"] + y_pad)
    # render longitude and latitude with locally correct proportions
    mean_latitude = float(segments["latitude"].mean())
    ax.set_aspect(1 / math.cos(math.radians(mean_latitude)))
    for row in segments.iter_rows(named=True):
        ax.annotate(
            "",
            xy=(row["longitude_end"], row["latitude_end"]),
            xytext=(row["longitude"], row["latitude"]),
            arrowprops={
                "arrowstyle": "->",
                "color": HIGHLIGHT_COLOR if row["highlight"] else MUTED_COLOR,
                "linewidth": 1.5,
                "mutation_scale": 10,
                "shrinkA": 0,
                "shrinkB": 0,
                "zorder": 3 if row["highlight"] else 2,
            },
        )
    rose_ax = fig.add_axes((0.4, 0.0, 0.4, 0.5), projection="polar")
    plot_orientation(
        bin_counts=bearings["bin_count"].to_numpy(),
        bin_centers=bearings["bin_center"].to_numpy(),
        ax=rose_ax,  # type: ignore[arg-type]
        color=MUTED_COLOR,
        margin_text={},
    )
    # transparent circle so segments remain visible beneath the rose
    rose_ax.patch.set_alpha(0.0)
    for patch, bin_index in zip(rose_ax.patches, bearings["bin_index"], strict=True):
        if bin_index == highlight_bin_index:
            patch.set_facecolor(HIGHLIGHT_COLOR)
    highlight_row = bearings.row(
        by_predicate=pl.col("bin_index") == highlight_bin_index, named=True
    )
    radius = math.sqrt(highlight_row["bin_count"] * num_bins / math.pi)
    rose_ax.text(
        x=math.radians(highlight_row["bin_center"]),
        y=radius * 0.78,
        s=highlight_bin_label,
        color="white",
        size=9,
        weight="bold",
        ha="center",
        va="center",
        # radial rotation so the label runs along the narrow petal
        rotation=90 - highlight_row["bin_center"],
        rotation_mode="anchor",
    )
    return fig
