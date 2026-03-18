"""
Elevation distribution histograms for ski areas.

Generates horizontal stacked bar charts showing the distribution of
run length across elevation bands, colored by difficulty grade.
"""

import textwrap
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.figure import Figure

from openskistats.models import (
    RunDifficultyConvention,
    SkiRunDifficulty,
)
from openskistats.plot import NARROW_SPACE
from openskistats.utils import (
    pl_condense_run_difficulty,
)

_DEFAULT_BIN_WIDTH: float = 25.0
"""Default elevation bin width in meters."""


def _get_elevation_segments(ski_area_id: str) -> pl.DataFrame:
    """
    Load run segments for a single ski area with elevation and difficulty data.

    Each segment is assigned to a single elevation bin by its midpoint
    elevation. This is valid because the vast majority of segments have
    a vertical span much smaller than the default 25 m bin width:
    in test data (Whaleback, Storrs Hill), the median vertical span is
    ~3 m, P95 is ~18 m, and < 3 % of segments exceed 25 m vertically.
    """
    from openskistats.analyze import load_runs_pl

    return (
        load_runs_pl(
            ski_area_filters=[pl.col("ski_area_id") == ski_area_id],
        )
        .with_columns(pl_condense_run_difficulty())
        .select(
            "run_difficulty_condensed",
            "run_coordinates_clean",
        )
        .explode("run_coordinates_clean")
        .unnest("run_coordinates_clean")
        .filter(pl.col("segment_hash").is_not_null())
        .with_columns(
            # distance_vertical = elevation_lag - elevation, so
            # midpoint ≈ elevation + distance_vertical / 2
            elevation_midpoint=pl.col("elevation")
            + pl.col("distance_vertical").truediv(2),
        )
        .filter(
            pl.col("elevation_midpoint").is_not_null(),
            pl.col("distance_3d").is_not_null(),
            pl.col("distance_3d") > 0,
        )
        .select(
            "run_difficulty_condensed",
            "elevation_midpoint",
            "distance_3d",
        )
        .collect()
    )


def get_elevation_histogram_data(
    segments: pl.DataFrame,
    bin_width: float = _DEFAULT_BIN_WIDTH,
) -> pl.DataFrame:
    """
    Bin segments by elevation and aggregate total 3D run length per bin,
    broken down by condensed difficulty.
    """
    if segments.is_empty():
        return pl.DataFrame(
            schema={
                "elevation_bin_center": pl.Float64,
                "run_difficulty_condensed": pl.String,
                "distance_3d": pl.Float64,
            }
        )

    min_elev = segments["elevation_midpoint"].min()
    max_elev = segments["elevation_midpoint"].max()
    assert min_elev is not None
    assert max_elev is not None
    bin_lower = np.floor(min_elev / bin_width) * bin_width
    bin_upper = np.ceil(max_elev / bin_width) * bin_width + bin_width
    breaks = np.arange(bin_lower, bin_upper, bin_width)

    return (
        segments.with_columns(
            elevation_bin_center=pl.col("elevation_midpoint")
            .cut(breaks=breaks, left_closed=True, include_breaks=True)
            .struct.field("breakpoint")
            - bin_width / 2,
        )
        .filter(pl.col("elevation_bin_center").is_not_null())
        .group_by("elevation_bin_center", "run_difficulty_condensed")
        .agg(pl.col("distance_3d").sum())
        .sort("elevation_bin_center", "run_difficulty_condensed")
    )


def _compute_median_elevation(segments: pl.DataFrame) -> float | None:
    """Weighted median elevation: 50 % of total run length above and below."""
    total = segments["distance_3d"].sum()
    if not total or total <= 0:
        return None
    sorted_segs = segments.sort("elevation_midpoint")
    idx = sorted_segs["distance_3d"].cum_sum().search_sorted(total / 2)
    return float(sorted_segs["elevation_midpoint"][min(idx, len(sorted_segs) - 1)])


def _draw_median_elevation_line(ax: plt.Axes, y: float) -> None:
    """
    Draw a line at the median elevation.

    TODO: styling TBD — using a simple dotted line for now.
    """
    ax.axhline(
        y=y,
        color="#6A0DAD",
        linewidth=1.0,
        linestyle=(0, (20, 5)),
        zorder=4,
        clip_on=False,
    )


def plot_elevation_histogram(
    ski_area_id: str,
    bin_width: float = _DEFAULT_BIN_WIDTH,
    convention: RunDifficultyConvention = RunDifficultyConvention.north_america,
    figsize: tuple[float, float] = (4, 4),
) -> Figure:
    """
    Create an elevation distribution histogram for a single ski area.

    Horizontal stacked bars with elevation on the y-axis and run length
    on the x-axis, colored by difficulty.
    """
    from openskistats.analyze import load_ski_areas_pl

    info: dict[str, Any] = load_ski_areas_pl(
        ski_area_filters=[pl.col("ski_area_id") == ski_area_id]
    ).row(0, named=True)

    segments = _get_elevation_segments(ski_area_id)
    histogram = get_elevation_histogram_data(segments, bin_width=bin_width)

    colormap = SkiRunDifficulty.colormap(
        condense=True, subtle=True, convention=convention
    )
    difficulties = SkiRunDifficulty.condensed_values()
    elevation_centers = histogram["elevation_bin_center"].unique().sort().to_numpy()

    # pivot so each difficulty is a column aligned to elevation bins
    pivoted = (
        histogram.pivot(
            on="run_difficulty_condensed",
            index="elevation_bin_center",
            values="distance_3d",
        )
        .sort("elevation_bin_center")
        .fill_null(0)
    )

    fig, ax = plt.subplots(figsize=figsize)

    cumulative = np.zeros(len(elevation_centers), dtype=np.float64)
    for diff in difficulties:
        if diff.value not in pivoted.columns:
            continue
        values = pivoted[diff.value].to_numpy()
        if values.sum() == 0:
            continue
        ax.barh(
            y=elevation_centers,
            width=values,
            height=bin_width,
            left=cumulative,
            color=colormap[diff],
            edgecolor="#292929",
            linewidth=0.4,
            label=diff.value,
            zorder=2,
        )
        cumulative += values

    ax.set_ylabel("Elevation (m)", fontsize=10)
    ax.set_xlabel("Skiable Distance (km)", fontsize=10)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x / 1_000:.1f}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    ski_area_name = info.get("ski_area_name", "")
    if ski_area_name:
        ax.set_title(
            "\n".join(textwrap.wrap(ski_area_name, width=30)),
            fontsize=14,
            fontweight="bold",
            pad=10,
        )

    # bottom-right: vertical drop, elevation range, and median elevation
    median_elev = _compute_median_elevation(segments)
    parts = []
    if (vd := info.get("vertical_drop")) is not None:
        parts.append(f"{vd:,.0f}{NARROW_SPACE}m vert drop")
    if (lo := info.get("min_elevation")) is not None and (
        hi := info.get("max_elevation")
    ) is not None:
        parts.append(f"{lo:,.0f}–{hi:,.0f}{NARROW_SPACE}m")
    if median_elev is not None:
        parts.append(f"{median_elev:,.0f}{NARROW_SPACE}m median elev")
    if parts:
        ax.text(
            0.97,
            0.03,
            "\n".join(parts),
            transform=ax.transAxes,
            fontsize=7,
            color="#95A5A6",
            va="bottom",
            ha="right",
        )

    ax.grid(axis="x", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(left=0)
    # snap y-axis to bar edges so there is no gap above or below
    ax.set_ylim(
        elevation_centers[0] - bin_width / 2,
        elevation_centers[-1] + bin_width / 2,
    )

    if median_elev is not None:
        _draw_median_elevation_line(ax, median_elev)

    fig.tight_layout()
    return fig


def plot_elevation_histogram_preview(
    ski_area_id: str,
    bin_width: float = _DEFAULT_BIN_WIDTH,
    figsize: tuple[float, float] = (1, 1),
) -> Figure:
    """
    Create a compact preview elevation histogram for a single ski area.

    Filled area chart, no title, no legend, no x-axis labels, no median line.
    Matches the preview rose pattern used in the webapp grid view.
    """
    segments = _get_elevation_segments(ski_area_id)
    histogram = (
        get_elevation_histogram_data(segments, bin_width=bin_width)
        .group_by("elevation_bin_center")
        .agg(pl.col("distance_3d").sum())
        .sort("elevation_bin_center")
    )
    elevation_centers = histogram["elevation_bin_center"].to_numpy()
    totals = histogram["distance_3d"].to_numpy()

    # pad with zero-width points at bin edges so fill extends to ylim
    y_padded = np.concatenate(
        [
            [elevation_centers[0] - bin_width / 2],
            elevation_centers,
            [elevation_centers[-1] + bin_width / 2],
        ]
    )
    x_padded = np.concatenate([[0], totals, [0]])

    fig, ax = plt.subplots(figsize=figsize)
    ax.fill_betweenx(
        y=y_padded,
        x1=0,
        x2=x_padded,
        color="#D4A0A7",
        edgecolor="#6b6b6b",
        linewidth=0.4,
        zorder=2,
    )
    ax.set_xlim(left=0)
    ax.set_xticks([])
    ax.set_yticks([])
    # snap y-axis to bar edges so there is no gap above or below
    ax.set_ylim(
        elevation_centers[0] - bin_width / 2,
        elevation_centers[-1] + bin_width / 2,
    )
    # overlay base and peak elevation labels directly on the chart
    y_lo, y_hi = elevation_centers[0], elevation_centers[-1]
    label_kwargs = {
        "fontsize": 7,
        "fontweight": "bold",
        "color": "#4a4a4a",
        "ha": "left",
        "zorder": 5,
        "bbox": {"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 1},
    }
    ax.text(
        0.04,
        0.02,
        f"{y_lo:,.0f}{NARROW_SPACE}m",
        transform=ax.transAxes,
        va="bottom",
        **label_kwargs,
    )
    ax.text(
        0.04,
        0.98,
        f"{y_hi:,.0f}{NARROW_SPACE}m",
        transform=ax.transAxes,
        va="top",
        **label_kwargs,
    )
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout(pad=0)
    return fig
