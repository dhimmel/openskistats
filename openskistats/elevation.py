"""
Elevation distribution histograms for ski areas.

Generates horizontal stacked bar charts showing the distribution of
run length across elevation bands, colored by difficulty grade.
"""

import textwrap
from typing import Any, Literal

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

ElevationMetric = Literal["distance_3d", "distance_vertical_drop"]
"""Metric for elevation histogram x-axis: 3-D run distance or skiable vertical."""

_DEFAULT_BIN_WIDTH: float = 25.0
"""Default elevation bin width in meters."""


def _get_elevation_segments(ski_area_id: str) -> pl.DataFrame:
    """
    Load run segments for a single ski area with elevation and difficulty data.

    Returns one row per segment with raw elevation bounds, vertical distance,
    3-D distance, and vert drop.  Downstream,
    :func:`get_elevation_histogram_data` splits each segment proportionally
    across all elevation bins it spans rather than assigning it to a single
    bin by midpoint.
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
        .filter(
            pl.col("elevation").is_not_null(),
            pl.col("distance_vertical").is_not_null(),
            pl.col("distance_3d").is_not_null(),
            pl.col("distance_3d") > 0,
        )
        .select(
            "run_difficulty_condensed",
            "elevation",
            "distance_vertical",
            "distance_3d",
            "distance_vertical_drop",
        )
        .collect()
    )


def _split_segment_across_bins(
    elevation: float,
    distance_vertical: float,
    value: float,
    bin_width: float = _DEFAULT_BIN_WIDTH,
) -> list[dict[str, float]]:
    """
    Split a single segment's metric value proportionally across elevation bins.

    The segment runs from ``elevation`` (upper end, since distance_vertical =
    elevation_lag - elevation) to ``elevation + distance_vertical``.  Each bin
    the segment spans receives a fraction of ``value`` equal to the overlap of
    the segment's vertical extent with that bin divided by the total vertical
    extent of the segment.

    Flat segments (|distance_vertical| < 1e-6) are assigned entirely to the
    bin containing ``elevation``.

    Returns a list of ``{"elevation_bin_center": float, "binned_value": float}``
    dicts, one per bin touched.
    """
    span = abs(distance_vertical)
    if span < 1e-6:
        bin_lo = np.floor(elevation / bin_width) * bin_width
        return [
            {
                "elevation_bin_center": float(bin_lo + bin_width / 2),
                "binned_value": float(value),
            }
        ]
    elev_lo = min(elevation, elevation + distance_vertical)
    elev_hi = max(elevation, elevation + distance_vertical)
    first_idx = int(np.floor(elev_lo / bin_width))
    last_idx = int(np.floor(elev_hi / bin_width))
    results: list[dict[str, float]] = []
    for i in range(first_idx, last_idx + 1):
        lo = i * bin_width
        hi = lo + bin_width
        overlap = min(elev_hi, hi) - max(elev_lo, lo)
        if overlap > 0:
            results.append(
                {
                    "elevation_bin_center": float(lo + bin_width / 2),
                    "binned_value": float(value * overlap / span),
                }
            )
    return results


def get_elevation_histogram_data(
    segments: pl.DataFrame,
    bin_width: float = _DEFAULT_BIN_WIDTH,
    metric: ElevationMetric = "distance_vertical_drop",
) -> pl.DataFrame:
    """
    Bin segments by elevation and aggregate the chosen metric per bin,
    broken down by condensed difficulty.

    Each segment is split proportionally across all elevation bins it spans,
    so long segments crossing bin boundaries are accurately distributed rather
    than assigned entirely to a single midpoint bin.

    The default metric is ``distance_vertical_drop`` (skiable vertical), which
    avoids over-representing low-angle runs relative to steep terrain —
    consistent with how combined vertical is used elsewhere in the project.
    Pass ``metric="distance_3d"`` to use 3-D run length instead.
    """
    if segments.is_empty():
        return pl.DataFrame(
            schema={
                "elevation_bin_center": pl.Float64,
                "run_difficulty_condensed": pl.String,
                metric: pl.Float64,
            }
        )

    _split_schema = pl.List(
        pl.Struct({"elevation_bin_center": pl.Float64, "binned_value": pl.Float64})
    )

    def _split_segment(row: dict[str, float]) -> list[dict[str, float]]:
        return _split_segment_across_bins(
            elevation=row["elevation"],
            distance_vertical=row["distance_vertical"],
            value=row["_v"],
            bin_width=bin_width,
        )

    return (
        segments.with_columns(pl.col(metric).alias("_v"))
        .with_columns(
            pl.struct("elevation", "distance_vertical", "_v")
            .map_elements(_split_segment, return_dtype=_split_schema)
            .alias("_bins")
        )
        .select("run_difficulty_condensed", "_bins")
        .explode("_bins")
        .unnest("_bins")
        .rename({"binned_value": metric})
        .filter(pl.col(metric) > 0)
        .group_by("elevation_bin_center", "run_difficulty_condensed")
        .agg(pl.col(metric).sum())
        .sort("elevation_bin_center", "run_difficulty_condensed")
    )


def _compute_median_elevation(segments: pl.DataFrame) -> float | None:
    """Weighted median elevation: 50 % of total 3-D run length above and below."""
    total = segments["distance_3d"].sum()
    if not total or total <= 0:
        return None
    sorted_segs = segments.with_columns(
        elevation_midpoint=pl.col("elevation") + pl.col("distance_vertical") / 2
    ).sort("elevation_midpoint")
    idx = sorted_segs["distance_3d"].cum_sum().search_sorted(total / 2)
    return float(sorted_segs["elevation_midpoint"][min(idx, len(sorted_segs) - 1)])


def get_shared_axis_bounds(
    ski_area_ids: list[str],
    bin_width: float = _DEFAULT_BIN_WIDTH,
    share_y: bool = True,
    share_x: bool = True,
    metric: ElevationMetric = "distance_vertical_drop",
) -> tuple[float | None, float | None, float | None]:
    """
    Compute shared axis bounds for comparing multiple ski areas.

    Returns ``(y_min, y_max, x_max)`` where:

    - ``y_min`` / ``y_max`` are elevation axis limits snapped to bin
      boundaries so that bars sit flush across all areas, or ``None`` if
      ``share_y=False``.
    - ``x_max`` is the metric axis limit rounded up to a clean boundary
      (nearest km for ``distance_3d``, nearest 100 m for
      ``distance_vertical_drop``), or ``None`` if ``share_x=False``.

    Pass all three values directly to :func:`plot_elevation_histogram` or
    :func:`plot_elevation_histogram_preview` (``None`` values are ignored).

    Parameters
    ----------
    share_y:
        Compute shared elevation (y) axis limits.
    share_x:
        Compute shared metric (x) axis limit.
    metric:
        Which metric to use for the x-axis; must match the value passed to
        the plot functions.
    """
    elev_mins: list[float] = []
    elev_maxes: list[float] = []
    bin_maxes: list[float] = []

    for ski_area_id in ski_area_ids:
        segments = _get_elevation_segments(ski_area_id)
        if segments.is_empty():
            continue
        if share_y:
            elev_mins.append(float(segments["elevation"].min()))
            elev_maxes.append(float(segments["elevation"].max()))
        if share_x:
            bin_max = (
                get_elevation_histogram_data(
                    segments, bin_width=bin_width, metric=metric
                )
                .group_by("elevation_bin_center")
                .agg(pl.col(metric).sum())[metric]
                .max()
            )
            bin_maxes.append(float(bin_max or 0.0))

    if share_y and not elev_mins:
        raise ValueError("No elevation data found for the supplied ski_area_ids.")
    if share_x and not bin_maxes:
        raise ValueError("No distance data found for the supplied ski_area_ids.")

    y_min = (
        float(np.floor(min(elev_mins) / bin_width) * bin_width - bin_width / 2)
        if share_y
        else None
    )
    y_max = (
        float(np.ceil(max(elev_maxes) / bin_width) * bin_width + bin_width / 2)
        if share_y
        else None
    )
    round_to = 1_000 if metric == "distance_3d" else 100
    x_max = float(np.ceil(max(bin_maxes) / round_to) * round_to) if share_x else None
    return y_min, y_max, x_max


def plot_elevation_histogram(
    ski_area_id: str,
    bin_width: float = _DEFAULT_BIN_WIDTH,
    convention: RunDifficultyConvention = RunDifficultyConvention.north_america,
    figsize: tuple[float, float] = (4, 4),
    y_min: float | None = None,
    y_max: float | None = None,
    x_max: float | None = None,
    metric: ElevationMetric = "distance_vertical_drop",
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
    histogram = get_elevation_histogram_data(
        segments, bin_width=bin_width, metric=metric
    )

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
            values=metric,
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
    if metric == "distance_3d":
        ax.set_xlabel("Skiable Distance (km)", fontsize=10)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x / 1_000:.1f}"))
    else:
        ax.set_xlabel("Skiable Vertical (m)", fontsize=10)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
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
    ax.set_xlim(left=0, right=x_max)
    # snap y-axis to bar edges so there is no gap above or below;
    # use caller-supplied bounds when provided for cross-area comparison
    ax.set_ylim(
        y_min if y_min is not None else elevation_centers[0] - bin_width / 2,
        y_max if y_max is not None else elevation_centers[-1] + bin_width / 2,
    )

    fig.tight_layout()
    return fig


def plot_elevation_histogram_preview(
    ski_area_id: str,
    bin_width: float = _DEFAULT_BIN_WIDTH,
    figsize: tuple[float, float] = (1, 1),
    y_min: float | None = None,
    y_max: float | None = None,
    x_max: float | None = None,
    metric: ElevationMetric = "distance_vertical_drop",
) -> Figure:
    """
    Create a compact preview elevation histogram for a single ski area.

    Mini histogram with no title, legend, or axis labels.
    Matches the preview rose pattern used in the webapp grid view.
    """
    segments = _get_elevation_segments(ski_area_id)
    histogram = (
        get_elevation_histogram_data(segments, bin_width=bin_width, metric=metric)
        .group_by("elevation_bin_center")
        .agg(pl.col(metric).sum())
        .sort("elevation_bin_center")
    )
    elevation_centers = histogram["elevation_bin_center"].to_numpy()
    totals = histogram[metric].to_numpy()

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(
        y=elevation_centers,
        width=totals,
        height=bin_width,
        color="#D4A0A7",
        edgecolor="none",
        zorder=2,
    )
    ax.set_xlim(left=0, right=x_max)
    ax.set_xticks([])
    ax.set_yticks([])
    # snap y-axis to bar edges so there is no gap above or below;
    # use caller-supplied bounds when provided for cross-area comparison
    ax.set_ylim(
        y_min if y_min is not None else elevation_centers[0] - bin_width / 2,
        y_max if y_max is not None else elevation_centers[-1] + bin_width / 2,
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
