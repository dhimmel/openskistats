"""
Elevation distribution histograms for ski areas.

Generates horizontal stacked bar charts showing the distribution of
run length across elevation bands, colored by difficulty grade, plus
latitude-binned elevation distributions across all ski-run segments.
"""

import textwrap
from collections.abc import Sequence
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import plotnine as pn
import polars as pl
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from mizani.formatters import comma_format

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
        .explode("run_coordinates_clean", empty_as_null=False, keep_nulls=False)
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


def _get_latitude_elevation_segments() -> pl.DataFrame:
    """
    Load segments belonging to ski areas for the global latitude analysis.

    Runs can be mapped outside a ski area in OpenSkiMap. Restricting this analysis
    to runs with at least one ski-area association matches the existing global
    latitude analyses in :mod:`openskistats.plot_runs`.
    """
    from openskistats.analyze import load_run_segments_pl

    return (
        load_run_segments_pl(
            run_filters=[pl.col("ski_area_ids").list.len() > 0],
        )
        .filter(
            pl.col("latitude").is_not_null(),
            pl.col("latitude").is_between(-90, 90),
            pl.col("elevation").is_not_null(),
            pl.col("distance_vertical").is_not_null(),
            pl.col("distance_3d").is_not_null(),
            pl.col("distance_3d") > 0,
            pl.col("distance_vertical_drop").is_not_null(),
        )
        .select(
            "latitude",
            "elevation",
            "distance_vertical",
            "distance_3d",
            "distance_vertical_drop",
        )
        .collect()
    )


def _allocate_segments_to_elevation_bins(
    segments: pl.DataFrame,
    *,
    bin_width: float,
    metric: ElevationMetric,
    group_columns: Sequence[str],
) -> pl.DataFrame:
    """Split segment metric values across elevation bins, retaining group columns."""
    bw = bin_width
    return (
        segments.with_columns(
            _elev_lo=pl.min_horizontal(
                "elevation", pl.col("elevation") + pl.col("distance_vertical")
            ),
            _elev_hi=pl.max_horizontal(
                "elevation", pl.col("elevation") + pl.col("distance_vertical")
            ),
            _span=pl.col("distance_vertical").abs(),
        )
        .with_columns(
            _first_idx=(pl.col("_elev_lo") / bw).floor().cast(pl.Int64),
            _last_idx=(pl.col("_elev_hi") / bw).floor().cast(pl.Int64),
        )
        .with_columns(
            _bin_idx=pl.int_ranges("_first_idx", pl.col("_last_idx") + 1),
        )
        .explode("_bin_idx", empty_as_null=False, keep_nulls=False)
        .with_columns(_bin_lo=pl.col("_bin_idx") * bw)
        .with_columns(
            _overlap=pl.min_horizontal("_elev_hi", pl.col("_bin_lo") + bw)
            - pl.max_horizontal("_elev_lo", "_bin_lo")
        )
        .select(
            *group_columns,
            (pl.col("_bin_lo") + bw / 2).alias("elevation_bin_center"),
            pl.when(pl.col("_span") < 1e-6)
            .then(pl.col(metric))
            .otherwise(pl.col(metric) * pl.col("_overlap") / pl.col("_span"))
            .cast(pl.Float64)
            .alias(metric),
        )
        .filter(pl.col(metric) > 0)
    )


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

    Bin expansion and proportional allocation are performed with native Polars
    expressions.

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

    return (
        _allocate_segments_to_elevation_bins(
            segments,
            bin_width=bin_width,
            metric=metric,
            group_columns=["run_difficulty_condensed"],
        )
        .group_by("elevation_bin_center", "run_difficulty_condensed")
        .agg(pl.col(metric).sum())
        .sort("elevation_bin_center", "run_difficulty_condensed")
    )


def get_elevation_by_latitude_data(segments: pl.DataFrame) -> pl.DataFrame:
    """
    Prepare weighted segment elevations within latitude bands.

    A segment's elevation is its midpoint elevation. Elevations are rounded to
    the nearest meter and identical values within a latitude band are collapsed
    by summing ``metric``. This keeps the weighted violin density equivalent at
    the scale of the chart while avoiding over a million duplicate observations.

    The northern and southern hemispheres are combined into 10-degree absolute
    latitude bands, matching the project's other global latitude analyses.
    """
    schema = {
        "latitude_bin_lower": pl.Float64,
        "latitude_bin_center": pl.Float64,
        "latitude_bin_upper": pl.Float64,
        "segment_elevation": pl.Float64,
        "distance_vertical_drop": pl.Float64,
    }
    if segments.is_empty():
        return pl.DataFrame(schema=schema)

    latitude_bin_width = 10.0
    binned = segments.filter(
        pl.col("latitude").is_not_null(),
        pl.col("latitude").is_between(-90, 90),
        pl.col("elevation").is_not_null(),
        pl.col("distance_vertical").is_not_null(),
        pl.col("distance_vertical_drop").is_not_null(),
        pl.col("distance_vertical_drop") > 0,
    ).with_columns(
        latitude_bin_lower=(pl.col("latitude").abs() / latitude_bin_width).floor()
        * latitude_bin_width,
        segment_elevation=(pl.col("elevation") + pl.col("distance_vertical") / 2).round(
            0
        ),
    )
    if binned.is_empty():
        return pl.DataFrame(schema=schema)

    return (
        binned.with_columns(
            latitude_bin_center=pl.col("latitude_bin_lower") + latitude_bin_width / 2,
            latitude_bin_upper=pl.col("latitude_bin_lower") + latitude_bin_width,
        )
        .group_by(
            "latitude_bin_lower",
            "latitude_bin_center",
            "latitude_bin_upper",
            "segment_elevation",
        )
        .agg(pl.col("distance_vertical_drop").sum())
        .select(*schema)
        .sort("latitude_bin_center", "segment_elevation")
    )


def plot_elevation_by_latitude_violins() -> pn.ggplot:
    """
    Plot the manuscript's elevation violins by absolute-latitude band.

    Plotnine's violin density is drawn horizontally so elevation is on the x-axis
    and 10-degree latitude bands are on the y-axis. The distributions are weighted
    by skiable vertical.
    """
    data = get_elevation_by_latitude_data(_get_latitude_elevation_segments())
    if data.is_empty():
        raise ValueError("No segment elevation data found.")

    latitude_bins = (
        data.select(
            "latitude_bin_lower",
            "latitude_bin_center",
            "latitude_bin_upper",
        )
        .unique()
        .sort("latitude_bin_center")
    )
    latitude_breaks = latitude_bins["latitude_bin_center"].to_list()
    latitude_labels = [
        f"{lower:g}–{upper:g}°"
        for lower, upper in latitude_bins.select(
            "latitude_bin_lower", "latitude_bin_upper"
        ).iter_rows()
    ]
    plot_data = data.to_pandas()
    plot_data["distance_vertical_drop"] = plot_data["distance_vertical_drop"].astype(
        object
    )
    return (
        pn.ggplot(
            # Plotnine passes weights to statsmodels, which normalizes them in
            # place. Object dtype makes its float conversion allocate a writable
            # array under pandas copy-on-write.
            data=plot_data,
            mapping=pn.aes(
                x="latitude_bin_center",
                y="segment_elevation",
                group="latitude_bin_center",
                weight="distance_vertical_drop",
            ),
        )
        + pn.geom_violin(
            width=9,
            scale="width",
            trim=True,
            bw=150,
            fill="#D4A0A7",
            color="#292929",
            size=0.4,
            draw_quantiles=[0.25, 0.5, 0.75],
            quantile_color="#B9828A",
            quantile_size=0.6,
            quantile_linetype="dotted",
        )
        + pn.coord_flip()
        + pn.scale_x_continuous(
            name="Absolute Latitude",
            breaks=latitude_breaks,
            labels=latitude_labels,
            expand=(0.03, 0.03),
        )
        + pn.scale_y_continuous(
            name="Segment Elevation (m)",
            labels=comma_format(),
            expand=(0.02, 0.02),
        )
        + pn.theme_bw()
        + pn.theme(
            figure_size=(3, 4),
            panel_grid_minor=pn.element_blank(),
        )
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


def _plot_elevation_detail(
    ax: Axes,
    ski_area_id: str,
    segments: pl.DataFrame,
    histogram: pl.DataFrame,
    bin_width: float,
    convention: RunDifficultyConvention,
    metric: ElevationMetric,
) -> Any:
    """Plot the full-detail elevation histogram with stacked difficulty bars."""
    from openskistats.analyze import load_ski_areas_pl

    info: dict[str, Any] = load_ski_areas_pl(
        ski_area_filters=[pl.col("ski_area_id") == ski_area_id]
    ).row(0, named=True)

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
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1_000:.1f}"))
    else:
        ax.set_xlabel("Skiable Vertical (m)", fontsize=10)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))

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
    return elevation_centers


def plot_elevation_histogram(
    ski_area_id: str,
    bin_width: float = _DEFAULT_BIN_WIDTH,
    convention: RunDifficultyConvention = RunDifficultyConvention.north_america,
    figsize: tuple[float, float] = (4, 4),
    y_min: float | None = None,
    y_max: float | None = None,
    x_max: float | None = None,
    metric: ElevationMetric = "distance_vertical_drop",
    preview: bool = False,
) -> Figure:
    """
    Create an elevation distribution histogram for a single ski area.

    Horizontal stacked bars with elevation on the y-axis and run length
    on the x-axis, colored by difficulty.

    When *preview* is ``True``, produce a compact mini-histogram with no
    title, axes labels, grid, stats, or spines — suitable for thumbnail
    grids.
    """
    segments = _get_elevation_segments(ski_area_id)
    histogram = get_elevation_histogram_data(
        segments, bin_width=bin_width, metric=metric
    )

    fig, ax = plt.subplots(figsize=figsize)

    if preview:
        # single-color bars grouped across all difficulties
        totals = (
            histogram.group_by("elevation_bin_center")
            .agg(pl.col(metric).sum())
            .sort("elevation_bin_center")
        )
        elevation_centers = totals["elevation_bin_center"].to_numpy()
        ax.barh(
            y=elevation_centers,
            width=totals[metric].to_numpy(),
            height=bin_width,
            color="#D4A0A7",
            edgecolor="#292929",
            linewidth=0.4,
            zorder=2,
        )
    else:
        elevation_centers = _plot_elevation_detail(
            ax, ski_area_id, segments, histogram, bin_width, convention, metric
        )

    ax.set_xlim(left=0, right=x_max)
    # snap y-axis to bar edges so there is no gap above or below;
    # use caller-supplied bounds when provided for cross-area comparison
    ax.set_ylim(
        y_min if y_min is not None else elevation_centers[0] - bin_width / 2,
        y_max if y_max is not None else elevation_centers[-1] + bin_width / 2,
    )

    if preview:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ("top", "right", "bottom"):
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_linewidth(0.5)
        fig.tight_layout(pad=0.3)
    else:
        fig.tight_layout()
    return fig


def plot_elevation_histogram_preview(
    ski_area_id: str,
    bin_width: float = 100.0,
    figsize: tuple[float, float] = (1, 1),
    y_min: float | None = None,
    y_max: float | None = None,
    x_max: float | None = None,
    metric: ElevationMetric = "distance_vertical_drop",
) -> Figure:
    """
    Create a compact preview elevation histogram for a single ski area.

    Thin wrapper around :func:`plot_elevation_histogram` with ``preview=True``.
    """
    return plot_elevation_histogram(
        ski_area_id,
        bin_width=bin_width,
        figsize=figsize,
        y_min=y_min,
        y_max=y_max,
        x_max=x_max,
        metric=metric,
        preview=True,
    )
