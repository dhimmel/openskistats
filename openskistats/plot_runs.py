from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import plotnine as pn
import polars as pl
from matplotlib.colors import TwoSlopeNorm
from mizani.formatters import comma_format

from openskistats.analyze import load_run_segments_pl, load_runs_pl
from openskistats.bearing import (
    cut_bearings_pl,
    get_cut_bearing_bins_df,
)
from openskistats.models import (
    SkiRunConvention,
    SkiRunDifficulty,
    ski_run_colors_bright,
    ski_run_colors_subtle,
)
from openskistats.plot import NARROW_SPACE, _add_polar_y_ticks
from openskistats.utils import (
    pl_condense_run_difficulty,
    pl_flip_bearing,
    pl_hemisphere,
)


@dataclass
class RunLatitudeBearingHistogram:
    num_latitude_bins: int = 30  # 3-degree bins
    num_bearing_bins: int = 90  # 4-degree bins
    prior_total_combined_vert: int = 20_000

    @property
    def latitude_abs_breaks(self) -> npt.NDArray[np.float64]:
        return np.linspace(0, 90, self.num_latitude_bins + 1, dtype=np.float64)

    @property
    def bearing_breaks(self) -> npt.NDArray[np.float64]:
        bearing_bin_df = get_cut_bearing_bins_df(num_bins=self.num_bearing_bins)
        return np.array(
            [*bearing_bin_df["bin_lower"], bearing_bin_df["bin_upper"].last()]
        )

    def get_latitude_bins_df(self, include_hemisphere: bool = False) -> pl.DataFrame:
        latitude_bins = pl.DataFrame(
            {
                "latitude_abs_bin_lower": self.latitude_abs_breaks[:-1],
                "latitude_abs_bin_upper": self.latitude_abs_breaks[1:],
            }
        ).with_columns(
            latitude_abs_bin_center=pl.mean_horizontal(
                "latitude_abs_bin_lower", "latitude_abs_bin_upper"
            )
        )
        if include_hemisphere:
            latitude_bins = pl.concat(
                [
                    latitude_bins.with_columns(hemisphere=pl.lit("north")),
                    latitude_bins.with_columns(hemisphere=pl.lit("south")),
                ]
            )
        return latitude_bins

    def get_grid_bins_df(self) -> pl.DataFrame:
        bearing_bins = get_cut_bearing_bins_df(num_bins=self.num_bearing_bins).select(
            pl.col("bin_index").alias("bearing_bin_index"),
            pl.col("bin_lower").alias("bearing_bin_lower"),
            pl.col("bin_upper").alias("bearing_bin_upper"),
            pl.col("bin_center").alias("bearing_bin_center"),
        )
        return self.get_latitude_bins_df().join(bearing_bins, how="cross")

    def load_and_filter_runs_pl(self) -> pl.LazyFrame:
        # filter for on-piste runs within a ski area
        # FIXME: remove freeride filter
        run_filters = [
            pl.col("ski_area_ids").list.len() > 0,
            # pl.col("run_difficulty").ne_missing(pl.lit("freeride")),
        ]
        return (
            load_run_segments_pl(run_filters=run_filters)
            .with_columns(
                latitude_abs=pl.col("latitude").abs(),
                hemisphere=pl_hemisphere(),
                bearing_poleward=pl_flip_bearing(),
            )
            .with_columns(
                pl.col("latitude_abs")
                .cut(
                    breaks=self.latitude_abs_breaks,
                    left_closed=True,
                    include_breaks=True,
                )
                .struct.field("breakpoint")
                .alias("latitude_abs_bin_upper"),
                cut_bearings_pl(
                    num_bins=self.num_bearing_bins, bearing_col="bearing_poleward"
                ).alias("bearing_bin_index"),
            )
        )

    def get_combined_vertical_prop_in_latitude_range(
        self, latitude_lower_bound: float, latitude_upper_bound: float
    ) -> float:
        """
        Get the proportion of skiable vert (combined vertical drop) that is within a given latitude range
        (inclusive/closed on both ends).
        The latitude range is not absolute, i.e. use negative latitude for southern hemisphere.
        While this is not a class/static method, the output should be invariant across instances.
        """
        latitude_lower_bound, latitude_upper_bound = sorted(
            [latitude_lower_bound, latitude_upper_bound]
        )
        metric_df = (
            self.load_and_filter_runs_pl()
            .with_columns(
                latitude_in_window=pl.col.latitude.is_between(
                    latitude_lower_bound, latitude_upper_bound
                )
            )
            .group_by("latitude_in_window")
            .agg(*self._get_agg_metrics())
            .with_columns(
                combined_vertical_prop=pl.col("combined_vertical").truediv(
                    pl.sum("combined_vertical")
                )
            )
            .collect()
            .filter(pl.col("latitude_in_window"))
        )
        if metric_df.is_empty():
            return 0.0
        return float(metric_df.select("combined_vertical_prop").item())

    def _get_agg_metrics(self) -> list[pl.Expr]:
        return [
            pl.count("segment_hash").alias("segment_count"),
            pl.col("distance_vertical_drop").sum().alias("combined_vertical").round(5),
        ]

    def get_latitude_histogram(self) -> pl.DataFrame:
        from openskistats.analyze import _get_bearing_summary_stats_pl

        histogram = (
            self.load_and_filter_runs_pl()
            .group_by("hemisphere", "latitude_abs_bin_upper")
            .agg(
                *self._get_agg_metrics(),
                _bearing_stats=pl.struct(
                    "bearing",
                    pl.col("distance_vertical_drop").alias("bearing_magnitude_cum"),
                    "hemisphere",
                ).map_batches(_get_bearing_summary_stats_pl, returns_scalar=True),
            )
            .unnest("_bearing_stats")
        )
        return (
            self.get_latitude_bins_df(include_hemisphere=True)
            .lazy()
            .join(histogram, how="left", on=["hemisphere", "latitude_abs_bin_upper"])
            .sort("hemisphere", "latitude_abs_bin_lower")
            .collect()
            .with_columns(
                pl.col("segment_count").fill_null(0),
                pl.col("combined_vertical").fill_null(0).round(5),
            )
        )

    def get_latitude_bearing_histogram(self) -> pl.DataFrame:
        histogram = (
            self.load_and_filter_runs_pl()
            .group_by("latitude_abs_bin_upper", "bearing_bin_index")
            .agg(*self._get_agg_metrics())
        )

        return (
            self.get_grid_bins_df()
            .lazy()
            .join(
                histogram,
                how="left",
                on=["latitude_abs_bin_upper", "bearing_bin_index"],
            )
            .sort("latitude_abs_bin_upper", "bearing_bin_index")
            .collect()
            .with_columns(
                pl.col("segment_count").fill_null(0),
                pl.col("combined_vertical").fill_null(0).round(5),
            )
            .with_columns(
                total_combined_vertical=pl.sum("combined_vertical").over(
                    "latitude_abs_bin_upper"
                ),
            )
            .with_columns(
                combined_vertical_prop=pl.col("combined_vertical").truediv(
                    "total_combined_vertical"
                ),
            )
            .with_columns(
                combined_vertical_enrichment=pl.col("combined_vertical_prop").mul(
                    self.num_bearing_bins
                ),
            )
            .with_columns(
                bearing_bin_center_radians=pl.col("bearing_bin_center").radians()
            )
            .with_columns(
                combined_vertical_prop_regularized=pl.col("combined_vertical").add(
                    self.prior_total_combined_vert / self.num_bearing_bins
                )
                / pl.col("total_combined_vertical").add(self.prior_total_combined_vert),
            )
            .with_columns(
                combined_vertical_enrichment_regularized=pl.col(
                    "combined_vertical_prop_regularized"
                ).mul(self.num_bearing_bins)
            )
        )

    def plot_latitude_histogram(self) -> pn.ggplot:
        return (
            pn.ggplot(
                data=self.get_latitude_histogram(),
                mapping=pn.aes(
                    x="latitude_abs_bin_center",
                    y="combined_vertical",
                    fill="hemisphere",
                ),
            )
            + pn.geom_col()
            + pn.scale_x_continuous(
                name="Absolute Latitude",
                breaks=np.linspace(0, 90, 10),
                labels=lambda values: [f"{x:.0f}°" for x in values],
                expand=(0, 0),
            )
            + pn.scale_y_continuous(
                labels=lambda values: [f"{x / 1_000:.0f}" for x in values],
                name="Combined Vertical (km)",
            )
            + pn.coord_flip()
            + pn.theme_bw()
            + pn.theme(
                figure_size=(2.5, 5),
                legend_position="inside",
                legend_position_inside=(0.97, 0.97),
            )
        )


@dataclass
class BearingByLatitudeBinMeshGrid:
    latitude_grid: npt.NDArray[np.float64]
    bearing_grid: npt.NDArray[np.float64]
    color_grid: npt.NDArray[np.float64]


def get_bearing_by_latitude_bin_mesh_grids() -> BearingByLatitudeBinMeshGrid:
    rlbh = RunLatitudeBearingHistogram()
    grid_pl = (
        rlbh.get_latitude_bearing_histogram()
        .with_columns(
            _pivot_value=pl.when(pl.col("total_combined_vertical") >= 10_000).then(
                pl.col("combined_vertical_enrichment_regularized")
            ),
        )
        .pivot(
            on="bearing_bin_lower",
            index="latitude_abs_bin_lower",
            values="_pivot_value",
            sort_columns=False,  # order of discovery
        )
        .sort("latitude_abs_bin_lower")
        .drop("latitude_abs_bin_lower")
    )
    latitude_grid, bearing_grid = np.meshgrid(
        rlbh.latitude_abs_breaks,
        rlbh.bearing_breaks,
        indexing="ij",
    )
    return BearingByLatitudeBinMeshGrid(
        latitude_grid=latitude_grid,
        bearing_grid=bearing_grid,
        color_grid=grid_pl.to_numpy(),
    )


def plot_bearing_by_latitude_bin() -> plt.Figure:
    """
    https://github.com/dhimmel/openskistats/issues/11
    """
    grids = get_bearing_by_latitude_bin_mesh_grids()
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    quad_mesh = ax.pcolormesh(
        np.deg2rad(grids.bearing_grid),
        grids.latitude_grid,
        grids.color_grid.clip(min=0, max=2.5),
        shading="flat",
        cmap="coolwarm",
        norm=TwoSlopeNorm(vmin=0, vcenter=1, vmax=2.5),
    )
    colorbar = plt.colorbar(quad_mesh, ax=ax, location="left", aspect=35, pad=0.053)
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(labelsize=8)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction("clockwise")
    ax.grid(visible=False)
    # TODO: reuse code in plot
    ax.set_xticks(ax.get_xticks())
    xticklabels = ["Poleward", "", "E", "", "Equatorward", "", "W", ""]
    ax.set_xticklabels(labels=xticklabels)
    ax.tick_params(axis="x", which="major", pad=-2)
    _add_polar_y_ticks(ax=ax, arc_color="black")

    return fig


def plot_run_difficulty_histograms_by_slope(
    condense_difficulty: bool = False,
    convention: SkiRunConvention = SkiRunConvention.north_america,
) -> pn.ggplot:
    difficulty_col = (
        "run_difficulty_condensed" if condense_difficulty else "run_difficulty"
    )
    run_stats = (
        load_runs_pl()
        .with_columns(
            run_difficulty=pl.col("run_difficulty")
            .fill_null(SkiRunDifficulty.other)
            .cast(pl.Enum(SkiRunDifficulty)),
            run_difficulty_condensed=pl_condense_run_difficulty(),
            run_slope=pl.col("vertical_drop")
            .truediv("combined_distance")
            .arctan()
            .degrees(),
        )
        .with_columns(
            run_slope_bin_center=pl.col("run_slope")
            .cut(
                breaks=pl.int_range(0, 51, 1, eager=True),
                left_closed=True,
                include_breaks=True,
            )
            .struct.field("breakpoint")
            .sub(0.5)
        )
        .group_by("run_slope_bin_center", difficulty_col)
        .agg(
            pl.count("run_id").alias("runs_count"),
            pl.col("combined_vertical").sum().alias("combined_vertical"),
        )
        .sort("run_slope_bin_center", difficulty_col)
        .filter(pl.col("run_slope_bin_center").is_not_null())
        .collect()
    )
    difficulty_stats = (
        run_stats.group_by(difficulty_col)
        .agg(
            pl.sum("runs_count"),
            pl.sum("combined_vertical"),
            pl.max("combined_vertical").mul(0.97).alias("bin_max_combined_vertical"),
        )
        .with_columns(
            label=pl.struct("runs_count", "combined_vertical").map_elements(
                lambda x: f"{x['runs_count']:,} runs\n{x['combined_vertical'] / 1_000:,.0f}{NARROW_SPACE}km vert",
                return_dtype=pl.String,
            )
        )
    )
    colormap = SkiRunDifficulty.colormap(
        condense=condense_difficulty, subtle=True, convention=convention
    )
    return (
        pn.ggplot(
            data=run_stats,
            mapping=pn.aes(
                x="run_slope_bin_center",
                y="combined_vertical",
                fill=difficulty_col,
            ),
        )
        + pn.geom_bar(
            stat="identity",
            width=1,
            color="#292929",
            size=0.3,
        )
        + pn.geom_label(
            pn.aes(x=40.8, y="bin_max_combined_vertical", label="label"),
            fill="#fff9e8",
            boxcolor="#8c8980",
            data=difficulty_stats,
            ha="right",
            va="top",
            size=8,
            alpha=0.8,
        )
        + pn.facet_grid(f"{difficulty_col} ~ .", scales="free_y")
        + pn.scale_fill_manual(
            values=colormap,
            limits=list(colormap),
            guide=None,
        )
        + pn.scale_x_continuous(
            name="\nSlope",  # newline to prevent overplotting bug when axis_text_y=pn.element_blank()
            breaks=np.arange(0, 90, 10),
            labels=lambda values: [f"{x:.0f}°" for x in values],
            expand=(0, 0),
        )
        + pn.scale_y_continuous(
            name="Combined Vertical (km)",
            labels=lambda values: [f"{x / 1_000:.0f}" for x in values],
        )
        + pn.coord_cartesian(xlim=(0, 42))
        + pn.theme_bw()
        + pn.theme(
            figure_size=(3, len(colormap)),
            axis_text_y=pn.element_blank(),
            axis_ticks_y=pn.element_blank(),
            strip_background=pn.element_rect(fill="#fef2e2"),
        )
    )


class DifficultyByConventionRunMetrics:
    @classmethod
    def get_difficulty_by_convention_metrics(cls) -> pl.DataFrame:
        # QUESTION: should we filter to runs in ski areas?
        return (
            load_runs_pl()
            .with_columns(
                pl.col("run_difficulty")
                .fill_null(SkiRunDifficulty.other)
                .cast(pl.Enum(SkiRunDifficulty)),
                pl.col("run_convention").cast(pl.Enum(SkiRunConvention)),
            )
            .group_by("run_difficulty", "run_convention")
            .agg(
                # FIXME: use standard aggregation functions
                pl.count("run_id").alias("runs_count"),
                pl.col("combined_vertical").sum().alias("combined_vertical"),
            )
            .collect()
            # get convention colors for run_convention, run_difficulty pair
            .with_columns(
                run_difficulty_color_subtle=pl.struct(
                    "run_convention", "run_difficulty"
                ).map_elements(
                    lambda x: SkiRunDifficulty[x["run_difficulty"]].color(
                        subtle=True, convention=x["run_convention"]
                    ),
                    return_dtype=pl.String,
                ),
                run_difficulty_color_bright=pl.struct(
                    "run_convention", "run_difficulty"
                ).map_elements(
                    lambda x: SkiRunDifficulty[x["run_difficulty"]].color(
                        subtle=False, convention=x["run_convention"]
                    ),
                    return_dtype=pl.String,
                ),
            )
        )

    @classmethod
    def plot_difficulty_by_convention(cls) -> pn.ggplot:
        return (
            pn.ggplot(
                data=cls.get_difficulty_by_convention_metrics(),
                mapping=pn.aes(
                    y="run_difficulty",
                    x="run_convention",
                    fill="run_difficulty_color_subtle",
                    color="run_difficulty_color_bright",
                    size="runs_count",
                ),
            )
            + pn.geom_point(stat="identity")
            + pn.scale_x_discrete(labels=SkiRunConvention.display_names(), name="")
            + pn.scale_y_discrete(limits=list(reversed(SkiRunDifficulty)), name="")
            + pn.scale_fill_identity(guide=None)
            + pn.scale_color_identity(guide=None)
            + pn.scale_size_continuous(
                name="Runs",
                range=(0.5, 8),
                guide=pn.guide_legend(
                    override_aes={
                        "color": ski_run_colors_bright.gray,
                        "fill": ski_run_colors_subtle.gray,
                    },
                ),
                labels=comma_format(),
            )
            + pn.coord_equal()
            + pn.theme_bw()
            + pn.theme(
                axis_text_x=pn.element_text(rotation=90, hjust=0.5),
                figure_size=(3, 3.2),
                legend_title=pn.element_text(hjust=0.5),
                legend_key=pn.element_blank(),
            )
        )
