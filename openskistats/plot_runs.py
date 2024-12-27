from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import plotnine as pn
import polars as pl
from matplotlib.colors import TwoSlopeNorm

from openskistats.analyze import load_runs_pl
from openskistats.bearing import (
    cut_bearings_pl,
    get_cut_bearing_bins_df,
)
from openskistats.utils import pl_flip_bearing, pl_hemisphere


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
        return (
            load_runs_pl()
            .filter(pl.col("run_uses").list.contains("downhill"))
            .explode("run_coordinates_clean")
            .unnest("run_coordinates_clean")
            .filter(pl.col("segment_hash").is_not_null())
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
                    pl.col("distance_vertical_drop").alias("bearing_magnitude_net"),
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
    # y-tick labeling
    latitude_ticks = np.arange(0, 91, 10)
    ax.set_yticks(latitude_ticks)
    ax.tick_params(axis="y", which="major", length=5, width=1)
    ax.set_yticklabels(
        [f"{r}°" if r in {0, 90} else "" for r in latitude_ticks],
        rotation=0,
        fontsize=7,
    )
    ax.set_rlabel_position(225)

    # Draw custom radial arcs for y-ticks
    for radius in latitude_ticks:
        theta_start = np.deg2rad(220)
        theta_end = np.deg2rad(230)
        theta = np.linspace(theta_start, theta_end, 100)
        ax.plot(
            theta,
            np.full_like(theta, radius),
            linewidth=1 if radius < 90 else 2,
            color="black",
        )

    return fig
