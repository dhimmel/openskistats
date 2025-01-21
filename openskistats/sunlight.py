import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from functools import cached_property, lru_cache
from operator import itemgetter
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import polars as pl
import pvlib
import requests
from matplotlib.collections import QuadMesh
from matplotlib.colorbar import Colorbar
from osmnx.plot import _get_fig_ax
from rich.progress import Progress

from openskistats.utils import (
    get_data_directory,
    get_hemisphere,
    running_in_ci,
    running_in_test,
)

SOLAR_CACHE_VERSION = 1
"""
Increment this version number to invalidate the solar irradiance cache.
"""

SOLSTICE_NORTH = date.fromisoformat("2024-12-21")
SOLSTICE_SOUTH = date.fromisoformat("2024-06-20")


def compute_solar_irradiance(
    latitude: float,
    longitude: float,
    elevation: float,
    slope: float,
    bearing: float,
    time_freq: str = "15min",
    extent: Literal["solstice", "season"] = "season",
) -> pl.DataFrame:
    """
    Compute daily clear-sky irradiance (W/m^2) for the winter solstice in the Northern Hemisphere.
    """
    assert slope is not None
    ski_season = SkiSeasonDatetimes(
        hemisphere=get_hemisphere(latitude),
        extent=extent,
        freq=time_freq,
    )
    # rounding as a hack to improve efficiency via caching
    latitude = round(latitude, 1)
    longitude = round(longitude, 1)
    elevation = round(elevation, -1)
    clearsky_df = get_clearsky(
        latitude=latitude,
        longitude=longitude,
        elevation=elevation,
        ski_season=ski_season,
    )
    # Calculate plane-of-array irradiance for each hour
    # Returns an OrderedDict when inputs are not pd.Series.
    # https://github.com/pvlib/pvlib-python/blob/9fb2eb3aa7984c6283252e53e3052fe2c2cee90e/pvlib/irradiance.py#L506-L507
    irrad_dict = pvlib.irradiance.get_total_irradiance(
        surface_tilt=slope,
        surface_azimuth=bearing,
        solar_zenith=clearsky_df["sun_apparent_zenith"],
        solar_azimuth=clearsky_df["sun_azimuth"],
        dni=clearsky_df["dni"],  # diffuse normal irradiance
        ghi=clearsky_df["ghi"],  # global horizontal irradiance
        dhi=clearsky_df["dhi"],  # direct horizontal irradiance
        surface_type="snow",
    )
    irrad_df = clearsky_df.hstack(pl.from_dict(irrad_dict)).with_columns(
        is_solstice=pl.col("datetime").dt.date() == ski_season.solstice,
    )
    return irrad_df


def collapse_solar_irradiance(
    irrad_df: pl.DataFrame,
) -> dict[str, float]:
    """Aggregate solar irradiance to solar irradiation."""

    def get_irradiation(df: pl.DataFrame) -> float:
        mean = df["poa_global"].mean()
        if mean is None:
            return -10_000.0  # sentinel value for missing data
        return 24 * float(mean) / 1_000

    return {
        "solar_irradiation_season": get_irradiation(irrad_df),
        "solar_irradiation_solstice": get_irradiation(
            irrad_df.filter(pl.col("is_solstice"))
        ),
    }


@dataclass(frozen=True)
class SkiSeasonDatetimes:
    hemisphere: Literal["north", "south"]
    extent: Literal["solstice", "season"]
    freq: str = "15min"

    @cached_property
    def ski_season_dates(self) -> tuple[date, date]:
        """
        Return open and closing dates for a typical ski season.
        """
        solstice = self.solstice
        match self.extent:
            case "solstice":
                return solstice, solstice
            case "season":
                return solstice - timedelta(days=20), solstice + timedelta(days=100)
            case _:
                raise ValueError("Invalid extent.")

    @cached_property
    def solstice(self) -> date:
        """Return the winter solstice date."""
        return {
            "north": SOLSTICE_NORTH,
            "south": SOLSTICE_SOUTH,
        }[self.hemisphere]

    @cached_property
    def freq_time_delta(self) -> pd.Timedelta:
        return pd.to_timedelta(self.freq)

    @cached_property
    def times_per_hour(self) -> float:
        return float(self.freq_time_delta.total_seconds()) / 3_600

    @cached_property
    def season_duration_days(self) -> int:
        date_open, date_close = self.ski_season_dates
        return (date_close - date_open).days + 1

    @cached_property
    def interpolated_range(self) -> pd.DatetimeIndex:
        """Interpolate ski season datetimes."""
        date_open, date_close = self.ski_season_dates
        return pd.date_range(
            start=date_open,
            # add one day to include the closing date
            end=date_close + timedelta(days=1),
            inclusive="left",
            freq=self.freq_time_delta,
            tz="UTC",
            unit="s",
        )


@lru_cache(maxsize=5_000)
def get_clearsky(
    latitude: float, longitude: float, elevation: float, ski_season: SkiSeasonDatetimes
) -> pl.DataFrame:
    location = pvlib.location.Location(
        latitude=latitude,
        longitude=longitude,
        altitude=elevation,
    )
    times = ski_season.interpolated_range
    solar_positions = location.get_solarposition(times, method="nrel_numpy").add_prefix(
        "sun_"
    )
    clearsky = location.get_clearsky(times, model="ineichen")
    return (
        pd.concat([solar_positions, clearsky], axis=1)
        .reset_index(names="datetime")
        .pipe(pl.from_pandas)
    )


def write_dartmouth_skiway_solar_irradiance() -> pl.DataFrame:
    from openskistats.analyze import load_run_segments_pl

    skiway_df = (
        load_run_segments_pl(
            ski_area_filters=[pl.col("ski_area_name").eq("Dartmouth Skiway")]
        )
        .with_columns(
            solar_irradiance=pl.when(pl.col("segment_hash").is_not_null())
            .then(pl.struct("latitude", "longitude", "elevation", "slope", "bearing"))
            .map_elements(
                lambda x: compute_solar_irradiance(
                    **x, time_freq="1h", extent="solstice"
                ).to_struct(),
                return_dtype=pl.List(
                    pl.Struct({"datetime": pl.Datetime(), "poa_global": pl.Float64})
                ),
                skip_nulls=True,
                strategy="thread_local",
            ),
        )
        .collect()
    )
    path = get_data_directory().joinpath("dartmouth_skiway_solar_irradiance.parquet")
    skiway_df.write_parquet(path)
    return skiway_df


def add_solar_irradiation_columns(
    run_segments: pl.DataFrame,
    skip_cache: bool = False,
    max_items: int | None = int(
        os.environ.get("OPENSKISTATS_SOLAR_SEGMENT_COUNT", "500")
    ),
) -> pl.DataFrame:
    """
    Adds three columns to a run coordinate/segment DataFrame:

    - solar_cache_version
    - solar_irradiation_season
    - solar_irradiation_solstice

    Unless clear_cache is True, a lookup of prior results is attempted because the computation is quite slow.
    """
    segments_cached = load_solar_irradiation_cache_pl(skip_cache=skip_cache)
    n_segments = run_segments["segment_hash"].drop_nulls().n_unique()
    segments_to_compute = (
        run_segments
        # remove segments that have already been computed
        .filter(pl.col("segment_hash").is_not_null())
        .join(
            segments_cached,
            on="segment_hash",
            how="anti",
        )
        .select(
            "segment_hash", "latitude", "longitude", "elevation", "slope", "bearing"
        )
        .unique(subset=["segment_hash"])
        .to_dicts()
    )
    logging.info(
        f"Solar irradiation requested for {n_segments:,} segments, {len(segments_to_compute):,} segments not in cache."
    )
    if max_items is not None:
        segments_to_compute = segments_to_compute[:max_items]
    logging.info(
        f"Computing solar irradiation for {len(segments_to_compute):,} segments after limiting to {max_items=}."
    )

    def _process_segment(segment: dict[str, Any]) -> dict[str, float]:
        segment_hash = segment.pop("segment_hash")
        result = compute_solar_irradiance(**segment).pipe(collapse_solar_irradiance)
        assert isinstance(result, dict)
        result["segment_hash"] = segment_hash
        result["solar_cache_version"] = SOLAR_CACHE_VERSION
        return result

    start_time = perf_counter()
    with ThreadPoolExecutor(max_workers=4) as executor, Progress() as progress:
        progress_task = progress.add_task(
            "Computing solar irradiation...",
            total=len(segments_to_compute),
        )
        results = []
        for result in executor.map(_process_segment, segments_to_compute):
            results.append(result)
            progress.advance(progress_task)
    total_time = perf_counter() - start_time
    if segments_to_compute:
        logging.info(
            f"Computed solar irradiation for {len(segments_to_compute):,} segments in {total_time / 60:.1f} minutes: "
            f"{total_time / len(segments_to_compute):.4f} seconds per segment."
        )
        logging.info(f"_get_clearsky lru_cache info: {get_clearsky.cache_info()}")
    segments_computed = pl.DataFrame(
        data=results, schema=_get_solar_irradiation_cache_schema()
    )
    return run_segments.join(
        pl.concat([segments_cached, segments_computed]),
        on="segment_hash",
        how="left",
    )


def _get_solar_irradiation_cache_schema() -> dict[str, pl.DataType]:
    return {
        "segment_hash": pl.UInt64,
        "solar_cache_version": pl.UInt8,
        "solar_irradiation_season": pl.Float32,
        "solar_irradiation_solstice": pl.Float32,
    }


def _get_runs_cache_path(skip_cache: bool = False) -> str | None | Path:
    from openskistats.analyze import get_runs_parquet_path

    if skip_cache or running_in_test():
        return None
    if running_in_ci():
        url = "https://github.com/dhimmel/openskistats/raw/data/runs.parquet"
        return url if requests.head(url).ok else None
    local_path = get_runs_parquet_path()
    if not local_path.exists():
        return None
    return local_path


def load_solar_irradiation_cache_pl(skip_cache: bool = False) -> pl.DataFrame:
    path = _get_runs_cache_path(skip_cache=skip_cache)
    if not path:
        return pl.DataFrame(
            data=[],
            schema=_get_solar_irradiation_cache_schema(),
        )
    logging.info(f"Loading solar irradiation cache from {path=}.")
    return (
        # NOTE: could use load_run_segments_pl if we allow custom paths
        pl.scan_parquet(source=path)
        .select("run_id", "run_coordinates_clean")
        .explode("run_coordinates_clean")
        .unnest("run_coordinates_clean")
        .filter(pl.col("segment_hash").is_not_null())
        .filter(pl.col("solar_cache_version") == SOLAR_CACHE_VERSION)
        .select(
            "segment_hash",
            "solar_cache_version",
            "solar_irradiation_season",
            "solar_irradiation_solstice",
        )
        .unique(subset=["segment_hash"])
        .collect()
    )


def get_solar_location_band(
    latitude: float, longitude: float, elevation: float
) -> pl.DataFrame:
    """
    Get the solar position for a location to plot the sun's path at key dates.
    This is a preliminary function which likely will be changed to support the exact data schema
    required for plotting the sun's path as a band between the solstice and closing day of a typical ski season.
    """
    times = SkiSeasonDatetimes(
        hemisphere=get_hemisphere(latitude),
        extent="season",
        freq="1min",
    ).interpolated_range
    return (
        pvlib.location.Location(
            latitude=latitude, longitude=longitude, altitude=elevation
        )
        .get_solarposition(times, method="nrel_numpy")
        .add_prefix("sun_")
        .reset_index(names="datetime")
        .pipe(pl.from_pandas)
        # restrict to daytime
        .filter(pl.col("sun_apparent_elevation") > 0)
        .select(
            "datetime",
            pl.col("datetime").dt.date().alias("date"),
            pl.col("datetime").dt.time().alias("time"),
            pl.col("sun_apparent_zenith").alias("sun_zenith"),
            "sun_azimuth",
            sun_azimuth_bin_center=pl.col("sun_azimuth")
            .cut(breaks=list(range(0, 361)), left_closed=True, include_breaks=True)
            .struct.field("breakpoint")
            .sub(0.5),
        )
        .group_by("sun_azimuth_bin_center")
        .agg(
            pl.count("sun_zenith").alias("sun_zenith_count"),
            pl.min("datetime").alias("datetime_min"),
            pl.max("datetime").alias("datetime_max"),
            pl.min("sun_zenith").alias("sun_zenith_min"),
            pl.max("sun_zenith").alias("sun_zenith_max"),
        )
        .sort("sun_azimuth_bin_center")
    )


@dataclass(frozen=True)
class SolarPolarPlot:
    """Base class for solar polar plots with shared functionality."""

    # when datetime is None, plot the entire season.
    # If datetime is not None, plot the solar irradiance at that date and time.
    date_time: datetime | None = None

    def plot(
        self,
        fig: plt.Figure | None = None,
        ax: plt.Axes | None = None,
        vmax: float | None = None,
    ) -> tuple[plt.Figure, QuadMesh]:
        fig, ax = _get_fig_ax(ax=ax, figsize=(3, 3), bgcolor=None, polar=True)
        radial_grid, bearing_grid, irradiance_grid = self.get_grids()
        mesh = self._create_polar_mesh(
            ax,
            bearing_grid,
            radial_grid,
            irradiance_grid,
            vmax=vmax,
        )
        self._setup_polar_plot(ax, colorbar=False)
        return fig, mesh

    def _setup_polar_plot(self, ax: plt.Axes, colorbar: bool = True) -> Colorbar | None:
        """Configure polar plot with standard formatting."""
        ax.set_theta_zero_location("N")
        ax.set_theta_direction("clockwise")
        ax.grid(visible=False)
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(labels=["N", "", "E", "", "S", "", "W", ""])
        ax.tick_params(axis="x", which="major", pad=-2)

        from openskistats.plot import _add_polar_y_ticks

        _add_polar_y_ticks(ax=ax)

        cb = None
        if colorbar:
            quad_mesh = ax.collections[0]  # Get the last added pcolormesh
            cb = plt.colorbar(quad_mesh, ax=ax, location="left", aspect=35, pad=0.053)
            cb.outline.set_visible(False)
            cb.ax.tick_params(labelsize=8)
        return cb

    def _create_polar_mesh(
        self,
        ax: plt.Axes,
        bearing_grid: npt.NDArray[np.float64],
        radial_grid: npt.NDArray[np.float64],
        value_grid: npt.NDArray[np.float64],
        vmax: float | None = None,
    ) -> QuadMesh:
        """Create polar mesh plot with standard formatting."""
        return ax.pcolormesh(
            np.deg2rad(bearing_grid),
            radial_grid,
            value_grid,
            shading="nearest",
            cmap="inferno" if self.date_time else "cividis",
            vmin=0,
            vmax=vmax,
        )

    def get_grids(
        self,
    ) -> tuple[
        npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
    ]:
        raise NotImplementedError


@dataclass(frozen=True)
class SlopeByBearingPlots(SolarPolarPlot):
    latitude: float = 43.785237
    longitude: float = -72.09891
    elevation: float = 280.24

    def get_clearsky(self) -> pl.DataFrame:
        df = get_clearsky(
            latitude=self.latitude,
            longitude=self.longitude,
            elevation=self.elevation,
            ski_season=SkiSeasonDatetimes("north", "season"),
        )
        if self.date_time:
            df = df.filter(pl.col("datetime").eq(self.date_time.astimezone(UTC)))
        return df

    def get_slopes_range(self) -> npt.NDArray[np.float64]:
        return np.arange(0, 90, 5, dtype=np.float64)

    def get_bearings_range(self) -> npt.NDArray[np.float64]:
        return np.arange(0, 360, 10, dtype=np.float64)

    def _get_grids_season(
        self,
    ) -> tuple[
        npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
    ]:
        clearsky_df = self.get_clearsky()
        slopes = self.get_slopes_range()
        bearings = self.get_bearings_range()
        bearing_grid, slope_grid = np.meshgrid(
            bearings,
            slopes,
            indexing="ij",
        )
        irradiance_grid = np.zeros(shape=(len(bearings), len(slopes)))
        for i, bearing in enumerate(bearings):
            for j, slope in enumerate(slopes):
                irrad_df = (
                    pvlib.irradiance.get_total_irradiance(
                        surface_tilt=slope,
                        surface_azimuth=bearing,
                        solar_zenith=clearsky_df["sun_apparent_zenith"].to_pandas(),
                        solar_azimuth=clearsky_df["sun_azimuth"].to_pandas(),
                        dni=clearsky_df["dni"].to_pandas(),
                        ghi=clearsky_df["ghi"].to_pandas(),
                        dhi=clearsky_df["dhi"].to_pandas(),
                        surface_type="snow",
                    )
                    .pipe(pl.from_pandas)
                    .with_columns(is_solstice=pl.lit(False))
                )
                if self.date_time:
                    irradiance_grid[i, j] = irrad_df.get_column("poa_global").item()
                else:
                    irradiance_grid[i, j] = collapse_solar_irradiance(irrad_df)[
                        "solar_irradiation_season"
                    ]
        return slope_grid, bearing_grid, irradiance_grid

    def _get_grids_datetime(
        self,
    ) -> tuple[
        npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
    ]:
        clearsky_df = self.get_clearsky()
        clearsky_info = clearsky_df.row(
            by_predicate=pl.lit(True),
            named=True,
        )
        slopes = self.get_slopes_range()
        bearings = self.get_bearings_range()
        bearing_grid, slope_grid = np.meshgrid(
            bearings,
            slopes,
            indexing="ij",
        )
        irradiance_grid = np.zeros(shape=(len(bearings), len(slopes)))
        for i, bearing in enumerate(bearings):
            irradiance = pvlib.irradiance.get_total_irradiance(
                surface_tilt=slopes,
                surface_azimuth=bearing,
                solar_zenith=clearsky_info["sun_apparent_zenith"],
                solar_azimuth=clearsky_info["sun_azimuth"],
                dni=clearsky_info["dni"],
                ghi=clearsky_info["ghi"],
                dhi=clearsky_info["dhi"],
                surface_type="snow",
            )["poa_global"]
            irradiance_grid[i, :] = irradiance
        return slope_grid, bearing_grid, irradiance_grid

    @lru_cache(maxsize=20)  # noqa: B019
    def get_grids(
        self,
    ) -> tuple[
        npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
    ]:
        if self.date_time:
            return self._get_grids_datetime()
        return self._get_grids_season()


@dataclass(frozen=True)
class LatitudeByBearingPlots(SolarPolarPlot):
    longitude: float = -72.09891
    elevation: float = 280.24
    slope: float = 15.0

    def get_clearsky(self, latitude: float) -> pl.DataFrame:
        df = get_clearsky(
            latitude=latitude,
            longitude=self.longitude,
            elevation=self.elevation,
            ski_season=SkiSeasonDatetimes("north", "season"),
        )
        if self.date_time:
            df = df.filter(pl.col.datetime.eq(self.date_time.astimezone(UTC)))
        return df

    @lru_cache(maxsize=20)  # noqa: B019
    def get_grids(
        self,
    ) -> tuple[
        npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
    ]:
        latitudes = np.arange(0, 90, 5)
        bearings = np.arange(0, 360, 10)
        bearing_grid, latitude_grid = np.meshgrid(
            bearings,
            latitudes,
            indexing="ij",
        )
        irradiance_grid = np.zeros(shape=(len(bearings), len(latitudes)))
        for j, latitude in enumerate(latitudes):
            clearsky_info = self.get_clearsky(latitude=float(latitude))
            for i, bearing in enumerate(bearings):
                irrad_df = pvlib.irradiance.get_total_irradiance(
                    surface_tilt=self.slope,
                    surface_azimuth=bearing,  # Vectorize across bearings
                    solar_zenith=clearsky_info["sun_apparent_zenith"].to_pandas(),
                    solar_azimuth=clearsky_info["sun_azimuth"].to_pandas(),
                    dni=clearsky_info["dni"].to_pandas(),
                    ghi=clearsky_info["ghi"].to_pandas(),
                    dhi=clearsky_info["dhi"].to_pandas(),
                    surface_type="snow",
                ).pipe(pl.from_pandas)
                if self.date_time:
                    irradiance_grid[i, j] = irrad_df.get_column("poa_global").item()
                else:
                    irradiance_grid[i, j] = irrad_df.with_columns(
                        is_solstice=pl.lit(False)
                    ).pipe(collapse_solar_irradiance)["solar_irradiation_season"]

        return latitude_grid, bearing_grid, irradiance_grid


def create_combined_solar_plots() -> plt.Figure:
    """Create a combined figure with multiple solar plots arranged in a 2x3 grid."""
    # Create main figure with two subfigures side by side
    fig = plt.figure(figsize=(17, 10), constrained_layout=True)
    subfigs = fig.subfigures(nrows=1, ncols=2, width_ratios=[2, 1])

    # Left subfigure for instant irradiance plots (4 plots)
    subfig_instant = subfigs[0]
    gs_instant = subfig_instant.add_gridspec(nrows=2, ncols=2)
    ax1 = subfig_instant.add_subplot(gs_instant[0, 0], projection="polar")
    ax2 = subfig_instant.add_subplot(gs_instant[0, 1], projection="polar")
    ax4 = subfig_instant.add_subplot(gs_instant[1, 0], projection="polar")
    ax5 = subfig_instant.add_subplot(gs_instant[1, 1], projection="polar")

    # Right subfigure for season average plots (2 plots)
    subfig_season = subfigs[1]
    gs_season = subfig_season.add_gridspec(nrows=2, ncols=1)
    ax3 = subfig_season.add_subplot(gs_season[0, 0], projection="polar")
    ax6 = subfig_season.add_subplot(gs_season[1, 0], projection="polar")

    datetime_solstice_morning = datetime.fromisoformat("2024-12-21 09:00:00-05:00")
    datetime_closing_afternoon = datetime.fromisoformat("2025-03-31 15:30:00-05:00")

    plotters = [
        SlopeByBearingPlots(date_time=datetime_solstice_morning),
        SlopeByBearingPlots(date_time=datetime_closing_afternoon),
        SlopeByBearingPlots(date_time=None),
        LatitudeByBearingPlots(date_time=datetime_solstice_morning),
        LatitudeByBearingPlots(date_time=datetime_closing_afternoon),
        LatitudeByBearingPlots(date_time=None),
    ]
    max_values = [plotter.get_grids()[2].max() for plotter in plotters]
    max_value_instant = max(itemgetter(0, 1, 3, 4)(max_values))
    max_value_season = max(itemgetter(2, 5)(max_values))

    # Plot instant irradiance plots
    _, mesh1 = plotters[0].plot(fig=fig, ax=ax1, vmax=max_value_instant)
    ax1.set_title("Winter Solstice Morning")
    _, mesh2 = plotters[1].plot(fig=fig, ax=ax2, vmax=max_value_instant)
    ax2.set_title("Season Close Afternoon")
    _, mesh4 = plotters[3].plot(fig=fig, ax=ax4, vmax=max_value_instant)
    _, mesh5 = plotters[4].plot(fig=fig, ax=ax5, vmax=max_value_instant)

    # Plot season average plots
    _, mesh3 = plotters[2].plot(fig=fig, ax=ax3, vmax=max_value_season)
    ax3.set_title("Season Average")
    _, mesh6 = plotters[5].plot(fig=fig, ax=ax6, vmax=max_value_season)

    # Add colorbars to each subfigure
    subfig_instant.colorbar(
        mesh1,
        ax=[ax1, ax2, ax4, ax5],
        label="Instant Irradiance (W/m²)",
        location="right",
        pad=0.05,
        aspect=40,
    )

    subfig_season.colorbar(
        mesh3,
        ax=[ax3, ax6],
        label="Daily Irradiation (kWh/m²)",
        location="right",
        pad=0.05,
        aspect=40,
    )

    return fig
