from dataclasses import dataclass
from datetime import date, timedelta
from functools import cached_property, lru_cache
from typing import Literal

import pandas as pd
import polars as pl
import pvlib

from openskistats.utils import get_data_directory, get_hemisphere

SOLSTICE_NORTH = date.fromisoformat("2024-12-21")
SOLSTICE_SOUTH = date.fromisoformat("2024-06-20")


def compute_solar_irradiance(
    latitude: float,
    longitude: float,
    elevation: float,
    slope: float,
    bearing: float,
    time_freq: str = "1h",
    extent: Literal["solstice", "season"] = "solstice",
    collapse: bool = True,
) -> float | pl.Series | None:
    """
    Compute daily clear-sky irradiance (W/m^2) for the winter solstice in the Northern Hemisphere.
    """
    if slope is None:
        return None

    ski_season = SkiSeasonDatetimes(
        freq=time_freq,
        hemisphere=get_hemisphere(latitude),
        extent=extent,
    )
    # rounding as a hack to improve efficiency via caching
    latitude = round(latitude, 0)
    longitude = round(longitude, 0)
    elevation = 100 * round(elevation / 100, 0)
    clearsky_df = get_clearsky(
        latitude=latitude,
        longitude=longitude,
        elevation=elevation,
        ski_season=ski_season,
    )
    # Calculate plane-of-array irradiance for each hour
    irrad_df = pvlib.irradiance.get_total_irradiance(
        surface_tilt=slope,
        surface_azimuth=bearing,
        solar_zenith=clearsky_df["apparent_zenith"],
        solar_azimuth=clearsky_df["azimuth"],
        dni=clearsky_df["dni"],  # diffuse normal irradiance
        ghi=clearsky_df["ghi"],  # global horizontal irradiance
        dhi=clearsky_df["dhi"],  # direct horizontal irradiance
        surface_type="snow",
    )
    if collapse:
        return (
            float(irrad_df["poa_global"].sum())
            * ski_season.times_per_hour
            / ski_season.season_duration_days
        )
    else:
        return pl.DataFrame(
            {
                "datetime": irrad_df.index,
                "poa_global": irrad_df["poa_global"],
            }
        ).to_struct(name="solar_irradiance")


@dataclass(frozen=True)
class SkiSeasonDatetimes:
    freq: str
    hemisphere: Literal["north", "south"]
    extent: Literal["solstice", "season"]

    @staticmethod
    def get_typical_ski_season_dates(
        hemisphere: Literal["north", "south"], extent: Literal["solstice", "season"]
    ) -> tuple[date, date]:
        """
        Return open and closing dates for a typical ski season.
        """
        match hemisphere, extent:
            case "north", "solstice":
                return SOLSTICE_NORTH, SOLSTICE_NORTH
            case "south", "solstice":
                return SOLSTICE_SOUTH, SOLSTICE_SOUTH
            case "north", "season":
                return SOLSTICE_NORTH - timedelta(days=20), SOLSTICE_NORTH + timedelta(
                    days=100
                )
            case "south", "season":
                return SOLSTICE_SOUTH - timedelta(days=20), SOLSTICE_SOUTH + timedelta(
                    days=100
                )
            case _:
                raise ValueError("Invalid hemisphere or extent")

    @cached_property
    def ski_season_dates(self) -> tuple[date, date]:
        return self.get_typical_ski_season_dates(
            hemisphere=self.hemisphere, extent=self.extent
        )

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


@lru_cache(maxsize=10_000)
def get_clearsky(
    latitude: float, longitude: float, elevation: float, ski_season: SkiSeasonDatetimes
) -> pd.DataFrame:
    location = pvlib.location.Location(
        latitude=latitude,
        longitude=longitude,
        altitude=elevation,
    )
    times = ski_season.interpolated_range
    solar_positions = location.get_solarposition(times)
    clearsky = location.get_clearsky(times, model="ineichen")
    return pd.concat([solar_positions, clearsky], axis=1)[
        [
            "apparent_zenith",
            "azimuth",
            "dni",
            "ghi",
            "dhi",
        ]
    ]


def write_dartmouth_skiway_solar_irradiance() -> pl.DataFrame:
    from openskistats.analyze import load_runs_pl

    skiway_df = (
        load_runs_pl()
        .filter(
            pl.col("ski_area_ids").list.contains(
                "74e0060a96e0399ace1b1e5ef5af1e5197a19752"
            )
        )  # Dartmouth Skiway
        .select("run_id", "run_coordinates_clean")
        .explode("run_coordinates_clean")
        .unnest("run_coordinates_clean")
        .with_columns(
            solar_irradiance=pl.struct(
                "latitude", "longitude", "elevation", "slope", "bearing"
            ).map_elements(
                lambda x: compute_solar_irradiance(
                    **x, time_freq="1h", extent="solstice", collapse=False
                ),
                return_dtype=pl.List(
                    pl.Struct({"datetime": pl.Datetime(), "poa_global": pl.Float64})
                ),
                returns_scalar=True,
                strategy="thread_local",
            ),
        )
        .collect()
    )
    path = get_data_directory().joinpath("dartmouth_skiway_solar_irradiance.parquet")
    skiway_df.write_parquet(path)
    return skiway_df


def compute_solar_irradiance_all_segments(cache_version: int = 1) -> pl.DataFrame:
    from openskistats.analyze import (
        get_display_ski_area_filters,
        load_runs_pl,
        load_ski_areas_pl,
    )

    path = get_data_directory().joinpath("runs_segments_solar_irradiance.parquet")
    ski_areas = (
        load_ski_areas_pl(ski_area_filters=get_display_ski_area_filters())
        .filter(pl.col("country") == "United States")
        .filter(pl.col("region") == "New Hampshire")
        .select("ski_area_id")
        .lazy()
    )
    segments = (
        load_runs_pl()
        .explode("ski_area_ids")
        .rename({"ski_area_ids": "ski_area_id"})
        .join(ski_areas, on="ski_area_id")
        .select("run_id", "run_coordinates_clean")
        .explode("run_coordinates_clean")
        .unnest("run_coordinates_clean")
        .filter(pl.col("segment_hash").is_not_null())
        .select(
            "segment_hash",
            "latitude",
            "longitude",
            "elevation",
            "slope",
            "bearing",
            pl.lit(cache_version).alias("cache_version"),
        )
        .unique(subset=["segment_hash"])
        .with_columns(
            solar_irradiance_solstice=pl.struct(
                "latitude", "longitude", "elevation", "slope", "bearing"
            ).map_elements(
                lambda x: compute_solar_irradiance(
                    **x, time_freq="1h", extent="solstice", collapse=True
                ),
                return_dtype=pl.Float64,
                returns_scalar=True,
                strategy="thread_local",
            ),
        )
        .collect()
    )
    segments.write_parquet(path)
    return segments
