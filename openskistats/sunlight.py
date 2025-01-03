from dataclasses import dataclass
from datetime import date, timedelta
from functools import cached_property
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
    def get_mean_irradiance(df: pl.DataFrame) -> float:
        return 24 * float(df["poa_global"].mean()) / 1_000

    return {
        "solar_irradiance_season": get_mean_irradiance(irrad_df),
        "solar_irradiance_solstice": get_mean_irradiance(
            irrad_df.filter(pl.col("is_solstice"))
        ),
    }


@dataclass(frozen=True)
class SkiSeasonDatetimes:
    hemisphere: Literal["north", "south"]
    extent: Literal["solstice", "season"]
    freq: str = "60min"

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


# @lru_cache(maxsize=20_000)
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
            solar_irradiance=pl.when(pl.col("segment_hash").is_not_null()).then(
                pl.struct(
                    "latitude", "longitude", "elevation", "slope", "bearing"
                ).map_elements(
                    lambda x: compute_solar_irradiance(
                        **x, time_freq="1h", extent="solstice"
                    ).to_struct(),
                    return_dtype=pl.List(
                        pl.Struct({"datetime": pl.Datetime(), "poa_global": pl.Float64})
                    ),
                    returns_scalar=True,
                    strategy="thread_local",
                )
            ),
        )
        .collect()
    )
    path = get_data_directory().joinpath("dartmouth_skiway_solar_irradiance.parquet")
    skiway_df.write_parquet(path)
    return skiway_df


SOLAR_IRRADIANCE_CACHE_VERSION = 1


def compute_solar_irradiance_all_segments(
    cache_version: int = SOLAR_IRRADIANCE_CACHE_VERSION, clear_cache: bool = False
) -> pl.DataFrame:
    from openskistats.analyze import (
        get_display_ski_area_filters,
        load_runs_pl,
        load_ski_areas_pl,
    )

    segments_cached = load_solar_irradiance_pl(skip_cache=clear_cache)
    path = get_data_directory().joinpath("runs_segments_solar_irradiance.parquet")
    ski_areas = (
        load_ski_areas_pl(ski_area_filters=get_display_ski_area_filters())
        .filter(pl.col("country") == "United States")
        .filter(pl.col("region") == "New Hampshire")
        .head(2)
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
            pl.lit(cache_version, dtype=pl.Int32).alias("cache_version"),
        )
        .unique(subset=["segment_hash"])
        .join(segments_cached, on=["segment_hash", "cache_version"], how="anti")
        .with_columns(
            _solar_irradiance=pl.when(pl.col("segment_hash").is_not_null()).then(
                pl.struct(
                    "latitude", "longitude", "elevation", "slope", "bearing"
                ).map_elements(
                    lambda x: compute_solar_irradiance(
                        **x, time_freq="1h", extent="solstice"
                    ).pipe(collapse_solar_irradiance),
                    return_dtype=pl.Struct(
                        {
                            "solar_irradiance_season": pl.Float64,
                            "solar_irradiance_solstice": pl.Float64,
                        },
                    ),
                    returns_scalar=True,
                    strategy="thread_local",
                )
            ),
        )
        .unnest("_solar_irradiance")
    )
    segments = pl.concat([segments_cached, segments], how="vertical")
    segments = segments.collect()
    segments.write_parquet(path)
    return segments


def load_solar_irradiance_pl(
    skip_cache: bool = False, apply_filter_select: bool = False
) -> pl.LazyFrame:
    path = get_data_directory().joinpath("runs_segments_solar_irradiance.parquet")
    if skip_cache or not path.exists():
        segments_cached = pl.DataFrame(
            data=[],
            schema={
                "segment_hash": pl.UInt64,
                "latitude": pl.Float64,
                "longitude": pl.Float64,
                "elevation": pl.Float64,
                "slope": pl.Float64,
                "bearing": pl.Float64,
                "cache_version": pl.Int32,
                "solar_irradiance_season": pl.Float64,
                "solar_irradiance_solstice": pl.Float64,
            },
        ).lazy()
    else:
        segments_cached = pl.scan_parquet(path)
    if apply_filter_select:
        segments_cached = segments_cached.filter(
            pl.col("cache_version") == SOLAR_IRRADIANCE_CACHE_VERSION
        ).select("segment_hash", "solar_irradiance_season", "solar_irradiance_solstice")
    return segments_cached
