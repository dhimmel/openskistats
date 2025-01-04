import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, timedelta
from functools import cached_property, lru_cache
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import pandas as pd
import polars as pl
import pvlib

from openskistats.utils import (
    get_data_directory,
    get_hemisphere,
    running_in_ci,
    running_in_test,
)

SOLAR_IRRADIANCE_CACHE_VERSION = 1
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


def add_solar_irradiance_columns(
    run_segments: pl.DataFrame,
    skip_cache: bool = True,
    max_items: int | None = 500,
) -> pl.DataFrame:
    """
    Adds three columns to a run coordinate/segment DataFrame:

    - solar_irradiance_cache_version
    - solar_irradiance_season
    - solar_irradiance_solstice

    Unless clear_cache is True, a lookup of prior results is attempted because the computation is quite slow.
    """
    segments_cached = load_solar_irradiance_cache_pl(skip_cache=skip_cache)
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
        f"Solar irradiance requested for {n_segments:,} segments, {len(segments_to_compute):,} segments not in cache."
    )
    if max_items is not None:
        segments_to_compute = segments_to_compute[:max_items]
    logging.info(
        f"Computing solar irradiance for {len(segments_to_compute):,} segments after limiting to {max_items=}."
    )

    def _process_segment(segment: dict[str, Any]) -> dict[str, float]:
        segment_hash = segment.pop("segment_hash")
        result = compute_solar_irradiance(**segment).pipe(collapse_solar_irradiance)
        assert isinstance(result, dict)
        result["segment_hash"] = segment_hash
        result["solar_irradiance_cache_version"] = SOLAR_IRRADIANCE_CACHE_VERSION
        return result

    start_time = perf_counter()
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(_process_segment, segments_to_compute))
    total_time = perf_counter() - start_time
    logging.info(
        f"Computed solar irradiance for {len(segments_to_compute):,} segments in {total_time / 60:.1f} minutes: "
        f"{total_time / len(segments_to_compute):.4f} seconds per segment."
    )
    segments_computed = pl.DataFrame(
        data=results, schema=_get_solar_irradiance_cache_schema()
    )
    return run_segments.join(
        pl.concat([segments_cached, segments_computed]),
        on="segment_hash",
        how="left",
    )


def _get_solar_irradiance_cache_schema() -> dict[str, pl.DataType]:
    return {
        "segment_hash": pl.UInt64,
        "solar_irradiance_cache_version": pl.UInt8,
        "solar_irradiance_season": pl.Float32,
        "solar_irradiance_solstice": pl.Float32,
    }


def _get_runs_cache_path(skip_cache: bool = False) -> str | None | Path:
    from openskistats.analyze import get_runs_parquet_path

    if skip_cache or running_in_test():
        return None
    if running_in_ci():
        # FIXME: return materialized data path
        return None
    local_path = get_runs_parquet_path()
    if not local_path.exists():
        return None
    return local_path


def load_solar_irradiance_cache_pl(skip_cache: bool = False) -> pl.DataFrame:
    path = _get_runs_cache_path(skip_cache=skip_cache)
    if not path:
        return pl.DataFrame(
            data=[],
            schema=_get_solar_irradiance_cache_schema(),
        )
    return (
        pl.scan_parquet(source=path)
        .select("run_id", "run_coordinates_clean")
        .explode("run_coordinates_clean")
        .unnest("run_coordinates_clean")
        .filter(pl.col("segment_hash").is_not_null())
        .filter(
            pl.col("solar_irradiance_cache_version") == SOLAR_IRRADIANCE_CACHE_VERSION
        )
        .select(
            "segment_hash",
            "solar_irradiance_cache_version",
            "solar_irradiance_season",
            "solar_irradiance_solstice",
        )
        .unique(subset=["segment_hash"])
        .collect()
    )
