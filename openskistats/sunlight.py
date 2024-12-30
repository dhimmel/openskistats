from functools import cache, lru_cache

import pandas as pd
import polars as pl
import pvlib

from openskistats.utils import get_data_directory


def compute_solar_irradiance(
    latitude: float,
    longitude: float,
    elevation: float,
    slope: float,
    bearing: float,
    time_freq: str = "1h",
    collapse: bool = True,
) -> float | pl.Series | None:
    """
    Compute daily clear-sky irradiance (W/m^2) for the winter solstice in the Northern Hemisphere.
    """
    if slope is None:
        return None
    # rounding as a hack to improve efficiency via caching
    latitude = round(latitude, 0)
    longitude = round(longitude, 0)
    elevation = 100 * round(elevation / 100, 0)
    clearsky_df = get_clearsky(
        latitude=latitude,
        longitude=longitude,
        elevation=elevation,
        time_freq=time_freq,
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
    time_freq_hours = pd.to_timedelta(time_freq).total_seconds() / 3_600
    if collapse:
        return float(irrad_df["poa_global"].sum() * time_freq_hours)
    else:
        return pl.DataFrame(
            {
                "datetime": irrad_df.index,
                "poa_global": irrad_df["poa_global"] * time_freq_hours,
            }
        ).to_struct()


@cache
def get_times(freq: str) -> pd.DatetimeIndex:
    # FIXME: dates based on hemisphere
    return pd.date_range(
        start="2024-12-21 00:00:00",
        end="2024-12-22 00:00:00",
        inclusive="left",
        freq=freq,
        tz="UTC",
    )


@lru_cache(maxsize=10_000)
def get_clearsky(
    latitude: float, longitude: float, elevation: float, time_freq: str
) -> pd.DataFrame:
    location = pvlib.location.Location(
        latitude=latitude,
        longitude=longitude,
        altitude=elevation,
    )
    times = get_times(freq=time_freq)
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
        # .drop_nulls(subset=["segment_hash"])
        .with_columns(
            solar_irradiance=pl.struct(
                "latitude", "longitude", "elevation", "slope", "bearing"
            ).map_elements(
                lambda x: compute_solar_irradiance(**x, collapse=False),
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
