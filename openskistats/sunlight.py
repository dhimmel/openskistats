from functools import cache, lru_cache

import pandas as pd
import pvlib


def compute_solar_irradiance(
    latitude: float,
    longitude: float,
    elevation: float,
    slope: float,
    bearing: float,
    time_freq: str = "1h",
) -> float:
    """
    Compute daily clear-sky irradiance (W/m^2) for the winter solstice in the Northern Hemisphere.
    """
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
    return float(irrad_df["poa_global"].sum() * time_freq_hours)


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
