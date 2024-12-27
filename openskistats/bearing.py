from typing import Literal

import numpy as np
import numpy.typing as npt
import polars as pl
from osmnx.bearing import calculate_bearing
from osmnx.distance import great_circle

from openskistats.models import BearingStatsModel, SkiAreaBearingDistributionModel


def add_spatial_metric_columns(
    df: pl.DataFrame | pl.LazyFrame, partition_by: str | list[str]
) -> pl.LazyFrame:
    """
    Add spatial metrics to a DataFrame of geographic coordinates.
    """
    for column in ["index", "latitude", "longitude", "elevation"]:
        assert column in df
    return (
        df.lazy()
        .with_columns(
            latitude_lag=pl.col("latitude")
            .shift(1)
            .over(partition_by, order_by="index"),
            longitude_lag=pl.col("longitude")
            .shift(1)
            .over(partition_by, order_by="index"),
            elevation_lag=pl.col("elevation")
            .shift(1)
            .over(partition_by, order_by="index"),
        )
        .with_columns(
            distance_vertical=pl.col("elevation_lag") - pl.col("elevation"),
            _coord_struct=pl.struct(
                "latitude_lag",
                "longitude_lag",
                "latitude",
                "longitude",
            ),
        )
        .with_columns(
            segment_hash=pl.when(pl.col("latitude_lag").is_not_null()).then(
                pl.col("_coord_struct").hash(seed=0)
            ),
            distance_vertical_drop=pl.col("distance_vertical").clip(lower_bound=0),
            distance_horizontal=pl.col("_coord_struct").map_batches(
                lambda x: great_circle(
                    lat1=x.struct.field("latitude_lag"),
                    lon1=x.struct.field("longitude_lag"),
                    lat2=x.struct.field("latitude"),
                    lon2=x.struct.field("longitude"),
                )
            ),
        )
        .with_columns(
            distance_3d=(
                pl.col("distance_horizontal") ** 2 + pl.col("distance_vertical") ** 2
            ).sqrt(),
            bearing=pl.col("_coord_struct").map_batches(
                lambda x: calculate_bearing(
                    lat1=x.struct.field("latitude_lag"),
                    lon1=x.struct.field("longitude_lag"),
                    lat2=x.struct.field("latitude"),
                    lon2=x.struct.field("longitude"),
                )
            ),
            gradient=pl.when(pl.col("distance_horizontal") > 0)
            .then(pl.col("distance_vertical"))
            .truediv("distance_horizontal"),
        )
        .with_columns(
            slope=pl.col("gradient").arctan().degrees(),
        )
        .drop("latitude_lag", "longitude_lag", "elevation_lag", "_coord_struct")
    )


bearing_labels = {
    0.0: "N",
    11.25: "NbE",
    22.5: "NNE",
    33.75: "NEbN",
    45.0: "NE",
    56.25: "NEbE",
    67.5: "ENE",
    78.75: "EbN",
    90.0: "E",
    101.25: "EbS",
    112.5: "ESE",
    123.75: "SEbE",
    135.0: "SE",
    146.25: "SEbS",
    157.5: "SSE",
    168.75: "SbE",
    180.0: "S",
    191.25: "SbW",
    202.5: "SSW",
    213.75: "SWbS",
    225.0: "SW",
    236.25: "SWbW",
    247.5: "WSW",
    258.75: "WbS",
    270.0: "W",
    281.25: "WbN",
    292.5: "WNW",
    303.75: "NWbW",
    315.0: "NW",
    326.25: "NWbN",
    337.5: "NNW",
    348.75: "NbW",
}
"""Bearing labels for 32-wind compass rose."""


def cut_bearings_pl(num_bins: int, bearing_col: str = "bearing") -> pl.Expr:
    """
    Get a Polars expression to bin bearings into `num_bins` bins.
    Returns bin index (1-indexed) as an integer.

    Prevents bin-edge effects around common values like 0 degrees and 90 degrees
    by centering the first bin due north (0 degrees) rather than starting the first bin at 0 degrees.
    For example, if `num_bins=36` is provided,
    then each bin will represent 10 degrees around the compass,
    with the first bin representing 355 degrees to 5 degrees.
    """
    assert num_bins > 0
    bin_centers = [i * 360 / num_bins for i in range(num_bins)]
    return (
        pl.col(bearing_col)
        # add a half bin width
        .add(180 / num_bins)
        .mod(360)
        .cut(
            breaks=bin_centers,
            # NOTE: in polars.Expr.cut labels must be one longer than breaks.
            # At ToW the 0000 bin never gets assigned,
            # although it's not clear why the extra label is for the first bin and not the last.
            labels=[f"{i:04d}" for i in range(num_bins + 1)],
            # left_closed for parity with np.histogram and osmnx.bearing._bearings_distribution
            # <https://github.com/gboeing/osmnx/blob/v2.0.0/osmnx/bearing.py#L240-L296>
            left_closed=True,
        )
        .cast(pl.Int16)
        .alias("bin_index")
    )


def cut_bearing_breakpoints_pl(
    num_bins: int, bin_index_col: str = "bin_index"
) -> list[pl.Expr]:
    """
    Get Polars expressions to calculate bearing bin breakpoints (bin_lower, bin_center, bin_upper)
    for the given number of bins based on the bin index in `bin_index_col`.
    """
    bin_center_expr = pl.col(bin_index_col).sub(1).mul(360 / num_bins)
    return [
        bin_center_expr.sub(180 / num_bins).mod(360).alias("bin_lower"),
        bin_center_expr.alias("bin_center"),
        bin_center_expr.add(180 / num_bins).alias("bin_upper"),
    ]


def get_cut_bearing_bins_df(num_bins: int) -> pl.DataFrame:
    """
    Get a DataFrame with bearing bin indexes and breakpoints.
    Useful since data grouped by bin_index might not have all bins represented.
    """
    return pl.DataFrame(
        data={"bin_index": [x + 1 for x in range(num_bins)]},
        schema={"bin_index": pl.Int16},
    ).with_columns(*cut_bearing_breakpoints_pl(num_bins=num_bins))


def get_bearing_histograms(
    bearings: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
) -> pl.DataFrame:
    """
    Get the bearing distributions of a graph as a pl.DataFrame.
    Returns multiple bearing histograms, one for each value in the the `num_bins` column.
    """
    bins = [2, 4, 8, 16, 32]
    return pl.concat(
        [
            get_bearing_histogram(bearings=bearings, weights=weights, num_bins=num_bins)
            for num_bins in bins
        ],
        how="vertical",
    )


def get_bearing_histogram(
    bearings: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
    num_bins: int,
) -> pl.DataFrame:
    """
    Modified from osmnx.bearing._bearings_distribution to accept non-graph input.
    Source at https://github.com/gboeing/osmnx/blob/v2.0.0/osmnx/bearing.py#L240-L296.
    Compute distribution of bearings across evenly spaced bins.

    Prevents bin-edge effects around common values like 0 degrees and 90
    degrees by initially creating twice as many bins as desired, then merging
    them in pairs. For example, if `num_bins=36` is provided, then each bin
    will represent 10 degrees around the compass, with the first bin
    representing 355 degrees to 5 degrees.
    """
    return (
        pl.DataFrame(data={"bearing": bearings, "weight": weights})
        .with_columns(cut_bearings_pl(num_bins=num_bins))
        .group_by("bin_index")
        .agg(
            bin_count=pl.sum("weight"),
        )
        .join(
            get_cut_bearing_bins_df(num_bins=num_bins),
            on="bin_index",
            how="right",
        )
        .with_columns(bin_count=pl.coalesce("bin_count", 0.0))
        .with_columns(bin_count_total=pl.sum("bin_count").over(pl.lit(True)))
        # nan values are not helpful here when bin_count_total is 0
        # Instead, set bin_proportion to 0, although setting to null could also make sense
        .with_columns(
            bin_proportion=pl.when(pl.col("bin_count_total") > 0)
            .then(pl.col("bin_count").truediv("bin_count_total"))
            .otherwise(0.0)
        )
        .drop("bin_count_total")
        .with_columns(pl.lit(num_bins).alias("num_bins"))
        .with_columns(
            pl.col("bin_center")
            .replace_strict(bearing_labels, default=None)
            .alias("bin_label")
        )
        .select(*SkiAreaBearingDistributionModel.model_fields)
    )


def get_bearing_summary_stats(
    bearings: list[float] | npt.NDArray[np.float64],
    net_magnitudes: list[float] | npt.NDArray[np.float64] | None = None,
    cum_magnitudes: list[float] | npt.NDArray[np.float64] | None = None,
    hemisphere: Literal["north", "south"] | None = None,
) -> BearingStatsModel:
    """
    Compute the mean bearing (i.e. average direction, mean angle)
    and mean bearing strength (i.e. resultant vector length, concentration, magnitude) from a set of bearings,
    with optional strengths and weights.

    bearings:
        An array or list of bearing angles in degrees. These represent directions, headings, or orientations.
    net_magnitudes:
        An array or list of weights (importance factors, influence coefficients, scaling factors) applied to each bearing.
        If None, all weights are assumed to be 1.
        These represent external weighting factors, priorities, or significance levels assigned to each bearing.
    cum_magnitudes:
        An array or list of combined verticals of the each bearing.
        If None, all combined verticals are assumed to be 1.
        These represent the total verticals of all the original group of segments attributing to this bearing.
    hemisphere:
        The hemisphere in which the bearings are located used to calculate poleward affinity.
        If None, poleward affinity is not calculated.

    Notes:
    - The function computes the mean direction by converting bearings to unit vectors (directional cosines and sines),
      scaling them by their strengths (magnitudes), and applying weights during summation.
    - The mean bearing strength is calculated as the magnitude (length, norm) of the resultant vector
      divided by the sum of weighted strengths, providing a normalized measure (ranging from 0 to 1)
      of how tightly the bearings are clustered around the mean direction.
    - The function handles edge cases where the sum of weights is zero,
      returning a mean bearing strength of 0.0 in such scenarios.

    Development chats:
    - https://chatgpt.com/share/6718521f-6768-8011-aed4-db345efb68b7
    - https://chatgpt.com/share/a2648aee-194b-4744-8a81-648d124d17f2
    """
    if net_magnitudes is None:
        net_magnitudes = np.ones_like(bearings, dtype=np.float64)
    if cum_magnitudes is None:
        cum_magnitudes = np.ones_like(bearings, dtype=np.float64)
    bearings = np.array(bearings, dtype=np.float64)
    net_magnitudes = np.array(net_magnitudes, dtype=np.float64)
    cum_magnitudes = np.array(cum_magnitudes, dtype=np.float64)
    assert bearings.shape == net_magnitudes.shape == cum_magnitudes.shape

    # Sum all vectors in their complex number form using weights and bearings
    total_complex = sum(net_magnitudes * np.exp(1j * np.deg2rad(bearings)))
    # Convert the result back to polar coordinates
    cum_magnitude = sum(cum_magnitudes)
    net_magnitude = np.abs(total_complex)
    alignment = net_magnitude / cum_magnitude if cum_magnitude > 1e-10 else 0.0
    mean_bearing_rad = np.angle(total_complex) if net_magnitude > 1e-10 else 0.0
    mean_bearing_deg = np.rad2deg(np.round(mean_bearing_rad, 10)) % 360

    if hemisphere == "north":
        # Northern Hemisphere: poleward is 0 degrees
        poleward_affinity = alignment * np.cos(mean_bearing_rad)
    elif hemisphere == "south":
        # Southern Hemisphere: poleward is 180 degrees
        poleward_affinity = -alignment * np.cos(mean_bearing_rad)
    else:
        poleward_affinity = None
    eastward_affinity = alignment * np.sin(mean_bearing_rad)

    return BearingStatsModel(
        bearing_mean=round(mean_bearing_deg, 7),
        bearing_alignment=round(alignment, 7),
        # plus zero to avoid -0.0 <https://stackoverflow.com/a/74383961/4651668>
        poleward_affinity=(
            round(poleward_affinity + 0, 7) if poleward_affinity is not None else None
        ),
        eastward_affinity=round(eastward_affinity + 0, 7),
        bearing_magnitude_net=round(net_magnitude, 7),
        bearing_magnitude_cum=round(cum_magnitude, 7),
    )


# def get_solar_intensity(
#     latitudes: list[float] | npt.NDArray[np.float64]
# ) -> pl.DataFrame:
    

HOURS_IN_DAY = 24
HOURS_START = 9  # Start time (9 AM)
HOURS_END = 16   # End time (4 PM)

def solar_declination(day_of_year: int) -> float:
    """
    Calculate the solar declination for a given day of the year.
    This uses the approximation formula for solar declination.
    """
    # Calculate the solar declination using the approximation formula
    # δ = 23.44 * sin(360° * (N + 10) / 365)
    # Where N is the day of the year (1 to 365)
    return 23.44 * np.sin(np.radians(360 * (day_of_year + 10) / 365))

def solar_intensity_at_time(latitude: float, declination: float, hour: int) -> float:
    """
    Calculate the solar intensity at a given time of day (hour) for a given latitude and solar declination.
    """
    # Convert latitude and declination to radians for trigonometric calculations
    latitude_rad = np.radians(latitude)
    declination_rad = np.radians(declination)

    # Hour angle (ω) for a given time of day (simple model)
    # Hour angle increases by 15 degrees per hour, starting from 12:00 PM.
    hour_angle = np.radians(15 * (hour - 12))

    # Calculate the solar zenith angle using the formula
    # cos(θ) = sin(δ) * sin(φ) + cos(δ) * cos(φ) * cos(ω)
    # Where:
    # δ = solar declination, φ = latitude, ω = hour angle
    solar_zenith = np.arccos(
        np.sin(declination_rad) * np.sin(latitude_rad) +
        np.cos(declination_rad) * np.cos(latitude_rad) * np.cos(hour_angle)
    )

    intensity = np.cos(solar_zenith)
    return max(intensity, 0)

def integrate_solar_intensity(latitude: float, day_of_year: int) -> float:
    """
    Integrate solar intensity from 9 AM to 4 PM.
    The function returns the average solar intensity over this period.
    """
    declination = solar_declination(day_of_year)
    intensities = [solar_intensity_at_time(latitude, declination, hour) for hour in range(HOURS_START, HOURS_END)]
    
    return np.mean(intensities)

def get_hemisphere(latitude: float) -> Literal["north", "south"] | None:
    if latitude is None:
        return None
    elif latitude >= 0:
        return "north"
    elif latitude < 0:
        return "south"
    

def get_solar_intensity(
    latitude: float
) -> float:
    """
    Calculate the average solar intensity from 9 AM to 4 PM for each latitude.
    
    Args:
        latitudes (list or ndarray): A list or numpy array of latitudes.
        hemisphere: lets us know if we are in the northern or southern hemisphere
    
    Returns:
        polars.DataFrame: A DataFrame with latitudes and their corresponding average solar intensity.
    """
    # For each latitude, calculate the solar intensity (assuming a fixed day of the year, e.g., 172nd day for mid-winter in southern hemisphere)
    # and the 355th day for mid-winter in northern hemisphere
    import logging
    logging.info(f"latitude is {latitude}")
    hemisphere: Literal["north", "south"] | None = None
    hemisphere = get_hemisphere(latitude)
    if hemisphere == "north":
        day_of_year = 355 # Mid-winter day (e.g., December 21)
    elif hemisphere == "south":
        day_of_year = 172 # Mid-winter day (e.g., June 21)
    else:
        logging.info(f"hemisphere has value: {hemisphere}")
        return None

    # TODO we could dynamically use the open and close dates of each resort
    avg_intensity = integrate_solar_intensity(latitude, day_of_year)
    return avg_intensity

def get_solar_intensity_batch(latitudes: pl.Series) -> pl.Series:
    return pl.Series([get_solar_intensity(latitude) for latitude in latitudes])
