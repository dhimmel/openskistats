from typing import Literal

import numpy as np
import numpy.typing as npt
import polars as pl
from osmnx.bearing import calculate_bearing
from osmnx.distance import great_circle

from openskistats.models import (
    BearingStatsModel,
    RunDifficultyConvention,
    SkiAreaBearingDistributionModel,
    SkiRunDifficulty,
)


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
            distance_vertical=pl.col("elevation_lag").sub("elevation"),
            _coord_struct=pl.struct(
                "latitude_lag",
                "longitude_lag",
                "latitude",
                "longitude",
            ),
        )
        .with_columns(
            # NOTE: Polars only guarantees hash stability within a single polars version.
            segment_hash=pl.when(pl.col("latitude_lag").is_not_null()).then(
                pl.col("_coord_struct").hash(seed=0)
            ),
            distance_vertical_drop=pl.col("distance_vertical").clip(lower_bound=0),
            # distance_vertical_gain=pl.col("distance_vertical").mul(-1).clip(lower_bound=0),
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
        # reduce precision to save on storage
        .cast(
            {
                "distance_vertical": pl.Float32,
                "distance_vertical_drop": pl.Float32,
                # "distance_vertical_gain": pl.Float32,
                "distance_horizontal": pl.Float32,
                "distance_3d": pl.Float32,
                "bearing": pl.Float32,
                "gradient": pl.Float32,
                "slope": pl.Float32,
            }
        )
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
    difficulties: npt.NDArray[np.str_],
) -> pl.DataFrame:
    """
    Get the bearing distributions of a graph as a pl.DataFrame.
    Returns multiple bearing histograms, one for each value in the the `num_bins` column.
    """
    bins = [2, 4, 8, 16, 32]
    return pl.concat(
        [
            get_bearing_histogram(
                bearings=bearings,
                weights=weights,
                num_bins=num_bins,
                difficulties=difficulties,
            )
            for num_bins in bins
        ],
        how="vertical",
    )


def get_bearing_histogram(
    bearings: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
    difficulties: npt.NDArray[np.str_],
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
        pl.DataFrame(
            data={
                "bearing": bearings,
                "weight": weights,
                "run_difficulty_condensed": difficulties,
            }
        )
        .with_columns(cut_bearings_pl(num_bins=num_bins))
        .group_by("bin_index")
        .agg(
            pl.sum("weight").alias("bin_count"),
            *[
                pl.when(pl.col("run_difficulty_condensed") == diff)
                .then("weight")
                .sum()
                .alias(f"bin_count_{diff.name}")
                for diff in SkiRunDifficulty.condensed_values()
            ],
        )
        .join(
            get_cut_bearing_bins_df(num_bins=num_bins),
            on="bin_index",
            how="right",
        )
        .with_columns(pl.selectors.starts_with("bin_count").fill_null(0.0))
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


def get_difficulty_color_to_bearing_bin_counts(
    bearing_pl: pl.DataFrame,
    convention: RunDifficultyConvention = RunDifficultyConvention.north_america,
) -> dict[str, pl.Series]:
    return (  # type: ignore [no-any-return]
        bearing_pl.select(
            "bin_count_other",
            "bin_count_easy",
            "bin_count_intermediate",
            "bin_count_advanced",
        )
        .select(
            pl.all().name.map(
                lambda x: SkiRunDifficulty(x.removeprefix("bin_count_")).color(
                    subtle=True,
                    convention=convention,
                )
            )
        )
        .to_dict()
    )


def get_bearing_summary_stats(
    bearings: list[float] | npt.NDArray[np.float64],
    cum_magnitudes: list[float] | npt.NDArray[np.float64] | None = None,
    alignments: list[float] | npt.NDArray[np.float64] | None = None,
    hemisphere: Literal["north", "south"] | None = None,
) -> BearingStatsModel:
    """
    Compute the mean bearing (i.e. average direction, mean angle)
    and mean bearing strength (i.e. resultant vector length, concentration, magnitude) from a set of bearings,
    with optional strengths and weights.

    bearings:
        An array or list of bearing angles in degrees. These represent directions, headings, or orientations.
    cum_magnitudes:
        An array or list of combined verticals of the each bearing.
        If None, all combined verticals are assumed to be 1.
        These represent the total verticals of all the original group of segments attributing to this bearing.
    alignments:
        An array or list of alignments between 0 and 1 useful for repeated bearing summarization.
        Defaults to 100% alignment (1.0) for all bearings when not specified.
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
    if cum_magnitudes is None:
        cum_magnitudes = np.ones_like(bearings, dtype=np.float64)
    if alignments is None:
        alignments = np.ones_like(bearings, dtype=np.float64)
    bearings = np.array(bearings, dtype=np.float64)
    cum_magnitudes = np.array(cum_magnitudes, dtype=np.float64)
    alignments = np.array(alignments, dtype=np.float64)
    assert bearings.shape == cum_magnitudes.shape == alignments.shape
    net_magnitudes = cum_magnitudes * alignments

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
