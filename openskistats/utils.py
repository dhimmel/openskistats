import os
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
import polars as pl

from openskistats.models import SkiRunDifficulty


def get_repo_directory() -> Path:
    return Path(__file__).parent.parent


def get_data_directory(testing: bool = False) -> Path:
    directory = (
        Path(__file__).parent.joinpath("tests", "data")
        if testing or running_in_test()
        else get_repo_directory().joinpath("data")
    )
    directory.mkdir(exist_ok=True)
    return directory


def get_images_directory() -> Path:
    """Directory for saving generated images."""
    directory = get_data_directory().joinpath("images")
    directory.mkdir(exist_ok=True)
    return directory


def get_images_data_directory() -> Path:
    """Directory for materializing data behind generated images."""
    directory = get_images_directory().joinpath("data")
    directory.mkdir(exist_ok=True)
    return directory


def get_website_source_directory() -> Path:
    return get_repo_directory().joinpath("website")


def running_in_test() -> bool:
    return "PYTEST_CURRENT_TEST" in os.environ


def running_in_ci() -> bool:
    return os.environ.get("GITHUB_ACTIONS") == "true"


def get_hemisphere(latitude: float) -> Literal["north", "south"]:
    if latitude > 0:
        return "north"
    if latitude < 0:
        return "south"
    raise ValueError(
        f"{latitude=} must be non-zero as equator is not supported value by downstream applications."
    )


def pl_hemisphere(latitude_col: str = "latitude") -> pl.Expr:
    return (
        pl.when(pl.col(latitude_col).gt(0))
        .then(pl.lit("north"))
        .when(pl.col(latitude_col).lt(0))
        .then(pl.lit("south"))
    )


def pl_flip_bearing(
    latitude_col: str = "latitude", bearing_col: str = "bearing"
) -> pl.Expr:
    """
    Flip bearings in the southern hemisphere across the hemisphere (east-west axis).
    Performs a latitudinal reflection or hemispherical flip of bearings in the southern hemisphere,
    while returning the original bearings in the northern hemisphere.
    """
    return (
        pl.when(pl.col(latitude_col).gt(0))
        .then(pl.col(bearing_col))
        .otherwise(pl.lit(180).sub(bearing_col).mod(360))
    )


def pl_weighted_mean(value_col: str, weight_col: str) -> pl.Expr:
    """
    Generate a Polars aggregation expression to take a weighted mean
    https://github.com/pola-rs/polars/issues/7499#issuecomment-2569748864
    """
    values = pl.col(value_col)
    weights = pl.when(values.is_not_null()).then(weight_col)
    return weights.dot(values).truediv(weights.sum()).fill_nan(None)


def pl_condense_run_difficulty(run_difficulty_col: str = "run_difficulty") -> pl.Expr:
    return (
        pl.col(run_difficulty_col)
        .fill_null(SkiRunDifficulty.other)
        .replace_strict(SkiRunDifficulty.condense())
        .cast(pl.Enum(SkiRunDifficulty))
        .alias("run_difficulty_condensed")
    )


def gini_coefficient(values: npt.NDArray[np.float64] | list[float]) -> float:
    """Compute the Gini coefficient of a list of values."""
    n = len(values)
    cumsum = np.cumsum(sorted(values))
    return float(1 - 2 * (cumsum.sum() / (n * cumsum[-1])) + n**-1)


def oxford_join(
    strings: list[str], sep: str = ", ", final_sep_extra: str = "and "
) -> str:
    """
    Join a list of strings with a separator,
    using a final separator for the last pair to reflect an Oxford comma style.
    """
    if not strings:
        return ""
    *head, final = strings
    if not head:
        return final
    if len(head) == 1:
        return f"{head[0]} {final_sep_extra}{final}"
    return f"{sep.join(head)}{sep}{final_sep_extra}{final}"


def get_request_headers() -> dict[str, str]:
    return {
        "From": "https://github.com/dhimmel/openskistats",
    }
