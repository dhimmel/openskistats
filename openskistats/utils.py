import os
from pathlib import Path

import numpy as np
import numpy.typing as npt
import polars as pl
import subprocess

def get_font_path(font_name: str, keywords: list[str] = None) -> Path | None:
    try:
        result = subprocess.check_output(f"fc-list | grep '{font_name}'", shell=True, stderr=subprocess.PIPE)
        font_list = result.decode('utf-8').strip().splitlines()
        
        for line in font_list:
            keywords_all_match_line = True
            if keywords is not None:
                keywords_match_line: list[bool] = [keyword in line for keyword in keywords]
                keywords_all_match_line: bool = False in keywords_match_line
            if 'Library/Fonts' in line and keywords_all_match_line:
                parts = line.split(":")
                if len(parts) > 1:
                    return Path(parts[0].strip())
        
        return None
    except subprocess.CalledProcessError as e:
        print(f"Error searching for fonts: {e}")
        return None

def get_repo_directory() -> Path:
    return Path(__file__).parent.parent


def get_data_directory(testing: bool = False) -> Path:
    directory = (
        Path(__file__).parent.joinpath("tests", "data")
        if testing or "PYTEST_CURRENT_TEST" in os.environ
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


def gini_coefficient(values: npt.NDArray[np.float64] | list[float]) -> float:
    """Compute the Gini coefficient of a list of values."""
    n = len(values)
    cumsum = np.cumsum(sorted(values))
    return float(1 - 2 * (cumsum.sum() / (n * cumsum[-1])) + n**-1)
