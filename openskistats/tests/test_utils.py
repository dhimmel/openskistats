import polars as pl
import pytest
from polars.testing import assert_frame_equal

from openskistats.utils import gini_coefficient, pl_flip_bearing, pl_weighted_mean


def test_pl_flip_bearing() -> None:
    expected_df = pl.DataFrame(
        [
            {"latitude": 10, "bearing": 0, "bearing_poleward": 0},
            {"latitude": 10, "bearing": 180, "bearing_poleward": 180},
            {"latitude": 10, "bearing": 45, "bearing_poleward": 45},
            {"latitude": -10, "bearing": 0, "bearing_poleward": 180},
            {"latitude": -10, "bearing": 5, "bearing_poleward": 175},
            {"latitude": -10, "bearing": 45, "bearing_poleward": 135},
            {"latitude": -10, "bearing": 90, "bearing_poleward": 90},
            {"latitude": -10, "bearing": 95, "bearing_poleward": 85},
            {"latitude": -10, "bearing": 180, "bearing_poleward": 0},
            {"latitude": -10, "bearing": 265, "bearing_poleward": 275},
            {"latitude": -10, "bearing": 270, "bearing_poleward": 270},
            {"latitude": -10, "bearing": 360, "bearing_poleward": 180},
        ]
    )
    output_df = expected_df.with_columns(bearing_poleward=pl_flip_bearing())
    assert_frame_equal(output_df, expected_df)


def test_pl_weighted_mean() -> None:
    df = pl.DataFrame(
        {
            "value": [0.0, 1.0, 2.0, 3.0, None, None],
            "weight": [0.0, 1.0, 3.0, None, 4.0, None],
        }
    )
    wtd_mean = df.select(pl_weighted_mean("value", "weight")).item(0, 0)
    assert wtd_mean == 1.75


@pytest.mark.parametrize(
    "values,expected",
    [
        pytest.param([5.0, 5.0, 5.0, 5.0], 0.0, id="equal_values"),
        pytest.param([100.0, 0.0, 0.0, 0.0], 0.75, id="max_inequality"),
        pytest.param([1.0, 2.0, 3.0, 4.0], 0.25, id="intermediate_distribution"),
    ],
)
def test_gini_coefficient(values: list[float], expected: float) -> None:
    assert gini_coefficient(values) == pytest.approx(expected)
