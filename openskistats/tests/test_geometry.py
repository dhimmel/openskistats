"""Tests for geometric operations on longitude-latitude coordinates."""

import math

import polars as pl
import pytest
from osmnx.distance import EARTH_RADIUS_M
from polars.testing import assert_frame_equal

from openskistats.geometry import simplify_coordinates, simplify_segments


@pytest.mark.parametrize(
    ("coordinates_meters", "tolerance_meters", "expected_indices"),
    [
        pytest.param(
            [(0.0, 0.0), (1.0, 0.1), (2.0, 0.0)],
            0.11,
            [0, 2],
            id="near-line-removes-middle",
        ),
        pytest.param(
            [(0.0, 0.0), (1.0, 1.0), (2.0, 0.0)],
            0.9,
            [0, 1, 2],
            id="bend-keeps-middle",
        ),
        pytest.param(
            [(0.0, 0.0), (1.0, 1.0)],
            1.0,
            [0, 1],
            id="endpoints-always-kept",
        ),
    ],
)
def test_simplify_coordinates(
    coordinates_meters: list[tuple[float, float]],
    tolerance_meters: float,
    expected_indices: list[int],
) -> None:
    degrees_per_meter = 180 / (math.pi * EARTH_RADIUS_M)
    coordinates = [
        (x * degrees_per_meter, y * degrees_per_meter) for x, y in coordinates_meters
    ]
    assert simplify_coordinates(coordinates, tolerance_meters) == [
        coordinates[index] for index in expected_indices
    ]


def test_simplify_segments_preserves_boundaries() -> None:
    segments = pl.DataFrame(
        {
            "run_id": ["run-a", "run-a", "run-a", "run-b"],
            "longitude": [0.0, 1.0, 2.0, 2.0],
            "latitude": [0.0, 0.0, 0.0, 0.0],
            "longitude_end": [1.0, 2.0, 3.0, 3.0],
            "latitude_end": [0.0, 0.0, 0.0, 0.0],
            "highlight": [False, False, True, False],
        }
    )
    expected = pl.DataFrame(
        {
            "run_id": ["run-a", "run-a", "run-b"],
            "highlight": [False, True, False],
            "longitude": [0.0, 2.0, 2.0],
            "latitude": [0.0, 0.0, 0.0],
            "longitude_end": [2.0, 3.0, 3.0],
            "latitude_end": [0.0, 0.0, 0.0],
        }
    )
    simplified = simplify_segments(
        segments=segments,
        group_columns=["run_id", "highlight"],
        tolerance_meters=0,
    )
    assert_frame_equal(simplified, expected)
