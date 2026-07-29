"""Tests for geometric operations on longitude-latitude coordinates."""

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from openskistats.geometry import (
    GeographicBounds,
    clip_polyline_to_bounds,
    meters_per_degree,
    simplify_coordinates,
    simplify_segments,
)


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
    degrees_per_meter = 1 / meters_per_degree(latitude=0.0).latitude
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


def test_meters_per_degree() -> None:
    equator = meters_per_degree(latitude=0.0)
    assert equator.longitude == equator.latitude
    assert equator.latitude == pytest.approx(111_195, rel=1e-3)
    subarctic = meters_per_degree(latitude=60.0)
    assert subarctic.longitude == pytest.approx(equator.longitude / 2)
    assert subarctic.latitude == equator.latitude


@pytest.mark.parametrize(
    ("vertices", "expected_pieces"),
    [
        pytest.param(
            [(-5.0, 5.0), (5.0, 5.0), (15.0, 5.0)],
            [[[0.0, 5.0], [5.0, 5.0], [10.0, 5.0]]],
            id="crossing-line-clipped-at-both-edges",
        ),
        pytest.param(
            [(-5.0, 5.0), (5.0, 5.0), (5.0, -5.0), (8.0, -5.0), (8.0, 5.0)],
            [[[0.0, 5.0], [5.0, 5.0], [5.0, 0.0]], [[8.0, 0.0], [8.0, 5.0]]],
            id="exit-and-reentry-splits-pieces",
        ),
        pytest.param(
            [(2.0, 2.0), (3.0, 7.0)],
            [[[2.0, 2.0], [3.0, 7.0]]],
            id="fully-inside-is-unchanged",
        ),
        pytest.param(
            [(-5.0, -5.0), (-1.0, -5.0)],
            [],
            id="fully-outside-is-empty",
        ),
        pytest.param(
            [(3.0, 3.0)],
            [],
            id="single-vertex-is-empty",
        ),
    ],
)
def test_clip_polyline_to_bounds(
    vertices: list[tuple[float, float]],
    expected_pieces: list[list[list[float]]],
) -> None:
    bounds = GeographicBounds(west=0.0, east=10.0, south=0.0, north=10.0)
    pieces = clip_polyline_to_bounds(
        vertices=np.asarray(vertices, dtype=np.float64),
        bounds=bounds,
    )
    assert pieces == expected_pieces
