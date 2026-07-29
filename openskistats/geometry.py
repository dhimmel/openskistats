"""Geometric operations on longitude-latitude coordinates and segments."""

import math
from collections.abc import Sequence
from itertools import pairwise

import polars as pl
from osmnx.distance import EARTH_RADIUS_M
from shapely import LineString


def simplify_coordinates(
    coordinates: list[tuple[float, float]],
    tolerance_meters: float,
) -> list[tuple[float, float]]:
    """
    Simplify longitude-latitude coordinates with the Douglas-Peucker algorithm,
    using a local equirectangular projection so the tolerance is in meters.
    Retained coordinates keep their exact input values.
    """
    if not coordinates:
        return []
    origin_longitude, origin_latitude = coordinates[0]
    midpoint_latitude = sum(latitude for _, latitude in coordinates) / len(coordinates)
    meters_per_degree_latitude = math.pi * EARTH_RADIUS_M / 180
    meters_per_degree_longitude = meters_per_degree_latitude * math.cos(
        math.radians(midpoint_latitude)
    )
    projected = [
        (
            (longitude - origin_longitude) * meters_per_degree_longitude,
            (latitude - origin_latitude) * meters_per_degree_latitude,
        )
        for longitude, latitude in coordinates
    ]
    simplified = LineString(projected).simplify(
        tolerance=tolerance_meters,
        preserve_topology=False,
    )
    projected_to_geographic = dict(zip(projected, coordinates, strict=True))
    return [projected_to_geographic[(float(x), float(y))] for x, y in simplified.coords]


def simplify_segments(
    segments: pl.DataFrame,
    group_columns: Sequence[str],
    tolerance_meters: float,
) -> pl.DataFrame:
    """
    Simplify contiguous sequences of segments defined by
    `longitude`, `latitude`, `longitude_end`, and `latitude_end` columns.
    Sequences break when any group column changes value
    or when a segment does not start where the previous segment ended,
    such that sequence boundary coordinates are always retained.
    Returns the group columns plus the four coordinate columns;
    other columns are dropped since simplification merges segments.
    """
    coordinate_columns = ["longitude", "latitude", "longitude_end", "latitude_end"]
    sequence_break = pl.any_horizontal(
        *(pl.col(column) != pl.col(column).shift() for column in group_columns),
        pl.col("longitude") != pl.col("longitude_end").shift(),
        pl.col("latitude") != pl.col("latitude_end").shift(),
    )
    rows = []
    for _, sequence in segments.with_columns(
        _sequence_id=sequence_break.fill_null(True).cum_sum()
    ).group_by("_sequence_id", maintain_order=True):
        coordinates = [
            (sequence["longitude"][0], sequence["latitude"][0]),
            *zip(sequence["longitude_end"], sequence["latitude_end"], strict=True),
        ]
        simplified = simplify_coordinates(
            coordinates=coordinates,
            tolerance_meters=tolerance_meters,
        )
        groups = {column: sequence[column][0] for column in group_columns}
        for start, end in pairwise(simplified):
            rows.append(
                groups
                | {
                    "longitude": start[0],
                    "latitude": start[1],
                    "longitude_end": end[0],
                    "latitude_end": end[1],
                }
            )
    schema = {
        column: segments.schema[column]
        for column in [*group_columns, *coordinate_columns]
    }
    return pl.DataFrame(rows, schema=schema)
