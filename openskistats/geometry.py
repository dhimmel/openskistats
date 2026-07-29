"""Geometric operations on longitude-latitude coordinates and segments."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np
import polars as pl
from osmnx.distance import EARTH_RADIUS_M
from shapely import LineString


@dataclass(frozen=True)
class GeographicBounds:
    """Fixed longitude and latitude bounds for a map canvas."""

    west: float
    east: float
    south: float
    north: float
    crs: str = "EPSG:4326"

    def local_data_aspect(self) -> float:
        """Return the latitude-to-longitude display scale at the map midpoint."""
        midpoint_latitude = (self.south + self.north) / 2
        return 1 / math.cos(math.radians(midpoint_latitude))

    def height_for_width(self, width: float) -> float:
        """Return the canvas height that preserves local geographic proportions."""
        longitude_span = self.east - self.west
        latitude_span = self.north - self.south
        geographic_width_to_height = longitude_span / (
            self.local_data_aspect() * latitude_span
        )
        return width / geographic_width_to_height

    def metadata_description(self) -> str:
        """Describe the coordinate reference system and bounding box."""
        return (
            f"{self.crs} bounds: "
            f"west={self.west}, east={self.east}, "
            f"south={self.south}, north={self.north}."
        )


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


def clip_segment_to_bounds(
    start: tuple[float, float],
    end: tuple[float, float],
    bounds: GeographicBounds,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Clip one line segment to a rectangular geographic extent."""
    x_start, y_start = start
    delta_x = end[0] - x_start
    delta_y = end[1] - y_start
    minimum_fraction = 0.0
    maximum_fraction = 1.0
    for direction, distance in (
        (-delta_x, x_start - bounds.west),
        (delta_x, bounds.east - x_start),
        (-delta_y, y_start - bounds.south),
        (delta_y, bounds.north - y_start),
    ):
        if direction == 0:
            if distance < 0:
                return None
            continue
        fraction = distance / direction
        if direction < 0:
            minimum_fraction = max(minimum_fraction, fraction)
        else:
            maximum_fraction = min(maximum_fraction, fraction)
        if minimum_fraction > maximum_fraction:
            return None
    return (
        (
            x_start + minimum_fraction * delta_x,
            y_start + minimum_fraction * delta_y,
        ),
        (
            x_start + maximum_fraction * delta_x,
            y_start + maximum_fraction * delta_y,
        ),
    )


def clip_polyline_to_bounds(
    vertices: np.ndarray[Any, np.dtype[np.float64]],
    bounds: GeographicBounds,
) -> list[list[list[float]]]:
    """Clip a polyline and return its nonempty pieces as GeoJSON coordinates."""
    pieces: list[list[list[float]]] = []
    current_piece: list[list[float]] = []
    for start_array, end_array in zip(vertices[:-1], vertices[1:], strict=True):
        clipped = clip_segment_to_bounds(
            start=(float(start_array[0]), float(start_array[1])),
            end=(float(end_array[0]), float(end_array[1])),
            bounds=bounds,
        )
        if clipped is None:
            if current_piece:
                pieces.append(current_piece)
                current_piece = []
            continue
        clipped_start = [round(value, 7) for value in clipped[0]]
        clipped_end = [round(value, 7) for value in clipped[1]]
        if current_piece and current_piece[-1] == clipped_start:
            current_piece.append(clipped_end)
        else:
            if current_piece:
                pieces.append(current_piece)
            current_piece = [clipped_start, clipped_end]
    if current_piece:
        pieces.append(current_piece)
    return pieces
