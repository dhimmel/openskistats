"""Geometric operations on longitude-latitude coordinates and segments."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np
import polars as pl
from osmnx.distance import EARTH_RADIUS_M
from shapely import LineString, clip_by_rect, get_parts


@dataclass(frozen=True)
class MetersPerDegree:
    """Local lengths in meters of one degree of longitude and latitude."""

    longitude: float
    latitude: float


def meters_per_degree(latitude: float) -> MetersPerDegree:
    """Spherical lengths in meters of one degree at the given latitude."""
    per_degree_latitude = math.pi * EARTH_RADIUS_M / 180
    return MetersPerDegree(
        longitude=per_degree_latitude * math.cos(math.radians(latitude)),
        latitude=per_degree_latitude,
    )


@dataclass(frozen=True)
class GeographicBounds:
    """Fixed longitude and latitude bounds for a map canvas."""

    west: float
    east: float
    south: float
    north: float
    crs: str = "EPSG:4326"

    @property
    def midpoint_latitude(self) -> float:
        return (self.south + self.north) / 2

    def local_data_aspect(self) -> float:
        """Return the latitude-to-longitude display scale at the map midpoint."""
        scale = meters_per_degree(self.midpoint_latitude)
        return scale.latitude / scale.longitude

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
    scale = meters_per_degree(midpoint_latitude)
    projected = [
        (
            (longitude - origin_longitude) * scale.longitude,
            (latitude - origin_latitude) * scale.latitude,
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


def clip_polyline_to_bounds(
    vertices: np.ndarray[Any, np.dtype[np.float64]],
    bounds: GeographicBounds,
) -> list[list[list[float]]]:
    """
    Clip a polyline to a rectangular geographic extent,
    returning its nonempty pieces as GeoJSON MultiLineString coordinates
    rounded to 7 decimal places.
    """
    if len(vertices) < 2:
        return []
    clipped = clip_by_rect(
        LineString(vertices), bounds.west, bounds.south, bounds.east, bounds.north
    )
    return [
        [[round(x, 7), round(y, 7)] for x, y in piece.coords]
        for piece in get_parts(clipped)
        if isinstance(piece, LineString) and not piece.is_empty
    ]
