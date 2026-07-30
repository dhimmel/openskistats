"""Static geographic context for the Dartmouth Skiway figure."""

from __future__ import annotations

import io
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import requests

from openskistats.geometry import (
    GeographicBounds,
    clip_polyline_to_bounds,
    meters_per_degree,
)
from openskistats.utils import get_repo_directory

if TYPE_CHECKING:
    from matplotlib.path import Path as MatplotlibPath

DARTMOUTH_CONTEXT_PATH = Path(__file__).parent.joinpath(
    "data", "dartmouth_skiway_context.geojson"
)
DARTMOUTH_CONTOURS_PATH = Path(__file__).parent.joinpath(
    "data", "dartmouth_skiway_contours.geojson"
)
MCLANE_FAMILY_LODGE_OSM_ID = 296382919
DARTMOUTH_SKIWAY_PARKING_LOT_OSM_ID = 602788499
DARTMOUTH_SKIWAY_SERVICE_ROAD_OSM_ID = 1147052531
APPALACHIAN_TRAIL_OSM_ID = 18319298
GRAFTON_TURNPIKE_OSM_WAY_IDS = (328497225, 532980281, 532980282, 1431999174)
DARTMOUTH_SKIWAY_WATER_OSM_WAY_IDS = (1144943274, 1475644343)
DARTMOUTH_CONTEXT_OSM_WAY_IDS = (
    MCLANE_FAMILY_LODGE_OSM_ID,
    DARTMOUTH_SKIWAY_PARKING_LOT_OSM_ID,
    DARTMOUTH_SKIWAY_SERVICE_ROAD_OSM_ID,
    *GRAFTON_TURNPIKE_OSM_WAY_IDS,
    *DARTMOUTH_SKIWAY_WATER_OSM_WAY_IDS,
)
DARTMOUTH_CONTEXT_OSM_RELATION_IDS = (APPALACHIAN_TRAIL_OSM_ID,)
USER_AGENT = "openskistats/0.1 (https://github.com/dhimmel/openskistats)"
DARTMOUTH_ELEVATION_SERVICE_URL = (
    "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer"
)
DARTMOUTH_CONTOUR_INTERVAL_METERS = 20
DARTMOUTH_INDEX_CONTOUR_INTERVAL_METERS = 100
DARTMOUTH_DEM_PIXEL_SIZE_METERS = 5


SKIWAY_MAP_BOUNDS = GeographicBounds(
    west=-72.1075,
    east=-72.0857,
    south=43.7774,
    north=43.7908,
)
"""Editable fixed map extent, stored in WGS 84 longitude and latitude."""


def _osm_way_to_geojson_feature(osm_data: dict[str, Any]) -> dict[str, Any]:
    """Convert one OSM API `way/full` response to a compact GeoJSON feature."""
    way = next(element for element in osm_data["elements"] if element["type"] == "way")
    nodes = {
        element["id"]: element
        for element in osm_data["elements"]
        if element["type"] == "node"
    }
    osm_id = way["id"]
    tags = way["tags"]
    coordinates = [
        [nodes[node_id]["lon"], nodes[node_id]["lat"]] for node_id in way["nodes"]
    ]
    if osm_id == MCLANE_FAMILY_LODGE_OSM_ID:
        feature_kind = "lodge"
    elif osm_id == DARTMOUTH_SKIWAY_PARKING_LOT_OSM_ID:
        feature_kind = "parking"
    elif osm_id == DARTMOUTH_SKIWAY_SERVICE_ROAD_OSM_ID:
        feature_kind = "parking_road"
    elif osm_id in DARTMOUTH_SKIWAY_WATER_OSM_WAY_IDS:
        feature_kind = "water"
    else:
        feature_kind = "road"
    is_polygon = feature_kind in {"lodge", "parking", "water"}
    geometry_type = "Polygon" if is_polygon else "LineString"
    geometry_coordinates = [coordinates] if is_polygon else coordinates
    return {
        "type": "Feature",
        "id": f"way/{osm_id}",
        "properties": {
            "feature_kind": feature_kind,
            "osm_id": osm_id,
            "osm_version": way["version"],
            "osm_timestamp": way["timestamp"],
            "name": tags.get("name"),
            "amenity": tags.get("amenity"),
            "building": tags.get("building"),
            "highway": tags.get("highway"),
            "intermittent": tags.get("intermittent"),
            "natural": tags.get("natural"),
            "parking": tags.get("parking"),
            "surface": tags.get("surface"),
            "water": tags.get("water"),
            "source": f"https://www.openstreetmap.org/way/{osm_id}",
        },
        "geometry": {
            "type": geometry_type,
            "coordinates": geometry_coordinates,
        },
    }


def _osm_relation_to_geojson_feature(
    osm_data: dict[str, Any],
    bounds: GeographicBounds,
) -> dict[str, Any]:
    """Convert one OSM route relation to a clipped GeoJSON feature."""
    relation = next(
        element for element in osm_data["elements"] if element["type"] == "relation"
    )
    ways = {
        element["id"]: element
        for element in osm_data["elements"]
        if element["type"] == "way"
    }
    nodes = {
        element["id"]: element
        for element in osm_data["elements"]
        if element["type"] == "node"
    }
    coordinates = []
    for member in relation["members"]:
        if member["type"] != "way":
            continue
        way = ways[member["ref"]]
        vertices = np.asarray(
            [
                [nodes[node_id]["lon"], nodes[node_id]["lat"]]
                for node_id in way["nodes"]
            ],
            dtype=np.float64,
        )
        coordinates.extend(clip_polyline_to_bounds(vertices=vertices, bounds=bounds))
    tags = relation["tags"]
    osm_id = relation["id"]
    return {
        "type": "Feature",
        "id": f"relation/{osm_id}",
        "properties": {
            "feature_kind": "trail",
            "osm_id": osm_id,
            "osm_version": relation["version"],
            "osm_timestamp": relation["timestamp"],
            "name": tags.get("name"),
            "network": tags.get("network"),
            "ref": tags.get("ref"),
            "route": tags.get("route"),
            "source": f"https://www.openstreetmap.org/relation/{osm_id}",
        },
        "geometry": {
            "type": "MultiLineString",
            "coordinates": coordinates,
        },
    }


def download_dartmouth_skiway_context() -> Path:
    """
    Refresh the committed OpenStreetMap context for the Dartmouth Skiway figure.

    This command is intended to be rerun infrequently and always manually.
    Plot generation reads the committed snapshot and never makes a network request.
    """
    features = []
    for osm_id in DARTMOUTH_CONTEXT_OSM_WAY_IDS:
        response = requests.get(
            f"https://api.openstreetmap.org/api/0.6/way/{osm_id}/full.json",
            headers={"User-Agent": USER_AGENT},
            timeout=60,
        )
        response.raise_for_status()
        features.append(_osm_way_to_geojson_feature(response.json()))
    for osm_id in DARTMOUTH_CONTEXT_OSM_RELATION_IDS:
        response = requests.get(
            f"https://api.openstreetmap.org/api/0.6/relation/{osm_id}/full.json",
            headers={"User-Agent": USER_AGENT},
            timeout=60,
        )
        response.raise_for_status()
        features.append(
            _osm_relation_to_geojson_feature(
                osm_data=response.json(),
                bounds=SKIWAY_MAP_BOUNDS,
            )
        )
    features.sort(
        key=lambda feature: (
            feature["properties"]["feature_kind"],
            feature["properties"]["osm_id"],
        )
    )
    feature_collection = {
        "type": "FeatureCollection",
        "properties": {
            "source": "https://www.openstreetmap.org",
            "copyright": "OpenStreetMap contributors",
            "license": "https://opendatacommons.org/licenses/odbl/1-0/",
            "latest_osm_feature_timestamp": max(
                feature["properties"]["osm_timestamp"] for feature in features
            ),
            "refresh_command": "pixi run openskistats download_dartmouth_context",
        },
        "features": features,
    }
    DARTMOUTH_CONTEXT_PATH.parent.mkdir(exist_ok=True)
    DARTMOUTH_CONTEXT_PATH.write_text(json.dumps(feature_collection, indent=2) + "\n")
    return DARTMOUTH_CONTEXT_PATH.relative_to(get_repo_directory())


def load_dartmouth_skiway_context() -> dict[str, Any]:
    """Load the committed OpenStreetMap context without network access."""
    return cast(dict[str, Any], json.loads(DARTMOUTH_CONTEXT_PATH.read_text()))


def _iter_contour_path_polylines(
    path: MatplotlibPath,
) -> list[np.ndarray[Any, np.dtype[np.float64]]]:
    """Split a compound Matplotlib contour path into disconnected polylines."""
    from matplotlib.path import Path as MatplotlibPath

    polylines = []
    current_vertices: list[tuple[float, float]] = []
    for vertices, code in path.iter_segments(simplify=False, curves=False):
        point = (float(vertices[-2]), float(vertices[-1]))
        if code == MatplotlibPath.MOVETO:
            if current_vertices:
                polylines.append(np.asarray(current_vertices, dtype=np.float64))
            current_vertices = [point]
        elif code == MatplotlibPath.LINETO:
            current_vertices.append(point)
        elif code == MatplotlibPath.CLOSEPOLY:
            current_vertices.append(current_vertices[0])
            polylines.append(np.asarray(current_vertices, dtype=np.float64))
            current_vertices = []
    if current_vertices:
        polylines.append(np.asarray(current_vertices, dtype=np.float64))
    return polylines


def _contour_features(
    elevations: np.ndarray[Any, np.dtype[np.float32]],
    extent: dict[str, float],
    bounds: GeographicBounds,
    contour_interval_meters: int,
    index_contour_interval_meters: int,
) -> list[dict[str, Any]]:
    """Convert an elevation grid into clipped GeoJSON contour features."""
    from matplotlib.figure import Figure

    height, width = elevations.shape
    longitude_step = (extent["xmax"] - extent["xmin"]) / width
    latitude_step = (extent["ymax"] - extent["ymin"]) / height
    longitudes = extent["xmin"] + longitude_step * (np.arange(width) + 0.5)
    latitudes = extent["ymin"] + latitude_step * (np.arange(height) + 0.5)
    minimum_level = (
        math.ceil(float(np.nanmin(elevations)) / contour_interval_meters)
        * contour_interval_meters
    )
    maximum_level = (
        math.floor(float(np.nanmax(elevations)) / contour_interval_meters)
        * contour_interval_meters
    )
    levels = np.arange(
        minimum_level,
        maximum_level + contour_interval_meters,
        contour_interval_meters,
    )

    figure = Figure()
    axes = figure.subplots()
    contour_set = axes.contour(
        longitudes,
        latitudes,
        np.flipud(elevations),
        levels=levels,
    )
    features = []
    for level, path in zip(contour_set.levels, contour_set.get_paths(), strict=True):
        lines = [
            piece
            for vertices in _iter_contour_path_polylines(path)
            for piece in clip_polyline_to_bounds(
                vertices=vertices,
                bounds=bounds,
            )
        ]
        if not lines:
            continue
        elevation_meters = int(level)
        features.append(
            {
                "type": "Feature",
                "id": f"contour/{elevation_meters}m",
                "properties": {
                    "elevation_m": elevation_meters,
                    "is_index": (elevation_meters % index_contour_interval_meters == 0),
                },
                "geometry": {
                    "type": "MultiLineString",
                    "coordinates": lines,
                },
            }
        )
    return features


def download_dartmouth_skiway_contours(
    bounds: GeographicBounds = SKIWAY_MAP_BOUNDS,
    *,
    contour_interval_meters: int = DARTMOUTH_CONTOUR_INTERVAL_METERS,
    index_contour_interval_meters: int = DARTMOUTH_INDEX_CONTOUR_INTERVAL_METERS,
    pixel_size_meters: int = DARTMOUTH_DEM_PIXEL_SIZE_METERS,
) -> Path:
    """
    Refresh static USGS 3DEP contours for the Dartmouth Skiway map extent.

    The temporary elevation raster is sampled slightly beyond `bounds` so contours
    can be clipped exactly at the requested edges.
    Only the resulting WGS 84 line geometry is saved in the repository.

    Alternative approaches considered:

    - The [USGS 3DEP ImageServer](https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer)
      contour function requires little local computation and accepts fixed intervals,
      but returns a raster that would remain rasterized in SVG and PDF output.
    - The [National Map contour MapServer](https://carto.nationalmap.gov/arcgis/rest/services/contours/MapServer)
      provides ready-made vectors, but its precomputed data can be older,
      its service proved unreliable, and its queries return long source features
      that still require clipping.
    - [Seamless3DEP](https://github.com/hyriver/seamless-3dep) and
      [Rasterio](https://github.com/rasterio/rasterio) provide maintained,
      reusable retrieval machinery, but add substantial dependencies without
      eliminating local vector generation and clipping.

    Downloading the current dynamic DEM and generating contours locally preserves
    exact interval control, current elevation data, and true vector output.
    """
    from PIL import Image

    if contour_interval_meters <= 0:
        raise ValueError("contour_interval_meters must be positive")
    if (
        index_contour_interval_meters <= 0
        or index_contour_interval_meters % contour_interval_meters
    ):
        raise ValueError(
            "index_contour_interval_meters must be a positive multiple of "
            "contour_interval_meters"
        )
    if pixel_size_meters <= 0:
        raise ValueError("pixel_size_meters must be positive")
    scale = meters_per_degree(bounds.midpoint_latitude)
    padding_meters = pixel_size_meters * 2
    request_bounds = {
        "west": bounds.west - padding_meters / scale.longitude,
        "east": bounds.east + padding_meters / scale.longitude,
        "south": bounds.south - padding_meters / scale.latitude,
        "north": bounds.north + padding_meters / scale.latitude,
    }
    width = math.ceil(
        (request_bounds["east"] - request_bounds["west"])
        * scale.longitude
        / pixel_size_meters
    )
    height = math.ceil(
        (request_bounds["north"] - request_bounds["south"])
        * scale.latitude
        / pixel_size_meters
    )
    export_parameters = {
        "bbox": ",".join(
            str(request_bounds[direction])
            for direction in ("west", "south", "east", "north")
        ),
        "bboxSR": "4326",
        "imageSR": "4326",
        "size": f"{width},{height}",
        "format": "tiff",
        "pixelType": "F32",
        "interpolation": "RSP_Bilinear",
        "adjustAspectRatio": "false",
        "f": "json",
    }
    export_response = requests.get(
        f"{DARTMOUTH_ELEVATION_SERVICE_URL}/exportImage",
        params=export_parameters,
        headers={"User-Agent": USER_AGENT},
        timeout=60,
    )
    export_response.raise_for_status()
    export = export_response.json()
    if "error" in export:
        raise RuntimeError(f"USGS elevation export failed: {export['error']}")
    raster_response = requests.get(
        export["href"],
        headers={"User-Agent": USER_AGENT},
        timeout=60,
    )
    raster_response.raise_for_status()
    with Image.open(io.BytesIO(raster_response.content)) as image:
        elevations = np.asarray(image, dtype=np.float32).copy()
    if elevations.ndim != 2:
        raise ValueError(
            f"Expected a single-band elevation raster, got {elevations.shape}"
        )

    extent = {
        direction: float(export["extent"][direction])
        for direction in ("xmin", "ymin", "xmax", "ymax")
    }
    features = _contour_features(
        elevations=elevations,
        extent=extent,
        bounds=bounds,
        contour_interval_meters=contour_interval_meters,
        index_contour_interval_meters=index_contour_interval_meters,
    )
    feature_collection = {
        "type": "FeatureCollection",
        "properties": {
            "source": DARTMOUTH_ELEVATION_SERVICE_URL,
            "source_name": "USGS 3D Elevation Program Bare Earth DEM",
            "downloaded_at": datetime.now(UTC).isoformat(),
            "horizontal_crs": bounds.crs,
            "vertical_crs": "EPSG:5703",
            "vertical_datum": "NAVD88",
            "vertical_units": "meters",
            "contour_interval_m": contour_interval_meters,
            "index_contour_interval_m": index_contour_interval_meters,
            "dem_pixel_size_m": pixel_size_meters,
            "bounds": {
                "west": bounds.west,
                "east": bounds.east,
                "south": bounds.south,
                "north": bounds.north,
            },
            "dem_elevation_range_m": [
                round(float(np.nanmin(elevations)), 3),
                round(float(np.nanmax(elevations)), 3),
            ],
            "refresh_command": "pixi run openskistats download_dartmouth_contours",
        },
        "features": features,
    }
    DARTMOUTH_CONTOURS_PATH.parent.mkdir(exist_ok=True)
    DARTMOUTH_CONTOURS_PATH.write_text(json.dumps(feature_collection, indent=2) + "\n")
    return DARTMOUTH_CONTOURS_PATH.relative_to(get_repo_directory())


def load_dartmouth_skiway_contours() -> dict[str, Any]:
    """Load the committed USGS contour snapshot without network access."""
    return cast(dict[str, Any], json.loads(DARTMOUTH_CONTOURS_PATH.read_text()))
