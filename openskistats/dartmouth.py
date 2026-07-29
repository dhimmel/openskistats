"""Static OpenStreetMap context for the Dartmouth Skiway figure."""

import json
from pathlib import Path
from typing import Any, cast

import requests

from openskistats.utils import get_repo_directory

DARTMOUTH_CONTEXT_PATH = Path(__file__).parent.joinpath(
    "data", "dartmouth_skiway_context.geojson"
)
DARTMOUTH_CONTEXT_OSM_WAY_IDS = (
    296382919,  # McLane Family Lodge
    328497225,  # Grafton Turnpike
    532980281,  # Grafton Turnpike
    532980282,  # Grafton Turnpike continuation
    1431999174,  # Grafton Turnpike connector
)
MCLANE_FAMILY_LODGE_OSM_ID = 296382919
USER_AGENT = "openskistats/0.1 (https://github.com/dhimmel/openskistats)"


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
    is_lodge = osm_id == MCLANE_FAMILY_LODGE_OSM_ID
    geometry_type = "Polygon" if is_lodge else "LineString"
    geometry_coordinates = [coordinates] if is_lodge else coordinates
    return {
        "type": "Feature",
        "id": f"way/{osm_id}",
        "properties": {
            "feature_kind": "lodge" if is_lodge else "road",
            "osm_id": osm_id,
            "osm_version": way["version"],
            "osm_timestamp": way["timestamp"],
            "name": tags["name"],
            "building": tags.get("building"),
            "highway": tags.get("highway"),
            "surface": tags.get("surface"),
            "source": f"https://www.openstreetmap.org/way/{osm_id}",
        },
        "geometry": {
            "type": geometry_type,
            "coordinates": geometry_coordinates,
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
