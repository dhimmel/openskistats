"""Tests for Dartmouth Skiway geographic data and plotting."""

from matplotlib.colors import to_rgba
from matplotlib.figure import Figure

from openskistats.skiway_data import (
    APPALACHIAN_TRAIL_OSM_ID,
    DARTMOUTH_CONTEXT_OSM_RELATION_IDS,
    DARTMOUTH_CONTEXT_OSM_WAY_IDS,
    DARTMOUTH_CONTOUR_INTERVAL_METERS,
    DARTMOUTH_INDEX_CONTOUR_INTERVAL_METERS,
    DARTMOUTH_SKIWAY_PARKING_LOT_OSM_ID,
    DARTMOUTH_SKIWAY_SERVICE_ROAD_OSM_ID,
    MCLANE_FAMILY_LODGE_OSM_ID,
    SKIWAY_MAP_BOUNDS,
    load_dartmouth_skiway_context,
    load_dartmouth_skiway_contours,
)
from openskistats.skiway_plot import (
    PARKING_COLOR,
    ROAD_COLOR,
    ROAD_LINEWIDTH,
    TRAIL_COLOR,
    TRAIL_LINEWIDTH,
    _plot_skiway_map_context,
)


def test_dartmouth_skiway_context_snapshot() -> None:
    context = load_dartmouth_skiway_context()
    features = context["features"]
    lodge_features = [
        feature
        for feature in features
        if feature["properties"]["feature_kind"] == "lodge"
    ]
    road_features = [
        feature
        for feature in features
        if feature["properties"]["feature_kind"] == "road"
    ]
    parking_road_features = [
        feature
        for feature in features
        if feature["properties"]["feature_kind"] == "parking_road"
    ]
    parking_features = [
        feature
        for feature in features
        if feature["properties"]["feature_kind"] == "parking"
    ]
    trail_features = [
        feature
        for feature in features
        if feature["properties"]["feature_kind"] == "trail"
    ]

    assert [feature["properties"]["osm_id"] for feature in lodge_features] == [
        MCLANE_FAMILY_LODGE_OSM_ID
    ]
    assert lodge_features[0]["geometry"]["type"] == "Polygon"
    assert (
        lodge_features[0]["geometry"]["coordinates"][0][0]
        == lodge_features[0]["geometry"]["coordinates"][0][-1]
    )
    assert [feature["properties"]["osm_id"] for feature in parking_features] == [
        DARTMOUTH_SKIWAY_PARKING_LOT_OSM_ID
    ]
    assert parking_features[0]["properties"]["amenity"] == "parking"
    assert parking_features[0]["geometry"]["type"] == "Polygon"
    assert (
        parking_features[0]["geometry"]["coordinates"][0][0]
        == parking_features[0]["geometry"]["coordinates"][0][-1]
    )
    assert [feature["properties"]["osm_id"] for feature in trail_features] == [
        APPALACHIAN_TRAIL_OSM_ID
    ]
    assert {feature["properties"]["osm_id"] for feature in trail_features} == set(
        DARTMOUTH_CONTEXT_OSM_RELATION_IDS
    )
    assert trail_features[0]["properties"]["ref"] == "AT"
    assert trail_features[0]["geometry"]["type"] == "MultiLineString"
    assert trail_features[0]["geometry"]["coordinates"]
    assert all(
        SKIWAY_MAP_BOUNDS.west <= longitude <= SKIWAY_MAP_BOUNDS.east
        and SKIWAY_MAP_BOUNDS.south <= latitude <= SKIWAY_MAP_BOUNDS.north
        for line in trail_features[0]["geometry"]["coordinates"]
        for longitude, latitude in line
    )
    assert [feature["properties"]["osm_id"] for feature in parking_road_features] == [
        DARTMOUTH_SKIWAY_SERVICE_ROAD_OSM_ID
    ]
    (service_road,) = parking_road_features
    assert service_road["properties"]["highway"] == "service"
    assert service_road["properties"]["surface"] == "asphalt"
    assert {feature["properties"]["osm_id"] for feature in road_features} == set(
        DARTMOUTH_CONTEXT_OSM_WAY_IDS
    ) - {
        MCLANE_FAMILY_LODGE_OSM_ID,
        DARTMOUTH_SKIWAY_PARKING_LOT_OSM_ID,
        DARTMOUTH_SKIWAY_SERVICE_ROAD_OSM_ID,
    }
    assert all(
        feature["properties"]["name"] == "Grafton Turnpike"
        and feature["geometry"]["type"] == "LineString"
        for feature in road_features
    )
    road_latitudes = [
        coordinate[1]
        for feature in road_features
        for coordinate in feature["geometry"]["coordinates"]
    ]
    assert min(road_latitudes) < 43.775
    assert max(road_latitudes) > 43.792


def test_dartmouth_skiway_contour_snapshot() -> None:
    contours = load_dartmouth_skiway_contours()
    properties = contours["properties"]
    assert properties["contour_interval_m"] == DARTMOUTH_CONTOUR_INTERVAL_METERS
    assert (
        properties["index_contour_interval_m"]
        == DARTMOUTH_INDEX_CONTOUR_INTERVAL_METERS
    )
    assert properties["horizontal_crs"] == SKIWAY_MAP_BOUNDS.crs
    assert properties["vertical_crs"] == "EPSG:5703"
    assert properties["vertical_datum"] == "NAVD88"
    assert properties["bounds"] == {
        "west": SKIWAY_MAP_BOUNDS.west,
        "east": SKIWAY_MAP_BOUNDS.east,
        "south": SKIWAY_MAP_BOUNDS.south,
        "north": SKIWAY_MAP_BOUNDS.north,
    }

    features = contours["features"]
    assert features
    assert all(
        feature["geometry"]["type"] == "MultiLineString"
        and feature["properties"]["elevation_m"] % DARTMOUTH_CONTOUR_INTERVAL_METERS
        == 0
        and feature["properties"]["is_index"]
        == (
            feature["properties"]["elevation_m"]
            % DARTMOUTH_INDEX_CONTOUR_INTERVAL_METERS
            == 0
        )
        for feature in features
    )
    coordinates = [
        coordinate
        for feature in features
        for line in feature["geometry"]["coordinates"]
        for coordinate in line
    ]
    assert all(
        SKIWAY_MAP_BOUNDS.west <= longitude <= SKIWAY_MAP_BOUNDS.east
        and SKIWAY_MAP_BOUNDS.south <= latitude <= SKIWAY_MAP_BOUNDS.north
        for longitude, latitude in coordinates
    )


def test_dartmouth_skiway_context_styles() -> None:
    figure = Figure()
    ax = figure.subplots()
    _plot_skiway_map_context(ax=ax, map_context=load_dartmouth_skiway_context())

    parking_patch = next(patch for patch in ax.patches if patch.get_zorder() == -0.5)
    road_lines = [line for line in ax.lines if line.get_linewidth() == ROAD_LINEWIDTH]
    parking_road_lines = [
        line for line in road_lines if line.get_color() == PARKING_COLOR
    ]
    trail_lines = [line for line in ax.lines if line.get_linewidth() == TRAIL_LINEWIDTH]

    assert parking_patch.get_facecolor() == to_rgba(PARKING_COLOR)
    assert road_lines
    assert len(parking_road_lines) == 1
    assert trail_lines
    assert parking_patch.get_zorder() < min(line.get_zorder() for line in road_lines)
    assert TRAIL_COLOR == ROAD_COLOR
    assert TRAIL_LINEWIDTH < ROAD_LINEWIDTH
    assert all(line.get_color() == TRAIL_COLOR for line in trail_lines)
