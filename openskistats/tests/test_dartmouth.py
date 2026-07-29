from openskistats.dartmouth import (
    DARTMOUTH_CONTEXT_OSM_WAY_IDS,
    DARTMOUTH_CONTOUR_INTERVAL_METERS,
    MCLANE_FAMILY_LODGE_OSM_ID,
    load_dartmouth_skiway_context,
    load_dartmouth_skiway_contours,
)
from openskistats.plot_dartmouth import SKIWAY_MAP_BOUNDS


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

    assert [feature["properties"]["osm_id"] for feature in lodge_features] == [
        MCLANE_FAMILY_LODGE_OSM_ID
    ]
    assert lodge_features[0]["geometry"]["type"] == "Polygon"
    assert (
        lodge_features[0]["geometry"]["coordinates"][0][0]
        == lodge_features[0]["geometry"]["coordinates"][0][-1]
    )
    assert {feature["properties"]["osm_id"] for feature in road_features} == set(
        DARTMOUTH_CONTEXT_OSM_WAY_IDS
    ) - {MCLANE_FAMILY_LODGE_OSM_ID}
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
