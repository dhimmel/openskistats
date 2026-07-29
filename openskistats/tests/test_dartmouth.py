from openskistats.dartmouth import (
    DARTMOUTH_CONTEXT_OSM_WAY_IDS,
    MCLANE_FAMILY_LODGE_OSM_ID,
    load_dartmouth_skiway_context,
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
